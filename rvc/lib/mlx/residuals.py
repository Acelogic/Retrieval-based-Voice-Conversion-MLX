import mlx.core as mx
import mlx.nn as nn
from rvc.lib.mlx.modules import WaveNet

LRELU_SLOPE = 0.1

class ResBlock(nn.Module):
    def __init__(
        self, channels: int, kernel_size: int = 3, dilations = (1, 3, 5)
    ):
        super().__init__()
        self.convs1 = []
        self.convs2 = []
        
        for d in dilations:
            # Padding calculation
            # (K*D - D)//2
            p = (kernel_size * d - d) // 2
            self.convs1.append(nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=p))
            self.convs2.append(nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=(kernel_size-1)//2))

        # Register as attributes
        for i, c in enumerate(self.convs1): setattr(self, f"c1_{i}", c)
        for i, c in enumerate(self.convs2): setattr(self, f"c2_{i}", c)

    def __call__(self, x, x_mask=None):
        for i in range(len(self.convs1)):
            c1 = getattr(self, f"c1_{i}")
            c2 = getattr(self, f"c2_{i}")
            
            x_residual = x
            x = nn.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None: x = x * x_mask
            
            x = c1(x)
            x = nn.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None: x = x * x_mask
            
            x = c2(x)
            x = x + x_residual
            
        if x_mask is not None: x = x * x_mask
        return x

class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.flows = []
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            # Flip is implicitly handled or we explicit module?
            # Original uses Flip() module. Logic: x.flip([1]) -> Time or Channel?
            # PyTorch: flip(x, [1]). x is (B, C, T) usually. So it flips Channels.
            # MLX x is (B, T, C). So we flip axis 2 (last axis).
            
        for i, f in enumerate(self.flows): setattr(self, f"flow_{i}", f)
        self.n_flows = n_flows

    def __call__(self, x, x_mask, g=None, reverse=False):
        # x: (N, L, C)
        
        iterator = range(self.n_flows)
        if reverse:
            iterator = reversed(iterator)
            
        for i in iterator:
            flow = getattr(self, f"flow_{i}")
            x, _ = flow(x, x_mask, g=g, reverse=reverse)
            
            # Flip logic after each flow step (except maybe last? check source. Source adds Flip module to list.)
            # Original: flows.append(Layer); flows.append(Flip)
            # My loop: implicit flip?
            # Wait, PyTorch ModuleList approach iterates all. 
            # I should explicitly do flip.
            # x is (N, L, C). Flip C -> axis=2
            x = mx.flip(x, axis=2) 
            
        return x

class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        p_dropout: float = 0,
        gin_channels: int = 0,
        mean_only: bool = False,
    ):
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only
        
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WaveNet(
             hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, p_dropout=p_dropout
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        
    def __call__(self, x, x_mask, g=None, reverse=False):
        # x: (N, L, C) -> transform to half channels axis=2
        x0 = x[:, :, :self.half_channels]
        x1 = x[:, :, self.half_channels:]
        
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        
        if not self.mean_only:
            m = stats[:, :, :self.half_channels]
            logs = stats[:, :, self.half_channels:]
        else:
            m = stats
            logs = mx.zeros_like(m)
            
        if not reverse:
            x1 = m + x1 * mx.exp(logs) * x_mask
            x = mx.concatenate([x0, x1], axis=2)
            logdet = mx.sum(logs, axis=[1, 2]) # Is this needed for inference? Usually no.
            return x, logdet
        else:
            x1 = (x1 - m) * mx.exp(-logs) * x_mask
            x = mx.concatenate([x0, x1], axis=2)
            return x, None
