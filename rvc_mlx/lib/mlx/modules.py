import mlx.core as mx
import mlx.nn as nn
from rvc_mlx.lib.mlx.commons import fused_add_tanh_sigmoid_multiply


class WaveNet(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd for proper padding."

        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        # MLX doesn't registers buffers in same way, but we can keep it as attribute
        self.n_channels_tensor = mx.array([hidden_channels])

        self.in_layers = []
        self.res_skip_layers = []
        # Dropout in inference is usually no-op, but MLX has nn.Dropout
        self.drop = nn.Dropout(p_dropout)

        if gin_channels:
            self.cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)

        dilations = [dilation_rate**i for i in range(n_layers)]
        paddings = [(kernel_size * d - d) // 2 for d in dilations]

        for i in range(n_layers):
            self.in_layers.append(
                nn.Conv1d(
                    hidden_channels,
                    2 * hidden_channels,
                    kernel_size,
                    stride=1,
                    padding=paddings[i],
                    dilation=dilations[i],
                )
            )
            res_skip_channels = (
                hidden_channels if i == n_layers - 1 else 2 * hidden_channels
            )
            self.res_skip_layers.append(
                nn.Conv1d(hidden_channels, res_skip_channels, 1)
            )

        # In MLX we explicitly add layers to self.layers or assign to attributes to register params
        # (Though assigning to a list isn't enough for auto-registration in Module.parameters(),
        # we need to assign to self attributes specifically or use ModuleList equivalent if available,
        # but MLX doesn't have ModuleList. We usually just use lists and manual iteration or setattr.)
        # Wait, simple list assignment DOES NOT register properties in MLX.
        # We need to assign them to attributes like self.layer_0, etc. or use a custom list wrapper.
        # The Pythonic way in MLX is often:
        self._in_layers = self.in_layers  # internal list
        for i, layer in enumerate(self.in_layers):
            setattr(self, f"in_layer_{i}", layer)

        self._res_skip_layers = self.res_skip_layers
        for i, layer in enumerate(self.res_skip_layers):
            setattr(self, f"res_skip_layer_{i}", layer)

    def __call__(self, x, x_mask, g=None):
        output = x  # Clone not needed for functional-like updates but variable reuse implies we track it.
        # Initialize output accumulator
        output_acc = mx.zeros_like(x)

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            # Access layers
            in_layer = getattr(self, f"in_layer_{i}")
            res_skip_layer = getattr(self, f"res_skip_layer_{i}")

            x_in = in_layer(x)

            g_l = 0
            if g is not None:
                g_l = g[
                    :,
                    :,
                    i * 2 * self.hidden_channels : (i + 1) * 2 * self.hidden_channels,
                ]

            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, self.n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = res_skip_layer(acts)

            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :, : self.hidden_channels]
                x = (x + res_acts) * x_mask
                output_acc = output_acc + res_skip_acts[:, :, self.hidden_channels :]
            else:
                output_acc = output_acc + res_skip_acts

        return output_acc * x_mask

    def remove_weight_norm(self):
        # Assumed handled by converter
        pass
