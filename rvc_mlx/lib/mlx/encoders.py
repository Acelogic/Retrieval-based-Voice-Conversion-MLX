import math
import mlx.core as mx
import mlx.nn as nn
from rvc_mlx.lib.mlx.commons import sequence_mask
from rvc_mlx.lib.mlx.modules import WaveNet
from rvc_mlx.lib.mlx.attentions import FFN, MultiHeadAttention

class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 10,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = []
        self.norm_layers_1 = []
        self.ffn_layers = []
        self.norm_layers_2 = []
        
        for _ in range(n_layers):
            self.attn_layers.append(
                MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, window_size=window_size)
            )
            self.norm_layers_1.append(nn.LayerNorm(hidden_channels))
            self.ffn_layers.append(
                 FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout)
            )
            self.norm_layers_2.append(nn.LayerNorm(hidden_channels))
            
        for i, l in enumerate(self.attn_layers): setattr(self, f"attn_{i}", l)
        for i, l in enumerate(self.norm_layers_1): setattr(self, f"norm1_{i}", l)
        for i, l in enumerate(self.ffn_layers): setattr(self, f"ffn_{i}", l)
        for i, l in enumerate(self.norm_layers_2): setattr(self, f"norm2_{i}", l)

    def __call__(self, x, x_mask):
        # x: (N, L, C)
        # attn_mask from x_mask: (N, 1, T, T) or similar
        # x_mask is (N, 1, L) usually? Or (N, L)?
        # commons.sequence_mask returns boolean (N, L)?? Check commons.py port.
        # commons.py: return x[None, :] < length[:, None]. That's (1, MAX) < (B, 1) -> (B, MAX). Yes (B, L).
        
        # Attn mask calculation
        # PyTorch: input x_mask (B, 1, L). 
        # MLX: let's assume x_mask is (B, L, 1) or (B, L) compatible.
        
        # attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        # (B, 1, L) * (B, L, 1) -> (B, L, L)
        
        # MLX: if x_mask is (B, L)
        # x_mask[:, None, :] * x_mask[:, :, None] -> (B, 1, L) * (B, L, 1) -> (B, L, L)
        # Then expand to (B, 1, L, L) for heads?
        
        x_mask_b = x_mask.astype(mx.float32)

        if x_mask.ndim == 2:
             # x_mask is (B, L)
             attn_mask = x_mask_b[:, None, :] * x_mask_b[:, :, None]  # (B, 1, L) * (B, L, 1) -> (B, L, L)
             # Add head dim -> (B, 1, L, L)
             attn_mask = attn_mask[:, None, :, :]
        elif x_mask.ndim == 3:
             # x_mask is (B, L, 1)
             attn_mask = x_mask_b * x_mask_b.transpose(0, 2, 1)  # (B, L, 1) * (B, 1, L) -> (B, L, L)
             attn_mask = attn_mask[:, None, :, :]  # -> (B, 1, L, L)
             
        x = x * x_mask # broadcast
        
        for i in range(self.n_layers):
            attn = getattr(self, f"attn_{i}")
            norm1 = getattr(self, f"norm1_{i}")
            ffn = getattr(self, f"ffn_{i}")
            norm2 = getattr(self, f"norm2_{i}")
            
            y = attn(x, x, attn_mask=attn_mask)
            y = self.drop(y)
            x = norm1(x + y)
            
            y = ffn(x, x_mask)
            y = self.drop(y)
            x = norm2(x + y)
            
        return x * x_mask

class TextEncoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        embedding_dim: int,
        f0: bool = True,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.emb_phone = nn.Linear(embedding_dim, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.1) # inplace=True in torch, auto in mlx
        
        if f0:
            self.emb_pitch = nn.Embedding(256, hidden_channels)
        else:
            self.emb_pitch = None
            
        self.encoder = Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def __call__(self, phone, pitch, lengths):
        # phone: (B, L, D) - phone embeddings from HuBERT
        
        x = self.emb_phone(phone)

        if pitch is not None and self.emb_pitch is not None:
            p_emb = self.emb_pitch(pitch)
            x = x + p_emb
            
        x = x * math.sqrt(self.hidden_channels)
        x = self.lrelu(x)

        # Keep (B, T, C) format for MLX
        # x_mask: (B, T)
        x_mask = sequence_mask(lengths, x.shape[1])
        # Unsqueeze for broadcasting: (B, T, 1)
        x_mask_expanded = x_mask[:, :, None]

        x = self.encoder(x, x_mask_expanded)

        # proj is Conv1d: input (B, T, C), output (B, T, Out*2)
        stats = self.proj(x) * x_mask_expanded

        # Transpose to PyTorch format (B, C, T) before splitting
        stats = stats.transpose(0, 2, 1)  # (B, T, C*2) -> (B, C*2, T)

        # Split on channel dimension (axis=1) to match PyTorch
        m, logs = mx.split(stats, 2, axis=1)

        # Return in PyTorch format: (B, C, T) for m and logs, (B, 1, T) for x_mask
        # Add dimension to x_mask to match PyTorch: (B, T) -> (B, 1, T)
        x_mask_out = x_mask[:, None, :]
        return m, logs, x_mask_out

class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WaveNet(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        
    def __call__(self, x, x_lengths, g=None):
        # x input: (B, L, C)
        x_mask = sequence_mask(x_lengths, x.shape[1])
        x_mask_b = x_mask[:, :, None] # (B, L, 1)
        
        x = self.pre(x) * x_mask_b
        x = self.enc(x, x_mask_b, g=g)
        stats = self.proj(x) * x_mask_b
        
        m, logs = mx.split(stats, 2, axis=-1)
        z = m + mx.random.normal(m.shape).astype(m.dtype) * mx.exp(logs)
        z = z * x_mask_b
        return z, m, logs, x_mask
