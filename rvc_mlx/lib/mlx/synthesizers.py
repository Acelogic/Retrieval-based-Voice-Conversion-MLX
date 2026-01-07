import mlx.core as mx
import mlx.nn as nn
from rvc_mlx.lib.mlx.generators import HiFiGANNSFGenerator
from rvc_mlx.lib.mlx.encoders import TextEncoder, PosteriorEncoder
from rvc_mlx.lib.mlx.residuals import ResidualCouplingBlock

class Synthesizer(nn.Module):
    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        spk_embed_dim: int,
        gin_channels: int,
        sr: int,
        use_f0: bool,
        text_enc_hidden_dim: int = 768,
        vocoder: str = "HiFi-GAN",
        **kwargs,
    ):
        super().__init__()
        self.segment_size = segment_size
        self.use_f0 = use_f0
        
        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            text_enc_hidden_dim,
            f0=use_f0,
        )
        
        # We only support HiFiGANNSFGenerator for this port as it's the primary one used.
        # Add checks if needed later.
        self.dec = HiFiGANNSFGenerator(
             inter_channels,
             resblock_kernel_sizes,
             resblock_dilation_sizes,
             upsample_rates,
             upsample_initial_channel,
             upsample_kernel_sizes,
             gin_channels,
             sr
        )
        
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        
        self.flow = ResidualCouplingBlock(
             inter_channels,
             hidden_channels,
             5,
             1,
             3,
             gin_channels=gin_channels
        )
        
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)
        
    def __call__(self, *args, **kwargs):
        # Forward pass (training logic) - usually not needed for inference loop in RVC 
        # unless full pipeline is run.
        # RVC Inference calls `infer()` directly usually? 
        # Let's check original. Original has `forward` and `@torch.jit.export infer`.
        # infer() is what's used.
        pass
        
    def infer(self, phone, phone_lengths, pitch, nsff0, sid, rate=None):
        # phone: (B, L, Dim)
        # pitch: (B, L)
        # nsff0: (B, L)
        # sid: (B,)

        g = self.emb_g(sid)[:, None, :] # unsqueeze -1 -> (B, 1, C)

        # enc_p returns: m_p (B, C, T), logs_p (B, C, T), x_mask (B, 1, T)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)

        # Expanding m_p with noise
        # m_p: (B, C, T), x_mask: (B, 1, T) - broadcasts correctly
        z_p = (m_p + mx.exp(logs_p) * mx.random.normal(m_p.shape).astype(m_p.dtype) * 0.0) * x_mask
        print(f"DEBUG: TextEncoder output - m_p: {m_p.shape} [{m_p.min().item():.6f}...{m_p.max().item():.6f}], logs_p: {logs_p.shape} [{logs_p.min().item():.6f}...{logs_p.max().item():.6f}], x_mask: {x_mask.shape}")
        print(f"DEBUG: z_p shape: {z_p.shape}, stats: min {z_p.min().item():.6f}, max {z_p.max().item():.6f}")

        # Rate / time stretching
        if rate is not None:
             head = int(z_p.shape[2] * (1.0 - rate.item()))
             z_p = z_p[:, :, head:]
             x_mask = x_mask[:, :, head:]
             if self.use_f0 and nsff0 is not None:
                 nsff0 = nsff0[:, head:]

        # Flow reverse
        # Flow expects (B, T, C) format, but we have (B, C, T)
        # Convert z_p from (B, C, T) to (B, T, C)
        z_p_mlx = z_p.transpose(0, 2, 1)  # (B, C, T) -> (B, T, C)
        x_mask_mlx = x_mask.transpose(0, 2, 1)  # (B, 1, T) -> (B, T, 1)

        z_mlx = self.flow(z_p_mlx, x_mask_mlx, g=g, reverse=True)

        # Convert back to (B, C, T) for decoder
        z = z_mlx.transpose(0, 2, 1)  # (B, T, C) -> (B, C, T)

        # Decoder
        # Input: z (B, C, T), x_mask (B, 1, T)
        # Output: o (B, T, 1) from generator
        o = self.dec(z * x_mask, nsff0, g=g)

        # o is already in (B, T, 1) format from generator
        return o, x_mask, (z, z_p, m_p, logs_p)
