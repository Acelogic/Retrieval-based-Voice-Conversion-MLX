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
            sr,
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
            inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels
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

        g = self.emb_g(sid)[:, None, :]  # unsqueeze -1 -> (B, 1, C)

        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)

        # Expanding m_p with noise
        z_p = (
            m_p
            + mx.exp(logs_p) * mx.random.normal(m_p.shape).astype(m_p.dtype) * 0.66666
        ) * x_mask[:, :, None]

        # Rate / time stretching
        if rate is not None:
            # Logic from original:
            # head = int(z_p.shape[2] * (1.0 - rate.item())) -> shape[2] is length in orig (B, C, L)
            # Here shape[1] is length (B, L, C)
            head = int(z_p.shape[1] * (1.0 - rate.item()))
            z_p = z_p[:, head:, :]
            x_mask = x_mask[:, head:]
            if self.use_f0 and nsff0 is not None:
                nsff0 = nsff0[:, head:]

        # Flow reverse
        # z = self.flow(z_p, x_mask, g=g, reverse=True)
        # Mask needs to be (B, L, 1) for flow
        z = self.flow(z_p, x_mask[:, :, None], g=g, reverse=True)

        # Decoder
        # o = self.dec(z * x_mask, nsff0, g=g)
        # z: (B, L, C)
        o = self.dec(z * x_mask[:, :, None], nsff0, g=g)

        return o, x_mask, (z, z_p, m_p, logs_p)
