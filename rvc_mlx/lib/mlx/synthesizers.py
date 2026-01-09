import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional
from rvc_mlx.lib.mlx.generators import HiFiGANNSFGenerator
from rvc_mlx.lib.mlx.encoders import TextEncoder, PosteriorEncoder
from rvc_mlx.lib.mlx.residuals import ResidualCouplingBlock
from rvc_mlx.lib.mlx.commons import rand_slice_segments, slice_segments

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
        
    def __call__(
        self,
        phone: mx.array,
        phone_lengths: mx.array,
        pitch: mx.array,
        pitchf: mx.array,
        y: mx.array,
        y_lengths: mx.array,
        ds: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array, Tuple]:
        """
        Training forward pass.

        Args:
            phone: Phone/content features (B, T, 768)
            phone_lengths: Phone sequence lengths (B,)
            pitch: Pitch features (B, T)
            pitchf: Pitch f0 (continuous) (B, T)
            y: Target spectrogram (B, spec_channels, T_spec)
            y_lengths: Spectrogram lengths (B,)
            ds: Speaker ID (B,) - optional, use default if not provided

        Returns:
            Tuple of:
                - o: Generated audio (B, T_audio, 1)
                - ids_slice: Slice indices used
                - x_mask: Input mask
                - y_mask: Output mask
                - (z, z_p, m_p, logs_p, m_q, logs_q): Latent variables for losses
        """
        # Speaker embedding
        if ds is not None:
            g = self.emb_g(ds)[:, None, :]  # (B, 1, gin_channels)
        else:
            g = None

        # Text encoder: encode phone features
        # m_p, logs_p: prior mean and log variance
        # Returns (B, C, T) format
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)

        # Posterior encoder: encode target spectrogram
        # y is (B, spec_channels, T_spec), needs to be (B, T_spec, spec_channels) for MLX
        y_mlx = y.transpose(0, 2, 1)  # (B, T, C)
        z, m_q, logs_q, y_mask = self.enc_q(y_mlx, y_lengths, g=g)
        # enc_q returns (B, T, C) format, convert to (B, C, T) for consistency
        z = z.transpose(0, 2, 1)
        m_q = m_q.transpose(0, 2, 1)
        logs_q = logs_q.transpose(0, 2, 1)
        # y_mask is (B, T), expand to (B, 1, T) for broadcasting
        y_mask = y_mask[:, None, :]  # (B, 1, T)

        # Flow: transform posterior to prior space
        # Flow expects (B, T, C) format
        z_flow = z.transpose(0, 2, 1)  # (B, C, T) -> (B, T, C)
        # y_mask is (B, 1, T), need (B, T, 1) for flow
        mask_flow = y_mask.transpose(0, 2, 1)  # (B, 1, T) -> (B, T, 1)
        z_p = self.flow(z_flow, mask_flow, g=g, reverse=False)
        z_p = z_p.transpose(0, 2, 1)  # Back to (B, C, T)

        # Random slice for decoder (training uses segments)
        # self.segment_size is already in frames (not samples)
        # It's set from the model config, typically 32 for RVC
        segment_size_frames = self.segment_size

        # Ensure we have enough frames
        z_lengths = y_lengths
        # z is in (B, C, T) PyTorch format, so use time_first=False
        z_slice, ids_slice = rand_slice_segments(
            z, z_lengths, segment_size_frames, time_first=False
        )

        # Also slice pitch features - pitchf is (B, T), expand and slice
        # pitchf[:, None, :] gives (B, 1, T) in PyTorch format
        pitchf_slice = slice_segments(
            pitchf[:, None, :] if pitchf.ndim == 2 else pitchf,
            ids_slice,
            segment_size_frames,
            time_first=False  # PyTorch format (B, C, T)
        )
        if pitchf_slice.ndim == 3:
            pitchf_slice = pitchf_slice.squeeze(1)

        # Decoder: generate audio from sliced latents
        # z_slice is (B, C, segment_frames)
        o = self.dec(z_slice, pitchf_slice, g=g)

        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def forward_generator_loss(
        self,
        phone: mx.array,
        phone_lengths: mx.array,
        pitch: mx.array,
        pitchf: mx.array,
        y: mx.array,
        y_lengths: mx.array,
        ds: Optional[mx.array] = None,
    ):
        """
        Forward pass returning values needed for generator loss computation.
        Same as __call__ but with clearer naming for training loops.
        """
        return self(phone, phone_lengths, pitch, pitchf, y, y_lengths, ds)

    # Alias for training compatibility
    forward = forward_generator_loss

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
