"""
FCPE (Fast Context-aware Pitch Estimation) - MLX Implementation Stub.

FCPE uses a PCmer (Pitch Conformer) transformer architecture for pitch estimation.
It provides state-of-the-art pitch detection with fast inference.

Architecture:
- Mel-spectrogram input (128 bins)
- Conv1d stack for initial processing
- PCmer transformer (12 layers) with:
  - LocalAttention (windowed attention)
  - FastAttention (kernel-based approximation)
  - ConformerConvModule (GLU + depthwise conv)
- Output: 360 pitch bin probabilities

Reference: torchfcpe - https://github.com/CNChTu/FCPE

NOTE: This is a stub implementation. Full transformer porting is complex.
For now, it falls back to RMVPE when available.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
import warnings

# Constants
PITCH_BINS = 360
SAMPLE_RATE = 16000
HOP_SIZE = 256  # FCPE uses 256 (16ms at 16kHz)
N_MELS = 128

# Frequency conversion
FMIN = 32.70  # C1
FMAX = 1975.5  # B6


def f0_to_cent(f0: np.ndarray) -> np.ndarray:
    """Convert F0 in Hz to cents."""
    return 1200.0 * np.log2(np.maximum(f0, 1e-5) / 10.0)


def cent_to_f0(cent: np.ndarray) -> np.ndarray:
    """Convert cents to F0 in Hz."""
    return 10.0 * (2 ** (cent / 1200.0))


class FCPE:
    """
    FCPE pitch extraction interface.

    This is a stub implementation that uses RMVPE as fallback.
    Full FCPE transformer model porting is complex and in progress.

    Example:
        >>> fcpe = FCPE()
        >>> f0 = fcpe.get_f0(audio, f0_min=50, f0_max=1100)
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        use_fallback: bool = True,
    ):
        """
        Initialize FCPE.

        Args:
            weights_path: Path to FCPE weights (not yet supported)
            use_fallback: If True, use RMVPE as fallback
        """
        self._model = None
        self._fallback = None
        self.use_fallback = use_fallback

        # Try to load weights
        if weights_path and Path(weights_path).exists():
            self._load_weights(weights_path)
        else:
            # Use RMVPE fallback
            if use_fallback:
                self._init_fallback()
            else:
                warnings.warn(
                    "FCPE weights not found and fallback disabled. "
                    "Run: python tools/convert_fcpe_weights.py"
                )

    def _init_fallback(self):
        """Initialize RMVPE as fallback."""
        try:
            from rvc_mlx.lib.mlx.rmvpe import RMVPE0Predictor
            self._fallback = RMVPE0Predictor()
            warnings.warn(
                "FCPE using RMVPE as fallback. "
                "Full FCPE model not yet implemented in MLX."
            )
        except (ImportError, Exception) as e:
            warnings.warn(f"RMVPE fallback not available: {e}")

    def _load_weights(self, weights_path: str):
        """Load FCPE weights (not yet implemented)."""
        warnings.warn("FCPE weight loading not yet implemented")

    def get_f0(
        self,
        audio: np.ndarray,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
        threshold: float = 0.006,
        decoder_mode: str = "local_argmax",
        **kwargs,
    ) -> np.ndarray:
        """
        Extract F0 from audio.

        Args:
            audio: Audio signal (1D numpy array, 16kHz)
            f0_min: Minimum F0 to detect (Hz)
            f0_max: Maximum F0 to detect (Hz)
            threshold: Confidence threshold (default 0.006 for FCPE)
            decoder_mode: "argmax" or "local_argmax"

        Returns:
            F0 contour (T,) in Hz, 0 for unvoiced
        """
        # Use fallback if available
        if self._fallback is not None:
            f0 = self._fallback.infer_from_audio(audio, thred=threshold * 5)  # Scale threshold
            return f0

        # Native FCPE implementation (stub - needs full model)
        raise NotImplementedError(
            "Full FCPE MLX model not yet implemented. "
            "Use use_fallback=True or wait for full implementation."
        )

    def infer(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        decoder_mode: str = "local_argmax",
        threshold: float = 0.006,
    ) -> np.ndarray:
        """
        Inference interface matching torchfcpe.

        Args:
            audio: Audio signal
            sr: Sample rate
            decoder_mode: Decoder mode
            threshold: Confidence threshold

        Returns:
            F0 contour in Hz
        """
        return self.get_f0(audio, threshold=threshold, decoder_mode=decoder_mode)


# ============================================================================
# Planned architecture components (stubs for future implementation)
# ============================================================================

class Wav2Mel:
    """
    Audio to Mel-spectrogram conversion.

    Uses STFT with:
    - n_fft: 1024
    - hop_size: 256
    - n_mels: 128
    - fmin: 40 Hz
    - fmax: 7600 Hz
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_size: int = 256,
        n_mels: int = 128,
        fmin: float = 40.0,
        fmax: float = 7600.0,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

        # Create mel filterbank
        import librosa
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert audio to mel-spectrogram.

        Args:
            audio: Audio signal (N,)

        Returns:
            Mel-spectrogram (T, n_mels)
        """
        import librosa

        # STFT
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.n_fft,
            window='hann',
            center=True,
        )
        magnitude = np.abs(stft)

        # Apply mel filterbank
        mel = self.mel_basis @ magnitude

        # Log compression
        log_mel = np.log(np.maximum(mel, 1e-5))

        return log_mel.T  # (T, n_mels)


class ConformerConvModule(nn.Module):
    """
    Conformer convolution module.

    Architecture:
    - LayerNorm
    - Pointwise conv (expand)
    - GLU activation
    - Depthwise conv
    - Swish activation
    - Pointwise conv (project)
    """

    def __init__(self, dim: int, expansion_factor: int = 2, kernel_size: int = 31):
        super().__init__()
        inner_dim = dim * expansion_factor
        self.norm = nn.LayerNorm(dim)
        self.pointwise_in = nn.Linear(dim, inner_dim * 2)
        self.depthwise = nn.Conv1d(
            inner_dim, inner_dim, kernel_size,
            padding=kernel_size // 2, groups=inner_dim
        )
        self.pointwise_out = nn.Linear(inner_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass."""
        x = self.norm(x)
        x = self.pointwise_in(x)

        # GLU activation
        x, gate = mx.split(x, 2, axis=-1)
        x = x * mx.sigmoid(gate)

        # Depthwise conv (requires transpose for Conv1d)
        x = x.transpose(0, 2, 1)  # (B, T, C) -> (B, C, T)
        x = self.depthwise(x)
        x = x.transpose(0, 2, 1)  # (B, C, T) -> (B, T, C)

        # Swish
        x = x * mx.sigmoid(x)

        x = self.pointwise_out(x)
        return x


class LocalAttentionModule(nn.Module):
    """
    Local attention with windowed attention.

    Note: Full implementation requires custom attention mask handling.
    """

    def __init__(self, dim: int, heads: int = 8, window_size: int = 256):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.window_size = window_size
        self.head_dim = dim // heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """Forward pass - standard attention for now."""
        B, T, C = x.shape

        # QKV projection
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, T, 3, self.heads, self.head_dim)
        qkv = qkv.transpose(0, 2, 3, 1, 4)  # (B, 3, H, T, D)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale

        if mask is not None:
            attn = attn + mask

        attn = mx.softmax(attn, axis=-1)
        out = attn @ v

        # Reshape and project
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        out = self.proj(out)

        return out


class FCPEModel(nn.Module):
    """
    Full FCPE model architecture (stub).

    Note: Complete implementation requires:
    - Local attention with proper windowing
    - Fast attention kernel
    - Weight loading from torchfcpe
    """

    def __init__(
        self,
        n_mels: int = 128,
        hidden_dim: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
    ):
        super().__init__()

        # Input projection
        self.input_stack = nn.Sequential(
            nn.Conv1d(n_mels, hidden_dim, kernel_size=3, padding=1),
            # GroupNorm not directly available in MLX
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )

        # Transformer layers (simplified)
        self.layers = [
            {
                'attn': LocalAttentionModule(hidden_dim, n_heads),
                'conv': ConformerConvModule(hidden_dim),
                'norm': nn.LayerNorm(hidden_dim),
            }
            for _ in range(n_layers)
        ]

        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, PITCH_BINS)

    def __call__(self, mel: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            mel: Mel-spectrogram (B, T, n_mels)

        Returns:
            Pitch probabilities (B, T, 360)
        """
        # Input projection
        x = mel.transpose(0, 2, 1)  # (B, T, C) -> (B, C, T)
        x = self.input_stack(x)
        x = x.transpose(0, 2, 1)  # (B, C, T) -> (B, T, C)

        # Transformer layers
        for layer in self.layers:
            x = x + layer['attn'](layer['norm'](x))
            x = x + layer['conv'](x)

        # Output
        x = self.output_norm(x)
        x = self.output_linear(x)
        x = mx.sigmoid(x)

        return x
