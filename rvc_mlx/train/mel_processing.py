"""
Mel Spectrogram Processing for RVC MLX Training

MLX implementation of STFT and mel spectrogram computation.
"""

import mlx.core as mx
import numpy as np
from typing import Optional, Tuple
import librosa


# Cache for mel filterbank
_mel_basis_cache = {}


def get_mel_basis(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: Optional[float],
) -> mx.array:
    """
    Get or create mel filterbank.

    Args:
        sample_rate: Audio sample rate
        n_fft: FFT size
        n_mels: Number of mel bands
        fmin: Minimum frequency
        fmax: Maximum frequency (None = sample_rate / 2)

    Returns:
        Mel filterbank matrix (n_mels, n_fft // 2 + 1)
    """
    key = (sample_rate, n_fft, n_mels, fmin, fmax)
    if key not in _mel_basis_cache:
        mel = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )
        _mel_basis_cache[key] = mx.array(mel.astype(np.float32))
    return _mel_basis_cache[key]


def hann_window(window_length: int) -> mx.array:
    """Create Hann window."""
    n = mx.arange(window_length, dtype=mx.float32)
    return 0.5 - 0.5 * mx.cos(2.0 * np.pi * n / window_length)


def reflect_pad_1d(x: mx.array, pad_left: int, pad_right: int) -> mx.array:
    """
    Apply 1D reflect padding to array.

    For array [a, b, c, d] with pad_left=2, pad_right=1:
    Result: [c, b, a, b, c, d, c]

    Args:
        x: Input array (T,) or (B, T)
        pad_left: Left padding amount
        pad_right: Right padding amount

    Returns:
        Padded array
    """
    if x.ndim == 1:
        x = x[None, :]
        squeeze = True
    else:
        squeeze = False

    batch_size, length = x.shape

    # Handle left padding - reflect without edge
    if pad_left > 0:
        # Take indices 1 to pad_left+1 and reverse
        left_pad = x[:, 1:pad_left+1][:, ::-1]
        x = mx.concatenate([left_pad, x], axis=1)

    # Handle right padding - reflect without edge
    if pad_right > 0:
        # Take indices -(pad_right+1) to -1 and reverse
        right_pad = x[:, -(pad_right+1):-1][:, ::-1]
        x = mx.concatenate([x, right_pad], axis=1)

    if squeeze:
        x = x[0]

    return x


def stft(
    y: mx.array,
    n_fft: int,
    hop_length: int,
    win_length: int,
    center: bool = True,
) -> mx.array:
    """
    Short-time Fourier Transform.

    Args:
        y: Input audio (B, T) or (T,)
        n_fft: FFT size
        hop_length: Hop size between frames
        win_length: Window length
        center: Whether to pad signal for centered frames

    Returns:
        Complex STFT (B, n_frames, n_fft // 2 + 1) or (n_frames, n_fft // 2 + 1)
    """
    # Handle 1D input
    if y.ndim == 1:
        y = y[None, :]
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size, signal_length = y.shape

    # Create window
    window = hann_window(win_length)

    # Pad window to n_fft if needed
    if win_length < n_fft:
        left_pad = (n_fft - win_length) // 2
        right_pad = n_fft - win_length - left_pad
        window = mx.pad(window, [(left_pad, right_pad)])

    # Center padding with reflect
    if center:
        pad_amount = n_fft // 2
        y = reflect_pad_1d(y, pad_amount, pad_amount)

    # Calculate number of frames
    n_frames = (y.shape[1] - n_fft) // hop_length + 1

    # Create frames using indexing
    frames = []
    for i in range(n_frames):
        start = i * hop_length
        frame = y[:, start:start + n_fft] * window
        frames.append(frame)

    # Stack frames: (B, n_frames, n_fft)
    frames = mx.stack(frames, axis=1)

    # FFT
    spec = mx.fft.rfft(frames, axis=-1)

    if squeeze_output:
        spec = spec[0]

    return spec


def spectrogram(
    y: mx.array,
    n_fft: int,
    hop_length: int,
    win_length: int,
    center: bool = True,
) -> mx.array:
    """
    Compute magnitude spectrogram.

    Args:
        y: Input audio (B, T) or (B, T, 1)
        n_fft: FFT size
        hop_length: Hop size
        win_length: Window length
        center: Center padding

    Returns:
        Magnitude spectrogram (B, n_frames, n_fft // 2 + 1)
    """
    # Handle (B, T, 1) format
    if y.ndim == 3 and y.shape[-1] == 1:
        y = y[:, :, 0]

    # Compute STFT
    spec = stft(y, n_fft, hop_length, win_length, center)

    # Magnitude - mx.abs works directly on complex arrays
    mag = mx.abs(spec) + 1e-9

    return mag


def mel_spectrogram(
    y: mx.array,
    n_fft: int = 1024,
    n_mels: int = 80,
    sample_rate: int = 22050,
    hop_length: int = 256,
    win_length: int = 1024,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    center: bool = True,
) -> mx.array:
    """
    Compute mel spectrogram.

    Args:
        y: Input audio (B, T) or (B, T, 1)
        n_fft: FFT size
        n_mels: Number of mel bands
        sample_rate: Audio sample rate
        hop_length: Hop size
        win_length: Window length
        fmin: Minimum frequency
        fmax: Maximum frequency
        center: Center padding

    Returns:
        Log mel spectrogram (B, n_frames, n_mels)
    """
    # Get magnitude spectrogram
    spec = spectrogram(y, n_fft, hop_length, win_length, center)

    # Get mel filterbank
    mel_basis = get_mel_basis(sample_rate, n_fft, n_mels, fmin, fmax)

    # Apply mel filterbank: (B, T, freq) @ (mels, freq).T -> (B, T, mels)
    mel = mx.matmul(spec, mel_basis.T)

    # Log scale with clamping
    mel = mx.log(mx.maximum(mel, mx.array(1e-5)))

    return mel


def dynamic_range_compression(x: mx.array, C: float = 1.0, clip_val: float = 1e-5) -> mx.array:
    """Apply dynamic range compression."""
    return mx.log(mx.maximum(x, mx.array(clip_val)) * C)


def dynamic_range_decompression(x: mx.array, C: float = 1.0) -> mx.array:
    """Apply dynamic range decompression."""
    return mx.exp(x) / C


class MelSpectrogramConfig:
    """Configuration for mel spectrogram computation."""

    def __init__(
        self,
        sample_rate: int = 40000,
        n_fft: int = 1024,
        n_mels: int = 80,
        hop_length: int = 320,
        win_length: int = 1024,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sample_rate / 2

    def compute(self, audio: mx.array) -> mx.array:
        """Compute mel spectrogram for audio."""
        return mel_spectrogram(
            audio,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            win_length=self.win_length,
            fmin=self.fmin,
            fmax=self.fmax,
        )


# Pre-configured mel configs for common sample rates
MEL_CONFIG_32K = MelSpectrogramConfig(sample_rate=32000, hop_length=320)
MEL_CONFIG_40K = MelSpectrogramConfig(sample_rate=40000, hop_length=400)
MEL_CONFIG_48K = MelSpectrogramConfig(sample_rate=48000, hop_length=480)


def get_mel_config(sample_rate: int) -> MelSpectrogramConfig:
    """Get mel config for sample rate."""
    configs = {
        32000: MEL_CONFIG_32K,
        40000: MEL_CONFIG_40K,
        48000: MEL_CONFIG_48K,
    }
    if sample_rate in configs:
        return configs[sample_rate]
    # Default: scale hop_length proportionally
    hop_length = int(sample_rate * 0.01)  # 10ms hop
    return MelSpectrogramConfig(sample_rate=sample_rate, hop_length=hop_length)
