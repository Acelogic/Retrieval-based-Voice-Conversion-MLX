"""
Shared test fixtures for pitch detection tests.

Provides:
- Synthetic audio generation (sine waves, silence)
- Sample rate and hop size configuration
- Test audio loading utilities
- Reference F0 computation helpers
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import os

# Test configuration
SAMPLE_RATE = 16000
HOP_SIZE = 160
F0_MIN = 50
F0_MAX = 1100

# Paths
FIXTURES_DIR = Path(__file__).parent / "fixtures"
AUDIO_FIXTURES_DIR = FIXTURES_DIR / "audio"


@pytest.fixture
def sample_rate():
    """Standard sample rate for pitch detection (16kHz)."""
    return SAMPLE_RATE


@pytest.fixture
def hop_size():
    """Standard hop size for pitch detection (160 samples = 10ms @ 16kHz)."""
    return HOP_SIZE


@pytest.fixture
def f0_range():
    """Standard F0 range (min, max) in Hz."""
    return (F0_MIN, F0_MAX)


def generate_sine_wave(
    frequency: float,
    duration: float = 1.0,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = 0.5,
) -> np.ndarray:
    """
    Generate a pure sine wave at the given frequency.

    Args:
        frequency: Frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        amplitude: Amplitude (0.0 to 1.0)

    Returns:
        Audio signal as float32 numpy array
    """
    t = np.arange(int(duration * sample_rate)) / sample_rate
    audio = amplitude * np.sin(2 * np.pi * frequency * t)
    return audio.astype(np.float32)


def generate_silence(
    duration: float = 1.0,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Generate silent audio.

    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        Zero array as float32 numpy array
    """
    return np.zeros(int(duration * sample_rate), dtype=np.float32)


def generate_chirp(
    f0_start: float,
    f0_end: float,
    duration: float = 1.0,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a linear frequency chirp signal.

    Args:
        f0_start: Starting frequency in Hz
        f0_end: Ending frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        amplitude: Amplitude (0.0 to 1.0)

    Returns:
        Tuple of (audio signal, expected F0 contour)
    """
    n_samples = int(duration * sample_rate)
    t = np.arange(n_samples) / sample_rate

    # Linear frequency sweep
    f0_contour = f0_start + (f0_end - f0_start) * t / duration

    # Instantaneous phase
    phase = 2 * np.pi * np.cumsum(f0_contour) / sample_rate
    audio = amplitude * np.sin(phase)

    return audio.astype(np.float32), f0_contour.astype(np.float32)


def generate_voiced_unvoiced(
    voiced_freq: float = 200.0,
    duration: float = 2.0,
    sample_rate: int = SAMPLE_RATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate audio with alternating voiced and unvoiced segments.

    Args:
        voiced_freq: Frequency for voiced segments in Hz
        duration: Total duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        Tuple of (audio signal, expected F0 contour with 0 for unvoiced)
    """
    n_samples = int(duration * sample_rate)
    n_frames = n_samples // HOP_SIZE

    audio = np.zeros(n_samples, dtype=np.float32)
    f0_contour = np.zeros(n_frames, dtype=np.float32)

    # Alternate between voiced (0.5s) and unvoiced (0.5s)
    segment_samples = int(0.5 * sample_rate)
    segment_frames = segment_samples // HOP_SIZE

    for i in range(int(duration / 0.5)):
        start_sample = i * segment_samples
        end_sample = min((i + 1) * segment_samples, n_samples)
        start_frame = i * segment_frames
        end_frame = min((i + 1) * segment_frames, n_frames)

        if i % 2 == 0:  # Voiced segment
            t = np.arange(end_sample - start_sample) / sample_rate
            audio[start_sample:end_sample] = 0.5 * np.sin(2 * np.pi * voiced_freq * t)
            f0_contour[start_frame:end_frame] = voiced_freq
        # Unvoiced segments remain as zeros

    return audio, f0_contour


@pytest.fixture
def sine_440hz():
    """1-second 440 Hz sine wave (A4 note)."""
    return generate_sine_wave(440.0, duration=1.0)


@pytest.fixture
def sine_100hz():
    """1-second 100 Hz sine wave (low male voice)."""
    return generate_sine_wave(100.0, duration=1.0)


@pytest.fixture
def sine_300hz():
    """1-second 300 Hz sine wave (high male/low female voice)."""
    return generate_sine_wave(300.0, duration=1.0)


@pytest.fixture
def silence_audio():
    """1-second of silence."""
    return generate_silence(duration=1.0)


@pytest.fixture
def chirp_100_400():
    """1-second chirp from 100 Hz to 400 Hz with expected F0 contour."""
    return generate_chirp(100.0, 400.0, duration=1.0)


@pytest.fixture
def voiced_unvoiced_audio():
    """2-second audio with alternating voiced (200 Hz) and unvoiced segments."""
    return generate_voiced_unvoiced(voiced_freq=200.0, duration=2.0)


@pytest.fixture
def real_voice_audio():
    """
    Load a real voice audio sample if available.
    Falls back to synthetic audio if no fixture exists.
    """
    voice_path = AUDIO_FIXTURES_DIR / "voice_sample.wav"
    if voice_path.exists():
        import soundfile as sf
        audio, sr = sf.read(voice_path)
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        return audio.astype(np.float32)
    else:
        # Fall back to synthetic "voice-like" audio with vibrato
        duration = 2.0
        n_samples = int(duration * SAMPLE_RATE)
        t = np.arange(n_samples) / SAMPLE_RATE

        # Base frequency with vibrato
        base_freq = 200.0
        vibrato_rate = 5.0  # Hz
        vibrato_depth = 10.0  # Hz
        freq = base_freq + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)

        # Generate audio
        phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
        audio = 0.5 * np.sin(phase)

        return audio.astype(np.float32)


def expected_f0_frames(audio_length: int, hop_size: int = HOP_SIZE) -> int:
    """Calculate expected number of F0 frames for given audio length."""
    return audio_length // hop_size


def f0_correlation(f0_a: np.ndarray, f0_b: np.ndarray) -> float:
    """
    Compute correlation between two F0 contours.
    Only considers voiced frames (where both F0 values > 0).

    Args:
        f0_a: First F0 contour
        f0_b: Second F0 contour

    Returns:
        Pearson correlation coefficient, or 0.0 if insufficient voiced frames
    """
    # Ensure same length
    min_len = min(len(f0_a), len(f0_b))
    f0_a = f0_a[:min_len]
    f0_b = f0_b[:min_len]

    # Only compare voiced frames
    voiced_mask = (f0_a > 0) & (f0_b > 0)
    n_voiced = voiced_mask.sum()

    if n_voiced < 10:  # Need at least 10 frames for meaningful correlation
        return 0.0

    f0_a_voiced = f0_a[voiced_mask]
    f0_b_voiced = f0_b[voiced_mask]

    # Compute correlation
    corr = np.corrcoef(f0_a_voiced, f0_b_voiced)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def f0_accuracy(
    f0_pred: np.ndarray,
    f0_true: np.ndarray,
    tolerance_cents: float = 50.0,
) -> float:
    """
    Compute F0 accuracy within a tolerance (in cents).

    Args:
        f0_pred: Predicted F0 contour
        f0_true: Ground truth F0 contour
        tolerance_cents: Tolerance in cents (default 50 cents = half semitone)

    Returns:
        Fraction of voiced frames within tolerance
    """
    # Ensure same length
    min_len = min(len(f0_pred), len(f0_true))
    f0_pred = f0_pred[:min_len]
    f0_true = f0_true[:min_len]

    # Only evaluate voiced frames
    voiced_mask = (f0_pred > 0) & (f0_true > 0)
    n_voiced = voiced_mask.sum()

    if n_voiced == 0:
        return 0.0

    # Convert to cents and compute error
    f0_pred_cents = 1200 * np.log2(f0_pred[voiced_mask] / 10.0)
    f0_true_cents = 1200 * np.log2(f0_true[voiced_mask] / 10.0)

    error_cents = np.abs(f0_pred_cents - f0_true_cents)
    accurate = error_cents <= tolerance_cents

    return float(accurate.sum()) / n_voiced


# Pytest configuration
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "parity: marks parity tests")
    config.addinivalue_line("markers", "requires_model: marks tests needing model weights")
