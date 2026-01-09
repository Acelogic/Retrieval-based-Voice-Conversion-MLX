"""
Voice-Specific Metrics for RVC MLX Training

Metrics for evaluating voice conversion quality.
"""

import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass


def compute_f0_accuracy(
    pred_f0: np.ndarray,
    target_f0: np.ndarray,
    threshold_cents: float = 50.0,
) -> float:
    """
    Compute F0 accuracy within threshold.

    Args:
        pred_f0: Predicted F0 values (Hz)
        target_f0: Target F0 values (Hz)
        threshold_cents: Threshold in cents (100 cents = 1 semitone)

    Returns:
        Accuracy (0.0 to 1.0) of frames within threshold
    """
    # Only compare voiced regions
    voiced_mask = (target_f0 > 0) & (pred_f0 > 0)

    if not np.any(voiced_mask):
        return 0.0

    # Convert to cents difference
    # cents = 1200 * log2(f1 / f2)
    pred_voiced = pred_f0[voiced_mask]
    target_voiced = target_f0[voiced_mask]

    # Avoid division by zero
    ratio = np.clip(pred_voiced / (target_voiced + 1e-8), 1e-8, 1e8)
    cents_diff = np.abs(1200 * np.log2(ratio))

    # Calculate accuracy
    accuracy = np.mean(cents_diff < threshold_cents)

    return float(accuracy)


def compute_mcd(
    pred_mel: np.ndarray,
    target_mel: np.ndarray,
    n_mfcc: int = 13,
) -> float:
    """
    Compute Mel Cepstral Distortion (MCD).

    Lower is better. Typical values: 3-6 dB for good synthesis.

    Args:
        pred_mel: Predicted mel spectrogram (T, n_mels)
        target_mel: Target mel spectrogram (T, n_mels)
        n_mfcc: Number of cepstral coefficients to use

    Returns:
        MCD in dB
    """
    # Ensure same length
    min_len = min(pred_mel.shape[0], target_mel.shape[0])
    pred_mel = pred_mel[:min_len]
    target_mel = target_mel[:min_len]

    # DCT-II to get cepstral coefficients (simplified)
    # Using direct difference on mel for approximation
    diff = pred_mel - target_mel

    # Use first n_mfcc coefficients (or all if less)
    n_use = min(n_mfcc, diff.shape[1])
    diff = diff[:, :n_use]

    # MCD formula: (10 / ln(10)) * sqrt(2 * sum((c1 - c2)^2))
    mcd = np.mean(np.sqrt(2 * np.sum(diff ** 2, axis=1)))
    mcd = mcd * (10 / np.log(10))

    return float(mcd)


def compute_spectrogram_correlation(
    pred_spec: np.ndarray,
    target_spec: np.ndarray,
) -> float:
    """
    Compute Pearson correlation between spectrograms.

    Args:
        pred_spec: Predicted spectrogram
        target_spec: Target spectrogram

    Returns:
        Correlation coefficient (-1 to 1, higher is better)
    """
    # Flatten
    pred_flat = pred_spec.flatten()
    target_flat = target_spec.flatten()

    # Ensure same length
    min_len = min(len(pred_flat), len(target_flat))
    pred_flat = pred_flat[:min_len]
    target_flat = target_flat[:min_len]

    # Correlation
    correlation = np.corrcoef(pred_flat, target_flat)[0, 1]

    if np.isnan(correlation):
        return 0.0

    return float(correlation)


def compute_snr(
    pred: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Compute Signal-to-Noise Ratio.

    Args:
        pred: Predicted signal
        target: Target signal

    Returns:
        SNR in dB
    """
    # Ensure same length
    min_len = min(len(pred), len(target))
    pred = pred[:min_len]
    target = target[:min_len]

    # Calculate noise
    noise = target - pred

    # Signal power
    signal_power = np.mean(target ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power < 1e-10:
        return 100.0  # Very high SNR

    snr = 10 * np.log10(signal_power / noise_power)

    return float(snr)


def compute_rmse(
    pred: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Compute Root Mean Squared Error.

    Args:
        pred: Predicted values
        target: Target values

    Returns:
        RMSE
    """
    min_len = min(len(pred), len(target))
    pred = pred[:min_len]
    target = target[:min_len]

    rmse = np.sqrt(np.mean((pred - target) ** 2))
    return float(rmse)


@dataclass
class VoiceMetrics:
    """Container for voice conversion metrics."""
    f0_accuracy: float = 0.0
    mcd: float = 0.0
    spec_correlation: float = 0.0
    snr: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "f0_accuracy": self.f0_accuracy,
            "mcd": self.mcd,
            "spec_correlation": self.spec_correlation,
            "snr": self.snr,
        }

    @classmethod
    def compute(
        cls,
        pred_audio: Optional[np.ndarray] = None,
        target_audio: Optional[np.ndarray] = None,
        pred_f0: Optional[np.ndarray] = None,
        target_f0: Optional[np.ndarray] = None,
        pred_mel: Optional[np.ndarray] = None,
        target_mel: Optional[np.ndarray] = None,
    ) -> "VoiceMetrics":
        """
        Compute all available metrics.

        Args:
            pred_audio: Predicted audio waveform
            target_audio: Target audio waveform
            pred_f0: Predicted F0
            target_f0: Target F0
            pred_mel: Predicted mel spectrogram
            target_mel: Target mel spectrogram

        Returns:
            VoiceMetrics instance
        """
        metrics = cls()

        # F0 accuracy
        if pred_f0 is not None and target_f0 is not None:
            metrics.f0_accuracy = compute_f0_accuracy(pred_f0, target_f0)

        # MCD
        if pred_mel is not None and target_mel is not None:
            metrics.mcd = compute_mcd(pred_mel, target_mel)
            metrics.spec_correlation = compute_spectrogram_correlation(pred_mel, target_mel)

        # SNR
        if pred_audio is not None and target_audio is not None:
            metrics.snr = compute_snr(pred_audio, target_audio)

        return metrics


def evaluate_voice_conversion(
    pred_audio: np.ndarray,
    target_audio: np.ndarray,
    sample_rate: int = 16000,
) -> VoiceMetrics:
    """
    Evaluate voice conversion quality.

    Args:
        pred_audio: Predicted audio
        target_audio: Target audio
        sample_rate: Audio sample rate

    Returns:
        VoiceMetrics with computed metrics
    """
    # Compute mel spectrograms for MCD
    try:
        import librosa
        pred_mel = librosa.feature.melspectrogram(y=pred_audio, sr=sample_rate, n_mels=80)
        target_mel = librosa.feature.melspectrogram(y=target_audio, sr=sample_rate, n_mels=80)
        pred_mel = librosa.power_to_db(pred_mel).T
        target_mel = librosa.power_to_db(target_mel).T
    except ImportError:
        pred_mel = None
        target_mel = None

    return VoiceMetrics.compute(
        pred_audio=pred_audio,
        target_audio=target_audio,
        pred_mel=pred_mel,
        target_mel=target_mel,
    )
