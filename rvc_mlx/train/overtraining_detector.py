"""
Overtraining Detection for RVC MLX Training

Based on AI Hub recommendations: monitors g/total loss to detect when
the model starts overtraining (loss plateaus or rises).
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class OvertTrainingStatus:
    """Status of overtraining detection."""
    is_overtraining: bool
    reason: str
    epochs_since_best: int
    best_loss: float
    current_loss: float
    trend: str  # "improving", "plateau", "rising"


class OvertrainingDetector:
    """
    Detects overtraining by monitoring g/total loss.

    Overtraining symptoms (from AI Hub):
    - Robotic sibilances
    - Inability to produce high-end harmonics
    - Loss plateaus or starts rising

    Detection strategy:
    1. Track g/total loss history
    2. Detect plateau: loss hasn't improved by min_delta for patience epochs
    3. Detect rising: loss consistently increases for rising_patience epochs
    """

    def __init__(
        self,
        patience: int = 10,
        rising_patience: int = 5,
        min_delta: float = 0.001,
        smoothing: float = 0.9,
    ):
        """
        Initialize overtraining detector.

        Args:
            patience: Epochs to wait for improvement before declaring plateau
            rising_patience: Consecutive rising epochs to trigger warning
            min_delta: Minimum loss decrease to count as improvement
            smoothing: EMA smoothing factor for loss (reduces noise)
        """
        self.patience = patience
        self.rising_patience = rising_patience
        self.min_delta = min_delta
        self.smoothing = smoothing

        # State
        self.loss_history: List[float] = []
        self.smoothed_history: List[float] = []
        self.best_loss: float = float("inf")
        self.best_epoch: int = 0
        self.epochs_since_best: int = 0
        self.rising_count: int = 0
        self.current_epoch: int = 0

    def update(self, g_total_loss: float) -> OvertTrainingStatus:
        """
        Update detector with new epoch's g/total loss.

        Args:
            g_total_loss: The g/total loss value for this epoch

        Returns:
            OvertTrainingStatus with detection results
        """
        self.current_epoch += 1
        self.loss_history.append(g_total_loss)

        # Compute smoothed loss (EMA)
        if len(self.smoothed_history) == 0:
            smoothed = g_total_loss
        else:
            smoothed = self.smoothing * self.smoothed_history[-1] + (1 - self.smoothing) * g_total_loss
        self.smoothed_history.append(smoothed)

        # Check for improvement
        if smoothed < self.best_loss - self.min_delta:
            self.best_loss = smoothed
            self.best_epoch = self.current_epoch
            self.epochs_since_best = 0
            self.rising_count = 0
            trend = "improving"
        else:
            self.epochs_since_best += 1

            # Check if loss is rising
            if len(self.smoothed_history) >= 2 and smoothed > self.smoothed_history[-2]:
                self.rising_count += 1
                trend = "rising"
            else:
                self.rising_count = 0
                trend = "plateau"

        # Determine overtraining status
        is_overtraining = False
        reason = ""

        if self.rising_count >= self.rising_patience:
            is_overtraining = True
            reason = f"Loss rising for {self.rising_count} consecutive epochs"
        elif self.epochs_since_best >= self.patience:
            is_overtraining = True
            reason = f"No improvement for {self.epochs_since_best} epochs (plateau)"

        return OvertTrainingStatus(
            is_overtraining=is_overtraining,
            reason=reason,
            epochs_since_best=self.epochs_since_best,
            best_loss=self.best_loss,
            current_loss=g_total_loss,
            trend=trend,
        )

    def get_recommendation(self) -> str:
        """Get recommendation based on current state."""
        if len(self.loss_history) < 5:
            return "Too early to make recommendations. Continue training."

        avg_recent = np.mean(self.loss_history[-5:])
        avg_early = np.mean(self.loss_history[:5]) if len(self.loss_history) >= 10 else avg_recent

        improvement = (avg_early - avg_recent) / avg_early * 100 if avg_early > 0 else 0

        if self.epochs_since_best > self.patience:
            return (
                f"Consider stopping training. Best loss ({self.best_loss:.4f}) "
                f"was at epoch {self.best_epoch}. Current: {self.loss_history[-1]:.4f}"
            )
        elif self.rising_count > 2:
            return (
                f"Warning: Loss appears to be rising. "
                f"Consider reducing learning rate or stopping soon."
            )
        elif improvement > 20:
            return f"Good progress! Loss improved {improvement:.1f}% from start."
        else:
            return "Training is progressing normally."

    def should_stop(self) -> Tuple[bool, str]:
        """
        Determine if training should stop.

        Returns:
            (should_stop, reason)
        """
        if len(self.loss_history) < 2:
            return False, ""

        # Rising loss for too long
        if self.rising_count >= self.rising_patience:
            return True, f"Loss has been rising for {self.rising_count} epochs. Overtraining detected."

        # Plateau for too long
        if self.epochs_since_best >= self.patience:
            return True, f"No improvement for {self.epochs_since_best} epochs. Stopping to prevent overtraining."

        return False, ""

    def get_best_epoch(self) -> int:
        """Get the epoch with the best loss."""
        return self.best_epoch

    def get_summary(self) -> dict:
        """Get summary statistics."""
        return {
            "current_epoch": self.current_epoch,
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "current_loss": self.loss_history[-1] if self.loss_history else None,
            "epochs_since_best": self.epochs_since_best,
            "rising_count": self.rising_count,
            "total_epochs": len(self.loss_history),
        }


def calculate_recommended_batch_size(dataset_duration_minutes: float) -> int:
    """
    Calculate recommended batch size based on dataset duration.

    Based on AI Hub recommendations:
    - 30+ minutes → batch size 8
    - Under 30 minutes → batch size 4

    Smaller batch sizes create noisier gradients which can help prevent
    overtraining on small/repetitive datasets.

    Args:
        dataset_duration_minutes: Total duration of dataset in minutes

    Returns:
        Recommended batch size (4 or 8)
    """
    if dataset_duration_minutes >= 30:
        return 8
    else:
        return 4


def estimate_dataset_duration(
    audio_dir: str,
    sample_rate: int = 40000,
) -> Tuple[float, int]:
    """
    Estimate total duration of audio files in a directory.

    Args:
        audio_dir: Directory containing audio files
        sample_rate: Expected sample rate

    Returns:
        (duration_minutes, file_count)
    """
    import os

    try:
        import librosa
        use_librosa = True
    except ImportError:
        use_librosa = False

    total_seconds = 0.0
    file_count = 0

    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in audio_extensions:
                file_path = os.path.join(root, file)
                file_count += 1

                if use_librosa:
                    try:
                        duration = librosa.get_duration(path=file_path)
                        total_seconds += duration
                    except Exception:
                        # Estimate based on file size (rough approximation)
                        size = os.path.getsize(file_path)
                        # Assume ~2 bytes per sample for 16-bit audio
                        total_seconds += size / (2 * sample_rate)
                else:
                    # Rough estimate based on file size
                    size = os.path.getsize(file_path)
                    total_seconds += size / (2 * sample_rate)

    return total_seconds / 60.0, file_count


def get_smart_batch_size(audio_dir: str, sample_rate: int = 40000, verbose: bool = True) -> int:
    """
    Get smart batch size recommendation based on dataset.

    Args:
        audio_dir: Directory containing audio files
        sample_rate: Expected sample rate
        verbose: Print recommendation details

    Returns:
        Recommended batch size
    """
    duration_minutes, file_count = estimate_dataset_duration(audio_dir, sample_rate)

    batch_size = calculate_recommended_batch_size(duration_minutes)

    if verbose:
        print(f"Dataset analysis:")
        print(f"  - Files: {file_count}")
        print(f"  - Duration: {duration_minutes:.1f} minutes")
        print(f"  - Recommended batch size: {batch_size}")
        if duration_minutes < 30:
            print(f"  - Note: Smaller batch (4) helps prevent overtraining on small datasets")
        else:
            print(f"  - Note: Larger batch (8) provides smoother gradients for bigger datasets")

    return batch_size
