"""
Aim Experiment Tracker for RVC MLX Training

Provides experiment tracking with voice-specific metrics and visualizations.
"""

import os
import numpy as np
from typing import Dict, Optional, Any, List
from dataclasses import dataclass

# Try to import Aim, gracefully handle if not installed
try:
    from aim import Run, Audio, Image
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False
    Run = None
    Audio = None
    Image = None


@dataclass
class TrackingConfig:
    """Configuration for experiment tracking."""
    experiment_name: str = "rvc_mlx_training"
    repo_path: Optional[str] = None
    log_system_metrics: bool = True
    log_interval: int = 50  # Log every N steps


class VoiceTrainingTracker:
    """
    Aim-based experiment tracker for voice conversion training.

    Tracks:
    - Training losses (generator, discriminator, mel, KL, etc.)
    - Voice-specific metrics (F0 accuracy, MCD, spectral correlation)
    - Audio samples
    - Spectrogram visualizations
    - Hyperparameters
    """

    def __init__(
        self,
        config: Optional[TrackingConfig] = None,
        experiment_name: Optional[str] = None,
        repo_path: Optional[str] = None,
    ):
        self.config = config or TrackingConfig(
            experiment_name=experiment_name or "rvc_mlx_training",
            repo_path=repo_path,
        )

        self.run = None
        self._step = 0

        if AIM_AVAILABLE:
            self._init_aim()
        else:
            print("Warning: Aim not installed. Tracking disabled. Install with: pip install aim")

    def _init_aim(self):
        """Initialize Aim run."""
        try:
            self.run = Run(
                repo=self.config.repo_path,
                experiment=self.config.experiment_name,
                log_system_params=self.config.log_system_metrics,
            )
            self.run["framework"] = "mlx"
            self.run["model"] = "rvc"
            print(f"Aim tracking initialized: {self.config.experiment_name}")
        except Exception as e:
            print(f"Failed to initialize Aim: {e}")
            self.run = None

    def log_hparams(self, hparams: Dict[str, Any]):
        """Log hyperparameters."""
        if self.run is None:
            return

        self.run["hparams"] = hparams

    def log_config(self, config: Any):
        """Log training configuration."""
        if self.run is None:
            return

        if hasattr(config, "__dict__"):
            config_dict = config.__dict__
        elif isinstance(config, dict):
            config_dict = config
        else:
            config_dict = {"config": str(config)}

        self.run["config"] = config_dict

    def log_scalar(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        context: Optional[Dict[str, str]] = None,
    ):
        """Log a scalar metric."""
        if self.run is None:
            return

        step = step if step is not None else self._step
        self.run.track(
            value=float(value),
            name=name,
            step=step,
            context=context or {},
        )

    def log_losses(
        self,
        losses: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = "loss",
    ):
        """Log multiple loss values."""
        if self.run is None:
            return

        step = step if step is not None else self._step
        for name, value in losses.items():
            self.log_scalar(f"{prefix}/{name}", value, step)

    def log_gradients(
        self,
        grad_norm_g: float,
        grad_norm_d: float,
        step: Optional[int] = None,
    ):
        """Log gradient norms."""
        self.log_scalar("grad/norm_g", grad_norm_g, step)
        self.log_scalar("grad/norm_d", grad_norm_d, step)

    def log_learning_rate(self, lr: float, step: Optional[int] = None):
        """Log learning rate."""
        self.log_scalar("lr", lr, step)

    def log_voice_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        """Log voice-specific metrics."""
        if self.run is None:
            return

        step = step if step is not None else self._step
        for name, value in metrics.items():
            self.log_scalar(f"voice/{name}", value, step)

    def log_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        name: str,
        step: Optional[int] = None,
    ):
        """Log audio sample."""
        if self.run is None or Audio is None:
            return

        step = step if step is not None else self._step

        # Ensure audio is 1D and float32
        if audio.ndim > 1:
            audio = audio.flatten()
        audio = audio.astype(np.float32)

        # Normalize if needed
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        try:
            self.run.track(
                Audio(audio, sample_rate=sample_rate),
                name=name,
                step=step,
            )
        except Exception as e:
            print(f"Failed to log audio: {e}")

    def log_spectrogram(
        self,
        spec: np.ndarray,
        name: str,
        step: Optional[int] = None,
    ):
        """Log spectrogram as image."""
        if self.run is None or Image is None:
            return

        step = step if step is not None else self._step

        try:
            # Convert spectrogram to image
            spec_img = self._spec_to_image(spec)
            self.run.track(
                Image(spec_img),
                name=name,
                step=step,
            )
        except Exception as e:
            print(f"Failed to log spectrogram: {e}")

    def _spec_to_image(self, spec: np.ndarray) -> np.ndarray:
        """Convert spectrogram to RGB image."""
        # Normalize to 0-1
        spec = spec - spec.min()
        if spec.max() > 0:
            spec = spec / spec.max()

        # Apply colormap (viridis-like)
        # Simple grayscale for now
        spec_uint8 = (spec * 255).astype(np.uint8)

        # Stack to RGB
        if spec_uint8.ndim == 2:
            spec_rgb = np.stack([spec_uint8] * 3, axis=-1)
        else:
            spec_rgb = spec_uint8

        return spec_rgb

    def log_epoch_summary(
        self,
        epoch: int,
        train_losses: Dict[str, float],
        val_losses: Optional[Dict[str, float]] = None,
        voice_metrics: Optional[Dict[str, float]] = None,
    ):
        """Log epoch summary."""
        if self.run is None:
            return

        # Log train losses
        for name, value in train_losses.items():
            self.log_scalar(f"epoch/train/{name}", value, epoch)

        # Log val losses
        if val_losses:
            for name, value in val_losses.items():
                self.log_scalar(f"epoch/val/{name}", value, epoch)

        # Log voice metrics
        if voice_metrics:
            for name, value in voice_metrics.items():
                self.log_scalar(f"epoch/voice/{name}", value, epoch)

    def step(self):
        """Increment step counter."""
        self._step += 1

    def close(self):
        """Close the tracking run."""
        if self.run is not None:
            self.run.close()
            print("Aim tracking closed")


class SimpleTracker:
    """
    Simple fallback tracker when Aim is not available.

    Logs metrics to console and optionally to files.
    """

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = log_dir
        self._step = 0
        self.history = {}

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        """Log a scalar metric."""
        step = step if step is not None else self._step

        if name not in self.history:
            self.history[name] = []
        self.history[name].append((step, value))

    def log_losses(self, losses: Dict[str, float], step: Optional[int] = None, prefix: str = "loss"):
        """Log multiple loss values."""
        for name, value in losses.items():
            self.log_scalar(f"{prefix}/{name}", value, step)

    def log_hparams(self, hparams: Dict[str, Any]):
        """Log hyperparameters."""
        print(f"Hyperparameters: {hparams}")

    def step(self):
        """Increment step counter."""
        self._step += 1

    def save(self):
        """Save history to file."""
        if self.log_dir:
            import json
            with open(os.path.join(self.log_dir, "metrics.json"), "w") as f:
                json.dump(self.history, f, indent=2)

    def close(self):
        """Close tracker."""
        self.save()


def create_tracker(
    experiment_name: str = "rvc_mlx",
    repo_path: Optional[str] = None,
    use_aim: bool = True,
) -> VoiceTrainingTracker:
    """
    Create a tracker instance.

    Args:
        experiment_name: Name of the experiment
        repo_path: Path to Aim repo
        use_aim: Whether to use Aim (falls back to simple if not available)

    Returns:
        Tracker instance
    """
    if use_aim and AIM_AVAILABLE:
        return VoiceTrainingTracker(
            experiment_name=experiment_name,
            repo_path=repo_path,
        )
    else:
        return VoiceTrainingTracker(
            experiment_name=experiment_name,
            repo_path=repo_path,
        )
