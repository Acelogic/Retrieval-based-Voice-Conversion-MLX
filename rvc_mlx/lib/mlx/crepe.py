"""
CREPE (Convolutional Representation for Pitch Estimation) - MLX Implementation.

CREPE is a deep neural network that estimates pitch by predicting probability
distributions over 360 pitch bins (20 cents each, from ~30 Hz to ~2000 Hz).

Two model variants:
- full: Higher accuracy, larger model (1024→128→128→128→256→512)
- tiny: Faster inference, smaller model (128→16→16→16→32→64)

Reference: https://github.com/marl/crepe
MLX port of torchcrepe: https://github.com/maxrmorrison/torchcrepe
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union

# Constants
PITCH_BINS = 360
SAMPLE_RATE = 16000
HOP_SIZE = 160  # ~10ms
WINDOW_SIZE = 1024  # 64ms

# Cents to frequency conversion
# cents = 1200 * log2(f / 10)
# f = 10 * 2^(cents / 1200)
CENTS_PER_BIN = 20
FMIN = 10.0  # Reference frequency


def _cents_to_frequency(cents: np.ndarray) -> np.ndarray:
    """Convert cents to frequency in Hz."""
    return FMIN * (2 ** (cents / 1200.0))


def _frequency_to_cents(freq: np.ndarray) -> np.ndarray:
    """Convert frequency in Hz to cents."""
    return 1200.0 * np.log2(freq / FMIN)


# Precompute cent values for each bin
CENTS = CENTS_PER_BIN * np.arange(PITCH_BINS) + 1997.3794084376191


class CREPEModel(nn.Module):
    """
    CREPE pitch estimation model in MLX.

    Architecture:
    - 6 Conv2d layers with BatchNorm and MaxPool
    - Final Linear classifier to 360 pitch bins
    - Sigmoid activation for probabilities

    Note: MLX uses (B, H, W, C) format, PyTorch uses (B, C, H, W)
    """

    def __init__(self, model: str = "full"):
        super().__init__()

        self.model_type = model

        # Model-specific parameters
        if model == "full":
            in_channels = [1, 1024, 128, 128, 128, 256]
            out_channels = [1024, 128, 128, 128, 256, 512]
            self.in_features = 2048
        elif model == "tiny":
            in_channels = [1, 128, 16, 16, 16, 32]
            out_channels = [128, 16, 16, 16, 32, 64]
            self.in_features = 256
        else:
            raise ValueError(f"Model {model} is not supported. Use 'full' or 'tiny'.")

        # Kernel sizes: first layer is (512, 1), rest are (64, 1)
        kernel_sizes = [(512, 1)] + [(64, 1)] * 5
        strides = [(4, 1)] + [(1, 1)] * 5

        # Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels[0],
            out_channels=out_channels[0],
            kernel_size=kernel_sizes[0],
            stride=strides[0],
        )
        self.conv1_BN = nn.BatchNorm(out_channels[0], eps=1e-3, momentum=0.0)

        # Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=in_channels[1],
            out_channels=out_channels[1],
            kernel_size=kernel_sizes[1],
            stride=strides[1],
        )
        self.conv2_BN = nn.BatchNorm(out_channels[1], eps=1e-3, momentum=0.0)

        # Layer 3
        self.conv3 = nn.Conv2d(
            in_channels=in_channels[2],
            out_channels=out_channels[2],
            kernel_size=kernel_sizes[2],
            stride=strides[2],
        )
        self.conv3_BN = nn.BatchNorm(out_channels[2], eps=1e-3, momentum=0.0)

        # Layer 4
        self.conv4 = nn.Conv2d(
            in_channels=in_channels[3],
            out_channels=out_channels[3],
            kernel_size=kernel_sizes[3],
            stride=strides[3],
        )
        self.conv4_BN = nn.BatchNorm(out_channels[3], eps=1e-3, momentum=0.0)

        # Layer 5
        self.conv5 = nn.Conv2d(
            in_channels=in_channels[4],
            out_channels=out_channels[4],
            kernel_size=kernel_sizes[4],
            stride=strides[4],
        )
        self.conv5_BN = nn.BatchNorm(out_channels[4], eps=1e-3, momentum=0.0)

        # Layer 6
        self.conv6 = nn.Conv2d(
            in_channels=in_channels[5],
            out_channels=out_channels[5],
            kernel_size=kernel_sizes[5],
            stride=strides[5],
        )
        self.conv6_BN = nn.BatchNorm(out_channels[5], eps=1e-3, momentum=0.0)

        # Classifier
        self.classifier = nn.Linear(self.in_features, PITCH_BINS)

    def _layer(
        self,
        x: mx.array,
        conv: nn.Conv2d,
        batch_norm: nn.BatchNorm,
        padding: Tuple[int, int, int, int] = (0, 0, 31, 32),
    ) -> mx.array:
        """
        Forward pass through one layer.

        Args:
            x: Input tensor (B, H, W, C) in MLX format
            conv: Conv2d layer
            batch_norm: BatchNorm layer
            padding: (left, right, top, bottom) padding

        Returns:
            Output after Conv -> ReLU -> BatchNorm -> MaxPool
        """
        # Apply padding: (left, right, top, bottom) -> for height dimension
        # MLX format is (B, H, W, C), so we pad H dimension
        left, right, top, bottom = padding
        if top > 0 or bottom > 0:
            # Pad height (axis 1)
            x = mx.pad(x, [(0, 0), (top, bottom), (0, 0), (0, 0)])

        # Conv2d
        x = conv(x)

        # ReLU
        x = nn.relu(x)

        # BatchNorm - MLX BatchNorm expects (B, ..., C)
        # Our shape is (B, H, W, C), so we need to reshape
        B, H, W, C = x.shape
        x_flat = x.reshape(B * H * W, C)
        x_flat = batch_norm(x_flat)
        x = x_flat.reshape(B, H, W, C)

        # MaxPool2d with kernel (2, 1) and stride (2, 1)
        # Pool over height dimension only
        x = x.reshape(B, H // 2, 2, W, C)
        x = x.max(axis=2)

        return x

    def __call__(self, x: mx.array, embed: bool = False) -> mx.array:
        """
        Forward pass.

        Args:
            x: Audio frames (B, 1024)
            embed: If True, return embedding instead of probabilities

        Returns:
            If embed=False: Pitch probabilities (B, 360)
            If embed=True: Embedding (B, H, W, C)
        """
        # Reshape to (B, H=1024, W=1, C=1) for Conv2d
        # MLX uses (B, H, W, C) format
        B = x.shape[0]
        x = x.reshape(B, WINDOW_SIZE, 1, 1)

        # Forward through first 5 layers
        x = self._layer(x, self.conv1, self.conv1_BN, padding=(0, 0, 254, 254))
        x = self._layer(x, self.conv2, self.conv2_BN, padding=(0, 0, 31, 32))
        x = self._layer(x, self.conv3, self.conv3_BN, padding=(0, 0, 31, 32))
        x = self._layer(x, self.conv4, self.conv4_BN, padding=(0, 0, 31, 32))
        x = self._layer(x, self.conv5, self.conv5_BN, padding=(0, 0, 31, 32))

        if embed:
            return x

        # Layer 6
        x = self._layer(x, self.conv6, self.conv6_BN, padding=(0, 0, 31, 32))

        # Flatten: (B, H, W, C) -> (B, H * W * C) = (B, in_features)
        x = x.reshape(B, self.in_features)

        # Classifier with sigmoid
        x = self.classifier(x)
        x = mx.sigmoid(x)

        return x


class CREPE:
    """
    CREPE pitch extraction interface.

    Example:
        >>> crepe = CREPE(model="full")
        >>> f0 = crepe.get_f0(audio, f0_min=50, f0_max=1100)
    """

    def __init__(
        self,
        model: str = "full",
        weights_path: Optional[str] = None,
    ):
        """
        Initialize CREPE.

        Args:
            model: Model variant ("full" or "tiny")
            weights_path: Path to weights file. If None, uses default location.
        """
        self.model_type = model
        self._model = CREPEModel(model=model)

        # Load weights
        if weights_path is None:
            weights_path = self._find_weights()

        if weights_path and Path(weights_path).exists():
            self._load_weights(weights_path)
        else:
            raise FileNotFoundError(
                f"CREPE weights not found. Expected at: {weights_path}\n"
                f"Run: python tools/convert_crepe_weights.py"
            )

    def _find_weights(self) -> str:
        """Find default weights path."""
        base_dir = Path(__file__).parent.parent.parent.parent  # rvc_mlx root
        weights_dir = base_dir / "weights"

        if self.model_type == "full":
            return str(weights_dir / "crepe_full.npz")
        else:
            return str(weights_dir / "crepe_tiny.npz")

    def _load_weights(self, weights_path: str):
        """Load weights from npz file."""
        weights = dict(np.load(weights_path))

        # Convert to MLX arrays
        params = {}
        for key, value in weights.items():
            params[key] = mx.array(value)

        # Update model parameters
        self._model.update(params)

    def get_f0(
        self,
        audio: np.ndarray,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
        return_periodicity: bool = False,
        threshold: float = 0.1,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Extract F0 from audio.

        Args:
            audio: Audio signal (1D numpy array, 16kHz)
            f0_min: Minimum F0 to detect (Hz)
            f0_max: Maximum F0 to detect (Hz)
            return_periodicity: Also return confidence/periodicity scores
            threshold: Periodicity threshold below which F0 is set to 0

        Returns:
            If return_periodicity=False: F0 contour (T,) in Hz
            If return_periodicity=True: Tuple of (F0, periodicity)
        """
        # Ensure float32
        audio = audio.astype(np.float32)

        # Frame the audio
        frames = self._frame_audio(audio)

        # Run inference
        probabilities = self._infer(frames)

        # Decode to F0 and periodicity
        f0, periodicity = self._decode(probabilities, f0_min, f0_max)

        # Apply filtering (median on periodicity, mean on F0)
        periodicity = self._median_filter(periodicity, 3)
        f0 = self._mean_filter(f0, 3)

        # Apply threshold
        f0[periodicity < threshold] = 0

        if return_periodicity:
            return f0, periodicity
        return f0

    def _frame_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Split audio into overlapping frames.

        Args:
            audio: Audio signal (N,)

        Returns:
            Frames (T, 1024) where T is number of frames
        """
        # Pad to ensure we have complete frames
        pad_length = WINDOW_SIZE // 2
        audio_padded = np.pad(audio, (pad_length, pad_length), mode='reflect')

        # Number of frames
        n_frames = 1 + (len(audio_padded) - WINDOW_SIZE) // HOP_SIZE

        # Extract frames
        frames = np.zeros((n_frames, WINDOW_SIZE), dtype=np.float32)
        for i in range(n_frames):
            start = i * HOP_SIZE
            end = start + WINDOW_SIZE
            frame = audio_padded[start:end]

            # Normalize each frame
            frame = frame - np.mean(frame)
            std = np.std(frame)
            if std > 1e-10:
                frame = frame / std

            frames[i] = frame

        return frames

    def _infer(self, frames: np.ndarray) -> np.ndarray:
        """
        Run model inference on frames.

        Args:
            frames: Audio frames (T, 1024)

        Returns:
            Pitch probabilities (T, 360)
        """
        # Convert to MLX
        frames_mx = mx.array(frames)

        # Batch processing (process all frames at once or in batches)
        batch_size = 512
        n_frames = frames_mx.shape[0]
        probabilities = []

        for i in range(0, n_frames, batch_size):
            batch = frames_mx[i:i + batch_size]
            probs = self._model(batch)
            mx.eval(probs)  # Force evaluation
            probabilities.append(np.array(probs))

        return np.concatenate(probabilities, axis=0)

    def _decode(
        self,
        probabilities: np.ndarray,
        f0_min: float,
        f0_max: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode probabilities to F0 and periodicity.

        Uses weighted average decoding (argmax with local averaging).

        Args:
            probabilities: (T, 360) probability distributions
            f0_min: Minimum F0
            f0_max: Maximum F0

        Returns:
            Tuple of (f0, periodicity)
        """
        # Mask bins outside frequency range
        f0_min_cents = _frequency_to_cents(f0_min)
        f0_max_cents = _frequency_to_cents(f0_max)

        valid_bins = (CENTS >= f0_min_cents) & (CENTS <= f0_max_cents)
        probabilities_masked = probabilities.copy()
        probabilities_masked[:, ~valid_bins] = 0

        # Get argmax (peak bin)
        peak_bins = np.argmax(probabilities_masked, axis=1)

        # Periodicity is the max probability
        periodicity = probabilities_masked[np.arange(len(peak_bins)), peak_bins]

        # Weighted average around peak (±4 bins)
        window_size = 4
        f0_cents = np.zeros(len(peak_bins), dtype=np.float32)

        for i, peak in enumerate(peak_bins):
            # Get window around peak
            start = max(0, peak - window_size)
            end = min(PITCH_BINS, peak + window_size + 1)

            probs_window = probabilities_masked[i, start:end]
            cents_window = CENTS[start:end]

            # Weighted average
            total_weight = probs_window.sum()
            if total_weight > 0:
                f0_cents[i] = (probs_window * cents_window).sum() / total_weight

        # Convert cents to Hz
        f0 = _cents_to_frequency(f0_cents)

        return f0.astype(np.float32), periodicity.astype(np.float32)

    @staticmethod
    def _median_filter(x: np.ndarray, window: int) -> np.ndarray:
        """Apply median filter."""
        from scipy.ndimage import median_filter
        return median_filter(x, size=window)

    @staticmethod
    def _mean_filter(x: np.ndarray, window: int) -> np.ndarray:
        """Apply mean filter."""
        from scipy.ndimage import uniform_filter1d
        return uniform_filter1d(x, size=window)
