"""
Unified Pitch Extractor Interface for RVC MLX.

Provides a consistent API for all pitch detection methods:
- RMVPE (default, MLX native)
- DIO, PM, HARVEST (PyWorld DSP)
- CREPE, CREPE-tiny (MLX neural network)
- FCPE (MLX neural network)

Example:
    >>> from rvc_mlx.lib.mlx.pitch_extractors import PitchExtractor
    >>> extractor = PitchExtractor(method="rmvpe")
    >>> f0 = extractor.extract(audio, f0_min=50, f0_max=1100)
"""

import numpy as np
from typing import Optional, Tuple, Union


class PitchExtractor:
    """
    Unified interface for pitch extraction methods.

    Supported methods:
    - "rmvpe": RMVPE (Robust Multi-scale Voiced Pitch Estimation) - MLX native
    - "dio": DIO (Distributed Inline-filter Operation) - PyWorld DSP
    - "pm": PM (DIO + StoneMask refinement) - PyWorld DSP
    - "harvest": HARVEST (High-accuracy pitch) - PyWorld DSP
    - "crepe": CREPE full model - MLX neural network
    - "crepe-tiny": CREPE tiny model - MLX neural network
    - "fcpe": FCPE (Fast Context-aware Pitch Estimation) - MLX neural network

    Args:
        method: Pitch detection method (default: "rmvpe")
        sample_rate: Audio sample rate in Hz (default: 16000)
        hop_size: Hop size in samples (default: 160, ~10ms at 16kHz)

    Example:
        >>> extractor = PitchExtractor(method="harvest")
        >>> f0 = extractor.extract(audio, f0_min=50, f0_max=1100)
    """

    # Supported methods
    METHODS = ["rmvpe", "dio", "pm", "harvest", "crepe", "crepe-tiny", "fcpe"]

    # Method categories
    PYWORLD_METHODS = ["dio", "pm", "harvest"]
    NEURAL_METHODS = ["rmvpe", "crepe", "crepe-tiny", "fcpe"]

    def __init__(
        self,
        method: str = "rmvpe",
        sample_rate: int = 16000,
        hop_size: int = 160,
    ):
        if method not in self.METHODS:
            raise ValueError(
                f"Unknown pitch method: {method}. "
                f"Supported methods: {self.METHODS}"
            )

        self.method = method
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self._model = None

        # Initialize the appropriate model/extractor
        self._init_model()

    def _init_model(self):
        """Initialize the pitch extraction model."""
        if self.method in self.PYWORLD_METHODS:
            self._init_pyworld()
        elif self.method == "rmvpe":
            self._init_rmvpe()
        elif self.method in ("crepe", "crepe-tiny"):
            self._init_crepe()
        elif self.method == "fcpe":
            self._init_fcpe()

    def _init_pyworld(self):
        """Initialize PyWorld extractor."""
        from rvc_mlx.lib.mlx.pyworld_pitch import PyWorldExtractor
        self._model = PyWorldExtractor(
            sample_rate=self.sample_rate,
            hop_size=self.hop_size,
        )

    def _init_rmvpe(self):
        """Initialize RMVPE model."""
        from rvc_mlx.lib.mlx.rmvpe import RMVPE0Predictor
        self._model = RMVPE0Predictor()

    def _init_crepe(self):
        """Initialize CREPE model."""
        try:
            from rvc_mlx.lib.mlx.crepe import CREPE
            model_type = "tiny" if self.method == "crepe-tiny" else "full"
            self._model = CREPE(model=model_type)
        except ImportError:
            raise ImportError(
                f"CREPE MLX implementation not available. "
                f"Make sure rvc_mlx.lib.mlx.crepe is implemented."
            )

    def _init_fcpe(self):
        """Initialize FCPE model."""
        try:
            from rvc_mlx.lib.mlx.fcpe import FCPE
            self._model = FCPE()
        except ImportError:
            raise ImportError(
                f"FCPE MLX implementation not available. "
                f"Make sure rvc_mlx.lib.mlx.fcpe is implemented."
            )

    def extract(
        self,
        audio: np.ndarray,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
        **kwargs,
    ) -> np.ndarray:
        """
        Extract F0 (fundamental frequency) from audio.

        Args:
            audio: Audio signal as 1D numpy array (mono, float32 or float64)
            f0_min: Minimum F0 to detect in Hz (default: 50)
            f0_max: Maximum F0 to detect in Hz (default: 1100)
            **kwargs: Additional method-specific arguments

        Returns:
            F0 contour as 1D numpy array in Hz.
            Unvoiced frames have F0 = 0.

        Note:
            The output length depends on the hop_size:
            n_frames ≈ len(audio) // hop_size
        """
        # Ensure audio is 1D
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)  # Convert stereo to mono

        # Dispatch to appropriate method
        if self.method == "dio":
            return self._model.dio(audio, f0_min, f0_max)
        elif self.method == "pm":
            return self._model.pm(audio, f0_min, f0_max)
        elif self.method == "harvest":
            return self._model.harvest(audio, f0_min, f0_max)
        elif self.method == "rmvpe":
            threshold = kwargs.get("threshold", 0.03)
            return self._model.infer_from_audio(audio, thred=threshold)
        elif self.method in ("crepe", "crepe-tiny"):
            return self._model.get_f0(audio, f0_min=f0_min, f0_max=f0_max, **kwargs)
        elif self.method == "fcpe":
            threshold = kwargs.get("threshold", 0.006)
            return self._model.get_f0(audio, f0_min=f0_min, f0_max=f0_max, threshold=threshold)

        raise ValueError(f"Unknown method: {self.method}")

    def extract_with_confidence(
        self,
        audio: np.ndarray,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract F0 with confidence/periodicity scores.

        Not all methods support confidence output.
        Methods without native confidence return ones for voiced frames.

        Args:
            audio: Audio signal
            f0_min: Minimum F0
            f0_max: Maximum F0

        Returns:
            Tuple of (f0, confidence) where:
            - f0: F0 contour in Hz (0 for unvoiced)
            - confidence: Confidence scores in [0, 1]
        """
        f0 = self.extract(audio, f0_min, f0_max, **kwargs)

        # For methods with native confidence
        if self.method in ("crepe", "crepe-tiny"):
            result = self._model.get_f0(
                audio, f0_min=f0_min, f0_max=f0_max,
                return_periodicity=True, **kwargs
            )
            if isinstance(result, tuple):
                return result

        # For methods without confidence, return 1.0 for voiced frames
        confidence = (f0 > 0).astype(np.float32)
        return f0, confidence

    @property
    def is_neural(self) -> bool:
        """Check if this is a neural network method."""
        return self.method in self.NEURAL_METHODS

    @property
    def is_pyworld(self) -> bool:
        """Check if this is a PyWorld method."""
        return self.method in self.PYWORLD_METHODS

    def __repr__(self) -> str:
        return f"PitchExtractor(method='{self.method}', sample_rate={self.sample_rate}, hop_size={self.hop_size})"


# Factory function
def create_extractor(
    method: str = "rmvpe",
    sample_rate: int = 16000,
    hop_size: int = 160,
) -> PitchExtractor:
    """
    Create a pitch extractor instance.

    Args:
        method: Pitch detection method
        sample_rate: Audio sample rate
        hop_size: Hop size in samples

    Returns:
        PitchExtractor instance
    """
    return PitchExtractor(method=method, sample_rate=sample_rate, hop_size=hop_size)


# Direct extraction functions
def extract_f0(
    audio: np.ndarray,
    method: str = "rmvpe",
    sample_rate: int = 16000,
    hop_size: int = 160,
    f0_min: float = 50.0,
    f0_max: float = 1100.0,
    **kwargs,
) -> np.ndarray:
    """
    Extract F0 from audio using specified method.

    Convenience function that creates extractor and extracts in one call.

    Args:
        audio: Audio signal
        method: Pitch method
        sample_rate: Sample rate
        hop_size: Hop size
        f0_min: Minimum F0
        f0_max: Maximum F0

    Returns:
        F0 contour in Hz
    """
    extractor = PitchExtractor(method=method, sample_rate=sample_rate, hop_size=hop_size)
    return extractor.extract(audio, f0_min=f0_min, f0_max=f0_max, **kwargs)
