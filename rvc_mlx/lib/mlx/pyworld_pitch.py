"""
PyWorld-based pitch extraction methods: DIO, PM (stonemask), HARVEST.

PyWorld is a fast, high-quality vocoder for speech analysis and synthesis.
These are traditional DSP-based methods (not neural networks).

Methods:
- DIO: Fast, coarse pitch estimation (real-time capable)
- HARVEST: More accurate pitch estimation (slower)
- PM: Pitch refinement using StoneMask (DIO + refinement)

Reference: https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder
"""

import numpy as np
from typing import Optional

try:
    import pyworld as pw
except ImportError:
    raise ImportError(
        "pyworld is required for DIO/PM/HARVEST pitch extraction. "
        "Install it with: pip install pyworld"
    )


class PyWorldExtractor:
    """
    Unified interface for PyWorld pitch extraction methods.

    Args:
        sample_rate: Audio sample rate in Hz (default: 16000)
        hop_size: Hop size in samples (default: 160, ~10ms at 16kHz)

    Example:
        >>> extractor = PyWorldExtractor(sample_rate=16000, hop_size=160)
        >>> f0 = extractor.dio(audio, f0_min=50, f0_max=1100)
        >>> f0 = extractor.harvest(audio, f0_min=50, f0_max=1100)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        hop_size: int = 160,
    ):
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        # Convert hop_size to frame_period in milliseconds
        self.frame_period = (hop_size / sample_rate) * 1000.0

    def dio(
        self,
        audio: np.ndarray,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
        channels_in_octave: float = 2.0,
        use_stonemask: bool = True,
    ) -> np.ndarray:
        """
        DIO (Distributed Inline-filter Operation) pitch estimation.

        Fast, coarse pitch estimation suitable for real-time applications.
        Optionally refined with StoneMask for better accuracy.

        Args:
            audio: Audio signal (1D numpy array)
            f0_min: Minimum F0 to detect (Hz)
            f0_max: Maximum F0 to detect (Hz)
            channels_in_octave: Frequency resolution (2 = semitone)
            use_stonemask: Apply StoneMask refinement (default: True)

        Returns:
            F0 contour as 1D numpy array (Hz), 0 for unvoiced frames
        """
        # PyWorld requires float64
        audio = self._ensure_float64(audio)

        # DIO extraction
        f0, t = pw.dio(
            audio,
            self.sample_rate,
            f0_floor=f0_min,
            f0_ceil=f0_max,
            channels_in_octave=channels_in_octave,
            frame_period=self.frame_period,
        )

        # Optional StoneMask refinement
        if use_stonemask:
            f0 = pw.stonemask(audio, f0, t, self.sample_rate)

        return f0.astype(np.float32)

    def harvest(
        self,
        audio: np.ndarray,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
    ) -> np.ndarray:
        """
        HARVEST pitch estimation.

        More accurate than DIO but slower (~10x). Better for offline processing.

        Args:
            audio: Audio signal (1D numpy array)
            f0_min: Minimum F0 to detect (Hz)
            f0_max: Maximum F0 to detect (Hz)

        Returns:
            F0 contour as 1D numpy array (Hz), 0 for unvoiced frames
        """
        audio = self._ensure_float64(audio)

        f0, t = pw.harvest(
            audio,
            self.sample_rate,
            f0_floor=f0_min,
            f0_ceil=f0_max,
            frame_period=self.frame_period,
        )

        return f0.astype(np.float32)

    def pm(
        self,
        audio: np.ndarray,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
    ) -> np.ndarray:
        """
        PM (Pitch Mark) estimation using DIO + StoneMask.

        Note: In the original PyWorld, "PM" refers to pitch marks for synthesis.
        Here we use it as an alias for DIO with StoneMask refinement,
        which provides a good balance of speed and accuracy.

        Args:
            audio: Audio signal (1D numpy array)
            f0_min: Minimum F0 to detect (Hz)
            f0_max: Maximum F0 to detect (Hz)

        Returns:
            F0 contour as 1D numpy array (Hz), 0 for unvoiced frames
        """
        # PM is essentially DIO + StoneMask (already default behavior)
        return self.dio(audio, f0_min, f0_max, use_stonemask=True)

    def get_spectral_envelope(
        self,
        audio: np.ndarray,
        f0: np.ndarray,
    ) -> np.ndarray:
        """
        Extract spectral envelope using CheapTrick algorithm.

        This is used for voice synthesis, not pitch detection.

        Args:
            audio: Audio signal
            f0: F0 contour from DIO or HARVEST

        Returns:
            Spectral envelope (frames, freq_bins)
        """
        audio = self._ensure_float64(audio)

        # Get time axis
        t = np.arange(len(f0)) * self.frame_period / 1000.0

        sp = pw.cheaptrick(audio, f0, t, self.sample_rate)
        return sp.astype(np.float32)

    def get_aperiodicity(
        self,
        audio: np.ndarray,
        f0: np.ndarray,
    ) -> np.ndarray:
        """
        Extract aperiodicity using D4C algorithm.

        This is used for voice synthesis, not pitch detection.

        Args:
            audio: Audio signal
            f0: F0 contour from DIO or HARVEST

        Returns:
            Aperiodicity spectrum (frames, freq_bins)
        """
        audio = self._ensure_float64(audio)

        # Get time axis
        t = np.arange(len(f0)) * self.frame_period / 1000.0

        ap = pw.d4c(audio, f0, t, self.sample_rate)
        return ap.astype(np.float32)

    @staticmethod
    def _ensure_float64(audio: np.ndarray) -> np.ndarray:
        """Convert audio to float64 (required by PyWorld)."""
        if audio.dtype != np.float64:
            audio = audio.astype(np.float64)
        return audio


# Convenience functions for direct use

def dio(
    audio: np.ndarray,
    sample_rate: int = 16000,
    hop_size: int = 160,
    f0_min: float = 50.0,
    f0_max: float = 1100.0,
) -> np.ndarray:
    """
    Extract F0 using DIO algorithm.

    Args:
        audio: Audio signal (1D numpy array)
        sample_rate: Sample rate in Hz
        hop_size: Hop size in samples
        f0_min: Minimum F0 (Hz)
        f0_max: Maximum F0 (Hz)

    Returns:
        F0 contour (Hz), 0 for unvoiced
    """
    extractor = PyWorldExtractor(sample_rate, hop_size)
    return extractor.dio(audio, f0_min, f0_max)


def harvest(
    audio: np.ndarray,
    sample_rate: int = 16000,
    hop_size: int = 160,
    f0_min: float = 50.0,
    f0_max: float = 1100.0,
) -> np.ndarray:
    """
    Extract F0 using HARVEST algorithm.

    Args:
        audio: Audio signal (1D numpy array)
        sample_rate: Sample rate in Hz
        hop_size: Hop size in samples
        f0_min: Minimum F0 (Hz)
        f0_max: Maximum F0 (Hz)

    Returns:
        F0 contour (Hz), 0 for unvoiced
    """
    extractor = PyWorldExtractor(sample_rate, hop_size)
    return extractor.harvest(audio, f0_min, f0_max)


def pm(
    audio: np.ndarray,
    sample_rate: int = 16000,
    hop_size: int = 160,
    f0_min: float = 50.0,
    f0_max: float = 1100.0,
) -> np.ndarray:
    """
    Extract F0 using DIO + StoneMask refinement.

    Args:
        audio: Audio signal (1D numpy array)
        sample_rate: Sample rate in Hz
        hop_size: Hop size in samples
        f0_min: Minimum F0 (Hz)
        f0_max: Maximum F0 (Hz)

    Returns:
        F0 contour (Hz), 0 for unvoiced
    """
    extractor = PyWorldExtractor(sample_rate, hop_size)
    return extractor.pm(audio, f0_min, f0_max)
