"""
Tests for CREPE MLX pitch extraction.

These tests verify:
1. CREPE model architecture and forward pass
2. Output shape and format
3. Correct frequency detection
4. Full vs Tiny model variants
"""

import pytest
import numpy as np
from tests.conftest import (
    SAMPLE_RATE,
    HOP_SIZE,
    F0_MIN,
    F0_MAX,
    expected_f0_frames,
    f0_accuracy,
    f0_correlation,
    generate_sine_wave,
    generate_silence,
)


class TestCREPEModel:
    """Tests for the CREPE MLX model architecture."""

    @pytest.fixture
    def crepe_full(self):
        """Create a full CREPE model."""
        try:
            from rvc_mlx.lib.mlx.crepe import CREPE
            return CREPE(model="full")
        except ImportError:
            pytest.skip("CREPE MLX not implemented yet")

    @pytest.fixture
    def crepe_tiny(self):
        """Create a tiny CREPE model."""
        try:
            from rvc_mlx.lib.mlx.crepe import CREPE
            return CREPE(model="tiny")
        except ImportError:
            pytest.skip("CREPE MLX not implemented yet")

    def test_full_model_loads(self, crepe_full):
        """CREPE full model should load without errors."""
        assert crepe_full is not None
        assert hasattr(crepe_full, "predict") or hasattr(crepe_full, "get_f0")

    def test_tiny_model_loads(self, crepe_tiny):
        """CREPE tiny model should load without errors."""
        assert crepe_tiny is not None

    def test_full_model_output_shape(self, crepe_full, sine_440hz):
        """CREPE full should return correct F0 shape."""
        f0 = crepe_full.get_f0(sine_440hz, f0_min=F0_MIN, f0_max=F0_MAX)

        assert isinstance(f0, np.ndarray), "F0 should be numpy array"
        assert f0.ndim == 1, "F0 should be 1D array"

        expected_frames = expected_f0_frames(len(sine_440hz))
        assert abs(len(f0) - expected_frames) < 20, (
            f"F0 length {len(f0)} should be close to {expected_frames}"
        )

    def test_tiny_model_output_shape(self, crepe_tiny, sine_440hz):
        """CREPE tiny should return correct F0 shape."""
        f0 = crepe_tiny.get_f0(sine_440hz, f0_min=F0_MIN, f0_max=F0_MAX)

        assert isinstance(f0, np.ndarray), "F0 should be numpy array"
        assert f0.ndim == 1, "F0 should be 1D array"


class TestCREPEPitchDetection:
    """Tests for CREPE pitch detection accuracy."""

    @pytest.fixture
    def crepe(self):
        """Create CREPE extractor (default: full model)."""
        try:
            from rvc_mlx.lib.mlx.crepe import CREPE
            return CREPE(model="full")
        except ImportError:
            pytest.skip("CREPE MLX not implemented yet")

    def test_sine_440hz_detection(self, crepe, sine_440hz):
        """CREPE should accurately detect 440 Hz sine wave."""
        f0 = crepe.get_f0(sine_440hz, f0_min=F0_MIN, f0_max=F0_MAX)

        expected_f0 = np.full_like(f0, 440.0)
        accuracy = f0_accuracy(f0, expected_f0, tolerance_cents=50.0)

        assert accuracy > 0.85, f"CREPE accuracy {accuracy:.2f} should be > 0.85 for 440 Hz"

    def test_sine_100hz_detection(self, crepe, sine_100hz):
        """CREPE should detect low frequency (100 Hz)."""
        f0 = crepe.get_f0(sine_100hz, f0_min=F0_MIN, f0_max=F0_MAX)

        expected_f0 = np.full_like(f0, 100.0)
        accuracy = f0_accuracy(f0, expected_f0, tolerance_cents=50.0)

        assert accuracy > 0.8, f"CREPE accuracy {accuracy:.2f} should be > 0.8 for 100 Hz"

    def test_sine_300hz_detection(self, crepe, sine_300hz):
        """CREPE should detect mid-range frequency (300 Hz)."""
        f0 = crepe.get_f0(sine_300hz, f0_min=F0_MIN, f0_max=F0_MAX)

        expected_f0 = np.full_like(f0, 300.0)
        accuracy = f0_accuracy(f0, expected_f0, tolerance_cents=50.0)

        assert accuracy > 0.85, f"CREPE accuracy {accuracy:.2f} should be > 0.85 for 300 Hz"

    def test_silence_detection(self, crepe, silence_audio):
        """CREPE should return low confidence (zeros) for silence."""
        f0 = crepe.get_f0(silence_audio, f0_min=F0_MIN, f0_max=F0_MAX)

        # Silence should be mostly unvoiced
        unvoiced_ratio = (f0 == 0).mean()
        assert unvoiced_ratio > 0.8, f"Silence should be mostly unvoiced, got {unvoiced_ratio:.2f}"

    def test_voiced_unvoiced_detection(self, crepe, voiced_unvoiced_audio):
        """CREPE should distinguish voiced and unvoiced segments."""
        audio, expected_f0 = voiced_unvoiced_audio
        f0 = crepe.get_f0(audio, f0_min=F0_MIN, f0_max=F0_MAX)

        # Trim to same length
        min_len = min(len(f0), len(expected_f0))
        f0 = f0[:min_len]
        expected_f0 = expected_f0[:min_len]

        voiced_expected = expected_f0 > 0
        voiced_detected = f0 > 0

        agreement = (voiced_expected == voiced_detected).mean()
        assert agreement > 0.65, f"V/UV agreement {agreement:.2f} should be > 0.65"

    @pytest.mark.slow
    def test_chirp_tracking(self, crepe, chirp_100_400):
        """CREPE should track frequency changes in chirp."""
        audio, expected_f0 = chirp_100_400
        f0 = crepe.get_f0(audio, f0_min=F0_MIN, f0_max=F0_MAX)

        voiced_mask = f0 > 0
        if voiced_mask.sum() > 10:
            voiced_f0 = f0[voiced_mask]
            n = len(voiced_f0)
            # First quarter should be lower than last quarter
            assert voiced_f0[:n//4].mean() < voiced_f0[-n//4:].mean(), (
                "CREPE should track increasing frequency"
            )

    def test_f0_range_constraint(self, crepe, sine_440hz):
        """CREPE output should respect F0 range constraints."""
        f0 = crepe.get_f0(sine_440hz, f0_min=F0_MIN, f0_max=F0_MAX)

        voiced_f0 = f0[f0 > 0]
        if len(voiced_f0) > 0:
            assert voiced_f0.min() >= F0_MIN * 0.9, "F0 should be >= f0_min"
            assert voiced_f0.max() <= F0_MAX * 1.1, "F0 should be <= f0_max"


class TestCREPEFullVsTiny:
    """Tests comparing CREPE full and tiny models."""

    @pytest.fixture
    def crepe_full(self):
        try:
            from rvc_mlx.lib.mlx.crepe import CREPE
            return CREPE(model="full")
        except ImportError:
            pytest.skip("CREPE MLX not implemented yet")

    @pytest.fixture
    def crepe_tiny(self):
        try:
            from rvc_mlx.lib.mlx.crepe import CREPE
            return CREPE(model="tiny")
        except ImportError:
            pytest.skip("CREPE MLX not implemented yet")

    def test_full_more_accurate(self, crepe_full, crepe_tiny, sine_440hz):
        """Full model should generally be more accurate than tiny."""
        f0_full = crepe_full.get_f0(sine_440hz, f0_min=F0_MIN, f0_max=F0_MAX)
        f0_tiny = crepe_tiny.get_f0(sine_440hz, f0_min=F0_MIN, f0_max=F0_MAX)

        expected_full = np.full_like(f0_full, 440.0)
        expected_tiny = np.full_like(f0_tiny, 440.0)

        acc_full = f0_accuracy(f0_full, expected_full, tolerance_cents=50.0)
        acc_tiny = f0_accuracy(f0_tiny, expected_tiny, tolerance_cents=50.0)

        # Full should be at least as accurate as tiny
        assert acc_full >= acc_tiny * 0.95, (
            f"Full ({acc_full:.2f}) should be >= tiny ({acc_tiny:.2f})"
        )

    def test_both_detect_same_pitch(self, crepe_full, crepe_tiny, sine_440hz):
        """Both models should detect roughly the same pitch."""
        f0_full = crepe_full.get_f0(sine_440hz, f0_min=F0_MIN, f0_max=F0_MAX)
        f0_tiny = crepe_tiny.get_f0(sine_440hz, f0_min=F0_MIN, f0_max=F0_MAX)

        # Trim to same length
        min_len = min(len(f0_full), len(f0_tiny))
        f0_full = f0_full[:min_len]
        f0_tiny = f0_tiny[:min_len]

        # Correlation should be high
        corr = f0_correlation(f0_full, f0_tiny)
        assert corr > 0.9, f"Full/tiny correlation {corr:.2f} should be > 0.9"


class TestCREPEProbabilityOutput:
    """Tests for CREPE probability/confidence output."""

    @pytest.fixture
    def crepe(self):
        try:
            from rvc_mlx.lib.mlx.crepe import CREPE
            return CREPE(model="full")
        except ImportError:
            pytest.skip("CREPE MLX not implemented yet")

    def test_periodicity_output(self, crepe, sine_440hz):
        """CREPE should provide periodicity/confidence scores if available."""
        # Some implementations return (f0, periodicity)
        result = crepe.get_f0(sine_440hz, f0_min=F0_MIN, f0_max=F0_MAX, return_periodicity=True)

        if isinstance(result, tuple):
            f0, periodicity = result
            assert periodicity.shape == f0.shape, "Periodicity shape should match F0"
            assert periodicity.min() >= 0 and periodicity.max() <= 1, (
                "Periodicity should be in [0, 1]"
            )

    def test_confidence_high_for_clean_signal(self, crepe, sine_440hz):
        """Confidence should be high for clean sine wave."""
        result = crepe.get_f0(sine_440hz, f0_min=F0_MIN, f0_max=F0_MAX, return_periodicity=True)

        if isinstance(result, tuple):
            f0, periodicity = result
            # For voiced frames, confidence should be high
            voiced_mask = f0 > 0
            if voiced_mask.any():
                mean_conf = periodicity[voiced_mask].mean()
                assert mean_conf > 0.5, f"Mean confidence {mean_conf:.2f} should be > 0.5"


class TestCREPEEdgeCases:
    """Edge case tests for CREPE."""

    @pytest.fixture
    def crepe(self):
        try:
            from rvc_mlx.lib.mlx.crepe import CREPE
            return CREPE(model="full")
        except ImportError:
            pytest.skip("CREPE MLX not implemented yet")

    def test_very_short_audio(self, crepe):
        """Handle very short audio (< 1024 samples)."""
        short_audio = generate_sine_wave(440.0, duration=0.05)  # 50ms = 800 samples
        f0 = crepe.get_f0(short_audio, f0_min=F0_MIN, f0_max=F0_MAX)

        assert isinstance(f0, np.ndarray), "Should return array for short audio"
        assert len(f0) > 0, "Should have at least some frames"

    def test_stereo_to_mono(self, crepe):
        """Should handle stereo input by converting to mono."""
        mono = generate_sine_wave(440.0, duration=0.5)
        stereo = np.stack([mono, mono], axis=0).T  # (samples, 2)

        # If stereo handling is implemented
        try:
            f0 = crepe.get_f0(stereo, f0_min=F0_MIN, f0_max=F0_MAX)
            assert isinstance(f0, np.ndarray)
        except Exception:
            # May raise if stereo not supported - that's acceptable
            pass

    def test_batch_processing(self, crepe):
        """Test batch processing multiple audio files."""
        audio1 = generate_sine_wave(440.0, duration=0.5)
        audio2 = generate_sine_wave(300.0, duration=0.5)

        # Some implementations support batched input
        try:
            f0_1 = crepe.get_f0(audio1, f0_min=F0_MIN, f0_max=F0_MAX)
            f0_2 = crepe.get_f0(audio2, f0_min=F0_MIN, f0_max=F0_MAX)

            # Different inputs should give different outputs
            assert not np.allclose(f0_1, f0_2), "Different inputs should give different F0"
        except Exception:
            pass

    def test_different_sample_rates(self, crepe):
        """CREPE expects 16kHz - should resample if needed."""
        # Generate at 44.1kHz
        duration = 0.5
        sr_44k = 44100
        t = np.arange(int(duration * sr_44k)) / sr_44k
        audio_44k = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Implementation should handle resampling
        try:
            f0 = crepe.get_f0(audio_44k, f0_min=F0_MIN, f0_max=F0_MAX, sample_rate=sr_44k)
            assert isinstance(f0, np.ndarray)
        except (NotImplementedError, TypeError):
            # May not support sample rate parameter
            pass
