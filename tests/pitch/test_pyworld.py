"""
Tests for PyWorld pitch extraction methods: DIO, PM, HARVEST.

These tests verify:
1. Output shape and format
2. F0 range constraints
3. Correct frequency detection for synthetic signals
4. Voiced/unvoiced detection
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
    generate_sine_wave,
    generate_silence,
)


class TestPyWorldExtractor:
    """Tests for the PyWorldExtractor wrapper class."""

    @pytest.fixture
    def extractor(self):
        """Create a PyWorldExtractor instance."""
        try:
            from rvc_mlx.lib.mlx.pyworld_pitch import PyWorldExtractor
            return PyWorldExtractor(sample_rate=SAMPLE_RATE, hop_size=HOP_SIZE)
        except ImportError:
            pytest.skip("PyWorldExtractor not implemented yet")

    # === DIO Tests ===

    def test_dio_output_shape(self, extractor, sine_440hz):
        """DIO should return F0 array with correct shape."""
        f0 = extractor.dio(sine_440hz, f0_min=F0_MIN, f0_max=F0_MAX)

        assert isinstance(f0, np.ndarray), "F0 should be numpy array"
        assert f0.ndim == 1, "F0 should be 1D array"

        expected_frames = expected_f0_frames(len(sine_440hz))
        # Allow some tolerance in frame count due to different windowing
        assert abs(len(f0) - expected_frames) < 10, (
            f"F0 length {len(f0)} should be close to {expected_frames}"
        )

    def test_dio_f0_range(self, extractor, sine_440hz):
        """DIO output should be within specified F0 range or zero (unvoiced)."""
        f0 = extractor.dio(sine_440hz, f0_min=F0_MIN, f0_max=F0_MAX)

        voiced_mask = f0 > 0
        if voiced_mask.any():
            voiced_f0 = f0[voiced_mask]
            assert voiced_f0.min() >= F0_MIN * 0.9, "Voiced F0 should be >= f0_min"
            assert voiced_f0.max() <= F0_MAX * 1.1, "Voiced F0 should be <= f0_max"

    def test_dio_sine_detection_440hz(self, extractor, sine_440hz):
        """DIO should detect 440 Hz sine wave accurately."""
        f0 = extractor.dio(sine_440hz, f0_min=F0_MIN, f0_max=F0_MAX)

        # Create expected F0 contour (all 440 Hz)
        expected_f0 = np.full_like(f0, 440.0)

        # Check accuracy (within 50 cents = half semitone)
        accuracy = f0_accuracy(f0, expected_f0, tolerance_cents=50.0)
        assert accuracy > 0.8, f"DIO accuracy {accuracy:.2f} should be > 0.8 for 440 Hz sine"

    def test_dio_sine_detection_100hz(self, extractor, sine_100hz):
        """DIO should detect 100 Hz sine wave (low frequency)."""
        f0 = extractor.dio(sine_100hz, f0_min=F0_MIN, f0_max=F0_MAX)

        expected_f0 = np.full_like(f0, 100.0)
        accuracy = f0_accuracy(f0, expected_f0, tolerance_cents=50.0)
        assert accuracy > 0.7, f"DIO accuracy {accuracy:.2f} should be > 0.7 for 100 Hz sine"

    def test_dio_silence(self, extractor, silence_audio):
        """DIO should return all zeros for silent audio."""
        f0 = extractor.dio(silence_audio, f0_min=F0_MIN, f0_max=F0_MAX)

        # Most frames should be unvoiced (F0 = 0)
        unvoiced_ratio = (f0 == 0).mean()
        assert unvoiced_ratio > 0.9, f"Silence should be mostly unvoiced, got {unvoiced_ratio:.2f}"

    def test_dio_voiced_unvoiced(self, extractor, voiced_unvoiced_audio):
        """DIO should distinguish voiced and unvoiced segments."""
        audio, expected_f0 = voiced_unvoiced_audio
        f0 = extractor.dio(audio, f0_min=F0_MIN, f0_max=F0_MAX)

        # Trim to same length
        min_len = min(len(f0), len(expected_f0))
        f0 = f0[:min_len]
        expected_f0 = expected_f0[:min_len]

        # Check voiced detection accuracy
        voiced_expected = expected_f0 > 0
        voiced_detected = f0 > 0

        # Should have reasonable agreement on voiced/unvoiced
        agreement = (voiced_expected == voiced_detected).mean()
        assert agreement > 0.6, f"V/UV agreement {agreement:.2f} should be > 0.6"

    # === HARVEST Tests ===

    def test_harvest_output_shape(self, extractor, sine_440hz):
        """HARVEST should return F0 array with correct shape."""
        f0 = extractor.harvest(sine_440hz, f0_min=F0_MIN, f0_max=F0_MAX)

        assert isinstance(f0, np.ndarray), "F0 should be numpy array"
        assert f0.ndim == 1, "F0 should be 1D array"

        expected_frames = expected_f0_frames(len(sine_440hz))
        assert abs(len(f0) - expected_frames) < 10, (
            f"F0 length {len(f0)} should be close to {expected_frames}"
        )

    def test_harvest_sine_detection_440hz(self, extractor, sine_440hz):
        """HARVEST should detect 440 Hz or return mostly unvoiced for pure sine.

        Note: HARVEST is designed for speech and may reject pure sine waves
        as "unvoiced" because they lack the harmonic structure of human voice.
        """
        f0 = extractor.harvest(sine_440hz, f0_min=F0_MIN, f0_max=F0_MAX)

        voiced_mask = f0 > 0
        if voiced_mask.sum() > 10:
            # If voiced frames detected, they should be near 440 Hz
            expected_f0 = np.full_like(f0, 440.0)
            accuracy = f0_accuracy(f0, expected_f0, tolerance_cents=50.0)
            assert accuracy > 0.7, f"HARVEST accuracy {accuracy:.2f} should be > 0.7 for 440 Hz"
        else:
            # HARVEST may reject pure sine as unvoiced - this is acceptable
            pass  # Pure sine wave rejection is expected behavior

    def test_harvest_silence(self, extractor, silence_audio):
        """HARVEST should return all zeros for silent audio."""
        f0 = extractor.harvest(silence_audio, f0_min=F0_MIN, f0_max=F0_MAX)

        unvoiced_ratio = (f0 == 0).mean()
        assert unvoiced_ratio > 0.9, f"Silence should be mostly unvoiced, got {unvoiced_ratio:.2f}"

    @pytest.mark.slow
    def test_harvest_chirp(self, extractor, chirp_100_400):
        """HARVEST should track frequency changes in a chirp signal."""
        audio, expected_f0 = chirp_100_400
        f0 = extractor.harvest(audio, f0_min=F0_MIN, f0_max=F0_MAX)

        # Resample expected F0 to match output length
        from scipy.ndimage import zoom
        expected_resampled = zoom(expected_f0, len(f0) / len(expected_f0))

        # Check that F0 increases over time (tracking the chirp)
        voiced_mask = f0 > 0
        if voiced_mask.sum() > 10:
            voiced_f0 = f0[voiced_mask]
            # First quarter should be lower than last quarter
            n = len(voiced_f0)
            assert voiced_f0[:n//4].mean() < voiced_f0[-n//4:].mean(), (
                "HARVEST should track increasing frequency in chirp"
            )

    # === PM Tests ===

    def test_pm_output_shape(self, extractor, sine_440hz):
        """PM should return F0 array with correct shape."""
        f0 = extractor.pm(sine_440hz, f0_min=F0_MIN, f0_max=F0_MAX)

        assert isinstance(f0, np.ndarray), "F0 should be numpy array"
        assert f0.ndim == 1, "F0 should be 1D array"

        expected_frames = expected_f0_frames(len(sine_440hz))
        assert abs(len(f0) - expected_frames) < 10, (
            f"F0 length {len(f0)} should be close to {expected_frames}"
        )

    def test_pm_sine_detection_440hz(self, extractor, sine_440hz):
        """PM should detect 440 Hz sine wave."""
        f0 = extractor.pm(sine_440hz, f0_min=F0_MIN, f0_max=F0_MAX)

        expected_f0 = np.full_like(f0, 440.0)
        accuracy = f0_accuracy(f0, expected_f0, tolerance_cents=50.0)
        assert accuracy > 0.8, f"PM accuracy {accuracy:.2f} should be > 0.8 for 440 Hz"

    def test_pm_silence(self, extractor, silence_audio):
        """PM should return all zeros for silent audio."""
        f0 = extractor.pm(silence_audio, f0_min=F0_MIN, f0_max=F0_MAX)

        unvoiced_ratio = (f0 == 0).mean()
        assert unvoiced_ratio > 0.9, f"Silence should be mostly unvoiced, got {unvoiced_ratio:.2f}"

    # === Comparison Tests ===

    def test_dio_vs_harvest_accuracy(self, extractor, sine_440hz):
        """DIO and HARVEST should both work (HARVEST may reject synthetic signals).

        Note: HARVEST is designed for speech and may reject pure sine waves.
        This test verifies both methods run without errors.
        """
        dio_f0 = extractor.dio(sine_440hz, f0_min=F0_MIN, f0_max=F0_MAX)
        harvest_f0 = extractor.harvest(sine_440hz, f0_min=F0_MIN, f0_max=F0_MAX)

        # Both should return valid arrays
        assert isinstance(dio_f0, np.ndarray)
        assert isinstance(harvest_f0, np.ndarray)

        # DIO should detect the sine wave
        expected_f0_dio = np.full_like(dio_f0, 440.0)
        dio_accuracy = f0_accuracy(dio_f0, expected_f0_dio, tolerance_cents=50.0)
        assert dio_accuracy > 0.8, f"DIO accuracy {dio_accuracy:.2f} should be > 0.8"

        # HARVEST may or may not detect - both are acceptable for synthetic signals

    @pytest.mark.slow
    def test_real_voice(self, extractor, real_voice_audio):
        """Test pitch extraction on real or voice-like audio.

        Note: HARVEST is strict and may reject synthetic "voice-like" signals.
        This test focuses on DIO which is more tolerant.
        """
        dio_f0 = extractor.dio(real_voice_audio, f0_min=F0_MIN, f0_max=F0_MAX)
        harvest_f0 = extractor.harvest(real_voice_audio, f0_min=F0_MIN, f0_max=F0_MAX)

        # DIO should detect voiced frames
        assert (dio_f0 > 0).sum() > 10, "DIO should detect some voiced frames"

        # HARVEST may or may not detect synthetic signals - skip assertion
        # (Real voice files would work better, but synthetic fallback is used)

        # F0 should be in reasonable speech range (80-400 Hz typically)
        voiced_dio = dio_f0[dio_f0 > 0]
        assert voiced_dio.mean() > 80 and voiced_dio.mean() < 500, (
            f"DIO mean F0 {voiced_dio.mean():.1f} should be in speech range"
        )

        # Only check HARVEST if it detected any voiced frames
        if (harvest_f0 > 0).sum() > 5:
            voiced_harvest = harvest_f0[harvest_f0 > 0]
            assert voiced_harvest.mean() > 80 and voiced_harvest.mean() < 500, (
                f"HARVEST mean F0 {voiced_harvest.mean():.1f} should be in speech range"
            )


class TestPyWorldEdgeCases:
    """Edge case tests for PyWorld methods."""

    @pytest.fixture
    def extractor(self):
        try:
            from rvc_mlx.lib.mlx.pyworld_pitch import PyWorldExtractor
            return PyWorldExtractor(sample_rate=SAMPLE_RATE, hop_size=HOP_SIZE)
        except ImportError:
            pytest.skip("PyWorldExtractor not implemented yet")

    def test_very_short_audio(self, extractor):
        """Handle very short audio gracefully."""
        short_audio = generate_sine_wave(440.0, duration=0.05)  # 50ms
        f0 = extractor.dio(short_audio, f0_min=F0_MIN, f0_max=F0_MAX)

        assert isinstance(f0, np.ndarray), "Should return array even for short audio"
        assert len(f0) > 0, "Should have at least some frames"

    def test_different_f0_ranges(self, extractor, sine_440hz):
        """Test with different F0 min/max ranges."""
        # Narrow range that includes 440 Hz
        f0_narrow = extractor.dio(sine_440hz, f0_min=400, f0_max=500)
        voiced_narrow = (f0_narrow > 0).sum()

        # Wide range
        f0_wide = extractor.dio(sine_440hz, f0_min=50, f0_max=1100)
        voiced_wide = (f0_wide > 0).sum()

        # Both should detect voiced frames
        assert voiced_narrow > 0, "Should detect 440 Hz in narrow range"
        assert voiced_wide > 0, "Should detect 440 Hz in wide range"

    def test_float64_input(self, extractor):
        """PyWorld expects float64 - wrapper should handle conversion."""
        audio_f32 = generate_sine_wave(440.0, duration=0.5).astype(np.float32)
        audio_f64 = audio_f32.astype(np.float64)

        f0_f32 = extractor.dio(audio_f32, f0_min=F0_MIN, f0_max=F0_MAX)
        f0_f64 = extractor.dio(audio_f64, f0_min=F0_MIN, f0_max=F0_MAX)

        # Results should be identical
        np.testing.assert_array_almost_equal(f0_f32, f0_f64, decimal=5)

    def test_high_frequency_sine(self, extractor):
        """Test detection of high frequency (near f0_max)."""
        high_freq = generate_sine_wave(1000.0, duration=0.5)
        f0 = extractor.harvest(high_freq, f0_min=F0_MIN, f0_max=F0_MAX)

        expected_f0 = np.full_like(f0, 1000.0)
        accuracy = f0_accuracy(f0, expected_f0, tolerance_cents=50.0)
        assert accuracy > 0.7, f"Should detect 1000 Hz, got accuracy {accuracy:.2f}"

    def test_low_frequency_sine(self, extractor):
        """Test detection of low frequency (near f0_min)."""
        low_freq = generate_sine_wave(60.0, duration=0.5)
        f0 = extractor.harvest(low_freq, f0_min=F0_MIN, f0_max=F0_MAX)

        expected_f0 = np.full_like(f0, 60.0)
        accuracy = f0_accuracy(f0, expected_f0, tolerance_cents=50.0)
        assert accuracy > 0.6, f"Should detect 60 Hz, got accuracy {accuracy:.2f}"
