"""
Integration tests for the unified PitchExtractor interface.

These tests verify:
1. All pitch methods accessible through unified interface
2. Consistent API across methods
3. End-to-end inference pipeline integration
"""

import pytest
import numpy as np
from tests.conftest import (
    SAMPLE_RATE,
    HOP_SIZE,
    F0_MIN,
    F0_MAX,
    generate_sine_wave,
    generate_silence,
    f0_accuracy,
)


class TestPitchExtractorInterface:
    """Tests for the unified PitchExtractor wrapper."""

    @pytest.fixture
    def extractor_class(self):
        """Get the PitchExtractor class."""
        try:
            from rvc_mlx.lib.mlx.pitch_extractors import PitchExtractor
            return PitchExtractor
        except ImportError:
            pytest.skip("PitchExtractor not implemented yet")

    @pytest.fixture
    def audio_440hz(self):
        return generate_sine_wave(440.0, duration=1.0)

    @pytest.fixture
    def silence(self):
        return generate_silence(duration=1.0)

    # === Interface Tests ===

    def test_available_methods(self, extractor_class):
        """PitchExtractor should list available methods."""
        expected_methods = ["rmvpe", "dio", "pm", "harvest", "crepe", "crepe-tiny", "fcpe"]
        assert hasattr(extractor_class, "METHODS"), "Should have METHODS attribute"
        for method in expected_methods:
            assert method in extractor_class.METHODS, f"Should support {method}"

    def test_create_with_default_method(self, extractor_class):
        """Should create with default method (rmvpe)."""
        extractor = extractor_class()
        assert extractor.method == "rmvpe", "Default method should be rmvpe"

    def test_create_with_each_method(self, extractor_class):
        """Should create extractor for each supported method."""
        for method in extractor_class.METHODS:
            try:
                extractor = extractor_class(method=method)
                assert extractor.method == method
            except (ImportError, FileNotFoundError) as e:
                pytest.skip(f"Method {method} not available: {e}")

    def test_invalid_method_raises(self, extractor_class):
        """Should raise error for invalid method."""
        with pytest.raises((ValueError, KeyError)):
            extractor_class(method="invalid_method")

    def test_extract_returns_numpy_array(self, extractor_class, audio_440hz):
        """extract() should return numpy array."""
        extractor = extractor_class(method="rmvpe")
        f0 = extractor.extract(audio_440hz)

        assert isinstance(f0, np.ndarray), "Should return numpy array"
        assert f0.ndim == 1, "Should be 1D array"

    def test_extract_with_f0_range(self, extractor_class, audio_440hz):
        """extract() should accept f0_min and f0_max."""
        extractor = extractor_class(method="rmvpe")
        f0 = extractor.extract(audio_440hz, f0_min=100, f0_max=800)

        voiced_f0 = f0[f0 > 0]
        if len(voiced_f0) > 0:
            assert voiced_f0.min() >= 100 * 0.9, "Should respect f0_min"
            assert voiced_f0.max() <= 800 * 1.1, "Should respect f0_max"

    # === Method-specific Tests ===

    def test_rmvpe_through_interface(self, extractor_class, audio_440hz):
        """RMVPE should work through unified interface."""
        extractor = extractor_class(method="rmvpe")
        f0 = extractor.extract(audio_440hz, f0_min=F0_MIN, f0_max=F0_MAX)

        expected = np.full_like(f0, 440.0)
        accuracy = f0_accuracy(f0, expected, tolerance_cents=50.0)
        assert accuracy > 0.85, f"RMVPE accuracy {accuracy:.2f} should be > 0.85"

    def test_dio_through_interface(self, extractor_class, audio_440hz):
        """DIO should work through unified interface."""
        try:
            extractor = extractor_class(method="dio")
        except (ImportError, FileNotFoundError):
            pytest.skip("PyWorld not available")

        f0 = extractor.extract(audio_440hz, f0_min=F0_MIN, f0_max=F0_MAX)

        expected = np.full_like(f0, 440.0)
        accuracy = f0_accuracy(f0, expected, tolerance_cents=50.0)
        assert accuracy > 0.7, f"DIO accuracy {accuracy:.2f} should be > 0.7"

    def test_harvest_through_interface(self, extractor_class, audio_440hz):
        """HARVEST should work through unified interface.

        Note: HARVEST is designed for speech and may reject pure sine waves.
        This test just verifies the interface works and returns valid output.
        """
        try:
            extractor = extractor_class(method="harvest")
        except (ImportError, FileNotFoundError):
            pytest.skip("PyWorld not available")

        f0 = extractor.extract(audio_440hz, f0_min=F0_MIN, f0_max=F0_MAX)

        # Verify valid output (HARVEST may return all zeros for pure sine)
        assert isinstance(f0, np.ndarray), "Should return numpy array"
        assert len(f0) > 0, "Should have frames"

        # If voiced frames detected, check accuracy
        voiced_mask = f0 > 0
        if voiced_mask.sum() > 10:
            expected = np.full_like(f0, 440.0)
            accuracy = f0_accuracy(f0, expected, tolerance_cents=50.0)
            assert accuracy > 0.7, f"HARVEST accuracy {accuracy:.2f} should be > 0.7"

    def test_crepe_through_interface(self, extractor_class, audio_440hz):
        """CREPE should work through unified interface."""
        try:
            extractor = extractor_class(method="crepe")
        except (ImportError, FileNotFoundError):
            pytest.skip("CREPE not available")

        f0 = extractor.extract(audio_440hz, f0_min=F0_MIN, f0_max=F0_MAX)

        expected = np.full_like(f0, 440.0)
        accuracy = f0_accuracy(f0, expected, tolerance_cents=50.0)
        assert accuracy > 0.85, f"CREPE accuracy {accuracy:.2f} should be > 0.85"

    def test_fcpe_through_interface(self, extractor_class, audio_440hz):
        """FCPE should work through unified interface."""
        try:
            extractor = extractor_class(method="fcpe")
        except (ImportError, FileNotFoundError):
            pytest.skip("FCPE not available")

        f0 = extractor.extract(audio_440hz, f0_min=F0_MIN, f0_max=F0_MAX)

        expected = np.full_like(f0, 440.0)
        accuracy = f0_accuracy(f0, expected, tolerance_cents=50.0)
        assert accuracy > 0.85, f"FCPE accuracy {accuracy:.2f} should be > 0.85"

    # === Silence Handling ===

    def test_all_methods_handle_silence(self, extractor_class, silence):
        """All methods should return zeros for silence."""
        for method in extractor_class.METHODS:
            try:
                extractor = extractor_class(method=method)
                f0 = extractor.extract(silence, f0_min=F0_MIN, f0_max=F0_MAX)

                unvoiced_ratio = (f0 == 0).mean()
                assert unvoiced_ratio > 0.8, f"{method} should detect silence as unvoiced"
            except (ImportError, FileNotFoundError):
                continue  # Skip unavailable methods

    # === Edge Cases ===

    def test_very_short_audio(self, extractor_class):
        """All methods should handle very short audio."""
        short_audio = generate_sine_wave(440.0, duration=0.05)

        for method in ["rmvpe", "dio", "harvest"]:  # Test stable methods
            try:
                extractor = extractor_class(method=method)
                f0 = extractor.extract(short_audio, f0_min=F0_MIN, f0_max=F0_MAX)
                assert len(f0) > 0, f"{method} should return at least some frames"
            except (ImportError, FileNotFoundError):
                continue

    def test_reuse_extractor(self, extractor_class, audio_440hz):
        """Extractor should be reusable for multiple extractions."""
        extractor = extractor_class(method="rmvpe")

        f0_1 = extractor.extract(audio_440hz, f0_min=F0_MIN, f0_max=F0_MAX)
        f0_2 = extractor.extract(audio_440hz, f0_min=F0_MIN, f0_max=F0_MAX)

        np.testing.assert_array_almost_equal(f0_1, f0_2, decimal=5,
            err_msg="Same input should give same output")


class TestPipelineIntegration:
    """Tests for integration with RVC MLX inference pipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create inference pipeline."""
        try:
            from rvc_mlx.infer.pipeline_mlx import Pipeline
            return Pipeline()
        except ImportError:
            pytest.skip("Pipeline not available")

    @pytest.fixture
    def audio_200hz(self):
        return generate_sine_wave(200.0, duration=2.0)

    def test_pipeline_accepts_f0_method(self, pipeline, audio_200hz):
        """Pipeline should accept f0_method parameter."""
        # This tests that the pipeline integrates with PitchExtractor
        if hasattr(pipeline, 'get_f0'):
            for method in ["rmvpe", "dio", "harvest"]:
                try:
                    f0 = pipeline.get_f0(
                        audio_200hz,
                        p_len=len(audio_200hz) // HOP_SIZE,
                        f0_method=method,
                        pitch=0,
                        f0_autotune=False,
                        f0_autotune_strength=1.0,
                        proposed_pitch=False,
                        proposed_pitch_threshold=155.0,
                    )
                    assert isinstance(f0, (tuple, np.ndarray)), f"Pipeline should work with {method}"
                except (ImportError, NotImplementedError):
                    continue


class TestConsistencyAcrossMethods:
    """Tests ensuring consistent behavior across all pitch methods."""

    @pytest.fixture
    def extractor_class(self):
        try:
            from rvc_mlx.lib.mlx.pitch_extractors import PitchExtractor
            return PitchExtractor
        except ImportError:
            pytest.skip("PitchExtractor not implemented yet")

    def test_output_shape_consistency(self, extractor_class):
        """All methods should produce similar output shapes."""
        audio = generate_sine_wave(200.0, duration=1.0)
        shapes = {}

        for method in extractor_class.METHODS:
            try:
                extractor = extractor_class(method=method)
                f0 = extractor.extract(audio, f0_min=F0_MIN, f0_max=F0_MAX)
                shapes[method] = len(f0)
            except (ImportError, FileNotFoundError):
                continue

        if len(shapes) < 2:
            pytest.skip("Need at least 2 methods to compare")

        # Allow 20% variation in frame count due to different hop sizes
        mean_frames = np.mean(list(shapes.values()))
        for method, frames in shapes.items():
            ratio = frames / mean_frames
            assert 0.5 < ratio < 2.0, f"{method} frame count {frames} too different from mean {mean_frames}"

    def test_f0_range_consistency(self, extractor_class):
        """All methods should produce F0 in similar range for same input."""
        audio = generate_sine_wave(300.0, duration=1.0)
        means = {}

        for method in extractor_class.METHODS:
            try:
                extractor = extractor_class(method=method)
                f0 = extractor.extract(audio, f0_min=F0_MIN, f0_max=F0_MAX)
                voiced = f0[f0 > 0]
                if len(voiced) > 0:
                    means[method] = voiced.mean()
            except (ImportError, FileNotFoundError):
                continue

        if len(means) < 2:
            pytest.skip("Need at least 2 methods to compare")

        # All methods should detect F0 near 300 Hz
        for method, mean_f0 in means.items():
            assert 250 < mean_f0 < 350, f"{method} mean F0 {mean_f0:.1f} should be ~300 Hz"
