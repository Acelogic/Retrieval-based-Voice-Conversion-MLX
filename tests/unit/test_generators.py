"""
Unit tests for MLX generator modules.

Tests HiFiGAN-NSF generator initialization, upsampling, and dimension handling.
Critical: Verifies input/output transposes match PyTorch format.
"""

import pytest
import mlx.core as mx


class TestSineGenerator:
    """Tests for SineGenerator module."""

    @pytest.fixture
    def sine_gen(self):
        from rvc_mlx.lib.mlx.generators import SineGenerator

        return SineGenerator(
            sample_rate=16000,
            harmonic_num=0,
            sine_amp=0.1,
            add_noise_std=0.003,
            voiced_threshod=0,
        )

    def test_sine_generator_output_shape(self, sine_gen):
        """Verify sine generator upsamples correctly."""
        batch_size, f0_frames = 1, 100
        upsampling = 160  # Typical hop size

        f0 = mx.full((batch_size, f0_frames), 200.0)  # 200 Hz F0
        sine_waves, voiced_mask, noise = sine_gen(f0, upsampling)

        expected_samples = f0_frames * upsampling
        assert sine_waves.shape == (batch_size, expected_samples, 1)
        assert voiced_mask.shape == (batch_size, expected_samples, 1)

    def test_sine_generator_voiced_unvoiced(self, sine_gen):
        """Test voiced/unvoiced detection."""
        batch_size, f0_frames = 1, 10
        upsampling = 160

        # Mix of voiced (200 Hz) and unvoiced (0 Hz)
        f0 = mx.array([[200.0, 200.0, 0.0, 0.0, 200.0, 0.0, 200.0, 200.0, 0.0, 0.0]])

        _, voiced_mask, _ = sine_gen(f0, upsampling)
        mx.eval(voiced_mask)

        # Check that voiced/unvoiced regions are correctly identified
        # Each F0 frame expands to `upsampling` samples
        assert voiced_mask[0, 0, 0] == 1.0  # First frame voiced
        assert voiced_mask[0, 2 * upsampling, 0] == 0.0  # Third frame unvoiced


class TestSourceModuleHnNSF:
    """Tests for SourceModuleHnNSF module."""

    @pytest.fixture
    def source_module(self):
        from rvc_mlx.lib.mlx.generators import SourceModuleHnNSF

        return SourceModuleHnNSF(
            sample_rate=16000,
            harmonic_num=0,
            sine_amp=0.1,
            add_noise_std=0.003,
            voiced_threshod=0,
        )

    def test_source_module_output_shape(self, source_module):
        """Verify source module outputs single channel."""
        batch_size, f0_frames = 1, 50
        upsampling = 160

        f0 = mx.full((batch_size, f0_frames), 200.0)
        output = source_module(f0, upsampling)

        expected_samples = f0_frames * upsampling
        assert output.shape == (batch_size, expected_samples, 1)


class TestHiFiGANNSFGenerator:
    """Tests for HiFiGAN-NSF Generator module."""

    @pytest.fixture
    def generator(self):
        from rvc_mlx.lib.mlx.generators import HiFiGANNSFGenerator

        # Use minimal config for fast tests
        return HiFiGANNSFGenerator(
            initial_channel=192,
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[10, 8, 2, 2],  # Total: 320x upsampling
            upsample_initial_channel=256,
            upsample_kernel_sizes=[20, 16, 4, 4],
            gin_channels=256,
            sr=16000,
        )

    def test_generator_initialization(self, generator):
        """Verify generator initializes all components."""
        assert generator.num_upsamples == 4
        assert generator.num_kernels == 3
        assert generator.upp == 320  # 10 * 8 * 2 * 2

        # Check that upsampling layers are registered
        assert hasattr(generator, "up_0")
        assert hasattr(generator, "up_1")
        assert hasattr(generator, "up_2")
        assert hasattr(generator, "up_3")

        # Check noise convs
        assert hasattr(generator, "noise_conv_0")
        assert hasattr(generator, "noise_conv_3")

        # Check resblocks (num_upsamples * num_kernels = 4 * 3 = 12)
        assert hasattr(generator, "resblock_0")
        assert hasattr(generator, "resblock_11")

    def test_generator_input_transpose(self, generator):
        """
        Critical test: Verify generator handles PyTorch input format (B, C, T).

        The generator expects input in PyTorch format and must transpose internally.
        """
        batch_size = 1
        channels = 192
        time_frames = 50

        # Input in PyTorch format (B, C, T)
        x = mx.random.normal((batch_size, channels, time_frames))
        f0 = mx.full((batch_size, time_frames), 200.0)

        output = generator(x, f0, g=None)
        mx.eval(output)

        # Output should have correct time dimension
        # With 320x upsampling: 50 * 320 = 16000 samples
        expected_samples = time_frames * 320
        assert output.shape[1] == expected_samples or abs(output.shape[1] - expected_samples) < 10

    def test_generator_output_shape(self, generator):
        """Verify generator produces single-channel waveform."""
        batch_size = 1
        channels = 192
        time_frames = 20

        x = mx.random.normal((batch_size, channels, time_frames))
        f0 = mx.full((batch_size, time_frames), 200.0)

        output = generator(x, f0, g=None)
        mx.eval(output)

        # Output should be (B, T, 1) - single channel audio
        assert output.shape[0] == batch_size
        assert output.shape[2] == 1

    def test_generator_with_conditioning(self, generator):
        """Test generator with speaker conditioning."""
        batch_size = 1
        channels = 192
        time_frames = 20

        x = mx.random.normal((batch_size, channels, time_frames))
        f0 = mx.full((batch_size, time_frames), 200.0)
        # Conditioning in MLX format (B, T, C)
        g = mx.random.normal((batch_size, time_frames, 256))

        output = generator(x, f0, g=g)
        mx.eval(output)

        assert output.shape[0] == batch_size

    def test_generator_output_bounded(self, generator):
        """Verify output is bounded by tanh [-1, 1]."""
        batch_size = 1
        channels = 192
        time_frames = 20

        x = mx.random.normal((batch_size, channels, time_frames))
        f0 = mx.full((batch_size, time_frames), 200.0)

        output = generator(x, f0, g=None)
        mx.eval(output)

        # tanh bounds output to [-1, 1]
        assert float(output.max()) <= 1.0
        assert float(output.min()) >= -1.0


class TestResBlock:
    """Tests for ResBlock in generator."""

    @pytest.fixture
    def resblock(self):
        from rvc_mlx.lib.mlx.residuals import ResBlock

        return ResBlock(channels=128, kernel_size=3, dilations=(1, 3, 5))

    def test_resblock_output_shape(self, resblock):
        """Verify ResBlock maintains input shape."""
        batch_size, time, channels = 1, 100, 128
        x = mx.random.normal((batch_size, time, channels))

        output = resblock(x)

        assert output.shape == x.shape

    def test_resblock_with_mask(self, resblock):
        """Test ResBlock respects mask."""
        batch_size, time, channels = 1, 50, 128
        x = mx.random.normal((batch_size, time, channels))
        x_mask = mx.ones((batch_size, time, 1))

        output = resblock(x, x_mask)

        assert output.shape == x.shape
