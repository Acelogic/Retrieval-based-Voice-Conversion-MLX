"""
Unit tests for MLX Synthesizer module.

Tests full synthesizer initialization, forward pass, and inference.
Validates integration of all components (TextEncoder, Flow, Decoder).
"""

import pytest
import mlx.core as mx


# Default synthesizer config matching typical RVC models
DEFAULT_CONFIG = {
    "spec_channels": 1025,
    "segment_size": 32,
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0.0,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "upsample_rates": [10, 10, 2, 2],  # Total: 400x
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "spk_embed_dim": 109,  # Number of speakers
    "gin_channels": 256,
    "sr": 40000,
    "use_f0": True,
    "text_enc_hidden_dim": 768,
}

# Minimal config for faster tests
MINIMAL_CONFIG = {
    "spec_channels": 513,
    "segment_size": 16,
    "inter_channels": 96,
    "hidden_channels": 96,
    "filter_channels": 384,
    "n_heads": 2,
    "n_layers": 2,
    "kernel_size": 3,
    "p_dropout": 0.0,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7],
    "resblock_dilation_sizes": [[1, 3], [1, 3]],
    "upsample_rates": [8, 8, 2, 2],  # Total: 256x
    "upsample_initial_channel": 256,
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "spk_embed_dim": 10,
    "gin_channels": 128,
    "sr": 16000,
    "use_f0": True,
    "text_enc_hidden_dim": 256,
}


class TestSynthesizerInitialization:
    """Tests for Synthesizer initialization."""

    def test_synthesizer_creates_all_components(self):
        """Verify synthesizer initializes all submodules."""
        from rvc_mlx.lib.mlx.synthesizers import Synthesizer

        synth = Synthesizer(**MINIMAL_CONFIG)

        assert hasattr(synth, "enc_p"), "Missing text encoder"
        assert hasattr(synth, "enc_q"), "Missing posterior encoder"
        assert hasattr(synth, "dec"), "Missing decoder"
        assert hasattr(synth, "flow"), "Missing flow"
        assert hasattr(synth, "emb_g"), "Missing speaker embedding"

    def test_synthesizer_use_f0_true(self):
        """Verify synthesizer with F0 enabled."""
        from rvc_mlx.lib.mlx.synthesizers import Synthesizer

        config = MINIMAL_CONFIG.copy()
        config["use_f0"] = True

        synth = Synthesizer(**config)

        assert synth.use_f0 is True
        assert synth.enc_p.emb_pitch is not None

    def test_synthesizer_use_f0_false(self):
        """Verify synthesizer without F0."""
        from rvc_mlx.lib.mlx.synthesizers import Synthesizer

        config = MINIMAL_CONFIG.copy()
        config["use_f0"] = False

        synth = Synthesizer(**config)

        assert synth.use_f0 is False
        assert synth.enc_p.emb_pitch is None


class TestSynthesizerInference:
    """Tests for Synthesizer inference mode."""

    @pytest.fixture
    def synthesizer(self):
        from rvc_mlx.lib.mlx.synthesizers import Synthesizer

        return Synthesizer(**MINIMAL_CONFIG)

    @pytest.mark.slow
    def test_infer_output_shape(self, synthesizer):
        """Verify inference produces correct output shape."""
        batch_size = 1
        seq_len = 50
        emb_dim = 256

        phone = mx.random.normal((batch_size, seq_len, emb_dim))
        phone_lengths = mx.array([seq_len])
        pitch = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        nsff0 = mx.full((batch_size, seq_len), 200.0)  # 200 Hz F0
        sid = mx.array([0])  # Speaker ID

        o, x_mask, _ = synthesizer.infer(phone, phone_lengths, pitch, nsff0, sid)
        mx.eval(o)

        # Output should be waveform (B, T, 1)
        assert o.ndim == 3
        assert o.shape[0] == batch_size
        assert o.shape[2] == 1

        # Time should be upsampled: seq_len * upsample_factor
        # 256x upsampling: 50 * 256 = 12800 samples (approximately)
        expected_samples = seq_len * 256  # 8*8*2*2 = 256
        assert abs(o.shape[1] - expected_samples) < 100

    @pytest.mark.slow
    def test_infer_with_rate(self, synthesizer):
        """Test inference with time rate adjustment."""
        batch_size = 1
        seq_len = 100
        emb_dim = 256

        phone = mx.random.normal((batch_size, seq_len, emb_dim))
        phone_lengths = mx.array([seq_len])
        pitch = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        nsff0 = mx.full((batch_size, seq_len), 200.0)
        sid = mx.array([0])
        rate = mx.array([0.5])  # Skip first 50%

        o, x_mask, _ = synthesizer.infer(phone, phone_lengths, pitch, nsff0, sid, rate=rate)
        mx.eval(o)

        # With rate=0.5, output should be approximately half the length
        full_len = seq_len * 256
        expected_len = full_len // 2
        assert abs(o.shape[1] - expected_len) < 100

    def test_infer_speaker_embedding(self, synthesizer):
        """Verify speaker embedding is applied."""
        batch_size = 1
        seq_len = 20
        emb_dim = 256

        phone = mx.random.normal((batch_size, seq_len, emb_dim))
        phone_lengths = mx.array([seq_len])
        pitch = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        nsff0 = mx.full((batch_size, seq_len), 200.0)

        # Different speaker IDs should produce different outputs
        sid0 = mx.array([0])
        sid1 = mx.array([1])

        o0, _, _ = synthesizer.infer(phone, phone_lengths, pitch, nsff0, sid0)
        o1, _, _ = synthesizer.infer(phone, phone_lengths, pitch, nsff0, sid1)

        mx.eval(o0, o1)

        # Outputs should differ for different speakers
        diff = mx.abs(o0 - o1).mean()
        assert float(diff) > 1e-6, "Different speakers should produce different outputs"


class TestSynthesizerTraining:
    """Tests for Synthesizer training forward pass."""

    @pytest.fixture
    def synthesizer(self):
        from rvc_mlx.lib.mlx.synthesizers import Synthesizer

        return Synthesizer(**MINIMAL_CONFIG)

    @pytest.mark.slow
    def test_training_forward_output_shapes(self, synthesizer):
        """Verify training forward pass output shapes."""
        batch_size = 1
        phone_len = 100  # Match spec_len for proper flow
        spec_len = 100
        emb_dim = 256
        spec_channels = 513

        phone = mx.random.normal((batch_size, phone_len, emb_dim))
        phone_lengths = mx.array([phone_len])
        pitch = mx.zeros((batch_size, phone_len), dtype=mx.int32)
        pitchf = mx.full((batch_size, phone_len), 200.0)
        y = mx.random.normal((batch_size, spec_channels, spec_len))  # (B, C, T)
        y_lengths = mx.array([spec_len])
        ds = mx.array([0])

        o, ids_slice, x_mask, y_mask, latents = synthesizer(
            phone, phone_lengths, pitch, pitchf, y, y_lengths, ds
        )
        mx.eval(o, x_mask, y_mask)

        # Output audio should be segment
        assert o.shape[0] == batch_size
        assert o.shape[2] == 1
        # Output time varies based on segment slicing, just check it exists
        assert o.shape[1] > 0

        # Masks
        assert x_mask.shape == (batch_size, 1, phone_len)
        assert y_mask.shape == (batch_size, 1, spec_len)

        # Latents
        z, z_p, m_p, logs_p, m_q, logs_q = latents
        assert z.shape[0] == batch_size
        assert m_p.shape == (batch_size, 96, phone_len)  # inter_channels=96

    @pytest.mark.slow
    def test_training_forward_without_ds(self, synthesizer):
        """Test training forward without speaker ID (uses None)."""
        batch_size = 1
        phone_len = 100  # Match spec_len
        spec_len = 100
        emb_dim = 256
        spec_channels = 513

        phone = mx.random.normal((batch_size, phone_len, emb_dim))
        phone_lengths = mx.array([phone_len])
        pitch = mx.zeros((batch_size, phone_len), dtype=mx.int32)
        pitchf = mx.full((batch_size, phone_len), 200.0)
        y = mx.random.normal((batch_size, spec_channels, spec_len))
        y_lengths = mx.array([spec_len])

        # Should work without speaker ID
        o, _, _, _, _ = synthesizer(
            phone, phone_lengths, pitch, pitchf, y, y_lengths, ds=None
        )
        mx.eval(o)

        assert o.shape[0] == batch_size


class TestSynthesizerWeightLoading:
    """Tests for Synthesizer weight loading."""

    def test_synthesizer_parameter_structure(self):
        """Verify synthesizer has expected parameter structure."""
        from rvc_mlx.lib.mlx.synthesizers import Synthesizer

        synth = Synthesizer(**MINIMAL_CONFIG)

        # Get all parameters
        params = synth.parameters()

        # Should have nested structure
        assert "enc_p" in params
        assert "enc_q" in params
        assert "dec" in params
        assert "flow" in params
        assert "emb_g" in params

    def test_synthesizer_trainable_parameters(self):
        """Verify synthesizer reports trainable parameters."""
        from rvc_mlx.lib.mlx.synthesizers import Synthesizer
        import mlx.core as mx
        import mlx.utils

        synth = Synthesizer(**MINIMAL_CONFIG)

        # Count parameters using MLX's tree_flatten
        def count_params(tree):
            total = 0
            leaves = mlx.utils.tree_flatten(tree)
            for _, arr in leaves:
                if isinstance(arr, mx.array):
                    total += arr.size
            return total

        total_params = count_params(synth.parameters())

        # Minimal config should still have substantial parameters
        # Reduced threshold for minimal config
        assert total_params > 100_000, f"Expected >100K params, got {total_params}"


class TestSynthesizerDimensionOrdering:
    """
    Tests for correct dimension ordering throughout synthesizer.

    Critical: Ensures PyTorch (B, C, T) <-> MLX (B, T, C) conversions are correct.
    """

    @pytest.fixture
    def synthesizer(self):
        from rvc_mlx.lib.mlx.synthesizers import Synthesizer

        return Synthesizer(**MINIMAL_CONFIG)

    def test_enc_p_output_format(self, synthesizer):
        """Verify TextEncoder outputs in PyTorch format (B, C, T)."""
        batch_size = 1
        seq_len = 30
        emb_dim = 256

        phone = mx.random.normal((batch_size, seq_len, emb_dim))
        pitch = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        lengths = mx.array([seq_len])

        m_p, logs_p, x_mask = synthesizer.enc_p(phone, pitch, lengths)

        # Should be (B, C, T) format
        assert m_p.shape == (batch_size, 96, seq_len), f"m_p wrong shape: {m_p.shape}"
        assert logs_p.shape == (batch_size, 96, seq_len)
        assert x_mask.shape == (batch_size, 1, seq_len)

    def test_flow_internal_format(self, synthesizer):
        """Verify flow uses MLX format (B, T, C) internally."""
        batch_size = 1
        seq_len = 30
        channels = 96

        # Flow input should be (B, T, C)
        x = mx.random.normal((batch_size, seq_len, channels))
        x_mask = mx.ones((batch_size, seq_len, 1))

        output = synthesizer.flow(x, x_mask, g=None, reverse=False)

        # Output should maintain (B, T, C) format
        assert output.shape == (batch_size, seq_len, channels)

    def test_dec_input_format(self, synthesizer):
        """Verify decoder expects PyTorch format (B, C, T)."""
        batch_size = 1
        time_frames = 20
        channels = 96

        # Decoder expects (B, C, T) input
        z = mx.random.normal((batch_size, channels, time_frames))
        f0 = mx.full((batch_size, time_frames), 200.0)

        output = synthesizer.dec(z, f0, g=None)
        mx.eval(output)

        # Output is (B, T_audio, 1) in MLX format
        assert output.shape[0] == batch_size
        assert output.shape[2] == 1
