"""
Unit tests for MLX encoder modules.

Tests TextEncoder and PosteriorEncoder output shapes and dimension ordering.
Critical: Verifies (B, C, T) output format matches PyTorch expectations.
"""

import pytest
import mlx.core as mx


class TestEncoder:
    """Tests for the base Encoder module."""

    @pytest.fixture
    def encoder(self):
        from rvc_mlx.lib.mlx.encoders import Encoder

        return Encoder(
            hidden_channels=192,
            filter_channels=768,
            n_heads=2,
            n_layers=2,  # Reduced for faster tests
            kernel_size=3,
            p_dropout=0.0,
            window_size=10,
        )

    def test_encoder_output_shape(self, encoder):
        """Verify encoder maintains input shape (B, T, C)."""
        batch_size, seq_len, hidden = 2, 50, 192
        x = mx.random.normal((batch_size, seq_len, hidden))
        x_mask = mx.ones((batch_size, seq_len, 1))

        output = encoder(x, x_mask)

        assert output.shape == (batch_size, seq_len, hidden)

    def test_encoder_respects_mask(self, encoder):
        """Verify encoder zeros out masked positions."""
        batch_size, seq_len, hidden = 1, 20, 192
        x = mx.random.normal((batch_size, seq_len, hidden))

        # Mask out last 5 positions
        x_mask = mx.ones((batch_size, seq_len, 1))
        x_mask = mx.concatenate(
            [x_mask[:, :15, :], mx.zeros((batch_size, 5, 1))], axis=1
        )

        output = encoder(x, x_mask)
        mx.eval(output)

        # Masked positions should be near zero
        masked_output = output[0, 15:, :]
        assert mx.abs(masked_output).max() < 1e-5


class TestTextEncoder:
    """Tests for TextEncoder module."""

    @pytest.fixture
    def text_encoder(self):
        from rvc_mlx.lib.mlx.encoders import TextEncoder

        return TextEncoder(
            out_channels=192,
            hidden_channels=192,
            filter_channels=768,
            n_heads=2,
            n_layers=2,  # Reduced for faster tests
            kernel_size=3,
            p_dropout=0.0,
            embedding_dim=256,
            f0=True,
        )

    def test_text_encoder_output_shape(self, text_encoder):
        """Verify TextEncoder outputs correct shape (B, C, T)."""
        batch_size, seq_len, emb_dim = 2, 100, 256
        phone = mx.random.normal((batch_size, seq_len, emb_dim))
        pitch = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        lengths = mx.array([seq_len, seq_len])

        m, logs, x_mask = text_encoder(phone, pitch, lengths)

        # Output should be in PyTorch format (B, C, T)
        assert m.shape == (batch_size, 192, seq_len), f"m shape {m.shape} != expected (B, C, T)"
        assert logs.shape == (batch_size, 192, seq_len), f"logs shape {logs.shape} != expected (B, C, T)"
        assert x_mask.shape == (batch_size, 1, seq_len), f"mask shape {x_mask.shape} != expected (B, 1, T)"

    def test_text_encoder_dimension_transpose(self, text_encoder):
        """
        Critical test: Verify dimension ordering matches PyTorch output format.

        MLX internal format is (B, T, C) but output must be (B, C, T) for
        compatibility with downstream modules.
        """
        batch_size, seq_len, emb_dim = 1, 50, 256
        phone = mx.random.normal((batch_size, seq_len, emb_dim))
        pitch = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        lengths = mx.array([seq_len])

        m, logs, x_mask = text_encoder(phone, pitch, lengths)

        # Verify channel dimension is axis 1 (not axis 2)
        # In (B, C, T) format, C=192 should be in position 1
        assert m.shape[1] == 192, "Channel dimension should be axis 1"
        assert m.shape[2] == seq_len, "Time dimension should be axis 2"

    def test_text_encoder_without_f0(self):
        """Test TextEncoder without pitch embeddings."""
        from rvc_mlx.lib.mlx.encoders import TextEncoder

        encoder = TextEncoder(
            out_channels=192,
            hidden_channels=192,
            filter_channels=768,
            n_heads=2,
            n_layers=2,
            kernel_size=3,
            p_dropout=0.0,
            embedding_dim=256,
            f0=False,
        )

        batch_size, seq_len, emb_dim = 1, 50, 256
        phone = mx.random.normal((batch_size, seq_len, emb_dim))
        lengths = mx.array([seq_len])

        m, logs, x_mask = encoder(phone, None, lengths)

        assert m.shape == (batch_size, 192, seq_len)

    def test_text_encoder_variable_lengths(self, text_encoder):
        """Test TextEncoder with variable sequence lengths."""
        batch_size, max_len, emb_dim = 2, 100, 256
        phone = mx.random.normal((batch_size, max_len, emb_dim))
        pitch = mx.zeros((batch_size, max_len), dtype=mx.int32)
        lengths = mx.array([80, 100])  # Different lengths

        m, logs, x_mask = text_encoder(phone, pitch, lengths)

        assert m.shape == (batch_size, 192, max_len)
        assert x_mask.shape == (batch_size, 1, max_len)


class TestPosteriorEncoder:
    """Tests for PosteriorEncoder module."""

    @pytest.fixture
    def posterior_encoder(self):
        from rvc_mlx.lib.mlx.encoders import PosteriorEncoder

        return PosteriorEncoder(
            in_channels=1025,
            out_channels=192,
            hidden_channels=192,
            kernel_size=5,
            dilation_rate=1,
            n_layers=8,
            gin_channels=256,
        )

    def test_posterior_encoder_output_shape(self, posterior_encoder):
        """Verify PosteriorEncoder outputs correct shapes."""
        batch_size, seq_len, in_ch = 1, 100, 1025
        x = mx.random.normal((batch_size, seq_len, in_ch))
        x_lengths = mx.array([seq_len])
        g = mx.random.normal((batch_size, seq_len, 256))

        z, m, logs, x_mask = posterior_encoder(x, x_lengths, g=g)

        assert z.shape == (batch_size, seq_len, 192)
        assert m.shape == (batch_size, seq_len, 192)
        assert logs.shape == (batch_size, seq_len, 192)
        assert x_mask.shape == (batch_size, seq_len)

    def test_posterior_encoder_without_conditioning(self, posterior_encoder):
        """Test PosteriorEncoder without speaker conditioning."""
        batch_size, seq_len, in_ch = 1, 50, 1025
        x = mx.random.normal((batch_size, seq_len, in_ch))
        x_lengths = mx.array([seq_len])

        z, m, logs, x_mask = posterior_encoder(x, x_lengths, g=None)

        assert z.shape == (batch_size, seq_len, 192)

    def test_posterior_encoder_sampling(self, posterior_encoder):
        """Verify z is sampled from N(m, exp(logs))."""
        batch_size, seq_len, in_ch = 1, 50, 1025
        x = mx.random.normal((batch_size, seq_len, in_ch))
        x_lengths = mx.array([seq_len])

        z1, m, logs, _ = posterior_encoder(x, x_lengths)
        z2, _, _, _ = posterior_encoder(x, x_lengths)

        mx.eval(z1, z2)

        # z should be stochastic (different each call due to random sampling)
        # Note: This may occasionally fail if random seeds align
        diff = mx.abs(z1 - z2).mean()
        assert float(diff) > 1e-6, "z should be stochastically sampled"
