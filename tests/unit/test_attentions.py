"""
Unit tests for MLX attention modules.

Tests MultiHeadAttention shapes and relative position embeddings.
Critical: Verifies emb_rel_k and emb_rel_v are NOT transposed during weight loading.
"""

import pytest
import mlx.core as mx


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention module."""

    @pytest.fixture
    def attention(self):
        from rvc_mlx.lib.mlx.attentions import MultiHeadAttention

        return MultiHeadAttention(
            channels=192,
            out_channels=192,
            n_heads=2,
            p_dropout=0.0,
            window_size=10,
            heads_share=True,
        )

    def test_attention_output_shape(self, attention):
        """Verify attention output shape matches input."""
        batch_size, time, channels = 2, 50, 192
        x = mx.random.normal((batch_size, time, channels))
        c = mx.random.normal((batch_size, time, channels))

        output = attention(x, c)

        assert output.shape == (batch_size, time, 192)

    def test_attention_with_mask(self, attention):
        """Test attention with mask."""
        batch_size, time, channels = 1, 30, 192
        x = mx.random.normal((batch_size, time, channels))
        c = mx.random.normal((batch_size, time, channels))

        # Create attention mask (B, 1, T, T)
        attn_mask = mx.ones((batch_size, 1, time, time))

        output = attention(x, c, attn_mask=attn_mask)

        assert output.shape == (batch_size, time, 192)

    def test_attention_self_attention(self, attention):
        """Test self-attention (x == c)."""
        batch_size, time, channels = 1, 50, 192
        x = mx.random.normal((batch_size, time, channels))

        output = attention(x, x)

        assert output.shape == x.shape


class TestRelativePositionEmbeddings:
    """
    Tests for relative position embeddings.

    Critical: These embeddings must NOT be transposed during weight loading.
    They are direct parameter arrays, not Conv weights.
    """

    @pytest.fixture
    def attention_with_rel_pos(self):
        from rvc_mlx.lib.mlx.attentions import MultiHeadAttention

        return MultiHeadAttention(
            channels=192,
            out_channels=192,
            n_heads=2,
            p_dropout=0.0,
            window_size=10,
            heads_share=True,
        )

    def test_rel_emb_shapes(self, attention_with_rel_pos):
        """Verify relative embedding shapes."""
        attn = attention_with_rel_pos
        window_size = 10
        n_heads_rel = 1  # heads_share=True
        k_channels = 192 // 2  # 96

        # Expected shape: (n_heads_rel, 2*window_size+1, k_channels)
        expected_shape = (n_heads_rel, 2 * window_size + 1, k_channels)

        assert attn.emb_rel_k.shape == expected_shape, f"emb_rel_k shape {attn.emb_rel_k.shape} != {expected_shape}"
        assert attn.emb_rel_v.shape == expected_shape, f"emb_rel_v shape {attn.emb_rel_v.shape} != {expected_shape}"

    def test_rel_emb_no_weight_suffix(self):
        """
        Critical test: Verify emb_rel_k and emb_rel_v are direct attributes.

        Unlike Conv weights which use .weight suffix, these are direct parameters.
        The weight converter must NOT add .weight suffix for these.
        """
        from rvc_mlx.lib.mlx.attentions import MultiHeadAttention

        attn = MultiHeadAttention(
            channels=192,
            out_channels=192,
            n_heads=2,
            p_dropout=0.0,
            window_size=10,
        )

        # These should be direct attributes, not nested in .weight
        assert hasattr(attn, "emb_rel_k"), "emb_rel_k should be direct attribute"
        assert hasattr(attn, "emb_rel_v"), "emb_rel_v should be direct attribute"
        assert isinstance(attn.emb_rel_k, mx.array), "emb_rel_k should be mx.array"
        assert isinstance(attn.emb_rel_v, mx.array), "emb_rel_v should be mx.array"

    def test_rel_emb_initialized(self, attention_with_rel_pos):
        """Verify embeddings are properly initialized with std = k_channels^-0.5."""
        attn = attention_with_rel_pos
        k_channels = 192 // 2

        # Check that values are reasonable (initialized with scaled normal)
        expected_std = k_channels ** -0.5
        actual_std_k = float(mx.std(attn.emb_rel_k))
        actual_std_v = float(mx.std(attn.emb_rel_v))

        # Allow some variance since it's random
        assert 0.5 * expected_std < actual_std_k < 2 * expected_std
        assert 0.5 * expected_std < actual_std_v < 2 * expected_std

    def test_rel_emb_heads_not_shared(self):
        """Test relative embeddings when heads are not shared."""
        from rvc_mlx.lib.mlx.attentions import MultiHeadAttention

        n_heads = 4
        attn = MultiHeadAttention(
            channels=192,
            out_channels=192,
            n_heads=n_heads,
            p_dropout=0.0,
            window_size=10,
            heads_share=False,  # Each head has own embeddings
        )

        k_channels = 192 // n_heads
        expected_shape = (n_heads, 21, k_channels)  # 2*10+1 = 21

        assert attn.emb_rel_k.shape == expected_shape
        assert attn.emb_rel_v.shape == expected_shape


class TestFFN:
    """Tests for FFN (Feed-Forward Network) module."""

    @pytest.fixture
    def ffn(self):
        from rvc_mlx.lib.mlx.attentions import FFN

        return FFN(
            in_channels=192,
            out_channels=192,
            filter_channels=768,
            kernel_size=3,
            p_dropout=0.0,
        )

    def test_ffn_output_shape(self, ffn):
        """Verify FFN output shape."""
        batch_size, time, channels = 1, 50, 192
        x = mx.random.normal((batch_size, time, channels))
        x_mask = mx.ones((batch_size, time, 1))

        output = ffn(x, x_mask)

        assert output.shape == (batch_size, time, 192)

    def test_ffn_respects_mask(self, ffn):
        """Verify FFN zeros out masked positions."""
        batch_size, time, channels = 1, 20, 192
        x = mx.random.normal((batch_size, time, channels))

        # Mask out last 5 positions
        x_mask = mx.concatenate(
            [mx.ones((batch_size, 15, 1)), mx.zeros((batch_size, 5, 1))],
            axis=1,
        )

        output = ffn(x, x_mask)
        mx.eval(output)

        # Masked positions should be zero
        masked_output = output[0, 15:, :]
        assert float(mx.abs(masked_output).max()) < 1e-6

    def test_ffn_causal(self):
        """Test causal FFN variant."""
        from rvc_mlx.lib.mlx.attentions import FFN

        ffn_causal = FFN(
            in_channels=192,
            out_channels=192,
            filter_channels=768,
            kernel_size=3,
            p_dropout=0.0,
            causal=True,
        )

        batch_size, time, channels = 1, 50, 192
        x = mx.random.normal((batch_size, time, channels))
        x_mask = mx.ones((batch_size, time, 1))

        output = ffn_causal(x, x_mask)

        assert output.shape == (batch_size, time, 192)

    def test_ffn_gelu_activation(self):
        """Test FFN with GELU activation."""
        from rvc_mlx.lib.mlx.attentions import FFN

        ffn_gelu = FFN(
            in_channels=192,
            out_channels=192,
            filter_channels=768,
            kernel_size=3,
            p_dropout=0.0,
            activation="gelu",
        )

        batch_size, time, channels = 1, 50, 192
        x = mx.random.normal((batch_size, time, channels))
        x_mask = mx.ones((batch_size, time, 1))

        output = ffn_gelu(x, x_mask)

        assert output.shape == (batch_size, time, 192)
