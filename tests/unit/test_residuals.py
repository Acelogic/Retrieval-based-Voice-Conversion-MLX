"""
Unit tests for MLX residual flow modules.

Tests ResidualCouplingBlock forward/reverse passes and flow layer indexing.
Critical: Verifies correct flip ordering in reverse mode (20% accuracy improvement).
"""

import pytest
import mlx.core as mx


class TestResidualCouplingLayer:
    """Tests for ResidualCouplingLayer module."""

    @pytest.fixture
    def coupling_layer(self):
        from rvc_mlx.lib.mlx.residuals import ResidualCouplingLayer

        return ResidualCouplingLayer(
            channels=192,
            hidden_channels=192,
            kernel_size=5,
            dilation_rate=1,
            n_layers=4,
            p_dropout=0.0,
            gin_channels=256,
            mean_only=True,
        )

    def test_coupling_layer_output_shape(self, coupling_layer):
        """Verify coupling layer maintains shape."""
        batch_size, time, channels = 1, 50, 192
        x = mx.random.normal((batch_size, time, channels))
        x_mask = mx.ones((batch_size, time, 1))
        g = mx.random.normal((batch_size, time, 256))

        output, logdet = coupling_layer(x, x_mask, g=g, reverse=False)

        assert output.shape == x.shape

    def test_coupling_layer_forward_reverse(self, coupling_layer):
        """
        Verify forward + reverse = identity (within numerical precision).

        This is critical for normalizing flow correctness.
        """
        batch_size, time, channels = 1, 50, 192
        x = mx.random.normal((batch_size, time, channels))
        x_mask = mx.ones((batch_size, time, 1))

        # Forward pass
        z, _ = coupling_layer(x, x_mask, g=None, reverse=False)

        # Reverse pass
        x_reconstructed, _ = coupling_layer(z, x_mask, g=None, reverse=True)

        mx.eval(x, x_reconstructed)

        # Should reconstruct original
        diff = mx.abs(x - x_reconstructed).max()
        assert float(diff) < 1e-4, f"Forward-reverse should be identity, diff={float(diff)}"

    def test_coupling_layer_splits_channels(self, coupling_layer):
        """Verify coupling layer correctly splits channels."""
        batch_size, time, channels = 1, 50, 192
        x = mx.random.normal((batch_size, time, channels))
        x_mask = mx.ones((batch_size, time, 1))

        output, _ = coupling_layer(x, x_mask, g=None, reverse=False)

        # First half should be unchanged (coupling only transforms second half)
        half = channels // 2
        x0_input = x[:, :, :half]
        x0_output = output[:, :, :half]

        mx.eval(x0_input, x0_output)
        diff = mx.abs(x0_input - x0_output).max()
        assert float(diff) < 1e-6, "First half should pass through unchanged"


class TestResidualCouplingBlock:
    """Tests for ResidualCouplingBlock module."""

    @pytest.fixture
    def coupling_block(self):
        from rvc_mlx.lib.mlx.residuals import ResidualCouplingBlock

        return ResidualCouplingBlock(
            channels=192,
            hidden_channels=192,
            kernel_size=5,
            dilation_rate=1,
            n_layers=4,
            n_flows=4,
            gin_channels=256,
        )

    def test_coupling_block_output_shape(self, coupling_block):
        """Verify coupling block maintains shape."""
        batch_size, time, channels = 1, 50, 192
        x = mx.random.normal((batch_size, time, channels))
        x_mask = mx.ones((batch_size, time, 1))
        g = mx.random.normal((batch_size, time, 256))

        output = coupling_block(x, x_mask, g=g, reverse=False)

        assert output.shape == x.shape

    def test_coupling_block_forward_reverse(self, coupling_block):
        """
        Critical test: Verify forward + reverse = identity.

        This validates the correct flip ordering in reverse mode
        (flip BEFORE flow in reverse, flip AFTER flow in forward).
        """
        batch_size, time, channels = 1, 50, 192
        x = mx.random.normal((batch_size, time, channels))
        x_mask = mx.ones((batch_size, time, 1))

        # Forward pass
        z = coupling_block(x, x_mask, g=None, reverse=False)

        # Reverse pass
        x_reconstructed = coupling_block(z, x_mask, g=None, reverse=True)

        mx.eval(x, x_reconstructed)

        # Should reconstruct original
        diff = mx.abs(x - x_reconstructed).max()
        assert float(diff) < 1e-3, f"Forward-reverse should be identity, diff={float(diff)}"

    def test_coupling_block_reverse_flip_order(self, coupling_block):
        """
        Critical test: Verify flip happens BEFORE flow in reverse mode.

        This is the fix that improved Swift correlation from 72% to 92%.
        In forward: flow -> flip
        In reverse: flip -> flow (NOT flow -> flip!)
        """
        batch_size, time, channels = 1, 20, 192
        x = mx.random.normal((batch_size, time, channels))
        x_mask = mx.ones((batch_size, time, 1))

        # Run forward
        z = coupling_block(x, x_mask, g=None, reverse=False)

        # Run reverse
        x_recon = coupling_block(z, x_mask, g=None, reverse=True)

        mx.eval(x, x_recon)

        # If flip order is wrong, reconstruction will be significantly different
        correlation = float(mx.sum(x * x_recon) / (mx.sqrt(mx.sum(x**2)) * mx.sqrt(mx.sum(x_recon**2))))
        assert correlation > 0.99, f"Correlation {correlation} suggests wrong flip order"

    def test_coupling_block_n_flows(self, coupling_block):
        """Verify correct number of flow layers."""
        assert coupling_block.n_flows == 4
        assert hasattr(coupling_block, "flow_0")
        assert hasattr(coupling_block, "flow_1")
        assert hasattr(coupling_block, "flow_2")
        assert hasattr(coupling_block, "flow_3")


class TestFlowLayerIndexMapping:
    """
    Tests for correct flow layer index mapping.

    Critical: PyTorch ResidualCouplingBlock interleaves Layer and Flip modules:
        - Index 0: Layer, Index 1: Flip, Index 2: Layer, Index 3: Flip...

    MLX uses only Layer modules with implicit flips:
        - PyTorch flow.flows.0 -> MLX flow.flow_0
        - PyTorch flow.flows.2 -> MLX flow.flow_1
        - PyTorch flow.flows.4 -> MLX flow.flow_2

    This mapping is handled in the weight converter (i // 2).
    """

    def test_mlx_flow_layer_naming(self):
        """Verify MLX uses consecutive flow layer naming."""
        from rvc_mlx.lib.mlx.residuals import ResidualCouplingBlock

        block = ResidualCouplingBlock(
            channels=192,
            hidden_channels=192,
            kernel_size=5,
            dilation_rate=1,
            n_layers=4,
            n_flows=4,
            gin_channels=0,
        )

        # MLX should have flow_0, flow_1, flow_2, flow_3 (consecutive)
        for i in range(4):
            assert hasattr(block, f"flow_{i}"), f"Missing flow_{i}"

    def test_pytorch_to_mlx_index_formula(self):
        """Verify PyTorch index to MLX index mapping formula."""
        # PyTorch indices: 0, 2, 4, 6 (even only, odd are Flip modules)
        # MLX indices: 0, 1, 2, 3

        pytorch_indices = [0, 2, 4, 6]
        expected_mlx = [0, 1, 2, 3]

        for pt_idx, mlx_idx in zip(pytorch_indices, expected_mlx):
            computed_mlx = pt_idx // 2
            assert computed_mlx == mlx_idx, f"PyTorch {pt_idx} -> MLX {mlx_idx}, got {computed_mlx}"
