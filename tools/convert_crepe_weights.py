#!/usr/bin/env python3
"""
Convert torchcrepe weights to MLX format.

Usage:
    python tools/convert_crepe_weights.py [--model full|tiny] [--output weights/]

This script extracts weights from the bundled torchcrepe package and
converts them to MLX-compatible format (npz).
"""

import argparse
import numpy as np
from pathlib import Path

try:
    import torch
    import torchcrepe
except ImportError:
    raise ImportError("torchcrepe is required. Install with: pip install torchcrepe")


def convert_crepe_weights(
    model_type: str = "full",
    output_dir: str = "weights",
) -> str:
    """
    Convert torchcrepe weights to MLX format.

    Args:
        model_type: "full" or "tiny"
        output_dir: Output directory for weights

    Returns:
        Path to saved weights file
    """
    print(f"Converting CREPE {model_type} weights to MLX format...")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load torchcrepe model (downloads weights if needed)
    model = torchcrepe.model.Crepe(model=model_type)
    model.eval()

    # Get state dict
    state_dict = model.state_dict()

    # Convert weights
    mlx_weights = {}

    for key, tensor in state_dict.items():
        value = tensor.numpy()

        # Handle different weight types
        if "conv" in key and "weight" in key:
            # Conv2d weights: PyTorch (Out, In, H, W) -> MLX (Out, H, W, In)
            value = np.transpose(value, (0, 2, 3, 1))

        elif "BN" in key:
            # BatchNorm: running_mean, running_var, weight, bias
            # MLX BatchNorm uses same names
            pass

        elif "classifier" in key:
            # Linear weights: PyTorch (Out, In) -> MLX (In, Out) for Linear
            if "weight" in key:
                value = np.transpose(value)

        # Create MLX-compatible key name
        mlx_key = key

        mlx_weights[mlx_key] = value.astype(np.float32)
        print(f"  {key}: {tensor.shape} -> {value.shape}")

    # Save weights
    output_file = output_path / f"crepe_{model_type}.npz"
    np.savez(str(output_file), **mlx_weights)

    print(f"\nSaved weights to: {output_file}")
    print(f"Total parameters: {sum(w.size for w in mlx_weights.values()):,}")

    return str(output_file)


def verify_weights(weights_path: str, model_type: str = "full"):
    """Verify converted weights can be loaded."""
    print(f"\nVerifying weights from {weights_path}...")

    weights = dict(np.load(weights_path))

    # Check expected keys
    expected_layers = ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "classifier"]
    for layer in expected_layers:
        weight_key = f"{layer}.weight"
        if layer != "classifier":
            bn_key = f"{layer}_BN.running_mean"
        else:
            bn_key = None

        assert weight_key in weights, f"Missing {weight_key}"
        if bn_key:
            assert bn_key in weights, f"Missing {bn_key}"

    print("All expected weights found!")

    # Print shapes
    print("\nWeight shapes:")
    for key in sorted(weights.keys()):
        print(f"  {key}: {weights[key].shape}")


def test_inference(model_type: str = "full"):
    """Test that converted weights produce same output as original."""
    print(f"\nTesting inference parity...")

    import torch

    # Generate test input
    np.random.seed(42)
    test_audio = np.random.randn(2, 1024).astype(np.float32)

    # PyTorch inference
    model_torch = torchcrepe.model.Crepe(model=model_type)
    model_torch.eval()

    with torch.no_grad():
        output_torch = model_torch(torch.from_numpy(test_audio))
        output_torch = output_torch.numpy()

    print(f"PyTorch output shape: {output_torch.shape}")
    print(f"PyTorch output range: [{output_torch.min():.4f}, {output_torch.max():.4f}]")

    # MLX inference (if available)
    try:
        from rvc_mlx.lib.mlx.crepe import CREPE
        import mlx.core as mx

        crepe_mlx = CREPE(model=model_type)

        # Frame audio (normally done in get_f0)
        frames_mlx = mx.array(test_audio)
        output_mlx = crepe_mlx._model(frames_mlx)
        mx.eval(output_mlx)
        output_mlx = np.array(output_mlx)

        print(f"MLX output shape: {output_mlx.shape}")
        print(f"MLX output range: [{output_mlx.min():.4f}, {output_mlx.max():.4f}]")

        # Compare
        diff = np.abs(output_torch - output_mlx)
        print(f"\nMax difference: {diff.max():.6f}")
        print(f"Mean difference: {diff.mean():.6f}")

        corr = np.corrcoef(output_torch.flatten(), output_mlx.flatten())[0, 1]
        print(f"Correlation: {corr:.6f}")

        if corr > 0.99:
            print("✓ PARITY ACHIEVED!")
        else:
            print("✗ Parity not achieved - check weight conversion")

    except ImportError as e:
        print(f"Skipping MLX inference test: {e}")
    except FileNotFoundError as e:
        print(f"Skipping MLX inference test (weights not found): {e}")


def main():
    parser = argparse.ArgumentParser(description="Convert CREPE weights to MLX")
    parser.add_argument(
        "--model",
        type=str,
        choices=["full", "tiny", "both"],
        default="both",
        help="Model variant to convert",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="weights",
        help="Output directory for weights",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify converted weights",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test inference parity",
    )

    args = parser.parse_args()

    models = ["full", "tiny"] if args.model == "both" else [args.model]

    for model_type in models:
        weights_path = convert_crepe_weights(model_type, args.output)

        if args.verify:
            verify_weights(weights_path, model_type)

        if args.test:
            test_inference(model_type)


if __name__ == "__main__":
    main()
