#!/usr/bin/env python3
"""
Convert RVC PyTorch model to MLX-compatible safetensors for iOS.
Usage: python convert_model_for_ios.py <path_to_model.pth> <output_path>
"""

import argparse
import sys
import os
import mlx.core as mx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rvc.lib.mlx.convert import convert_weights


def main():
    parser = argparse.ArgumentParser(
        description="Convert RVC PyTorch model to MLX-compatible safetensors for iOS."
    )
    parser.add_argument("input_model", help="Path to input .pth model file")
    parser.add_argument("output_path", help="Path to output .safetensors file")

    args = parser.parse_args()

    input_path = args.input_model
    output_path = args.output_path

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"Converting: {input_path}")
    print(f"Output: {output_path}")
    print()

    # Convert
    print("Loading PyTorch model...")
    try:
        mlx_weights, config = convert_weights(input_path)
    except Exception as e:
        print(f"Conversion failed: {e}")
        sys.exit(1)

    print(f"Converted {len(mlx_weights)} weights")
    # print(f"Config: {config}") # Too verbose
    print()

    # Save as safetensors
    print(f"Saving to {output_path}...")
    mx.save_safetensors(output_path, mlx_weights)

    print("✅ Conversion complete!")


if __name__ == "__main__":
    main()
