#!/usr/bin/env python3
"""
Convert RVC PyTorch model to MLX-compatible safetensors for iOS.
Usage: python convert_model_for_ios.py <path_to_model.pth> <output_path>
"""

import sys
import os
from rvc.lib.mlx.convert import convert_weights
import mlx.core as mx

def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_model_for_ios.py <input_model.pth> <output_coder.safetensors>")
        print()
        print("Example:")
        print("  python convert_model_for_ios.py \\")
        print("    '/Users/mcruz/Library/Application Support/Replay/com.replay.Replay/models/Slim Shady/model.pth' \\")
        print("    './coder_slimshady.safetensors'")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Converting: {input_path}")
    print(f"Output: {output_path}")
    print()

    # Convert
    print("Loading PyTorch model...")
    mlx_weights, config = convert_weights(input_path)

    print(f"Converted {len(mlx_weights)} weights")
    print(f"Config: {config}")
    print()

    # Save as safetensors
    print(f"Saving to {output_path}...")
    mx.save_safetensors(output_path, mlx_weights)

    print("✅ Conversion complete!")
    print()
    print("Next steps:")
    print("1. Copy the output file to your iOS project's Assets folder")
    print("2. Replace the existing coder.safetensors")
    print("3. Rebuild and run the iOS app")

if __name__ == "__main__":
    main()
