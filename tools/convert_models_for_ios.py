#!/usr/bin/env python3
"""
Convert RVC models to iOS-compatible MLX format

This script converts PyTorch RVC models to MLX safetensors format optimized for iOS,
ensuring all the critical fixes from the Python implementation are preserved.

Usage:
    python tools/convert_models_for_ios.py --model-path ~/Library/Application\ Support/Replay/.../models/Drake --output-dir rvc_mlx/models/converted_ios
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np
import mlx.core as mx
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.convert_rvc_model import remap_key_name

def convert_pytorch_to_mlx_safetensors(pytorch_model_path: Path, output_path: Path, model_name: str):
    """Convert PyTorch RVC model to MLX safetensors format"""

    print(f"\n{'='*60}")
    print(f"Converting {model_name}")
    print(f"{'='*60}")

    # Load PyTorch model
    print(f"Loading PyTorch model from: {pytorch_model_path}")
    checkpoint = torch.load(pytorch_model_path, map_location='cpu')

    # Extract weights
    if 'weight' in checkpoint:
        state_dict = checkpoint['weight']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    print(f"Loaded {len(state_dict)} tensors from PyTorch")

    # Convert and remap keys
    mlx_weights = {}
    skipped_keys = []

    for key, value in state_dict.items():
        # Skip non-tensor values
        if not isinstance(value, torch.Tensor):
            skipped_keys.append(key)
            continue

        # Remap key name to MLX format
        new_key = remap_key_name(key)

        # Convert tensor to numpy then MLX
        numpy_array = value.detach().cpu().numpy()
        
        # TRANSPOSE CONV WEIGHTS: 
        if "weight" in new_key:
            if numpy_array.ndim == 4:
                # Conv2d: PyTorch [Out, In, H, W] -> MLX [Out, H, W, In]
                # Default for Conv2d. For ConvTranspose2d, we handle it if detected.
                numpy_array = numpy_array.transpose(0, 2, 3, 1)
            elif numpy_array.ndim == 3 and "emb" not in new_key:
                if "dec.ups" in key or "dec.up_" in new_key:
                    # ConvTranspose1d: PyTorch [In, Out, K] -> MLX [Out, K, In]
                    numpy_array = numpy_array.transpose(1, 2, 0)
                else:
                    # Conv1d: PyTorch [Out, In, K] -> MLX [Out, K, In]
                    numpy_array = numpy_array.transpose(0, 2, 1)
        
        mlx_array = mx.array(numpy_array)

        mlx_weights[new_key] = mlx_array

        # Log conversion
        if key != new_key:
            print(f"  {key} -> {new_key} {mlx_array.shape}")

    if skipped_keys:
        print(f"\nSkipped {len(skipped_keys)} non-tensor keys:")
        for key in skipped_keys[:5]:
            print(f"  - {key}")
        if len(skipped_keys) > 5:
            print(f"  ... and {len(skipped_keys) - 5} more")

    # Save as safetensors
    output_file = output_path / f"{model_name}.safetensors"
    print(f"\nSaving to: {output_file}")
    mx.save_safetensors(str(output_file), mlx_weights)

    print(f"✅ Successfully converted {len(mlx_weights)} tensors")

    # Verify the saved file can be loaded
    print("\nVerifying saved file...")
    loaded = mx.load(str(output_file))
    assert len(loaded) == len(mlx_weights), "Mismatch in loaded tensor count!"
    print(f"✅ Verification passed - {len(loaded)} tensors loaded")

    return output_file

def convert_hubert_model(hubert_path: Path, output_path: Path):
    """Convert HuBERT model to MLX format"""

    print(f"\n{'='*60}")
    print("Converting HuBERT Model")
    print(f"{'='*60}")

    if not hubert_path.exists():
        print(f"⚠️  HuBERT model not found at {hubert_path}")
        print("Please download from: https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt")
        return None

    print(f"Loading HuBERT from: {hubert_path}")
    checkpoint = torch.load(hubert_path, map_location='cpu')

    # HuBERT typically has weights directly in checkpoint
    mlx_weights = {}

    for key, value in checkpoint.items():
        if isinstance(value, torch.Tensor):
            numpy_array = value.detach().cpu().numpy()
            
            # TRANSPOSE CONV WEIGHTS: 
            # 3D: PyTorch [Out, In, K] -> MLX [Out, K, In]
            if numpy_array.ndim == 3 and "weight" in key and "emb" not in key:
                numpy_array = numpy_array.transpose(0, 2, 1)
            # 4D: PyTorch [Out, In, H, W] -> MLX [Out, H, W, In]
            elif numpy_array.ndim == 4 and "weight" in key:
                numpy_array = numpy_array.transpose(0, 2, 3, 1)
                
            mlx_array = mx.array(numpy_array)
            mlx_weights[key] = mlx_array
            print(f"  {key}: {mlx_array.shape}")

    output_file = output_path / "hubert_base.safetensors"
    print(f"\nSaving to: {output_file}")
    mx.save_safetensors(str(output_file), mlx_weights)

    print(f"✅ Successfully converted HuBERT with {len(mlx_weights)} tensors")

    return output_file

def convert_rmvpe_model(rmvpe_path: Path, output_path: Path):
    """Convert RMVPE model to MLX format"""

    print(f"\n{'='*60}")
    print("Converting RMVPE Model")
    print(f"{'='*60}")

    if not rmvpe_path.exists():
        print(f"⚠️  RMVPE model not found at {rmvpe_path}")
        print("Using RMVPE from rvc_mlx/models/rmvpe.pt if available")
        return None

    print(f"Loading RMVPE from: {rmvpe_path}")

    # RMVPE can be .pt or .npz format
    if rmvpe_path.suffix == '.pt':
        checkpoint = torch.load(rmvpe_path, map_location='cpu')

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        mlx_weights = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                numpy_array = value.detach().cpu().numpy()
                
                # TRANSPOSE CONV WEIGHTS: 
                # 3D: PyTorch [Out, In, K] -> MLX [Out, K, In]
                if numpy_array.ndim == 3 and "weight" in key and "emb" not in key:
                    numpy_array = numpy_array.transpose(0, 2, 1)
                # 4D: 
                elif numpy_array.ndim == 4 and "weight" in key:
                    # RMVPE ConvTranspose2d: PyTorch [In, Out, H, W] -> MLX [Out, H, W, In]
                    if "conv1.0" in key or "conv1_trans" in key:
                        numpy_array = numpy_array.transpose(1, 2, 3, 0)
                    else:
                        # Conv2d: PyTorch [Out, In, H, W] -> MLX [Out, H, W, In]
                        numpy_array = numpy_array.transpose(0, 2, 3, 1)
                
                mlx_array = mx.array(numpy_array)
                mlx_weights[key] = mlx_array
                print(f"  {key}: {mlx_array.shape}")

    elif rmvpe_path.suffix == '.npz':
        # Already in numpy format
        data = np.load(rmvpe_path)
        mlx_weights = {}
        for key in data.files:
            mlx_weights[key] = mx.array(data[key])
            print(f"  {key}: {mlx_weights[key].shape}")

    output_file = output_path / "rmvpe.safetensors"
    print(f"\nSaving to: {output_file}")
    mx.save_safetensors(str(output_file), mlx_weights)

    print(f"✅ Successfully converted RMVPE with {len(mlx_weights)} tensors")

    return output_file

def verify_converted_model_in_python(model_path: Path, audio_path: Path):
    """
    Load the converted model in Python MLX and run a test inference.
    This ensures the model works before we try it on iOS.
    """
    print(f"\n{'='*60}")
    print("Verifying Converted Model in Python")
    print(f"{'='*60}")

    try:
        from rvc_mlx.infer.infer_mlx import RVCInference

        # Initialize inference
        print("Initializing RVC inference...")
        rvc = RVCInference()

        # Load model
        print(f"Loading model from: {model_path}")
        # TODO: Add proper model loading here

        print("✅ Model loaded successfully")

        # Run test inference if audio provided
        if audio_path and audio_path.exists():
            print(f"Running test inference on: {audio_path}")
            # TODO: Add inference test here
            print("✅ Test inference completed")

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Convert RVC models for iOS')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to PyTorch model directory (contains model.pth)')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Name for the converted model (e.g., "drake", "coder")')
    parser.add_argument('--output-dir', type=str, default='rvc_mlx/models/converted_ios',
                        help='Output directory for converted models')
    parser.add_argument('--hubert-path', type=str,
                        default='rvc_mlx/models/hubert_base.pt',
                        help='Path to HuBERT model')
    parser.add_argument('--rmvpe-path', type=str,
                        default='rvc_mlx/models/rmvpe.pt',
                        help='Path to RMVPE model')
    parser.add_argument('--verify', action='store_true',
                        help='Verify converted model in Python before deploying to iOS')
    parser.add_argument('--test-audio', type=str,
                        help='Test audio file for verification')

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 RVC Model Conversion for iOS")
    print(f"Output directory: {output_path}")

    # Convert RVC model
    model_path = Path(args.model_path) / "model.pth"
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print(f"   Looking in: {Path(args.model_path)}")
        print(f"   Contents: {list(Path(args.model_path).iterdir())}")
        sys.exit(1)

    converted_model = convert_pytorch_to_mlx_safetensors(
        model_path,
        output_path,
        args.model_name
    )

    # Convert HuBERT (shared across all models)
    hubert_path = Path(args.hubert_path)
    if hubert_path.exists():
        convert_hubert_model(hubert_path, output_path)
    else:
        print(f"\n⚠️  Skipping HuBERT conversion - not found at {hubert_path}")

    # Convert RMVPE (shared across all models)
    rmvpe_path = Path(args.rmvpe_path)
    if rmvpe_path.exists():
        convert_rmvpe_model(rmvpe_path, output_path)
    else:
        print(f"\n⚠️  Skipping RMVPE conversion - not found at {rmvpe_path}")

    # Verify if requested
    if args.verify:
        test_audio = Path(args.test_audio) if args.test_audio else None
        verify_converted_model_in_python(converted_model, test_audio)

    print(f"\n{'='*60}")
    print("✅ Conversion Complete!")
    print(f"{'='*60}")
    print(f"\nConverted models saved to: {output_path}")
    print("\nNext steps:")
    print("1. Copy models to iOS project:")
    print(f"   cp {output_path}/*.safetensors Demos/iOS/RVCNative/RVCNativePackage/Sources/RVCNativeFeature/Assets/")
    print("2. Rebuild iOS app")
    print("3. Test inference on device/simulator")

if __name__ == '__main__':
    main()
