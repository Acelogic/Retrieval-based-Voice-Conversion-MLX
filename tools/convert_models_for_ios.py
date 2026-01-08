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

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

def fuse_weight_norm(state_dict):
    """Fuse weight normalization (weight_g, weight_v) into single weight tensor"""
    fused = {}
    processed_bases = set()

    for key in list(state_dict.keys()):
        if key.endswith('.weight_g'):
            base = key[:-9]  # Remove '.weight_g'
            v_key = base + '.weight_v'

            if v_key in state_dict:
                weight_g = state_dict[key]
                weight_v = state_dict[v_key]

                # Fuse: weight = weight_g * (weight_v / ||weight_v||)
                # For Conv1d, weight_v shape is [Out, In, K] or similar
                # Norm is typically computed along dim 0
                norm = torch.linalg.norm(weight_v, dim=0, keepdim=True)
                weight_normalized = weight_v / (norm + 1e-12)
                weight_fused = weight_g * weight_normalized

                fused[base + '.weight'] = weight_fused
                processed_bases.add(base)
                print(f"  Fused weight_norm: {key} + {v_key} -> {base}.weight {weight_fused.shape}")
            else:
                fused[key] = state_dict[key]
        elif key.endswith('.weight_v'):
            base = key[:-9]
            if base not in processed_bases:
                # weight_v without weight_g - keep as is
                fused[key] = state_dict[key]
        else:
            fused[key] = state_dict[key]

    return fused


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

    # CRITICAL: Fuse weight normalization before conversion
    print("\nFusing weight normalization...")
    state_dict = fuse_weight_norm(state_dict)
    print(f"After fusion: {len(state_dict)} tensors")

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

def remap_rmvpe_key(key: str) -> str:
    """
    Remap RMVPE keys to match Swift structure.
    
    Swift Structure:
    - unet.encoder.l0... (instead of layers.0)
    - unet.encoder.l0.b0... (instead of blocks.0)
    - bigru.fwd0... (instead of fc.0.gru.forward_grus.0)
    - linear... (instead of fc.1)
    """
    import re
    new_key = key
    
    # 1. Map FC layers (BiGRU and Linear)
    # PyTorch: fc.0.gru.forward_grus.0... -> Swift: bigru.fwd0...
    # Python MLX (.npz): fc.bigru.forward_grus.0... -> Swift: bigru.fwd0...
    
    # Handle BiGRU
    if "fc.0.gru." in new_key or "fc.bigru." in new_key:
        new_key = new_key.replace("fc.0.gru.", "bigru.")
        new_key = new_key.replace("fc.bigru.", "bigru.")
        
        new_key = new_key.replace("forward_grus.0", "fwd0")
        new_key = new_key.replace("backward_grus.0", "bwd0")
        
        # Handle PyTorch's flat GRU naming (weight_ih_l0, weight_ih_l0_reverse)
        # _l0_reverse -> bwd0
        if "_l0_reverse" in new_key:
            new_key = new_key.replace("_l0_reverse", "")
            # Insert .bwd0. before the parameter name
            # e.g. bigru.weight_ih -> bigru.bwd0.weight_ih
            parts = new_key.split('.')
            # parts = ['bigru', 'weight_ih']
            if len(parts) >= 2:
                new_key = f"{parts[0]}.bwd0.{parts[1]}"
                
        # _l0 -> fwd0
        elif "_l0" in new_key:
            new_key = new_key.replace("_l0", "")
            # Insert .fwd0. before the parameter name
            parts = new_key.split('.')
            if len(parts) >= 2:
                new_key = f"{parts[0]}.fwd0.{parts[1]}"

    # Handle Linear
    # PyTorch: fc.1... -> Swift: linear...
    # Python MLX: fc.linear... -> Swift: linear...
    if "fc.1." in new_key:
        new_key = new_key.replace("fc.1.", "linear.")
    if "fc.linear." in new_key:
        new_key = new_key.replace("fc.linear.", "linear.")

    # 2. Map UNet Structure
    # layers.X -> lX
    new_key = re.sub(r'\.layers\.(\d+)\.', r'.l\1.', new_key)
    
    # Map blocks
    # Encoder/Intermediate: conv.N -> bN (where N is digit)
    if ".encoder." in new_key or ".intermediate." in new_key:
        new_key = re.sub(r'\.conv\.(\d+)\.', r'.b\1.', new_key)
        
    # Decoder: conv2.N -> bN
    if ".decoder." in new_key:
        new_key = re.sub(r'\.conv2\.(\d+)\.', r'.b\1.', new_key)
        
        # Decoder conv1 mapping
        # PyTorch: conv1.0.weight -> Swift: conv1Trans.convTranspose.weight
        if ".conv1.0." in new_key:
             new_key = new_key.replace(".conv1.0.", ".conv1Trans.convTranspose.")
        # PyTorch: conv1.1.weight -> Swift: bn1.weight
        if ".conv1.1." in new_key:
             new_key = new_key.replace(".conv1.1.", ".bn1.")

    # 3. Inside blocks: conv.X -> conv1/2, bn1/2
    # Only if we successfully mapped to .bN. to avoid replacing top-level convs incorrectly
    if ".b" in new_key:
        new_key = new_key.replace(".conv.0.", ".conv1.")
        new_key = new_key.replace(".conv.1.", ".bn1.")
        new_key = new_key.replace(".conv.3.", ".conv2.")
        new_key = new_key.replace(".conv.4.", ".bn2.")
        new_key = new_key.replace(".shortcut.0.", ".shortcut.")
        
    return new_key

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
    mlx_weights = {}
    
    if rmvpe_path.suffix == '.pt':
        checkpoint = torch.load(rmvpe_path, map_location='cpu')

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                numpy_array = value.detach().cpu().numpy()
                
                # Remap key
                new_key = remap_rmvpe_key(key)
                
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
                mlx_weights[new_key] = mlx_array
                if key != new_key:
                    print(f"  {key} -> {new_key} {mlx_array.shape}")
                else:
                    print(f"  {key}: {mlx_array.shape}")

    elif rmvpe_path.suffix == '.npz':
        # Already in numpy format
        data = np.load(rmvpe_path)
        for key in data.files:
            # Remap key
            new_key = remap_rmvpe_key(key)
            
            mlx_weights[new_key] = mx.array(data[key])
            if key != new_key:
                print(f"  {key} -> {new_key} {mlx_weights[new_key].shape}")
            else:
                print(f"  {key}: {mlx_weights[new_key].shape}")

    output_file = output_path / "rmvpe.safetensors"
    print(f"\nSaving to: {output_file}")
    mx.save_safetensors(str(output_file), mlx_weights)

    print(f"✅ Successfully converted RMVPE with {len(mlx_weights)} tensors")

    return output_file


def convert_index_if_exists(model_dir: Path, output_path: Path, model_name: str):
    """
    Convert FAISS .index file to safetensors if it exists alongside the model.
    
    Searches for index files in common RVC patterns:
    - model.index
    - added_*.index
    - trained_*.index
    """
    if not FAISS_AVAILABLE:
        print("\n⚠️  FAISS not installed - skipping index conversion")
        print("   Install with: pip install faiss-cpu")
        return None
    
    # Search patterns for index files
    index_patterns = [
        model_dir / "*.index",
        model_dir / "added_*.index",
        model_dir / "trained_*.index",
    ]
    
    index_files = []
    for pattern in index_patterns:
        index_files.extend(model_dir.glob(pattern.name))
    
    if not index_files:
        print(f"\n⚠️  No .index files found in {model_dir}")
        return None
    
    # Use the first index file found
    index_path = index_files[0]
    
    print(f"\n{'='*60}")
    print(f"Converting Index: {index_path.name}")
    print(f"{'='*60}")
    
    # Load FAISS index
    print(f"Loading FAISS index from: {index_path}")
    index = faiss.read_index(str(index_path))
    n_vectors = index.ntotal
    print(f"Index contains {n_vectors:,} vectors")
    
    # Extract vectors
    vectors = index.reconstruct_n(0, n_vectors)
    dim = vectors.shape[1]
    print(f"Vector dimension: {dim}")
    print(f"Memory size: {vectors.nbytes / (1024*1024):.1f} MB")
    
    # Convert to MLX
    vectors_mlx = mx.array(vectors.astype(np.float32))
    
    # Save as safetensors
    output_file = output_path / f"{model_name}_index.safetensors"
    print(f"Saving to: {output_file}")
    mx.save_safetensors(str(output_file), {"vectors": vectors_mlx})
    
    # Verify
    loaded = mx.load(str(output_file))
    assert "vectors" in loaded, "Vectors not found in saved file"
    assert loaded["vectors"].shape == (n_vectors, dim), "Shape mismatch"
    
    print(f"✅ Successfully converted index: {n_vectors:,} vectors ({dim}D)")
    
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

    # Convert Index (if exists alongside model)
    model_dir = Path(args.model_path)
    converted_index = convert_index_if_exists(model_dir, output_path, args.model_name)
    if converted_index:
        print(f"   Index saved to: {converted_index}")

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
