#!/usr/bin/env python3
"""
Check if weights are loading correctly by comparing a specific layer.
"""

import torch
import mlx.core as mx
import numpy as np

def main():
    # Load PyTorch model
    pt_path = "/Users/mcruz/Library/Application Support/Replay/com.replay.Replay/models/Drake/model.pth"
    print(f"Loading PyTorch model from {pt_path}...")
    pt_ckpt = torch.load(pt_path, map_location="cpu")
    pt_weights = pt_ckpt["weight"]

    # Load MLX model
    mlx_path = "rvc_mlx/models/checkpoints/Drake.npz"
    print(f"Loading MLX model from {mlx_path}...")
    mlx_weights = mx.load(mlx_path)

    # Compare a simple layer: emb_phone (Linear layer)
    # PyTorch key: enc_p.emb_phone.weight, shape (192, 768)
    # MLX key: enc_p.emb_phone.weight, shape (192, 768) - Linear weights aren't transposed

    print("\n--- Checking enc_p.emb_phone.weight ---")
    pt_key = "enc_p.emb_phone.weight"
    mlx_key = "enc_p.emb_phone.weight"

    if pt_key in pt_weights:
        pt_w = pt_weights[pt_key].numpy()
        print(f"PyTorch shape: {pt_w.shape}")
        print(f"PyTorch range: [{pt_w.min():.6f}, {pt_w.max():.6f}], mean: {pt_w.mean():.6f}")
    else:
        print(f"❌ {pt_key} not found in PyTorch model!")
        return

    if mlx_key in mlx_weights:
        mlx_w = np.array(mlx_weights[mlx_key])
        print(f"MLX shape: {mlx_w.shape}")
        print(f"MLX range: [{mlx_w.min():.6f}, {mlx_w.max():.6f}], mean: {mlx_w.mean():.6f}")
    else:
        print(f"❌ {mlx_key} not found in MLX model!")
        return

    # Compare
    if pt_w.shape != mlx_w.shape:
        print(f"❌ Shape mismatch!")
        return

    diff = np.abs(pt_w - mlx_w)
    print(f"\nDifference:")
    print(f"  Max: {diff.max():.10f}")
    print(f"  Mean: {diff.mean():.10f}")
    print(f"  RMSE: {np.sqrt(np.mean(diff**2)):.10f}")

    if diff.max() < 1e-6:
        print("  ✅ Weights match perfectly!")
    elif diff.max() < 1e-3:
        print("  ⚠️  Small difference (acceptable)")
    else:
        print("  ❌ Large difference!")

    # Check Conv1d layer (should be transposed)
    print("\n--- Checking dec.conv_pre.weight (Conv1d) ---")
    pt_key = "dec.conv_pre.weight"
    mlx_key = "dec.conv_pre.weight"

    if pt_key in pt_weights and mlx_key in mlx_weights:
        pt_w = pt_weights[pt_key].numpy()
        mlx_w = np.array(mlx_weights[mlx_key])

        print(f"PyTorch shape: {pt_w.shape} (Out, In, K)")
        print(f"MLX shape: {mlx_w.shape} (should be Out, K, In)")

        # Transpose MLX back to PyTorch format to compare
        mlx_w_pt_format = mlx_w.transpose(0, 2, 1)  # (Out, K, In) -> (Out, In, K)

        if pt_w.shape == mlx_w_pt_format.shape:
            diff = np.abs(pt_w - mlx_w_pt_format)
            print(f"\nDifference (after transposing MLX back):")
            print(f"  Max: {diff.max():.10f}")
            print(f"  Mean: {diff.mean():.10f}")

            if diff.max() < 1e-6:
                print("  ✅ Conv1d weights match (transpose is correct)!")
            else:
                print("  ❌ Conv1d weights don't match!")
        else:
            print(f"❌ Shape mismatch even after transpose: {mlx_w_pt_format.shape} vs {pt_w.shape}")

    # Check pitch embedding (embedding layer)
    print("\n--- Checking enc_p.emb_pitch.weight (Embedding) ---")
    pt_key = "enc_p.emb_pitch.weight"
    mlx_key = "enc_p.emb_pitch.weight"

    if pt_key in pt_weights and mlx_key in mlx_weights:
        pt_w = pt_weights[pt_key].numpy()
        mlx_w = np.array(mlx_weights[mlx_key])

        print(f"PyTorch shape: {pt_w.shape}")
        print(f"MLX shape: {mlx_w.shape}")

        diff = np.abs(pt_w - mlx_w)
        print(f"\nDifference:")
        print(f"  Max: {diff.max():.10f}")
        print(f"  Mean: {diff.mean():.10f}")

        if diff.max() < 1e-6:
            print("  ✅ Embedding weights match!")
        else:
            print("  ❌ Embedding weights don't match!")

if __name__ == "__main__":
    main()
