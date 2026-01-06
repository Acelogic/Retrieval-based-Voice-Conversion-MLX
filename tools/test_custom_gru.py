#!/usr/bin/env python3
"""
Test custom PyTorchGRU implementation against actual PyTorch GRU.
"""

import sys
import os
import torch
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.getcwd())

from rvc_mlx.lib.mlx.pytorch_gru import PyTorchGRU

def main():
    print("=== Testing Custom PyTorchGRU Implementation ===\n")

    # Load PyTorch weights
    pt_path = "rvc/models/predictors/rmvpe.pt"
    pt_ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)

    # Test parameters
    batch_size = 1
    seq_len = 100
    input_size = 384
    hidden_size = 256

    # Create test input
    np.random.seed(42)
    test_input = np.random.randn(batch_size, seq_len, input_size).astype(np.float32)

    print(f"Test input: shape={test_input.shape}\n")

    # === PyTorch GRU ===
    print("--- PyTorch GRU ---")
    pt_gru = torch.nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=False)
    pt_gru.weight_ih_l0 = torch.nn.Parameter(pt_ckpt['fc.0.gru.weight_ih_l0'])
    pt_gru.weight_hh_l0 = torch.nn.Parameter(pt_ckpt['fc.0.gru.weight_hh_l0'])
    pt_gru.bias_ih_l0 = torch.nn.Parameter(pt_ckpt['fc.0.gru.bias_ih_l0'])
    pt_gru.bias_hh_l0 = torch.nn.Parameter(pt_ckpt['fc.0.gru.bias_hh_l0'])
    pt_gru.eval()

    with torch.no_grad():
        test_input_torch = torch.from_numpy(test_input).half()
        pt_output, _ = pt_gru(test_input_torch)

    pt_output_np = pt_output.numpy()
    print(f"Output: shape={pt_output_np.shape}, range=[{pt_output_np.min():.4f}, {pt_output_np.max():.4f}]")
    print(f"  mean={pt_output_np.mean():.6f}, std={pt_output_np.std():.6f}")

    # === Custom MLX PyTorchGRU ===
    print("\n--- Custom MLX PyTorchGRU ---")
    mlx_gru = PyTorchGRU(input_size, hidden_size, bias=True)

    # Load weights directly from PyTorch
    mlx_gru.weight_ih = mx.array(pt_ckpt['fc.0.gru.weight_ih_l0'].numpy())
    mlx_gru.weight_hh = mx.array(pt_ckpt['fc.0.gru.weight_hh_l0'].numpy())
    mlx_gru.bias_ih = mx.array(pt_ckpt['fc.0.gru.bias_ih_l0'].numpy())
    mlx_gru.bias_hh = mx.array(pt_ckpt['fc.0.gru.bias_hh_l0'].numpy())

    test_input_mlx = mx.array(test_input)
    mlx_output = mlx_gru(test_input_mlx)
    mlx_output_np = np.array(mlx_output)

    print(f"Output: shape={mlx_output_np.shape}, range=[{mlx_output_np.min():.4f}, {mlx_output_np.max():.4f}]")
    print(f"  mean={mlx_output_np.mean():.6f}, std={mlx_output_np.std():.6f}")

    # === Comparison ===
    print("\n--- Comparison ---")
    diff = np.abs(pt_output_np - mlx_output_np)
    rmse = np.sqrt(np.mean((pt_output_np - mlx_output_np) ** 2))
    corr = np.corrcoef(pt_output_np.flatten(), mlx_output_np.flatten())[0, 1]

    print(f"Absolute difference:")
    print(f"  max={diff.max():.6f}, mean={diff.mean():.6f}, std={diff.std():.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Correlation: {corr:.6f}")

    if diff.max() < 1e-4:
        print("\n✅ Perfect match!")
    elif diff.max() < 0.01:
        print("\n✅ Excellent match (within float precision)!")
    elif diff.max() < 0.1:
        print("\n⚠️  Good match (small differences)")
    else:
        print("\n❌ Significant differences remain")

        # Show first few values for debugging
        print("\nFirst 3 timesteps, first 5 features:")
        print("PyTorch:")
        print(pt_output_np[0, :3, :5])
        print("MLX:")
        print(mlx_output_np[0, :3, :5])

if __name__ == "__main__":
    main()
