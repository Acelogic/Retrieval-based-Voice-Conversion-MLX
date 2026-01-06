#!/usr/bin/env python3
"""
Test if MLX GRU produces the same output as PyTorch GRU with the same weights and input.
"""

import sys
import os
import torch
import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.getcwd())

def test_gru():
    print("=== Testing GRU Forward Pass ===\n")

    # Load weights
    pt_path = "rvc/models/predictors/rmvpe.pt"
    pt_ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)

    mlx_path = "rvc_mlx/models/predictors/rmvpe_mlx.npz"
    mlx_weights = np.load(mlx_path)

    # Create test input
    batch_size = 1
    seq_len = 100
    input_size = 384
    hidden_size = 256

    np.random.seed(42)
    test_input = np.random.randn(batch_size, seq_len, input_size).astype(np.float32)

    print(f"Test input: shape={test_input.shape}, range=[{test_input.min():.4f}, {test_input.max():.4f}]")

    # === PyTorch GRU ===
    print("\n--- PyTorch GRU ---")
    pt_gru = torch.nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=False)

    # Load weights
    pt_gru.weight_ih_l0 = torch.nn.Parameter(pt_ckpt['fc.0.gru.weight_ih_l0'])
    pt_gru.weight_hh_l0 = torch.nn.Parameter(pt_ckpt['fc.0.gru.weight_hh_l0'])
    pt_gru.bias_ih_l0 = torch.nn.Parameter(pt_ckpt['fc.0.gru.bias_ih_l0'])
    pt_gru.bias_hh_l0 = torch.nn.Parameter(pt_ckpt['fc.0.gru.bias_hh_l0'])
    pt_gru.eval()

    with torch.no_grad():
        test_input_torch = torch.from_numpy(test_input).half()  # Convert to float16
        pt_output, pt_hidden = pt_gru(test_input_torch)

    pt_output_np = pt_output.numpy()
    print(f"Output: shape={pt_output_np.shape}, range=[{pt_output_np.min():.4f}, {pt_output_np.max():.4f}]")
    print(f"  mean={pt_output_np.mean():.4f}, std={pt_output_np.std():.4f}")

    # === MLX GRU ===
    print("\n--- MLX GRU ---")
    mlx_gru = nn.GRU(input_size, hidden_size, bias=True)

    # Load weights
    mlx_gru.update({
        'Wx': mx.array(mlx_weights['fc.bigru.forward_grus.0.Wx']),
        'Wh': mx.array(mlx_weights['fc.bigru.forward_grus.0.Wh']),
        'b': mx.array(mlx_weights['fc.bigru.forward_grus.0.b']),
        'bhn': mx.array(mlx_weights['fc.bigru.forward_grus.0.bhn']),
    })
    mlx_gru.eval()

    test_input_mlx = mx.array(test_input)
    mlx_output = mlx_gru(test_input_mlx)

    mlx_output_np = np.array(mlx_output)
    print(f"Output: shape={mlx_output_np.shape}, range=[{mlx_output_np.min():.4f}, {mlx_output_np.max():.4f}]")
    print(f"  mean={mlx_output_np.mean():.4f}, std={mlx_output_np.std():.4f}")

    # === Comparison ===
    print("\n--- Comparison ---")
    diff = np.abs(pt_output_np - mlx_output_np)
    print(f"Absolute difference:")
    print(f"  max={diff.max():.6f}, mean={diff.mean():.6f}, std={diff.std():.6f}")

    rmse = np.sqrt(np.mean((pt_output_np - mlx_output_np) ** 2))
    corr = np.corrcoef(pt_output_np.flatten(), mlx_output_np.flatten())[0, 1]
    print(f"RMSE: {rmse:.6f}")
    print(f"Correlation: {corr:.6f}")

    if diff.max() < 1e-4:
        print("\n✅ GRU outputs match!")
    elif diff.max() < 0.01:
        print("\n⚠️  Small differences (likely float precision)")
    else:
        print("\n❌ Significant differences!")

        # Debug: Check if it's a bias issue
        print("\nDetailed analysis:")
        print(f"PyTorch output first 5 timesteps, first 5 features:")
        print(pt_output_np[0, :5, :5])
        print(f"MLX output first 5 timesteps, first 5 features:")
        print(mlx_output_np[0, :5, :5])

if __name__ == "__main__":
    test_gru()
