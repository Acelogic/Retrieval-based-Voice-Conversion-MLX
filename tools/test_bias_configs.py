#!/usr/bin/env python3
"""
Test different bias configurations to find the correct one for MLX GRU.
"""

import sys
import os
import torch
import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.getcwd())

def test_bias_config(config_name, mlx_b, mlx_bhn, test_input, pt_output_np):
    """Test a specific bias configuration."""
    input_size = 384
    hidden_size = 256

    mlx_path = "rvc_mlx/models/predictors/rmvpe_mlx.npz"
    mlx_weights = np.load(mlx_path)

    mlx_gru = nn.GRU(input_size, hidden_size, bias=True)
    mlx_gru.update({
        'Wx': mx.array(mlx_weights['fc.bigru.forward_grus.0.Wx']),
        'Wh': mx.array(mlx_weights['fc.bigru.forward_grus.0.Wh']),
        'b': mx.array(mlx_b),
        'bhn': mx.array(mlx_bhn),
    })
    mlx_gru.eval()

    test_input_mlx = mx.array(test_input)
    mlx_output = mlx_gru(test_input_mlx)
    mlx_output_np = np.array(mlx_output)

    diff = np.abs(pt_output_np - mlx_output_np)
    rmse = np.sqrt(np.mean((pt_output_np - mlx_output_np) ** 2))
    corr = np.corrcoef(pt_output_np.flatten(), mlx_output_np.flatten())[0, 1]

    print(f"\n{config_name}:")
    print(f"  Max diff: {diff.max():.6f}, RMSE: {rmse:.6f}, Corr: {corr:.6f}")

    return diff.max()

def main():
    print("=== Testing Different Bias Configurations ===\n")

    # Load PyTorch weights
    pt_path = "rvc/models/predictors/rmvpe.pt"
    pt_ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)

    pt_bias_ih = pt_ckpt['fc.0.gru.bias_ih_l0'].numpy()
    pt_bias_hh = pt_ckpt['fc.0.gru.bias_hh_l0'].numpy()

    H = 256
    print(f"PyTorch biases:")
    print(f"  bias_ih: {pt_bias_ih.shape}, range=[{pt_bias_ih.min():.4f}, {pt_bias_ih.max():.4f}]")
    print(f"  bias_hh: {pt_bias_hh.shape}, range=[{pt_bias_hh.min():.4f}, {pt_bias_hh.max():.4f}]")

    # Create test input
    batch_size = 1
    seq_len = 100
    input_size = 384
    hidden_size = 256

    np.random.seed(42)
    test_input = np.random.randn(batch_size, seq_len, input_size).astype(np.float32)

    # Get PyTorch output
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
    print(f"\nPyTorch output: range=[{pt_output_np.min():.4f}, {pt_output_np.max():.4f}], mean={pt_output_np.mean():.4f}")

    # Test different configurations
    configs = []

    # Config 1: Current (bias_ih + bias_hh[:2H], bias_hh[2H:])
    b1 = pt_bias_ih.copy()
    b1[:2*H] += pt_bias_hh[:2*H]
    bhn1 = pt_bias_hh[2*H:]
    configs.append(("Config 1: Current (b_ih + b_hh[:2H], b_hh[2H:])", b1, bhn1))

    # Config 2: Only bias_ih, bias_hh[2H:]
    configs.append(("Config 2: Only (b_ih, b_hh[2H:])", pt_bias_ih, pt_bias_hh[2*H:]))

    # Config 3: bias_ih + full bias_hh, but zero bhn
    b3 = pt_bias_ih + pt_bias_hh
    bhn3 = np.zeros_like(pt_bias_hh[2*H:])
    configs.append(("Config 3: (b_ih + b_hh, zeros)", b3, bhn3))

    # Config 4: bias_ih only, zeros for bhn
    configs.append(("Config 4: (b_ih only, zeros)", pt_bias_ih, np.zeros_like(pt_bias_hh[2*H:])))

    # Config 5: Full bias_hh only, bias_hh[2H:] for bhn
    configs.append(("Config 5: (b_hh only, b_hh[2H:])", pt_bias_hh, pt_bias_hh[2*H:]))

    # Config 6: Try not combining, just use bias_ih for b, and bias_hh for... wait, that doesn't make sense

    # Config 7: bias_ih, zero bhn
    configs.append(("Config 6: (b_ih, zeros)", pt_bias_ih, np.zeros_like(pt_bias_hh[2*H:])))

    # Config 8: Try bias_ih, bias_hh[:H] (just reset gate from hh)?
    configs.append(("Config 7: (b_ih, b_hh[:H])", pt_bias_ih, pt_bias_hh[:H]))

    # Config 9: All zeros to understand baseline
    configs.append(("Config 8: (zeros, zeros)", np.zeros_like(pt_bias_ih), np.zeros_like(pt_bias_hh[2*H:])))

    best_config = None
    best_diff = float('inf')

    for config_name, b, bhn in configs:
        diff = test_bias_config(config_name, b, bhn, test_input, pt_output_np)
        if diff < best_diff:
            best_diff = diff
            best_config = config_name

    print(f"\n{'='*60}")
    print(f"Best configuration: {best_config}")
    print(f"Best max diff: {best_diff:.6f}")

    if best_diff < 1e-4:
        print("✅ Found exact match!")
    elif best_diff < 0.01:
        print("⚠️  Close match (likely numerical precision)")
    else:
        print("❌ No good match found - MLX GRU may have fundamental differences")

if __name__ == "__main__":
    main()
