#!/usr/bin/env python3
"""
Compare GRU weights between PyTorch and MLX to verify correct conversion.
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.getcwd())

def compare_weights():
    print("=== Comparing GRU Weights ===\n")

    # Load PyTorch model
    pt_path = "rvc/models/predictors/rmvpe.pt"
    pt_ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)

    # Load MLX weights
    mlx_path = "rvc_mlx/models/predictors/rmvpe_mlx.npz"
    mlx_weights = np.load(mlx_path)

    print("PyTorch GRU keys:")
    gru_keys_pt = [k for k in pt_ckpt.keys() if 'fc.0.gru' in k]
    for k in sorted(gru_keys_pt):
        print(f"  {k}: {tuple(pt_ckpt[k].shape)}")

    print("\nMLX GRU keys:")
    gru_keys_mlx = [k for k in mlx_weights.keys() if 'bigru' in k]
    for k in sorted(gru_keys_mlx):
        print(f"  {k}: {tuple(mlx_weights[k].shape)}")

    # Compare forward GRU weights
    print("\n=== Forward GRU ===")

    # Weight_ih (input-hidden) -> Wx
    pt_Wx = pt_ckpt['fc.0.gru.weight_ih_l0'].numpy()
    mlx_Wx = mlx_weights['fc.bigru.forward_grus.0.Wx']
    print(f"\nWx (input-hidden):")
    print(f"  PyTorch: {pt_Wx.shape}, range=[{pt_Wx.min():.6f}, {pt_Wx.max():.6f}], mean={pt_Wx.mean():.6f}")
    print(f"  MLX:     {mlx_Wx.shape}, range=[{mlx_Wx.min():.6f}, {mlx_Wx.max():.6f}], mean={mlx_Wx.mean():.6f}")
    if pt_Wx.shape == mlx_Wx.shape:
        diff = np.abs(pt_Wx - mlx_Wx)
        print(f"  Difference: max={diff.max():.6f}, mean={diff.mean():.6f}")
        if diff.max() < 1e-5:
            print(f"  ✅ MATCH!")
        else:
            print(f"  ❌ MISMATCH!")

    # Weight_hh (hidden-hidden) -> Wh
    pt_Wh = pt_ckpt['fc.0.gru.weight_hh_l0'].numpy()
    mlx_Wh = mlx_weights['fc.bigru.forward_grus.0.Wh']
    print(f"\nWh (hidden-hidden):")
    print(f"  PyTorch: {pt_Wh.shape}, range=[{pt_Wh.min():.6f}, {pt_Wh.max():.6f}], mean={pt_Wh.mean():.6f}")
    print(f"  MLX:     {mlx_Wh.shape}, range=[{mlx_Wh.min():.6f}, {mlx_Wh.max():.6f}], mean={mlx_Wh.mean():.6f}")
    if pt_Wh.shape == mlx_Wh.shape:
        diff = np.abs(pt_Wh - mlx_Wh)
        print(f"  Difference: max={diff.max():.6f}, mean={diff.mean():.6f}")
        if diff.max() < 1e-5:
            print(f"  ✅ MATCH!")
        else:
            print(f"  ❌ MISMATCH!")

    # Biases
    pt_b_ih = pt_ckpt['fc.0.gru.bias_ih_l0'].numpy()
    pt_b_hh = pt_ckpt['fc.0.gru.bias_hh_l0'].numpy()
    mlx_b = mlx_weights['fc.bigru.forward_grus.0.b']
    mlx_bhn = mlx_weights['fc.bigru.forward_grus.0.bhn']

    H = 256  # hidden size
    print(f"\nBias (combined):")
    print(f"  PyTorch bias_ih: {pt_b_ih.shape}, range=[{pt_b_ih.min():.6f}, {pt_b_ih.max():.6f}]")
    print(f"  PyTorch bias_hh: {pt_b_hh.shape}, range=[{pt_b_hh.min():.6f}, {pt_b_hh.max():.6f}]")
    print(f"  MLX b:    {mlx_b.shape}, range=[{mlx_b.min():.6f}, {mlx_b.max():.6f}]")
    print(f"  MLX bhn:  {mlx_bhn.shape}, range=[{mlx_bhn.min():.6f}, {mlx_bhn.max():.6f}]")

    # Check if MLX b matches PyTorch combined biases
    pt_b_combined = pt_b_ih.copy()
    pt_b_combined[:2*H] += pt_b_hh[:2*H]
    print(f"\n  Expected MLX b (bias_ih + bias_hh[:2H]):")
    print(f"    range=[{pt_b_combined.min():.6f}, {pt_b_combined.max():.6f}]")

    diff_b = np.abs(mlx_b - pt_b_combined)
    print(f"  Difference: max={diff_b.max():.6f}, mean={diff_b.mean():.6f}")
    if diff_b.max() < 1e-5:
        print(f"  ✅ MATCH!")
    else:
        print(f"  ❌ MISMATCH!")
        # Show which gates mismatch
        for gate_name, start, end in [("reset", 0, H), ("update", H, 2*H), ("new", 2*H, 3*H)]:
            gate_diff = np.abs(mlx_b[start:end] - pt_b_combined[start:end])
            print(f"    {gate_name} gate: max_diff={gate_diff.max():.6f}")

    # Check bhn
    pt_bhn_expected = pt_b_hh[2*H:]
    diff_bhn = np.abs(mlx_bhn - pt_bhn_expected)
    print(f"\n  MLX bhn vs PyTorch bias_hh[2H:]:")
    print(f"  Difference: max={diff_bhn.max():.6f}, mean={diff_bhn.mean():.6f}")
    if diff_bhn.max() < 1e-5:
        print(f"  ✅ MATCH!")
    else:
        print(f"  ❌ MISMATCH!")

    # Check Linear layer
    print("\n\n=== Linear Layer ===")
    pt_linear_weight = pt_ckpt['fc.1.weight'].numpy()
    pt_linear_bias = pt_ckpt['fc.1.bias'].numpy()
    mlx_linear_weight = mlx_weights['fc.linear.weight']
    mlx_linear_bias = mlx_weights['fc.linear.bias']

    print(f"Weight:")
    print(f"  PyTorch: {pt_linear_weight.shape}, range=[{pt_linear_weight.min():.6f}, {pt_linear_weight.max():.6f}]")
    print(f"  MLX:     {mlx_linear_weight.shape}, range=[{mlx_linear_weight.min():.6f}, {mlx_linear_weight.max():.6f}]")
    if pt_linear_weight.shape == mlx_linear_weight.shape:
        diff = np.abs(pt_linear_weight - mlx_linear_weight)
        print(f"  Difference: max={diff.max():.6f}, mean={diff.mean():.6f}")
        if diff.max() < 1e-3:  # Allow for float16 quantization
            print(f"  ✅ MATCH (within float16 tolerance)!")
        else:
            print(f"  ❌ MISMATCH!")

    print(f"\nBias:")
    print(f"  PyTorch: {pt_linear_bias.shape}, range=[{pt_linear_bias.min():.6f}, {pt_linear_bias.max():.6f}]")
    print(f"  MLX:     {mlx_linear_bias.shape}, range=[{mlx_linear_bias.min():.6f}, {mlx_linear_bias.max():.6f}]")
    if pt_linear_bias.shape == mlx_linear_bias.shape:
        diff = np.abs(pt_linear_bias - mlx_linear_bias)
        print(f"  Difference: max={diff.max():.6f}, mean={diff.mean():.6f}")
        if diff.max() < 1e-3:
            print(f"  ✅ MATCH (within float16 tolerance)!")
        else:
            print(f"  ❌ MISMATCH!")

if __name__ == "__main__":
    compare_weights()
