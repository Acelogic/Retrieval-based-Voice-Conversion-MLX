#!/usr/bin/env python3
"""
Debug grouped convolution behavior difference.
"""
import os
import sys
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.getcwd())

print("="*60)
print("GROUPED CONVOLUTION DEBUG")
print("="*60)

# Simple test case
np.random.seed(42)
in_channels = 768
kernel_size = 128
groups = 16
in_per_group = in_channels // groups  # 48
padding = 64

# Simple input: (1, 24, 768)
seq_len = 24
x_np = np.random.randn(1, seq_len, in_channels).astype(np.float32)

# Simple weights: (Out, K, In/G) for MLX
w_np = np.random.randn(in_channels, kernel_size, in_per_group).astype(np.float32) * 0.1
b_np = np.zeros(in_channels, dtype=np.float32)

print(f"Input shape: {x_np.shape}")
print(f"Weight shape: {w_np.shape}")
print(f"Groups: {groups}")

# ===== PyTorch =====
print("\n--- PyTorch Grouped Conv1d ---")
import torch
import torch.nn as nn

x_pt = torch.from_numpy(x_np).permute(0, 2, 1)  # (1, 768, 24) for PT
# PT expects (Out, In/G, K)
w_pt = torch.from_numpy(w_np.transpose(0, 2, 1))  # (768, 48, 128)
b_pt = torch.from_numpy(b_np)

with torch.no_grad():
    conv_pt = nn.functional.conv1d(x_pt, w_pt, bias=b_pt, stride=1, padding=padding, groups=groups)

conv_pt = conv_pt.permute(0, 2, 1)  # (1, L, 768)
conv_pt_np = conv_pt.numpy()
print(f"Output shape: {conv_pt.shape}")
print(f"Output range: [{conv_pt_np.min():.4f}, {conv_pt_np.max():.4f}]")
print(f"Output mean: {conv_pt_np.mean():.4f}")

# ===== MLX =====
print("\n--- MLX Grouped Conv1d ---")
import mlx.core as mx

x_mlx = mx.array(x_np)  # (1, 24, 768)
w_mlx = mx.array(w_np)  # (768, 128, 48)

conv_mlx = mx.conv1d(x_mlx, w_mlx, stride=1, padding=padding, groups=groups)
mx.eval(conv_mlx)
conv_mlx_np = np.array(conv_mlx)
print(f"Output shape: {conv_mlx.shape}")
print(f"Output range: [{conv_mlx_np.min():.4f}, {conv_mlx_np.max():.4f}]")
print(f"Output mean: {conv_mlx_np.mean():.4f}")

# Compare
print("\n--- Comparison ---")
# Trim to same length
min_len = min(conv_pt_np.shape[1], conv_mlx_np.shape[1])
pt_trimmed = conv_pt_np[:, :min_len, :]
mlx_trimmed = conv_mlx_np[:, :min_len, :]

corr = np.corrcoef(pt_trimmed.flatten(), mlx_trimmed.flatten())[0, 1]
diff = np.abs(pt_trimmed - mlx_trimmed)
print(f"Correlation: {corr:.6f}")
print(f"Max diff: {diff.max():.4f}")
print(f"Mean diff: {diff.mean():.4f}")

if corr < 0.95:
    print("\n❌ GROUPED CONV MISMATCH!")
    print("The mx.conv1d with groups behaves differently from PyTorch!")
else:
    print("\n✅ Grouped conv matches well")
