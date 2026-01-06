#!/usr/bin/env python3
"""
Debug the positional conv embedding specifically.
"""
import os
import sys
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.getcwd())

# Create a simple test input
np.random.seed(42)
test_input = np.random.randn(1, 24, 768).astype(np.float32)

print("="*60)
print("POSITIONAL CONV EMBEDDING DEBUG")
print("="*60)

# ===== PyTorch =====
print("\n--- PyTorch ---")
import torch
from rvc.lib.utils import load_embedding

h = load_embedding("contentvec", None).eval()
model = h.hubert if hasattr(h, "hubert") else h

pos_conv = model.encoder.pos_conv_embed

# Get the actual weight after weight_norm is resolved
print(f"Pos conv type: {type(pos_conv.conv)}")
w_pt = pos_conv.conv.weight.detach().numpy()
b_pt = pos_conv.conv.bias.detach().numpy()
print(f"Weight shape: {w_pt.shape}")
print(f"Weight range: [{w_pt.min():.4f}, {w_pt.max():.4f}]")
print(f"Bias shape: {b_pt.shape}")
print(f"Bias range: [{b_pt.min():.4f}, {b_pt.max():.4f}]")

x_pt = torch.from_numpy(test_input)
with torch.no_grad():
    y_pt = pos_conv(x_pt)
y_pt_np = y_pt.numpy()
print(f"\nOutput shape: {y_pt.shape}")
print(f"Output range: [{y_pt_np.min():.4f}, {y_pt_np.max():.4f}]")

# ===== MLX =====
print("\n--- MLX ---")
import mlx.core as mx
from rvc_mlx.lib.mlx.hubert import HubertModel, HubertConfig

config = HubertConfig(classifier_proj_size=768)
hubert_mlx = HubertModel(config)
hubert_mlx.load_weights("rvc_mlx/models/embedders/contentvec/hubert_mlx.npz", strict=False)
mx.eval(hubert_mlx.parameters())

pos_conv_mlx = hubert_mlx.encoder.pos_conv_embed
w_mlx = np.array(pos_conv_mlx.weight)
b_mlx = np.array(pos_conv_mlx.bias)
print(f"Weight shape: {w_mlx.shape}")
print(f"Weight range: [{w_mlx.min():.4f}, {w_mlx.max():.4f}]")
print(f"Bias shape: {b_mlx.shape}")
print(f"Bias range: [{b_mlx.min():.4f}, {b_mlx.max():.4f}]")

x_mlx = mx.array(test_input)
y_mlx = pos_conv_mlx(x_mlx)
mx.eval(y_mlx)
y_mlx_np = np.array(y_mlx)
print(f"\nOutput shape: {y_mlx.shape}")
print(f"Output range: [{y_mlx_np.min():.4f}, {y_mlx_np.max():.4f}]")

# ===== Compare weights =====
print("\n--- Weight Comparison ---")
# PyTorch: (Out, In/G, K) = (768, 48, 128)
# MLX: (Out, K, In/G) = (768, 128, 48)
# So MLX's weight transposed back should match PyTorch
w_mlx_transposed = w_mlx.transpose(0, 2, 1)  # (768, 48, 128)
print(f"PT weight shape: {w_pt.shape}")
print(f"MLX weight (transposed): {w_mlx_transposed.shape}")

weight_corr = np.corrcoef(w_pt.flatten(), w_mlx_transposed.flatten())[0, 1]
print(f"Weight correlation: {weight_corr:.6f}")

bias_corr = np.corrcoef(b_pt.flatten(), b_mlx.flatten())[0, 1]
print(f"Bias correlation: {bias_corr:.6f}")

# ===== Test just the conv operation =====
print("\n--- Raw Conv Test (no residual) ---")

# PyTorch forward (simplified)
with torch.no_grad():
    x_pt_t = x_pt.permute(0, 2, 1)  # (1, 768, 24) for PT Conv1d
    conv_out_pt = pos_conv.conv(x_pt_t)
    conv_out_pt = conv_out_pt.permute(0, 2, 1)  # (1, L, 768)
    # Crop and apply GELU
    conv_out_pt = torch.nn.functional.gelu(conv_out_pt[:, :-1, :])

conv_out_pt_np = conv_out_pt.numpy()
print(f"PT conv output shape: {conv_out_pt.shape}")
print(f"PT conv output range: [{conv_out_pt_np.min():.4f}, {conv_out_pt_np.max():.4f}]")

# MLX forward (simplified)
conv_out_mlx = mx.conv1d(
    x_mlx,
    pos_conv_mlx.weight,
    stride=1,
    padding=pos_conv_mlx.padding,
    groups=pos_conv_mlx.groups
)
if pos_conv_mlx.bias is not None:
    conv_out_mlx = conv_out_mlx + pos_conv_mlx.bias
import mlx.nn as nn_mlx
conv_out_mlx = nn_mlx.gelu(conv_out_mlx[:, :-1, :])
mx.eval(conv_out_mlx)

conv_out_mlx_np = np.array(conv_out_mlx)
print(f"MLX conv output shape: {conv_out_mlx.shape}")
print(f"MLX conv output range: [{conv_out_mlx_np.min():.4f}, {conv_out_mlx_np.max():.4f}]")

conv_corr = np.corrcoef(conv_out_pt_np.flatten(), conv_out_mlx_np.flatten())[0, 1]
print(f"\nRaw conv output correlation: {conv_corr:.6f}")

# Final outputs
print("\n--- Final Pos Embed Outputs ---")
output_corr = np.corrcoef(y_pt_np.flatten(), y_mlx_np.flatten())[0, 1]
print(f"Final output correlation: {output_corr:.6f}")
print(f"Max diff: {np.abs(y_pt_np - y_mlx_np).max():.4f}")
