#!/usr/bin/env python3
"""
Check if HuBERT weights in npz match PyTorch weights.
"""
import numpy as np
import torch 
import mlx.core as mx
import sys
import os

sys.path.insert(0, os.getcwd())

# Load PyTorch weights
from rvc.lib.utils import load_embedding
h = load_embedding('contentvec', None).eval()
model = h.hubert if hasattr(h, 'hubert') else h
w_pt = model.encoder.pos_conv_embed.conv.weight.detach().numpy()
b_pt = model.encoder.pos_conv_embed.conv.bias.detach().numpy()

print('PyTorch pos_conv weights:')
print(f'  Weight shape: {w_pt.shape}')  # (768, 48, 128)
print(f'  Weight[0,0,:5]: {w_pt[0,0,:5]}')
print(f'  Bias[:5]: {b_pt[:5]}')

# Load MLX weights directly from file
weights = dict(np.load('rvc_mlx/models/embedders/contentvec/hubert_mlx.npz'))
print(f'\nMLX npz keys containing pos_conv: {[k for k in weights.keys() if "pos_conv" in k]}')

if 'encoder.pos_conv_embed.weight' in weights:
    w_mlx = weights['encoder.pos_conv_embed.weight']
    print(f'\nMLX pos_conv weight:')
    print(f'  Shape: {w_mlx.shape}')  # Should be (768, 128, 48)
    # Transpose to match PT for comparison
    w_mlx_t = w_mlx.transpose(0, 2, 1)  # (768, 48, 128)
    print(f'  Weight[0,0,:5] (transposed): {w_mlx_t[0,0,:5]}')
    
    # Compare
    corr = np.corrcoef(w_pt.flatten(), w_mlx_t.flatten())[0, 1]
    print(f'\n  Weight correlation: {corr:.6f}')
    
if 'encoder.pos_conv_embed.bias' in weights:
    b_mlx = weights['encoder.pos_conv_embed.bias']
    print(f'\n  MLX bias[:5]: {b_mlx[:5]}')
    print(f'  Bias correlation: {np.corrcoef(b_pt.flatten(), b_mlx.flatten())[0, 1]:.6f}')
