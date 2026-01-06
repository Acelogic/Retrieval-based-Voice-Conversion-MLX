#!/usr/bin/env python3
"""Quick test of TextEncoder weight loading."""
import numpy as np
import sys
import os
sys.path.insert(0, os.getcwd())

import mlx.core as mx
from rvc_mlx.lib.mlx.synthesizers import Synthesizer as MLX_Synthesizer
from rvc_mlx.infer.infer_mlx import remap_keys

# Model config for Slim Shady (from checkpoint)
kwargs = {
    'spec_channels': 1025, 'segment_size': 32,
    'inter_channels': 192, 'hidden_channels': 192,
    'filter_channels': 768, 'n_heads': 2,
    'n_layers': 6, 'kernel_size': 3,
    'p_dropout': 0, 'resblock': '1',
    'resblock_kernel_sizes': [3, 7, 11], 
    'resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    'upsample_rates': [12, 10, 2, 2], 
    'upsample_initial_channel': 512,
    'upsample_kernel_sizes': [24, 20, 4, 4], 
    'spk_embed_dim': 1,
    'gin_channels': 256, 'sr': 48000,
    'use_f0': True, 'text_enc_hidden_dim': 768, 'vocoder': 'NSF',
}

# Load MLX model
net_g = MLX_Synthesizer(**kwargs)
weights = mx.load('weights/slim_shady.npz')
weights = remap_keys(dict(weights))
net_g.load_weights(list(weights.items()), strict=False)
mx.eval(net_g.parameters())

print('MLX Synthesizer loaded')
print(f'emb_phone weight shape: {net_g.enc_p.emb_phone.weight.shape}')
print(f'emb_phone weight[:5]: {np.array(net_g.enc_p.emb_phone.weight).flatten()[:5]}')

# Test forward pass
np.random.seed(42)
phone = mx.array(np.random.randn(1, 10, 768).astype(np.float32))
pitch = mx.array([[100, 120, 130, 140, 150, 160, 170, 180, 190, 200]]).astype(mx.int32)
lengths = mx.array([10], dtype=mx.int32)

m, logs, mask = net_g.enc_p(phone, pitch, lengths)
mx.eval(m, logs, mask)

print(f'\nMLX enc_p output:')
print(f'  m shape: {m.shape}')
print(f'  m range: [{np.array(m).min():.4f}, {np.array(m).max():.4f}]')
print(f'  m mean: {np.array(m).mean():.4f}')
