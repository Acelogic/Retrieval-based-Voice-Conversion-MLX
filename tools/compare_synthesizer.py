#!/usr/bin/env python3
"""
Compare Synthesizer outputs with identical inputs.
"""
import numpy as np
import torch
import mlx.core as mx
import sys
import os

sys.path.insert(0, os.getcwd())

# Create identical test inputs
np.random.seed(42)
seq_len = 100

# Phone features (from HuBERT) 
phone = np.random.randn(1, seq_len, 768).astype(np.float32)

# F0 - simple sine wave
t = np.linspace(0, 1, seq_len)
f0 = (220 + 50 * np.sin(2 * np.pi * 2 * t)).astype(np.float32)

# Coarse pitch
f0_mel = 1127 * np.log(1 + f0 / 700)
f0_mel_min = 1127 * np.log(1 + 50 / 700)
f0_mel_max = 1127 * np.log(1 + 1100 / 700)
pitch = np.clip((f0_mel - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1, 1, 255).astype(np.int64)

# Lengths
lengths = np.array([seq_len])

print('Test inputs:')
print(f'  phone: {phone.shape}')
print(f'  f0: {f0.shape}')
print(f'  pitch: {pitch.shape}')

# ===== PyTorch Synthesizer =====
print('\n--- PyTorch Synthesizer ---')

from rvc.lib.algorithm.synthesizers import Synthesizer as PT_Synthesizer

pt_model_path = '/Users/mcruz/Library/Application Support/Replay/com.replay.Replay/models/Slim Shady/model.pth'
pt_ckpt = torch.load(pt_model_path, map_location='cpu')
config = pt_ckpt.get('config', pt_ckpt.get('params', {}))

if isinstance(config, list):
    kwargs = {
        'spec_channels': config[0], 'segment_size': config[1],
        'inter_channels': config[2], 'hidden_channels': config[3],
        'filter_channels': config[4], 'n_heads': config[5],
        'n_layers': config[6], 'kernel_size': config[7],
        'p_dropout': config[8], 'resblock': config[9],
        'resblock_kernel_sizes': config[10], 'resblock_dilation_sizes': config[11],
        'upsample_rates': config[12], 'upsample_initial_channel': config[13],
        'upsample_kernel_sizes': config[14], 'spk_embed_dim': config[15],
        'gin_channels': config[16], 'sr': config[17] if len(config) > 17 else 40000,
        'use_f0': True, 'text_enc_hidden_dim': 768, 'vocoder': 'NSF',
    }
else:
    kwargs = dict(config)

net_g_pt = PT_Synthesizer(**kwargs)
net_g_pt.load_state_dict(pt_ckpt['weight'], strict=False)
net_g_pt.eval()
net_g_pt.remove_weight_norm()

# Convert inputs to PyTorch
phone_pt = torch.from_numpy(phone)
pitch_pt = torch.from_numpy(pitch[None, :]).long()
f0_pt = torch.from_numpy(f0[None, :])
lengths_pt = torch.from_numpy(lengths).long()
sid_pt = torch.tensor([0]).long()

with torch.no_grad():
    audio_pt, _, _ = net_g_pt.infer(phone_pt, lengths_pt, pitch_pt, f0_pt, sid_pt)

audio_pt_np = audio_pt.squeeze().numpy()
print(f'  Output shape: {audio_pt_np.shape}')
print(f'  Output range: [{audio_pt_np.min():.4f}, {audio_pt_np.max():.4f}]')
print(f'  Output RMS: {np.sqrt(np.mean(audio_pt_np**2)):.4f}')

# ===== MLX Synthesizer =====
print('\n--- MLX Synthesizer ---')

from rvc_mlx.lib.mlx.synthesizers import Synthesizer as MLX_Synthesizer
from rvc_mlx.infer.infer_mlx import remap_keys

net_g_mlx = MLX_Synthesizer(**kwargs)

# Load and remap weights
weights = mx.load('weights/slim_shady.npz')
weights = remap_keys(dict(weights))
net_g_mlx.load_weights(list(weights.items()), strict=False)
mx.eval(net_g_mlx.parameters())

phone_mlx = mx.array(phone)
pitch_mlx = mx.array(pitch.astype(np.int32))[None, :]
f0_mlx = mx.array(f0)[None, :]
lengths_mlx = mx.array(lengths.astype(np.int32))
sid_mlx = mx.array([0], dtype=mx.int32)

audio_mlx, _, _ = net_g_mlx.infer(phone_mlx, lengths_mlx, pitch_mlx, f0_mlx, sid_mlx)
mx.eval(audio_mlx)

audio_mlx_np = np.array(audio_mlx).squeeze()
print(f'  Output shape: {audio_mlx_np.shape}')
print(f'  Output range: [{audio_mlx_np.min():.4f}, {audio_mlx_np.max():.4f}]')
print(f'  Output RMS: {np.sqrt(np.mean(audio_mlx_np**2)):.4f}')

# ===== Compare =====
print('\n--- Comparison ---')
min_len = min(len(audio_pt_np), len(audio_mlx_np))
pt_trim = audio_pt_np[:min_len]
mlx_trim = audio_mlx_np[:min_len]

corr = np.corrcoef(pt_trim, mlx_trim)[0, 1]
print(f'Correlation: {corr:.6f}')
print(f'Max diff: {np.abs(pt_trim - mlx_trim).max():.4f}')
print(f'Mean diff: {np.abs(pt_trim - mlx_trim).mean():.4f}')

if corr > 0.95:
    print('\n✅ Synthesizer outputs match well!')
elif corr > 0.8:
    print('\n⚠️ Synthesizer outputs have moderate correlation')
else:
    print('\n❌ Synthesizer outputs diverge significantly')
