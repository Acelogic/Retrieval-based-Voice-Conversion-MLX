#!/usr/bin/env python3
"""
Compare Synthesizer layer-by-layer between PyTorch and MLX.
"""
import numpy as np
import torch
import mlx.core as mx
import sys
import os

sys.path.insert(0, os.getcwd())

def compare(name, pt, mlx):
    pt_np = pt.detach().cpu().numpy() if hasattr(pt, 'detach') else np.asarray(pt)
    mlx_np = np.array(mlx)
    
    if pt_np.shape != mlx_np.shape:
        print(f"  {name}: SHAPE MISMATCH PT={pt_np.shape} MLX={mlx_np.shape}")
        return
    
    corr = np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1]
    diff = np.abs(pt_np - mlx_np)
    status = "✅" if corr > 0.99 else ("⚠️" if corr > 0.9 else "❌")
    print(f"  {name}: {status} corr={corr:.4f}, max_diff={diff.max():.4f}")

# Create identical test inputs
np.random.seed(42)
seq_len = 50

phone = np.random.randn(1, seq_len, 768).astype(np.float32)
t = np.linspace(0, 1, seq_len)
f0 = (220 + 50 * np.sin(2 * np.pi * 2 * t)).astype(np.float32)
f0_mel = 1127 * np.log(1 + f0 / 700)
f0_mel_min = 1127 * np.log(1 + 50 / 700)
f0_mel_max = 1127 * np.log(1 + 1100 / 700)
pitch = np.clip((f0_mel - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1, 1, 255).astype(np.int64)
lengths = np.array([seq_len])

print('=' * 60)
print('SYNTHESIZER LAYER-BY-LAYER COMPARISON')
print('=' * 60)

# ===== Load PyTorch =====
from rvc.lib.algorithm.synthesizers import Synthesizer as PT_Synthesizer

pt_model_path = '/Users/mcruz/Library/Application Support/Replay/com.replay.Replay/models/Slim Shady/model.pth'
pt_ckpt = torch.load(pt_model_path, map_location='cpu', weights_only=False)
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

# ===== Load MLX =====
from rvc_mlx.lib.mlx.synthesizers import Synthesizer as MLX_Synthesizer
from rvc_mlx.infer.infer_mlx import remap_keys

net_g_mlx = MLX_Synthesizer(**kwargs)
weights = mx.load('weights/slim_shady.npz')
weights = remap_keys(dict(weights))
net_g_mlx.load_weights(list(weights.items()), strict=False)
mx.eval(net_g_mlx.parameters())

# Convert inputs
phone_pt = torch.from_numpy(phone)
pitch_pt = torch.from_numpy(pitch[None, :]).long()
f0_pt = torch.from_numpy(f0[None, :])
lengths_pt = torch.from_numpy(lengths).long()
sid_pt = torch.tensor([0]).long()

phone_mlx = mx.array(phone)
pitch_mlx = mx.array(pitch.astype(np.int32))[None, :]
f0_mlx = mx.array(f0)[None, :]
lengths_mlx = mx.array(lengths.astype(np.int32))
sid_mlx = mx.array([0], dtype=mx.int32)

# ===== Compare Text Encoder (enc_p) =====
print('\n--- Stage 1: Text Encoder (enc_p) ---')

with torch.no_grad():
    m_p_pt, logs_p_pt, x_mask_pt = net_g_pt.enc_p(phone_pt, pitch_pt, lengths_pt)

m_p_mlx, logs_p_mlx, x_mask_mlx = net_g_mlx.enc_p(phone_mlx, pitch_mlx, lengths_mlx)
mx.eval(m_p_mlx, logs_p_mlx, x_mask_mlx)

compare("enc_p m_p (mean)", m_p_pt, m_p_mlx)
compare("enc_p logs_p (log-var)", logs_p_pt, logs_p_mlx)

# ===== Compare Decoder (dec) =====
print('\n--- Stage 2: Decoder (dec) ---')

with torch.no_grad():
    # Use m_p as the "z" input to decoder
    audio_pt = net_g_pt.dec(m_p_pt, f0_pt, g=None)

audio_mlx = net_g_mlx.dec(m_p_pt.numpy(), np.array(f0_mlx), g=None)  # Use PT outputs as input
mx.eval(audio_mlx)

# Convert for comparison
audio_pt_np = audio_pt.squeeze().numpy()
audio_mlx_np = np.array(audio_mlx).squeeze()

print(f"  dec PT output: shape={audio_pt_np.shape}, range=[{audio_pt_np.min():.4f}, {audio_pt_np.max():.4f}]")
print(f"  dec MLX output: shape={audio_mlx_np.shape}, range=[{audio_mlx_np.min():.4f}, {audio_mlx_np.max():.4f}]")

min_len = min(len(audio_pt_np), len(audio_mlx_np))
corr = np.corrcoef(audio_pt_np[:min_len], audio_mlx_np[:min_len])[0, 1]
print(f"  dec correlation: {corr:.4f}")
