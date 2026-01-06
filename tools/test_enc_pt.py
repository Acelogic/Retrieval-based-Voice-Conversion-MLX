#!/usr/bin/env python3
"""Quick test of PyTorch TextEncoder."""
import numpy as np
import sys
import os
sys.path.insert(0, os.getcwd())

import torch
from rvc.lib.algorithm.synthesizers import Synthesizer as PT_Synthesizer

# Load PyTorch model
pt_path = '/Users/mcruz/Library/Application Support/Replay/com.replay.Replay/models/Slim Shady/model.pth'
pt_ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)

config = pt_ckpt.get('config', pt_ckpt.get('params', {}))
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

net_g = PT_Synthesizer(**kwargs)
net_g.load_state_dict(pt_ckpt['weight'], strict=False)
net_g.eval()

print('PyTorch Synthesizer loaded')
print(f'emb_phone weight shape: {net_g.enc_p.emb_phone.weight.shape}')
print(f'emb_phone weight[:5]: {net_g.enc_p.emb_phone.weight.detach().numpy().flatten()[:5]}')

# Test forward pass - same random seed
np.random.seed(42)
phone = torch.from_numpy(np.random.randn(1, 10, 768).astype(np.float32))
pitch = torch.tensor([[100, 120, 130, 140, 150, 160, 170, 180, 190, 200]], dtype=torch.long)
lengths = torch.tensor([10], dtype=torch.long)

with torch.no_grad():
    m, logs, mask = net_g.enc_p(phone, pitch, lengths)

print(f'\nPyTorch enc_p output:')
print(f'  m shape: {m.shape}')
m_np = m.numpy()
print(f'  m range: [{m_np.min():.4f}, {m_np.max():.4f}]')
print(f'  m mean: {m_np.mean():.4f}')
