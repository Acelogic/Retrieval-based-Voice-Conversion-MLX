#!/usr/bin/env python3
"""
Compare BiGRU outputs between PyTorch and MLX using real CNN outputs.
"""

import sys
import os
import torch
import librosa
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.getcwd())

# Force reload
for mod in list(sys.modules.keys()):
    if 'rvc_mlx' in mod:
        del sys.modules[mod]

from rvc_mlx.lib.mlx.rmvpe import RMVPE0Predictor as MLXRMVPE

# Get CNN output from MLX
audio, sr = librosa.load("test-audio/coder_audio_stock.wav", sr=16000, mono=True)
audio_short = audio[:sr // 2]  # 0.5 seconds

mlx_pred = MLXRMVPE()
mel = mlx_pred.mel_spectrogram(audio_short)

n_frames = mel.shape[-1]
pad_curr = 32 * ((n_frames - 1) // 32 + 1) - n_frames
mel_padded = mx.pad(mel, ((0, 0), (0, pad_curr)), mode='constant')
mel_input = mel_padded.transpose(1, 0)[None, :, :, None]

model_mlx = mlx_pred.model
x_mlx, concat = model_mlx.unet.encoder(mel_input)
x_mlx = model_mlx.unet.intermediate(x_mlx)
x_mlx = model_mlx.unet.decoder(x_mlx, concat)
x_mlx = model_mlx.cnn(x_mlx)
x_mlx = x_mlx.transpose(0, 1, 3, 2)
B, T, C, M = x_mlx.shape
cnn_out_mlx = x_mlx.reshape(B, T, C * M)
cnn_out_mlx_np = np.array(cnn_out_mlx)

print("CNN output (MLX):")
print(f"  shape: {cnn_out_mlx_np.shape}")
print(f"  range: [{cnn_out_mlx_np.min():.4f}, {cnn_out_mlx_np.max():.4f}]")
print(f"  mean: {cnn_out_mlx_np.mean():.6f}, std: {cnn_out_mlx_np.std():.6f}")

# Get BiGRU output from MLX
bigru_out_mlx = model_mlx.fc.bigru(cnn_out_mlx)
bigru_out_mlx_np = np.array(bigru_out_mlx)[0]  # Remove batch dim

print("\nBiGRU output (MLX):")
print(f"  shape: {bigru_out_mlx_np.shape}")
print(f"  range: [{bigru_out_mlx_np.min():.4f}, {bigru_out_mlx_np.max():.4f}]")
print(f"  mean: {bigru_out_mlx_np.mean():.6f}, std: {bigru_out_mlx_np.std():.6f}")

# Now run PyTorch BiGRU on the SAME CNN output
print("\n--- PyTorch BiGRU on same CNN input ---")

# Load PyTorch weights
pt_path = "rvc/models/predictors/rmvpe.pt"
pt_ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)

# Create PyTorch BiGRU
pt_bigru = torch.nn.GRU(384, 256, batch_first=True, bidirectional=True)

# Load forward direction weights
pt_bigru.weight_ih_l0 = torch.nn.Parameter(pt_ckpt['fc.0.gru.weight_ih_l0'])
pt_bigru.weight_hh_l0 = torch.nn.Parameter(pt_ckpt['fc.0.gru.weight_hh_l0'])
pt_bigru.bias_ih_l0 = torch.nn.Parameter(pt_ckpt['fc.0.gru.bias_ih_l0'])
pt_bigru.bias_hh_l0 = torch.nn.Parameter(pt_ckpt['fc.0.gru.bias_hh_l0'])

# Load reverse direction weights
pt_bigru.weight_ih_l0_reverse = torch.nn.Parameter(pt_ckpt['fc.0.gru.weight_ih_l0_reverse'])
pt_bigru.weight_hh_l0_reverse = torch.nn.Parameter(pt_ckpt['fc.0.gru.weight_hh_l0_reverse'])
pt_bigru.bias_ih_l0_reverse = torch.nn.Parameter(pt_ckpt['fc.0.gru.bias_ih_l0_reverse'])
pt_bigru.bias_hh_l0_reverse = torch.nn.Parameter(pt_ckpt['fc.0.gru.bias_hh_l0_reverse'])

pt_bigru.eval()

# Run PyTorch BiGRU on MLX's CNN output
with torch.no_grad():
    cnn_out_torch = torch.from_numpy(cnn_out_mlx_np).half()  # Use float16
    bigru_out_pt, _ = pt_bigru(cnn_out_torch)

bigru_out_pt_np = bigru_out_pt.numpy()

print("BiGRU output (PyTorch):")
print(f"  shape: {bigru_out_pt_np.shape}")
print(f"  range: [{bigru_out_pt_np.min():.4f}, {bigru_out_pt_np.max():.4f}]")
print(f"  mean: {bigru_out_pt_np.mean():.6f}, std: {bigru_out_pt_np.std():.6f}")

# Compare
print("\n=== Comparison ===")
diff = np.abs(bigru_out_mlx_np - bigru_out_pt_np)
print(f"Max diff: {diff.max():.6f}")
print(f"Mean diff: {diff.mean():.6f}")
print(f"RMSE: {np.sqrt(np.mean((bigru_out_mlx_np - bigru_out_pt_np) ** 2)):.6f}")

if diff.max() < 0.01:
    print("\n✅ BiGRU outputs match!")
else:
    print("\n❌ BiGRU outputs differ significantly!")
    print("\nFirst timestep comparison:")
    print(f"  PT: {bigru_out_pt_np[0, :5]}")
    print(f"  MLX: {bigru_out_mlx_np[0, :5]}")
