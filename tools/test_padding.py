#!/usr/bin/env python3
"""
Test if reflect padding matches PyTorch.
"""

import sys
import os
import torch
import torch.nn.functional as F
import librosa
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.getcwd())

# Force reload
for mod in list(sys.modules.keys()):
    if 'rvc_mlx' in mod:
        del sys.modules[mod]

from rvc_mlx.lib.mlx.rmvpe import RMVPE0Predictor

def main():
    print("=== Testing Reflect Padding ===\n")

    # Load audio
    audio, sr = librosa.load("test-audio/coder_audio_stock.wav", sr=16000, mono=True)
    audio = audio[:sr // 2]  # 0.5 seconds

    # MLX mel spectrogram
    pred = RMVPE0Predictor()
    mel_mlx = pred.mel_spectrogram(audio)
    mel_mlx_np = np.array(mel_mlx)

    print(f"MLX mel shape: {mel_mlx_np.shape}")
    print(f"MLX mel range: [{mel_mlx_np.min():.4f}, {mel_mlx_np.max():.4f}]")

    # Compute padding
    n_frames = mel_mlx.shape[-1]
    pad_curr = 32 * ((n_frames - 1) // 32 + 1) - n_frames
    print(f"\nn_frames: {n_frames}, pad_curr: {pad_curr}")

    # MLX reflect padding (my implementation)
    if pad_curr > 0:
        mel_np = np.array(mel_mlx)
        if pad_curr <= n_frames:
            reflected = mel_np[:, -(pad_curr)::][:, ::-1]
        else:
            reflected_parts = []
            remaining = pad_curr
            while remaining > 0:
                chunk_size = min(remaining, n_frames)
                reflected_parts.append(mel_np[:, -chunk_size:][:, ::-1])
                remaining -= chunk_size
            reflected = np.concatenate(reflected_parts, axis=1)[:, :pad_curr]
        mel_padded_mlx = np.concatenate([mel_np, reflected], axis=1)
    else:
        mel_padded_mlx = mel_mlx_np

    print(f"\nMLX padded shape: {mel_padded_mlx.shape}")
    print(f"MLX padded range: [{mel_padded_mlx.min():.4f}, {mel_padded_mlx.max():.4f}]")

    # PyTorch reflect padding
    mel_pt = torch.from_numpy(mel_mlx_np).float()
    mel_padded_pt = F.pad(mel_pt, (0, pad_curr), mode='reflect')
    mel_padded_pt_np = mel_padded_pt.numpy()

    print(f"\nPyTorch padded shape: {mel_padded_pt_np.shape}")
    print(f"PyTorch padded range: [{mel_padded_pt_np.min():.4f}, {mel_padded_pt_np.max():.4f}]")

    # Compare
    diff = np.abs(mel_padded_mlx - mel_padded_pt_np)
    print(f"\n--- Comparison ---")
    print(f"Max diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")
    print(f"RMSE: {np.sqrt(np.mean(diff ** 2)):.6f}")

    if diff.max() < 0.001:
        print("✅ Reflect padding matches PyTorch!")
    else:
        print("❌ Reflect padding differs from PyTorch")
        # Show where they differ
        print(f"\nOriginal last 5 frames:")
        print(mel_mlx_np[0, -5:])
        print(f"\nMLX padded first 5 padding frames:")
        print(mel_padded_mlx[0, n_frames:n_frames+5])
        print(f"\nPyTorch padded first 5 padding frames:")
        print(mel_padded_pt_np[0, n_frames:n_frames+5])

if __name__ == "__main__":
    main()
