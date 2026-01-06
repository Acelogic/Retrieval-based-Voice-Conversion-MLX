#!/usr/bin/env python3
"""
Debug the very first layers to find where divergence starts.
"""

import sys
import os
import torch
import librosa
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "rvc")

# Force reload
for mod in list(sys.modules.keys()):
    if 'rvc_mlx' in mod or 'rvc.lib' in mod:
        del sys.modules[mod]

from rvc.lib.predictors.RMVPE import RMVPE0Predictor as PyTorchRMVPE
from rvc_mlx.lib.mlx.rmvpe import RMVPE0Predictor as MLXRMVPE

def compare(name, mlx_val, pt_val):
    """Compare MLX and PyTorch values."""
    mlx_np = np.array(mlx_val) if isinstance(mlx_val, mx.array) else mlx_val
    pt_np = pt_val.cpu().numpy() if isinstance(pt_val, torch.Tensor) else pt_val

    # Remove batch dim if present
    if mlx_np.ndim > 3 and mlx_np.shape[0] == 1:
        mlx_np = mlx_np[0]
    if pt_np.ndim > 3 and pt_np.shape[0] == 1:
        pt_np = pt_np[0]

    print(f"\n{name}:")
    print(f"  MLX: shape={mlx_np.shape}, range=[{mlx_np.min():.4f}, {mlx_np.max():.4f}], mean={mlx_np.mean():.4f}")
    print(f"  PT:  shape={pt_np.shape}, range=[{pt_np.min():.4f}, {pt_np.max():.4f}], mean={pt_np.mean():.4f}")

    # For comparison, we need to account for NHWC vs NCHW
    if mlx_np.ndim == 3 and pt_np.ndim == 3:
        # MLX: (H, W, C), PT: (C, H, W)
        mlx_np_chw = mlx_np.transpose(2, 0, 1)
        min_shape = tuple(min(m, p) for m, p in zip(mlx_np_chw.shape, pt_np.shape))
        slices = tuple(slice(0, s) for s in min_shape)
        diff = np.abs(mlx_np_chw[slices] - pt_np[slices])
    else:
        diff = np.abs(mlx_np.ravel()[:1000] - pt_np.ravel()[:1000])  # Compare first 1000 elements

    max_diff = diff.max()
    print(f"  Max diff: {max_diff:.6f}")

    return max_diff < 0.01

def main():
    print("=== Debugging First Layers ===\n")

    # Load audio
    audio, sr = librosa.load("test-audio/coder_audio_stock.wav", sr=16000, mono=True)
    audio = audio[:sr // 2]  # 0.5 seconds

    # Initialize predictors
    mlx_pred = MLXRMVPE()
    pt_pred = PyTorchRMVPE("rvc/models/predictors/rmvpe.pt", device="cpu")

    # Get mel spectrograms
    print("--- Mel Spectrogram ---")
    mel_mlx = mlx_pred.mel_spectrogram(audio)
    audio_pt = torch.from_numpy(audio).float().unsqueeze(0)
    mel_pt = pt_pred.mel_extractor(audio_pt, center=True)

    compare("Mel", mel_mlx, mel_pt.squeeze(0))

    # Prepare inputs for model
    print("\n--- Preparing Model Inputs ---")

    # MLX: mel2hidden does padding and reshaping
    n_frames = mel_mlx.shape[-1]
    pad_curr = 32 * ((n_frames - 1) // 32 + 1) - n_frames

    # MLX padding
    if pad_curr > 0:
        mel_np = np.array(mel_mlx)
        if pad_curr <= n_frames - 1:
            reflected = mel_np[:, -(pad_curr+1):-1][:, ::-1]
        else:
            reflected_parts = []
            remaining = pad_curr
            offset = 1
            while remaining > 0:
                available = n_frames - offset
                if available <= 0:
                    offset = 1
                    available = n_frames - offset
                chunk_size = min(remaining, available)
                reflected_parts.append(mel_np[:, -(offset+chunk_size):-offset][:, ::-1])
                remaining -= chunk_size
                offset += chunk_size
            reflected = np.concatenate(reflected_parts, axis=1)[:, :pad_curr]
        mel_padded_mlx = mx.array(np.concatenate([mel_np, reflected], axis=1))
    else:
        mel_padded_mlx = mel_mlx

    # PyTorch padding
    mel_padded_pt = torch.nn.functional.pad(mel_pt, (0, pad_curr), mode='reflect')

    compare("Mel (padded)", mel_padded_mlx, mel_padded_pt.squeeze(0))

    # Reshape for model input
    # MLX: (128, T) -> (1, T, 128, 1)
    mel_input_mlx = mel_padded_mlx.transpose(1, 0)[None, :, :, None]

    # PyTorch: (1, 128, T) -> (1, 1, T, 128)
    mel_input_pt = mel_padded_pt.unsqueeze(1).transpose(2, 3)

    print(f"\nMLX input: shape={mel_input_mlx.shape}")
    print(f"PT input:  shape={mel_input_pt.shape}")

    # First layer: encoder BatchNorm
    print("\n--- Encoder BatchNorm ---")
    with torch.no_grad():
        bn_out_mlx = mlx_pred.model.unet.encoder.bn(mel_input_mlx)
        bn_out_pt = pt_pred.model.unet.encoder.bn(mel_input_pt)

    compare("BatchNorm output", bn_out_mlx, bn_out_pt.squeeze(0))

    # First conv block
    print("\n--- First ResEncoderBlock ---")
    with torch.no_grad():
        layer0_mlx = mlx_pred.model.unet.encoder.layers[0]
        layer0_pt = pt_pred.model.unet.encoder.layers[0]

        # First ConvBlockRes
        block0_mlx = layer0_mlx.blocks[0]
        block0_pt = layer0_pt.conv[0]

        # Conv1
        conv1_out_mlx = block0_mlx.conv1(bn_out_mlx)
        conv1_out_pt = block0_pt.conv[0](bn_out_pt)
        compare("Conv1 output", conv1_out_mlx, conv1_out_pt.squeeze(0))

        # BN1
        bn1_out_mlx = block0_mlx.bn1(conv1_out_mlx)
        bn1_out_pt = block0_pt.conv[1](conv1_out_pt)
        compare("BN1 output", bn1_out_mlx, bn1_out_pt.squeeze(0))

        # ReLU1
        relu1_out_mlx = block0_mlx.act1(bn1_out_mlx)
        relu1_out_pt = block0_pt.conv[2](bn1_out_pt)
        compare("ReLU1 output", relu1_out_mlx, relu1_out_pt.squeeze(0))

if __name__ == "__main__":
    main()
