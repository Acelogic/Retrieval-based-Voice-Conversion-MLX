#!/usr/bin/env python3
"""
Debug the complete first encoder block step-by-step.
"""

import sys
import os
import torch
import librosa
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "rvc")

for mod in list(sys.modules.keys()):
    if 'rvc_mlx' in mod or 'rvc.lib' in mod:
        del sys.modules[mod]

from rvc.lib.predictors.RMVPE import RMVPE0Predictor as PyTorchRMVPE
from rvc_mlx.lib.mlx.rmvpe import RMVPE0Predictor as MLXRMVPE

def compare(name, mlx_val, pt_val):
    mlx_np = np.array(mlx_val)
    pt_np = pt_val.cpu().numpy() if isinstance(pt_val, torch.Tensor) else pt_val

    # Handle batch dim
    if mlx_np.shape[0] == 1 and pt_np.shape[0] == 1:
        mlx_np = mlx_np[0]
        pt_np = pt_np[0]

    # MLX is NHWC, PT is NCHW - for comparison convert MLX to CHW
    if mlx_np.ndim == 3 and pt_np.ndim == 3:
        mlx_np = mlx_np.transpose(2, 0, 1)  # HWC -> CHW

    # Compute diff
    min_shape = tuple(min(m, p) for m, p in zip(mlx_np.shape, pt_np.shape))
    slices = tuple(slice(0, s) for s in min_shape)
    diff = np.abs(mlx_np[slices] - pt_np[slices])

    print(f"\n{name}:")
    print(f"  MLX: {mlx_np.shape}, range=[{mlx_np.min():.4f}, {mlx_np.max():.4f}], mean={mlx_np.mean():.4f}")
    print(f"  PT:  {pt_np.shape}, range=[{pt_np.min():.4f}, {pt_np.max():.4f}], mean={pt_np.mean():.4f}")
    print(f"  Diff: max={diff.max():.6f}, mean={diff.mean():.6f}")

    return diff.max()

def main():
    print("=== Debugging First Encoder Block ===\n")

    # Load audio
    audio, sr = librosa.load("test-audio/coder_audio_stock.wav", sr=16000, mono=True)
    audio = audio[:sr // 2]

    mlx_pred = MLXRMVPE()
    mel_mlx = mlx_pred.mel_spectrogram(audio)

    n_frames = mel_mlx.shape[-1]
    pad_curr = 32 * ((n_frames - 1) // 32 + 1) - n_frames

    # Prepare inputs
    if pad_curr > 0:
        mel_np = np.array(mel_mlx)
        reflected = mel_np[:, -(pad_curr+1):-1][:, ::-1]
        mel_padded = mx.array(np.concatenate([mel_np, reflected], axis=1))
    else:
        mel_padded = mel_mlx

    mel_input_mlx = mel_padded.transpose(1, 0)[None, :, :, None]

    pt_pred = PyTorchRMVPE("rvc/models/predictors/rmvpe.pt", device="cpu")
    audio_pt = torch.from_numpy(audio).float().unsqueeze(0)
    mel_pt = pt_pred.mel_extractor(audio_pt, center=True)
    mel_pt_padded = torch.nn.functional.pad(mel_pt, (0, pad_curr), mode='reflect')
    mel_input_pt = mel_pt_padded.unsqueeze(1).transpose(2, 3)

    # Encoder input BN
    x_mlx = mlx_pred.model.unet.encoder.bn(mel_input_mlx)
    with torch.no_grad():
        x_pt = pt_pred.model.unet.encoder.bn(mel_input_pt)

    compare("After input BatchNorm", x_mlx, x_pt)

    # First encoder block - step by step
    block_mlx = mlx_pred.model.unet.encoder.layers[0].blocks[0]
    block_pt = pt_pred.model.unet.encoder.layers[0].conv[0]

    # Save input for residual
    residual_mlx = x_mlx
    residual_pt = x_pt

    # Conv1
    out_mlx = block_mlx.conv1(x_mlx)
    with torch.no_grad():
        out_pt = block_pt.conv[0](x_pt)
    compare("Conv1", out_mlx, out_pt)

    # BN1
    out_mlx = block_mlx.bn1(out_mlx)
    with torch.no_grad():
        out_pt = block_pt.conv[1](out_pt)
    compare("BN1", out_mlx, out_pt)

    # ReLU1
    out_mlx = block_mlx.act1(out_mlx)
    with torch.no_grad():
        out_pt = block_pt.conv[2](out_pt)
    compare("ReLU1", out_mlx, out_pt)

    # Conv2
    out_mlx = block_mlx.conv2(out_mlx)
    with torch.no_grad():
        out_pt = block_pt.conv[3](out_pt)
    compare("Conv2", out_mlx, out_pt)

    # BN2
    out_mlx = block_mlx.bn2(out_mlx)
    with torch.no_grad():
        out_pt = block_pt.conv[4](out_pt)
    compare("BN2", out_mlx, out_pt)

    # ReLU2
    out_mlx = block_mlx.act2(out_mlx)
    with torch.no_grad():
        out_pt = block_pt.conv[5](out_pt)
    compare("ReLU2", out_mlx, out_pt)

    # Shortcut (if needed)
    if block_mlx.shortcut is not None:
        residual_mlx = block_mlx.shortcut(residual_mlx)
        with torch.no_grad():
            residual_pt = block_pt.shortcut(residual_pt)
        compare("Shortcut", residual_mlx, residual_pt)

    # Residual add
    final_mlx = out_mlx + residual_mlx
    with torch.no_grad():
        final_pt = out_pt + residual_pt
    compare("After residual add", final_mlx, final_pt)

    # Pool
    pool_mlx = mlx_pred.model.unet.encoder.layers[0].pool(final_mlx)
    with torch.no_grad():
        pool_pt = pt_pred.model.unet.encoder.layers[0].pool(final_pt)
    compare("After pool", pool_mlx, pool_pt)

if __name__ == "__main__":
    main()
