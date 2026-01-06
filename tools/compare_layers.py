#!/usr/bin/env python3
"""
Compare intermediate layer outputs between PyTorch and MLX RMVPE.
"""

import sys
import os
import torch
import librosa
import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "rvc")

from rvc.lib.predictors.RMVPE import RMVPE0Predictor as PyTorchRMVPE
from rvc_mlx.lib.mlx.rmvpe import RMVPE0Predictor as MLXRMVPE
import mlx.core as mx

def compare_layers(audio_path):
    """Compare layer-by-layer outputs."""
    print("=== Layer-by-Layer Comparison ===\n")

    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    print(f"Audio: {audio.shape}, SR: {sr}\n")

    # PyTorch
    print("--- PyTorch Forward ---")
    pt_rmvpe = PyTorchRMVPE("rvc/models/predictors/rmvpe.pt", device="cpu")
    audio_torch = torch.from_numpy(audio).float().unsqueeze(0)
    mel_pt = pt_rmvpe.mel_extractor(audio_torch, center=True)

    print(f"Mel: {mel_pt.shape}, range=[{mel_pt.min():.4f}, {mel_pt.max():.4f}]")

    # Forward through model
    with torch.no_grad():
        # Manually forward through layers
        model_pt = pt_rmvpe.model

        # Through UNet
        # PyTorch expects (N, C, H, W) format
        x_pt = mel_pt.unsqueeze(1)  # (N, 1, 128, T)
        x_pt = x_pt.transpose(2, 3)  # (N, 1, T, 128)
        x_pt, concat_tensors = model_pt.unet.encoder(x_pt)
        print(f"After encoder: {x_pt.shape}, range=[{x_pt.min():.4f}, {x_pt.max():.4f}]")

        x_pt = model_pt.unet.intermediate(x_pt)
        print(f"After intermediate: {x_pt.shape}, range=[{x_pt.min():.4f}, {x_pt.max():.4f}]")

        x_pt = model_pt.unet.decoder(x_pt, concat_tensors)
        print(f"After decoder: {x_pt.shape}, range=[{x_pt.min():.4f}, {x_pt.max():.4f}]")

        # CNN
        x_pt = model_pt.cnn(x_pt)
        print(f"After CNN: {x_pt.shape}, range=[{x_pt.min():.4f}, {x_pt.max():.4f}]")

        # Reshape for GRU
        # PyTorch: (N, C, H, W) -> (N, H, C*W)
        x_pt = x_pt.transpose(1, 2).flatten(-2)
        print(f"After reshape (BiGRU input): {x_pt.shape}, range=[{x_pt.min():.4f}, {x_pt.max():.4f}]")
        pt_bigru_input = x_pt.cpu().numpy()

        # GRU
        x_pt = model_pt.fc[0](x_pt)
        print(f"After BiGRU: {x_pt.shape}, range=[{x_pt.min():.4f}, {x_pt.max():.4f}]")
        pt_bigru_output = x_pt.cpu().numpy()

        # Linear
        x_pt = model_pt.fc[1](x_pt)
        print(f"After Linear (pre-sigmoid): {x_pt.shape}, range=[{x_pt.min():.4f}, {x_pt.max():.4f}]")
        print(f"  Stats: mean={x_pt.mean():.4f}, std={x_pt.std():.4f}")
        pt_linear_output = x_pt.cpu().numpy()

        # Dropout + Sigmoid
        x_pt = model_pt.dropout(x_pt)
        x_pt = model_pt.sigmoid(x_pt)
        print(f"After Sigmoid: {x_pt.shape}, range=[{x_pt.min():.6f}, {x_pt.max():.6f}]")
        pt_sigmoid_output = x_pt.cpu().numpy()

    # MLX
    print("\n--- MLX Forward ---")
    mlx_rmvpe = MLXRMVPE()
    mel_mlx = mlx_rmvpe.mel_spectrogram(audio)

    mel_mlx_np = np.array(mel_mlx)
    print(f"Mel: {mel_mlx_np.shape}, range=[{mel_mlx_np.min():.4f}, {mel_mlx_np.max():.4f}]")

    # Prepare input
    n_frames = mel_mlx.shape[-1]
    pad_curr = 32 * ((n_frames - 1) // 32 + 1) - n_frames
    mel_padded = mx.pad(mel_mlx, ((0, 0), (0, pad_curr)), mode='constant')
    mel_input = mel_padded.transpose(1, 0)[None, :, :, None]

    # Forward through model
    model_mlx = mlx_rmvpe.model

    x_mlx, concat_tensors = model_mlx.unet.encoder(mel_input)
    x_mlx_np = np.array(x_mlx)
    print(f"After encoder: {x_mlx.shape}, range=[{x_mlx_np.min():.4f}, {x_mlx_np.max():.4f}]")

    x_mlx = model_mlx.unet.intermediate(x_mlx)
    x_mlx_np = np.array(x_mlx)
    print(f"After intermediate: {x_mlx.shape}, range=[{x_mlx_np.min():.4f}, {x_mlx_np.max():.4f}]")

    x_mlx = model_mlx.unet.decoder(x_mlx, concat_tensors)
    x_mlx_np = np.array(x_mlx)
    print(f"After decoder: {x_mlx.shape}, range=[{x_mlx_np.min():.4f}, {x_mlx_np.max():.4f}]")

    x_mlx = model_mlx.cnn(x_mlx)
    x_mlx_np = np.array(x_mlx)
    print(f"After CNN: {x_mlx.shape}, range=[{x_mlx_np.min():.4f}, {x_mlx_np.max():.4f}]")

    x_mlx = x_mlx.transpose(0, 1, 3, 2)
    B, T, C, M = x_mlx.shape
    x_mlx = x_mlx.reshape(B, T, C * M)
    x_mlx_np = np.array(x_mlx)
    print(f"After reshape (BiGRU input): {x_mlx.shape}, range=[{x_mlx_np.min():.4f}, {x_mlx_np.max():.4f}]")
    mlx_bigru_input = x_mlx_np

    x_mlx = model_mlx.fc.bigru(x_mlx)
    x_mlx_np = np.array(x_mlx)
    print(f"After BiGRU: {x_mlx.shape}, range=[{x_mlx_np.min():.4f}, {x_mlx_np.max():.4f}]")
    mlx_bigru_output = x_mlx_np

    x_mlx = model_mlx.fc.linear(x_mlx)
    x_mlx_np = np.array(x_mlx)
    print(f"After Linear (pre-sigmoid): {x_mlx.shape}, range=[{x_mlx_np.min():.4f}, {x_mlx_np.max():.4f}]")
    print(f"  Stats: mean={x_mlx_np.mean():.4f}, std={x_mlx_np.std():.4f}")
    mlx_linear_output = x_mlx_np

    x_mlx = model_mlx.dropout(x_mlx)
    x_mlx = model_mlx.sigmoid(x_mlx)
    x_mlx_np = np.array(x_mlx)
    print(f"After Sigmoid: {x_mlx.shape}, range=[{x_mlx_np.min():.6f}, {x_mlx_np.max():.6f}]")
    mlx_sigmoid_output = x_mlx_np

    # Compare
    print("\n=== Comparison ===")

    # Match dimensions (take minimum length)
    min_t = min(pt_bigru_input.shape[1], mlx_bigru_input.shape[1])

    def compare_arrays(name, arr1, arr2, crop_time=True):
        if crop_time:
            arr1 = arr1[:, :min_t]
            arr2 = arr2[:, :min_t]
        diff = np.abs(arr1 - arr2)
        rmse = np.sqrt(np.mean((arr1 - arr2) ** 2))
        corr = np.corrcoef(arr1.flatten(), arr2.flatten())[0, 1]
        print(f"\n{name}:")
        print(f"  Max diff: {diff.max():.6f}, Mean diff: {diff.mean():.6f}")
        print(f"  RMSE: {rmse:.6f}, Correlation: {corr:.6f}")
        return diff.max()

    compare_arrays("BiGRU input", pt_bigru_input, mlx_bigru_input)
    compare_arrays("BiGRU output", pt_bigru_output, mlx_bigru_output)
    compare_arrays("Linear output", pt_linear_output, mlx_linear_output)
    compare_arrays("Sigmoid output", pt_sigmoid_output, mlx_sigmoid_output)

    # Analyze BiGRU input statistics
    print("\n=== BiGRU Input Statistics ===")
    print(f"PyTorch: mean={pt_bigru_input.mean():.6f}, std={pt_bigru_input.std():.6f}")
    print(f"MLX:     mean={mlx_bigru_input[:, :min_t].mean():.6f}, std={mlx_bigru_input[:, :min_t].std():.6f}")

    # Analyze BiGRU output statistics
    print("\n=== BiGRU Output Statistics ===")
    print(f"PyTorch: mean={pt_bigru_output.mean():.6f}, std={pt_bigru_output.std():.6f}")
    print(f"MLX:     mean={mlx_bigru_output[:, :min_t].mean():.6f}, std={mlx_bigru_output[:, :min_t].std():.6f}")

if __name__ == "__main__":
    compare_layers("test-audio/coder_audio_stock.wav")
