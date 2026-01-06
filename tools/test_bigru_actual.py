#!/usr/bin/env python3
"""
Test BiGRU with actual CNN output to see if it matches PyTorch.
"""

import sys
import os
import torch
import librosa
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "rvc")

from rvc.lib.predictors.RMVPE import RMVPE0Predictor as PyTorchRMVPE
from rvc_mlx.lib.mlx.rmvpe import RMVPE0Predictor as MLXRMVPE

def main():
    print("=== Testing BiGRU with Actual CNN Output ===\n")

    # Load short audio for faster testing
    audio, sr = librosa.load("test-audio/coder_audio_stock.wav", sr=16000, mono=True)
    audio = audio[:sr]  # 1 second
    print(f"Audio: {audio.shape}\n")

    # === Get CNN output from MLX ===
    print("--- MLX CNN Output ---")
    mlx_predictor = MLXRMVPE()
    mel_mlx = mlx_predictor.mel_spectrogram(audio)

    n_frames = mel_mlx.shape[-1]
    pad_curr = 32 * ((n_frames - 1) // 32 + 1) - n_frames
    mel_padded = mx.pad(mel_mlx, ((0, 0), (0, pad_curr)), mode='constant')
    mel_input = mel_padded.transpose(1, 0)[None, :, :, None]

    model_mlx = mlx_predictor.model
    x_mlx, concat = model_mlx.unet.encoder(mel_input)
    x_mlx = model_mlx.unet.intermediate(x_mlx)
    x_mlx = model_mlx.unet.decoder(x_mlx, concat)
    x_mlx = model_mlx.cnn(x_mlx)
    x_mlx = x_mlx.transpose(0, 1, 3, 2)
    B, T, C, M = x_mlx.shape
    cnn_output_mlx = x_mlx.reshape(B, T, C * M)

    cnn_output_mlx_np = np.array(cnn_output_mlx)[0]  # Remove batch dim
    print(f"Shape: {cnn_output_mlx_np.shape}")
    print(f"Range: [{cnn_output_mlx_np.min():.4f}, {cnn_output_mlx_np.max():.4f}]")
    print(f"Mean: {cnn_output_mlx_np.mean():.4f}, Std: {cnn_output_mlx_np.std():.4f}")

    # === Test with MLX BiGRU ===
    print("\n--- MLX BiGRU Output ---")
    bigru_output_mlx = model_mlx.fc.bigru(cnn_output_mlx)
    bigru_output_mlx_np = np.array(bigru_output_mlx)[0]
    print(f"Shape: {bigru_output_mlx_np.shape}")
    print(f"Range: [{bigru_output_mlx_np.min():.4f}, {bigru_output_mlx_np.max():.4f}]")
    print(f"Mean: {bigru_output_mlx_np.mean():.4f}, Std: {bigru_output_mlx_np.std():.4f}")

    # === Get PyTorch outputs for comparison ===
    print("\n--- PyTorch Outputs ---")
    try:
        pt_predictor = PyTorchRMVPE("rvc/models/predictors/rmvpe.pt", device="cpu")
        audio_torch = torch.from_numpy(audio).float().unsqueeze(0)
        mel_pt = pt_predictor.mel_extractor(audio_torch, center=True)

        with torch.no_grad():
            model_pt = pt_predictor.model
            # Get CNN output
            x_pt = mel_pt.unsqueeze(1).transpose(2, 3)
            x_pt, concat_pt = model_pt.unet.encoder(x_pt)
            x_pt = model_pt.unet.intermediate(x_pt)
            x_pt = model_pt.unet.decoder(x_pt, concat_pt)
            x_pt = model_pt.cnn(x_pt)
            cnn_output_pt = x_pt.transpose(1, 2).flatten(-2)

            cnn_output_pt_np = cnn_output_pt[0].cpu().numpy()
            print(f"CNN output shape: {cnn_output_pt_np.shape}")
            print(f"CNN output range: [{cnn_output_pt_np.min():.4f}, {cnn_output_pt_np.max():.4f}]")
            print(f"CNN output mean: {cnn_output_pt_np.mean():.4f}, std: {cnn_output_pt_np.std():.4f}")

            # BiGRU
            bigru_output_pt = model_pt.fc[0](cnn_output_pt)
            bigru_output_pt_np = bigru_output_pt[0].cpu().numpy()
            print(f"\nBiGRU output shape: {bigru_output_pt_np.shape}")
            print(f"BiGRU output range: [{bigru_output_pt_np.min():.4f}, {bigru_output_pt_np.max():.4f}]")
            print(f"BiGRU output mean: {bigru_output_pt_np.mean():.4f}, std: {bigru_output_pt_np.std():.4f}")

            # Linear
            linear_output_pt = model_pt.fc[1](bigru_output_pt)
            linear_output_pt_np = linear_output_pt[0].cpu().numpy()
            print(f"\nLinear output range: [{linear_output_pt_np.min():.4f}, {linear_output_pt_np.max():.4f}]")
            print(f"Linear output mean: {linear_output_pt_np.mean():.4f}, std: {linear_output_pt_np.std():.4f}")

            # Compare BiGRU outputs
            print("\n--- BiGRU Comparison ---")
            min_len = min(bigru_output_mlx_np.shape[0], bigru_output_pt_np.shape[0])
            mlx_crop = bigru_output_mlx_np[:min_len]
            pt_crop = bigru_output_pt_np[:min_len]

            diff = np.abs(mlx_crop - pt_crop)
            print(f"Max diff: {diff.max():.6f}")
            print(f"Mean diff: {diff.mean():.6f}")
            print(f"RMSE: {np.sqrt(np.mean((mlx_crop - pt_crop) ** 2)):.6f}")

            if diff.max() < 0.01:
                print("✅ BiGRU outputs match!")
            else:
                print("❌ BiGRU outputs differ significantly")

    except Exception as e:
        print(f"PyTorch comparison failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
