#!/usr/bin/env python3
"""
RMVPE debugging tool to compare intermediate layer outputs between PyTorch and MLX.

This helps identify exactly where the numerical divergence occurs.
"""

import os
import sys
import numpy as np
import librosa

sys.path.insert(0, os.getcwd())

def compare_mel_spectrograms(audio_path):
    """Compare mel spectrogram generation between implementations."""
    print("\n=== Comparing Mel Spectrograms ===")

    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    print(f"Audio shape: {audio.shape}, SR: {sr}")

    # MLX mel
    from rvc_mlx.lib.mlx.rmvpe import RMVPE0Predictor
    mlx_predictor = RMVPE0Predictor()
    mel_mlx = mlx_predictor.mel_spectrogram(audio)
    mel_mlx_np = np.array(mel_mlx)

    # PyTorch mel (if available)
    try:
        import torch
        # Import PyTorch RMVPE
        sys.path.insert(0, "rvc")
        from lib.predictors.RMVPE import RMVPE0Predictor as PyTorchRMVPE

        pt_predictor = PyTorchRMVPE("rvc/models/predictors/rmvpe.pt", device="cpu")

        # Get mel from PyTorch
        audio_torch = torch.from_numpy(audio).float()
        mel_pt = pt_predictor.mel_spectrogram(audio_torch.unsqueeze(0))
        mel_pt_np = mel_pt.squeeze(0).numpy()

        print(f"MLX mel shape: {mel_mlx_np.shape}")
        print(f"PyTorch mel shape: {mel_pt_np.shape}")
        print(f"MLX mel range: [{mel_mlx_np.min():.4f}, {mel_mlx_np.max():.4f}], mean: {mel_mlx_np.mean():.4f}")
        print(f"PyTorch mel range: [{mel_pt_np.min():.4f}, {mel_pt_np.max():.4f}], mean: {mel_pt_np.mean():.4f}")

        # Compare
        min_len = min(mel_mlx_np.shape[1], mel_pt_np.shape[1])
        mel_mlx_crop = mel_mlx_np[:, :min_len]
        mel_pt_crop = mel_pt_np[:, :min_len]

        diff = np.abs(mel_mlx_crop - mel_pt_crop)
        print(f"Mel difference: max={diff.max():.6f}, mean={diff.mean():.6f}, std={diff.std():.6f}")

        return mel_mlx_np, mel_pt_np
    except Exception as e:
        print(f"PyTorch comparison skipped: {e}")
        print(f"MLX mel shape: {mel_mlx_np.shape}")
        print(f"MLX mel range: [{mel_mlx_np.min():.4f}, {mel_mlx_np.max():.4f}], mean: {mel_mlx_np.mean():.4f}")
        return mel_mlx_np, None


def debug_mlx_model_weights(weights_path):
    """Examine MLX model weight statistics."""
    print("\n=== MLX Model Weight Statistics ===")

    weights = np.load(weights_path)
    print(f"Weight file contains {len(weights.keys())} entries")

    # Check key layers
    key_layers = [
        'fc.linear.weight',
        'fc.linear.bias',
        'cnn.weight',
        'cnn.bias',
    ]

    for key in key_layers:
        if key in weights:
            w = weights[key]
            print(f"{key}: shape={w.shape}, range=[{w.min():.6f}, {w.max():.6f}], mean={w.mean():.6f}, std={w.std():.6f}")
        else:
            print(f"{key}: NOT FOUND")

    # Check BiGRU weights
    gru_keys = [k for k in weights.keys() if 'bigru' in k or 'gru' in k.lower()]
    print(f"\nFound {len(gru_keys)} GRU-related keys")
    for key in sorted(gru_keys)[:10]:  # Show first 10
        w = weights[key]
        print(f"  {key}: shape={w.shape}, std={w.std():.6f}")


def debug_mlx_forward_pass(audio_path):
    """Run MLX forward pass and inspect intermediate outputs."""
    print("\n=== MLX Forward Pass Debug ===")

    from rvc_mlx.lib.mlx.rmvpe import RMVPE0Predictor
    import mlx.core as mx

    predictor = RMVPE0Predictor()

    # Load audio and get mel
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    mel = predictor.mel_spectrogram(audio)
    print(f"Mel input: shape={mel.shape}, range=[{np.array(mel).min():.4f}, {np.array(mel).max():.4f}]")

    # Prepare input
    n_frames = mel.shape[-1]
    pad_curr = 32 * ((n_frames - 1) // 32 + 1) - n_frames
    mel_padded = mx.pad(mel, ((0, 0), (0, pad_curr)), mode='constant')
    mel_input = mel_padded.transpose(1, 0)[None, :, :, None]

    print(f"Model input: shape={mel_input.shape}")

    # Forward through UNet
    model = predictor.model
    print("\n--- UNet Forward ---")
    x, concat_tensors = model.unet.encoder(mel_input)
    print(f"After encoder: shape={x.shape}, range=[{np.array(x).min():.4f}, {np.array(x).max():.4f}]")

    x = model.unet.intermediate(x)
    print(f"After intermediate: shape={x.shape}, range=[{np.array(x).min():.4f}, {np.array(x).max():.4f}]")

    x = model.unet.decoder(x, concat_tensors)
    print(f"After decoder: shape={x.shape}, range=[{np.array(x).min():.4f}, {np.array(x).max():.4f}]")

    # CNN
    print("\n--- CNN Forward ---")
    x = model.cnn(x)
    print(f"After CNN: shape={x.shape}, range=[{np.array(x).min():.4f}, {np.array(x).max():.4f}]")

    # Reshape for FC
    x = x.transpose(0, 1, 3, 2)
    B, T, C, M = x.shape
    x = x.reshape(B, T, C * M)
    print(f"After reshape: shape={x.shape}, range=[{np.array(x).min():.4f}, {np.array(x).max():.4f}]")

    # BiGRU
    if model.fc.has_gru:
        print("\n--- BiGRU Forward ---")
        x = model.fc.bigru(x)
        print(f"After BiGRU: shape={x.shape}, range=[{np.array(x).min():.4f}, {np.array(x).max():.4f}]")

    # Linear
    print("\n--- Linear Forward ---")
    x = model.fc.linear(x)
    print(f"After Linear (pre-sigmoid): shape={x.shape}, range=[{np.array(x).min():.4f}, {np.array(x).max():.4f}]")
    print(f"  Stats: mean={np.array(x).mean():.4f}, std={np.array(x).std():.4f}")

    # Dropout
    x = model.dropout(x)
    print(f"After Dropout: range=[{np.array(x).min():.4f}, {np.array(x).max():.4f}]")

    # Sigmoid
    x = model.sigmoid(x)
    x_np = np.array(x)
    print(f"After Sigmoid: shape={x.shape}, range=[{x_np.min():.6f}, {x_np.max():.6f}]")
    print(f"  Stats: mean={x_np.mean():.6f}, std={x_np.std():.6f}")

    # Analyze sigmoid output distribution
    print("\nSigmoid output distribution:")
    print(f"  Values > 0.1: {np.sum(x_np > 0.1)} / {x_np.size} ({100 * np.sum(x_np > 0.1) / x_np.size:.2f}%)")
    print(f"  Values > 0.03: {np.sum(x_np > 0.03)} / {x_np.size} ({100 * np.sum(x_np > 0.03) / x_np.size:.2f}%)")
    print(f"  Values > 0.01: {np.sum(x_np > 0.01)} / {x_np.size} ({100 * np.sum(x_np > 0.01) / x_np.size:.2f}%)")

    return x_np


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Debug RMVPE MLX implementation")
    parser.add_argument("--audio", type=str, default="test-audio/coder_audio_stock.wav",
                        help="Audio file to test")
    parser.add_argument("--weights", type=str, default="rvc_mlx/models/predictors/rmvpe_mlx.npz",
                        help="MLX weights file")
    parser.add_argument("--check-weights", action="store_true", help="Check weight statistics")
    parser.add_argument("--check-mel", action="store_true", help="Compare mel spectrograms")
    parser.add_argument("--check-forward", action="store_true", help="Debug forward pass")
    parser.add_argument("--all", action="store_true", help="Run all checks")
    args = parser.parse_args()

    if args.all:
        args.check_weights = True
        args.check_mel = True
        args.check_forward = True

    if not any([args.check_weights, args.check_mel, args.check_forward]):
        # Default: run forward pass debug
        args.check_forward = True

    if args.check_weights:
        debug_mlx_model_weights(args.weights)

    if args.check_mel:
        compare_mel_spectrograms(args.audio)

    if args.check_forward:
        debug_mlx_forward_pass(args.audio)

    print("\n" + "="*60)
    print("Debug complete!")


if __name__ == "__main__":
    main()
