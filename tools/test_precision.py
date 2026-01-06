#!/usr/bin/env python3
"""
Test the impact of float16 vs float32 precision on RMVPE inference.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.getcwd())

def test_precision():
    print("=== Testing Float16 vs Float32 Precision ===\n")

    # Load current float16 weights
    weights_path = "rvc_mlx/models/predictors/rmvpe_mlx.npz"
    weights = np.load(weights_path)

    print(f"Current weights dtype: {weights['fc.linear.weight'].dtype}")
    print(f"Number of weight tensors: {len(weights.keys())}\n")

    # Convert to float32
    print("Converting to float32...")
    weights_f32 = {}
    for key in weights.keys():
        val = weights[key]
        if val.dtype == np.float16:
            weights_f32[key] = val.astype(np.float32)
        else:
            weights_f32[key] = val

    # Save float32 version
    f32_path = "rvc_mlx/models/predictors/rmvpe_mlx_f32.npz"
    np.savez(f32_path, **weights_f32)
    print(f"Saved float32 weights to {f32_path}\n")

    # Test inference with both
    from rvc_mlx.lib.mlx.rmvpe import RMVPE0Predictor
    import librosa
    import mlx.core as mx

    audio_path = "test-audio/coder_audio_stock.wav"
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return

    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    print(f"Testing on audio: {audio.shape}\n")

    # Test float16
    print("--- Float16 Inference ---")
    predictor_f16 = RMVPE0Predictor(weights_path)
    f0_f16 = predictor_f16.infer_from_audio(audio, thred=0.03)

    # Get intermediate values
    mel = predictor_f16.mel_spectrogram(audio)
    hidden_f16 = predictor_f16.mel2hidden(mel)
    hidden_f16_np = np.array(hidden_f16)

    print(f"F0 mean: {np.mean(f0_f16[f0_f16 > 0]):.2f} Hz")
    print(f"Voiced frames: {np.sum(f0_f16 > 0)} / {len(f0_f16)}")
    print(f"Hidden (sigmoid) max: {hidden_f16_np.max():.6f}")
    print(f"Hidden (sigmoid) values > 0.03: {np.sum(hidden_f16_np > 0.03)}\n")

    # Test float32
    print("--- Float32 Inference ---")
    predictor_f32 = RMVPE0Predictor(f32_path)
    f0_f32 = predictor_f32.infer_from_audio(audio, thred=0.03)

    mel = predictor_f32.mel_spectrogram(audio)
    hidden_f32 = predictor_f32.mel2hidden(mel)
    hidden_f32_np = np.array(hidden_f32)

    print(f"F0 mean: {np.mean(f0_f32[f0_f32 > 0]):.2f} Hz")
    print(f"Voiced frames: {np.sum(f0_f32 > 0)} / {len(f0_f32)}")
    print(f"Hidden (sigmoid) max: {hidden_f32_np.max():.6f}")
    print(f"Hidden (sigmoid) values > 0.03: {np.sum(hidden_f32_np > 0.03)}\n")

    # Compare
    print("--- Comparison ---")
    f0_diff = np.mean(np.abs(f0_f16 - f0_f32))
    hidden_diff = np.mean(np.abs(hidden_f16_np - hidden_f32_np))

    print(f"F0 mean absolute difference: {f0_diff:.4f} Hz")
    print(f"Hidden mean absolute difference: {hidden_diff:.6f}")

    if f0_diff < 1.0:
        print("\n✅ Precision impact is minimal (< 1 Hz difference)")
    elif f0_diff < 10.0:
        print("\n⚠️  Moderate precision impact (1-10 Hz difference)")
    else:
        print("\n❌ Significant precision impact (> 10 Hz difference)")

if __name__ == "__main__":
    test_precision()
