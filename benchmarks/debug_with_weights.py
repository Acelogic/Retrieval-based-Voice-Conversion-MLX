#!/usr/bin/env python3
"""
Debug with actual loaded weights to see if that affects channel count.
"""

import os
import sys
import numpy as np
import mlx.core as mx

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(__file__))

from rvc.lib.mlx.rmvpe import RMVPE0Predictor

print("=" * 80)
print("Debug E2E with Loaded Weights")
print("=" * 80)

weights_file = "rvc/models/predictors/rmvpe_mlx.npz"

print(f"\nLoading RMVPE from {weights_file}...")
predictor = RMVPE0Predictor(weights_path=weights_file)

# Get the E2E model
e2e_model = predictor.model

print("\nE2E Model structure:")
print(f"  UNet encoder layers: {len(e2e_model.unet.encoder.layers)}")
print(f"  UNet intermediate layers: {len(e2e_model.unet.intermediate.layers)}")
print(f"  UNet decoder layers: {len(e2e_model.unet.decoder.layers)}")
print(f"  CNN: {e2e_model.cnn}")

# Create test mel spectrogram matching actual inference
print("\nCreating test mel spectrogram for 5 seconds of audio...")
# 5 seconds at 16kHz = 80,000 samples
# With hop_length=160, n_fft=2048: frames = 1 + (80000 - 2048) // 160 = 488 frames
# But let's use the actual mel_spectrogram function
audio = np.random.randn(16000 * 5).astype(np.float32)
print(f"Audio shape: {audio.shape}")

# Use predictor's mel_spectrogram function
mel = predictor.mel_spectrogram(audio)
print(f"Mel spectrogram shape: {mel.shape}")

# Process through mel2hidden pipeline
print("\nProcessing through mel2hidden...")
try:
    # Reshape mel same way as mel2hidden does
    mel_mx = mx.array(mel)
    mel_mx = mel_mx.transpose(1, 0)[None, :, :, None]  # (1, T, n_mels, 1)
    print(f"Reshaped mel_mx: {mel_mx.shape}")

    # Pass through full E2E model
    print("\nE2E forward (full pipeline)...")
    try:
        output = e2e_model(mel_mx)
        print(f"✅ E2E output shape: {output.shape}")
    except Exception as e2e_err:
        print(f"❌ E2E failed: {e2e_err}")

        # Also try UNet alone for comparison
        print("\n  Testing UNet alone...")
        try:
            unet_out = e2e_model.unet(mel_mx)
            print(f"  UNet output shape: {unet_out.shape}")
            print(f"  UNet output channels: {unet_out.shape[-1]}")
        except Exception as unet_err:
            print(f"  UNet also failed: {unet_err}")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 80)
