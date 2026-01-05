#!/usr/bin/env python3
"""
Debug script to trace UNet channel dimensions through the network.
"""

import os
import sys
import numpy as np
import mlx.core as mx

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(__file__))

from rvc.lib.mlx.rmvpe import RMVPE0Predictor, DeepUnet

print("=" * 80)
print("UNet Channel Dimension Debugging")
print("=" * 80)

# Create UNet with same params as RMVPE
# E2E(4, 1, (2, 2)) means:
# n_blocks=4, n_gru=1, kernel_size=(2,2)
# DeepUnet defaults: en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16

print("\nInitializing UNet...")
unet = DeepUnet(
    kernel_size=(2, 2),
    n_blocks=4,
    en_de_layers=5,
    inter_layers=4,
    in_channels=1,
    en_out_channels=16
)

print("\n" + "─" * 80)
print("Expected Architecture")
print("─" * 80)
print("\nEncoder (5 layers, en_out_channels=16):")
print("  Layer 0: 1 → 16")
print("  Layer 1: 16 → 32")
print("  Layer 2: 32 → 64")
print("  Layer 3: 64 → 128")
print("  Layer 4: 128 → 256")
print(f"  Encoder out_channel: {unet.encoder.out_channel}")

print("\nIntermediate (4 layers):")
print(f"  Input channels: {unet.encoder.out_channel // 2}")
print(f"  Output channels: {unet.encoder.out_channel}")

print("\nDecoder (5 layers):")
print(f"  Starting channels: {unet.encoder.out_channel}")
curr = unet.encoder.out_channel
for i in range(5):
    next_channels = curr // 2
    print(f"  Layer {i}: {curr} → {next_channels}")
    curr = next_channels
print(f"  Expected final output: 16 channels")

print("\n" + "─" * 80)
print("Actual Layer Structure")
print("─" * 80)

print(f"\nEncoder layers: {len(unet.encoder.layers)}")
print(f"Intermediate layers: {len(unet.intermediate.layers)}")
print(f"Decoder layers: {len(unet.decoder.layers)}")

print("\n" + "─" * 80)
print("Test Forward Pass")
print("─" * 80)

# Create test input
print("\nCreating test input: (1, 512, 128, 1)")
test_input = mx.random.normal((1, 512, 128, 1))

try:
    # Forward through encoder
    print("\nEncoder forward...")
    x, concat_tensors = unet.encoder(test_input)
    print(f"  Encoder output shape: {x.shape}")
    print(f"  Concat tensors: {len(concat_tensors)} tensors")
    for i, t in enumerate(concat_tensors):
        print(f"    Tensor {i}: {t.shape}")

    # Forward through intermediate
    print("\nIntermediate forward...")
    x = unet.intermediate(x)
    print(f"  Intermediate output shape: {x.shape}")

    # Forward through decoder
    print("\nDecoder forward...")
    x = unet.decoder(x, concat_tensors)
    print(f"  Decoder output shape: {x.shape}")

    print("\n" + "─" * 80)
    print("Result")
    print("─" * 80)

    expected_channels = 16
    actual_channels = x.shape[-1]

    if actual_channels == expected_channels:
        print(f"✅ SUCCESS! Output has {actual_channels} channels (expected {expected_channels})")
    else:
        print(f"❌ FAILED! Output has {actual_channels} channels (expected {expected_channels})")
        print(f"\nChannel mismatch: {actual_channels} != {expected_channels}")
        print("This explains why CNN fails (expects 16 input channels)")

except Exception as e:
    print(f"\n❌ ERROR during forward pass: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
