#!/usr/bin/env python3
"""
Test script to verify the Butterworth filter coefficients match scipy.signal.butter
"""

import numpy as np
from scipy import signal
import json

# Design 5th order Butterworth high-pass filter
# Cutoff: 48Hz, Sample rate: 16000Hz
b, a = signal.butter(5, 48, btype='high', fs=16000, output='ba')

# Convert to second-order sections (SOS) for cascade implementation
sos = signal.butter(5, 48, btype='high', fs=16000, output='sos')

print("Butterworth 5th Order High-Pass Filter (48Hz @ 16kHz)")
print("=" * 60)
print("\nBA Coefficients:")
print(f"b (numerator): {b}")
print(f"a (denominator): {a}")

print("\n\nSecond-Order Sections (SOS):")
print("Format: [b0, b1, b2, a0, a1, a2]")
for i, section in enumerate(sos):
    print(f"\nSection {i}:")
    print(f"  b0={section[0]:.10f}, b1={section[1]:.10f}, b2={section[2]:.10f}")
    print(f"  a0={section[3]:.10f}, a1={section[4]:.10f}, a2={section[5]:.10f}")
    # Normalize by a0 (which should be 1.0)
    if section[3] != 1.0:
        print(f"  Normalized (a0={section[3]}):")
        b_norm = section[:3] / section[3]
        a_norm = section[3:] / section[3]
        print(f"  b0={b_norm[0]:.10f}, b1={b_norm[1]:.10f}, b2={b_norm[2]:.10f}")
        print(f"  a1={a_norm[1]:.10f}, a2={a_norm[2]:.10f}")

# Test with sample audio
print("\n\nTest with sample signal:")
print("=" * 60)
t = np.linspace(0, 1, 16000, endpoint=False)
# Mix of 50Hz (should be attenuated) and 200Hz (should pass)
test_signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 200 * t)

# Apply filter (filtfilt for zero-phase)
filtered = signal.filtfilt(b, a, test_signal)

print(f"Input RMS: {np.sqrt(np.mean(test_signal**2)):.6f}")
print(f"Output RMS: {np.sqrt(np.mean(filtered**2)):.6f}")
print(f"Attenuation factor: {np.sqrt(np.mean(filtered**2)) / np.sqrt(np.mean(test_signal**2)):.6f}")

# Verify our Swift coefficients
print("\n\nSwift Implementation Verification:")
print("=" * 60)
print("Copy these coefficients to Swift:")
for i, section in enumerate(sos):
    # In Direct Form II, we use [b0, b1, b2, a1, a2] (a0 is assumed to be 1)
    b0, b1, b2, a0, a1, a2 = section
    if abs(a0 - 1.0) > 1e-10:
        # Normalize
        b0, b1, b2 = b0/a0, b1/a0, b2/a0
        a1, a2 = a1/a0, a2/a0
    
    print(f"\nSection {i}:")
    print(f"  let b0_{i}: Double = {b0}")
    print(f"  let b1_{i}: Double = {b1}")
    print(f"  let b2_{i}: Double = {b2}")
    print(f"  let a1_{i}: Double = {a1}")
    print(f"  let a2_{i}: Double = {a2}")
