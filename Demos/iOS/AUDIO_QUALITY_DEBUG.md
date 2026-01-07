# iOS RVC Audio Quality Debugging

## Status: STABLE (Functional but Noisy)

We have successfully achieved voice conversion on iOS! The audio contains the correct pitch and phonemes, confirming the model architecture is functional.

**Known Issue:** Background "robotic hum" or noise in silence regions.

---

## Critical Fixes Implemented

### 1. The "Feature Scrambling" Bug (SOLVED)
**Symptoms:** 100% voiced frames (noise) or 0% voiced (silence).
**Root Cause:**
- Python RMVPE outputs CNN features as `[N, T, 128, 3]`.
- Python does `transpose(0, 1, 3, 2)` -> `[N, T, 3, 128]` before flattening.
- Swift was missing this transpose, flattening interleaved features `[128, 3]`.
**Fix:** Added `x = x.transposed(axes: [0, 1, 3, 2])` in `RMVPE.swift`.

### 2. PyTorch GRU Parity (SOLVED)
**Symptoms:** Incorrect gate activation.
**Root Cause:** MLXNN `GRU` uses different bias logic than PyTorch.
**Fix:** Implemented custom `PyTorchGRU` class in `RMVPE.swift` that matches PyTorch's `bias_ih` and `bias_hh` application exactly.

### 3. Weight Format (SOLVED)
**Fix:** Updated `tools/convert_rmvpe_weights.py` to:
- Convert snake_case `forward_grus` to camelCase `forwardGRUs`.
- Keep PyTorch specific weight names (`weight_ih`, etc.).

---

## Tuning Attempts (Reverted)

### Median Filter
**Attempt:** Applied size-3 median filter to F0 curve.
**Result:** Regression. Smearing of artifacts.
**Status:** Reverted.

### Threshold Adjustment
**Attempt:** Increased `thred` from 0.03 to 0.10.
**Result:** Amplitude collapse. Loss of voice energy.
**Status:** Reverted to 0.03.

---

## Current Configuration

- **Model:** RVC v2 (RMVPE + HuBERT + Synthesizer)
- **Precision:** Float32 (iOS default)
- **RMVPE Threshold:** 0.03 (Standard)
- **F0 Filter:** None (Raw output)
- **Output:** ~16kHz (Upsampled by AudioProcessor)

## Next Steps for Quality

1. **Noise Gating:** Implement a post-processing noise gate to silence audio when input volume is low (kills the "hum" in silence).
2. **Volume Envelope:** Implement RMS mixing to match output volume envelope to input (standard RVC feature).
