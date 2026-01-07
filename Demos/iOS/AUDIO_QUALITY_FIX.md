# Audio Quality & Parity Fixes Implemented

## 1. SineGenerator Fix (Critical)
**Issue:** Continuous robotic noise / "Green Block" waveform.
**Root Cause:** Missing random phase offset in `SineGenerator`. Harmonics were synthesized with 0 phase, causing constructive interference artifacts.
**Fix:** Added random phase initialization matching Python `generators.py`.
- **File:** `RVCModel.swift`

## 2. Weight Auto-Transposition (Critical)
**Issue:** Garbage output or Crashes (Channel Mismatch).
**Root Cause:** `safetensors` files from PyTorch have `Conv1d` weights as `[Out, In, K]`. MLX expects `[Out, K, In]`.
**Fix:** Implemented automatic weight transposition in `RVCInference.swift` using heuristic detection:
- **General layers:** Flipped if `dim2 < dim1` (K < In).
- **noise_conv layers:** Flipped ONLY if `dim1 == 1` (PyTorch format).
- **File:** `RVCInference.swift`

## 3. Audio Preprocessing (Parity)
**Issue:** Muffled audio, low-frequency rumble, artifacts.
**Fixes:**
- **High-Pass Filter:** Added 48Hz Butterworth (5th order) filter to remove DC offset.
- **Feature Protection:** Implemented voicing-based blending (`protect` param) to fix unvoiced artifacts.
- **File:** `AudioProcessor.swift`, `RVCInference.swift`

## 4. Output Post-processing (Parity)
**Issue:** Clipping or Volume issues.
**Fixes:**
- **Normalization:** Added Peak Normalization (scale max to 0.99) preventing clipping.
- **Volume Envelope:** Mixing logic verified.
- **File:** `AudioProcessor.swift`

## 5. RMVPE & Weights Architecture (Critical)
**Issue:** Fatal crashes (`precondition failed`, `channel mismatch`) or silent output.
**Root Cause:** RMVPE has a complex U-Net structure with specific channel doubling in the intermediate layer and specialized weight formats.
**Fixes:**
- **ConvTranspose2d Weights:** Corrected transposition to `(1, 2, 3, 0)` to map PyTorch `[In, Out, H, W]` to MLX `[Out, H, W, In]`.
- **BatchNorm 5D Bug:** Removed redundant dimension expansion ensuring input stays 4D `[B, T, mels, 1]`.
- **Architecture Parity:** Corrected `DeepUnet` to double channels in the `Intermediate` layer and propagate them to the `Decoder`.
- **Input Padding:** Implemented reflect padding in `RMVPE.infer` to ensure input length is a multiple of 32.
- **Synthesizer Aligment:** Updated `ConvTranspose1d` weights for upscaling with `(1, 2, 0)` transpose.
- **Files:** `RMVPE.swift`, `RVCInference.swift`, `convert_models_for_ios.py`

## 6. RMVPE BatchNorm Running Statistics Fix (Critical - Jan 7, 2026)
**Issue:** RMVPE producing `NaN` outputs and flat (0 Hz) F0 predictions. Signal explosion in encoder (values reaching 1e18).
**Root Cause:** MLX Swift's `BatchNorm` class does not expose `runningMean` and `runningVar` as loadable parameters via `parameters()`, unlike PyTorch and MLX Python. When weights were loaded via `update(parameters:)`, only trainable parameters (weight, bias) were updated, leaving running stats at default initialization (mean=0, var=1). This caused catastrophic normalization failure.

**Signal Explosion Evidence:**
- **Python (Working):** Input BN output range `[-2.00, 1.95]`, stable encoder outputs
- **Swift (Broken):** Input BN output range `[-7.90, 0.51]`, Layer 1 max: 737, Layer 2 max: 3.2M, Layer 3: 5.4e12 → `NaN`

**Solution Implemented:**
Created `CustomBatchNorm` class in `RMVPE.swift` that:
- Exposes `runningMean` and `runningVar` as explicit `MLXArray` properties
- Properties are loadable via `update(parameters:)`
- Properly implements training and eval modes
- Correctly applies normalization formula: `(x - runningMean) / sqrt(runningVar + eps) * weight + bias`

**Changes Made:**
1. **CustomBatchNorm Class** (`RMVPE.swift:7-71`): Stores running stats as explicit properties with `setTrainingMode()` method
2. **Replaced All BatchNorm Instances:**
   - `ConvBlockRes`: bn1, bn2 → `CustomBatchNorm`
   - `Encoder`: input bn → `CustomBatchNorm`
   - `ResDecoderBlock`: bn1 → `CustomBatchNorm`
3. **Fixed Epsilon Mismatch:** Encoder BN epsilon changed from `1e-3` to `1e-5` to match Python
4. **Weight Loading Update** (`RVCInference.swift`):
   - Running stats remapped from snake_case (`running_mean`) to camelCase (`runningMean`)
   - Added `setTrainingMode(false)` call to set all CustomBatchNorm to eval mode
5. **Training Mode Propagation:** Added `setTrainingMode()` method to RMVPE that recursively sets eval mode on all CustomBatchNorm instances

**Files Modified:**
- `RMVPE.swift`: CustomBatchNorm class, replaced all BatchNorm, fixed epsilon
- `RVCInference.swift`: Updated RMVPE loading to use setTrainingMode()

**Verification:** Running stats (mean=-5.84, var=4.54) are now properly loaded and used during inference.

## Status
✅ **RMVPE & HuBERT Stable**. The core inference pipeline is now numerically stable. RMVPE produces high-accuracy F0 with proper BatchNorm statistics, and HuBERT features are correctly aligned.
