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

## Status
✅ **RMVPE & HuBERT Stable**. The core inference pipeline is now numerically stable. RMVPE produces high-accuracy F0, and HuBERT features are correctly aligned. Final focus is on fine-tuning Synthesizer broadcast shapes.
