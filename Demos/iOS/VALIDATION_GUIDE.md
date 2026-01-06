# iOS MLX Validation Guide

**Status:** Phase 3 - Component Testing Framework Ready
**Date:** 2026-01-06

## Overview

This guide describes how to validate that the iOS Swift/MLX implementation achieves numerical parity with the Python MLX implementation (0.986 correlation target).

---

## Phase 3.1: Component-Level Testing ✅ Framework Ready

### Test Data Generation

**Script:** `tools/export_ios_test_data.py`

This script exports intermediate outputs from the Python MLX implementation for comparison:

```bash
# Export test data from Python
python3 tools/export_ios_test_data.py \
    --audio test-audio/coder_audio_stock.wav \
    --output-dir ios_test_data
```

**Outputs generated:**
- `input_audio.npy` - Input audio waveform (216,100 samples, 13.51s)
- `hubert_features.npy` - HuBERT features (1, 675, 256)
- `rmvpe_f0.npy` - RMVPE F0 contour (1351,)
- `rmvpe_hidden.npy` - RMVPE hidden states (1, 1351, 360)
- `metadata.json` - Test configuration

**Test audio used:** `test-audio/coder_audio_stock.wav`
- Duration: 13.51 seconds
- Sample rate: 16kHz
- Format: Mono WAV

### Swift Validation Script

**Script:** `tools/validate_ios_parity.swift`

Swift script to load numpy test data and compare with iOS outputs.

**Status:** Framework created, needs model loading implementation

**TODO:**
1. Implement .safetensors loading in Swift
2. Run HuBERT inference on iOS
3. Run RMVPE inference on iOS
4. Compute correlation metrics:
   - Pearson correlation
   - RMSE (Root Mean Square Error)
   - MAE (Mean Absolute Error)
5. Target: >0.98 correlation for each component

---

## Expected Results

Based on the Python MLX implementation (commit df081a66):

### HuBERT Correlation
- **Target:** >0.98 correlation with PyTorch
- **Python achieved:** Improved from 0.815 to 0.926 after GELU fix
- **iOS fixes applied:**
  - layerNormEps: 1e-5 (matches Python)
  - Added missing GELU in positional conv

### RMVPE F0 Accuracy
- **Target:** <2.0 cents error
- **Python achieved:** 1.5 cents error after weighted averaging fix
- **iOS fixes applied:**
  - Complete decode() rewrite with weighted averaging
  - Correct F0 formula: `10 * 2^(cents/1200)`

### Synthesizer TextEncoder
- **Target:** Exact dimension match
- **Python issue:** "dimension format mismatch (B,C,T vs B,T,C)" - RESOLVED
- **iOS fixes applied:**
  - Transpose stats before splitting
  - Return (m, logs) as (B, C, T)
  - Return xMask as (B, 1, T)

---

## Validation Metrics

### 1. Component Correlation

For each component (HuBERT, RMVPE, TextEncoder):

```
Correlation = Pearson(Python_output, iOS_output)
```

**Pass criteria:** Correlation ≥ 0.98

### 2. Error Metrics

```
RMSE = sqrt(mean((Python - iOS)^2))
MAE = mean(|Python - iOS|)
```

**Pass criteria:**
- RMSE < 0.01 (1% of typical value range)
- MAE < 0.005

### 3. F0 Specific Metrics

For RMVPE F0 estimation:

```
Cents Error = 1200 * log2(f0_iOS / f0_Python)
```

**Pass criteria:** Mean cents error < 2.0 cents

---

## Phase 3.2: End-to-End Validation

### Full Pipeline Test

**TODO:** Once components pass validation, run full pipeline:

1. Load same audio file in Python and iOS
2. Run full RVC inference (HuBERT → TextEncoder → Generator)
3. Compute output audio correlation
4. Compare spectrograms

**Target:** >0.986 correlation (matching Python benchmark)

### Test Cases

1. **Short utterance** (3-5 seconds)
   - Quick validation
   - Easy to debug

2. **Medium utterance** (10-15 seconds)
   - Full feature test
   - Current test audio: 13.51s

3. **Long utterance** (30+ seconds)
   - Stress test
   - Memory management validation

---

## Current Status

### ✅ Completed

1. **Phase 1:** Model conversion pipeline
   - `tools/convert_models_for_ios.py` ✅
   - Drake model converted (457 tensors) ✅

2. **Phase 2:** All ML component fixes
   - HuBERT: layerNormEps, GELU activation ✅
   - RMVPE: decode() weighted averaging, F0 formula ✅
   - Synthesizer: TextEncoder dimension fix ✅

3. **Phase 3.1:** Testing framework
   - Python export script ✅
   - Test data generated ✅
   - Swift validation script (framework) ✅

### ⏳ In Progress

4. **Phase 3.1:** Component validation
   - Load .safetensors in Swift validation script
   - Run HuBERT and compare outputs
   - Run RMVPE and compare outputs
   - Document correlation metrics

5. **Phase 3.2:** End-to-end validation
   - Full pipeline test
   - Spectrogram analysis
   - Audio quality assessment

---

## Implementation Notes

### Loading .safetensors in Swift

The Swift/MLX library supports loading safetensors:

```swift
import MLX

let weights = MLX.loadArrays(url: modelURL)
// Returns [String: MLXArray] dictionary
```

### Model Paths

iOS models should be at:
```
Demos/iOS/RVCNative/RVCNativePackage/Sources/RVCNativeFeature/Assets/
├── hubert_base.safetensors  (361MB)
├── rmvpe.safetensors        (173MB)
└── drake.safetensors        (55MB)
```

### Memory Considerations

- HuBERT: ~361MB
- RMVPE: ~173MB
- Drake: ~55MB
- **Total:** ~589MB model weights
- **Runtime:** Add activation buffers (estimate 200-400MB)
- **Target:** <2GB total for 30s audio

---

## Success Criteria

### Phase 3 Complete When:

- [ ] HuBERT correlation ≥ 0.98
- [ ] RMVPE F0 cents error < 2.0
- [ ] RMVPE correlation ≥ 0.98
- [ ] TextEncoder correlation ≥ 0.98
- [ ] Full pipeline correlation ≥ 0.98
- [ ] Output audio sounds identical to Python

### Final Validation:

**"Can you hear the difference?"**
- If no audible difference → SUCCESS ✅
- Numerical metrics are guides, but perceptual quality is the ultimate test

---

## Next Steps

1. **Implement safetensors loading in Swift validation script**
2. **Run component validation tests**
3. **Document correlation results**
4. **If correlation < 0.98, debug differences:**
   - Check weight loading
   - Verify activation functions
   - Compare intermediate layer outputs
5. **Run end-to-end test**
6. **Optimize for iOS** (Phase 4)

---

## References

- Python MLX implementation: `rvc_mlx/lib/mlx/`
- iOS Swift implementation: `Demos/iOS/RVCNative/RVCNativePackage/Sources/RVCNativeFeature/RVC/`
- Benchmark results: `benchmarks/BENCHMARK_RESULTS.md`
- Python correlation: 0.986 (commit df081a66)
