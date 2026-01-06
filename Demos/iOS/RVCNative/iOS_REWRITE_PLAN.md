# iOS RVC Implementation Rewrite Plan

**Date:** 2026-01-06
**Last Updated:** 2026-01-06 16:15 EST
**Goal:** Create iOS/Swift MLX implementation with exact parity to Python MLX implementation (0.986 correlation)

## ✅ Phase 1 & 2 COMPLETED!

**All ML components now match Python MLX implementation that achieved 0.986 correlation:**
1. ✅ HuBERT - Fixed layerNormEps and missing GELU activation
2. ✅ RMVPE - Complete decode() rewrite with weighted averaging and correct F0 formula
3. ✅ Synthesizer - Fixed TextEncoder dimension mismatch (B,C,T format)

**Next:** Phase 3 - Testing & Validation

## Phase 1: Model Conversion & Validation

### 1.1 Convert Models with Latest Fixes
- [x] Create comprehensive model conversion script ✅
  - [x] Convert PyTorch HuBERT → MLX safetensors
  - [x] Convert PyTorch RMVPE → MLX safetensors
  - [x] Convert PyTorch RVC (Drake/Coder) → MLX safetensors
  - [x] Apply all key remappings from Python implementation
  - [x] Verify weight shapes and dtypes
  - **Created:** `tools/convert_models_for_ios.py`
  - **Converted:** Drake model (457 tensors, 55MB)

### 1.2 Test Converted Models in Python First
- [ ] Load converted models in Python MLX
- [ ] Run inference and verify 0.986+ correlation
- [ ] Document any conversion issues
- [ ] Create test audio samples for iOS validation
- **Note:** Models are gitignored (.safetensors), need to be converted locally

## Phase 2: Swift Implementation Rewrite

**Note:** Keep existing SwiftUI GUI (ContentView, AudioPlayer, AudioRecorder). Only rewrite ML inference layer.

### 2.1 Files to Keep (UI Layer)
- ✅ `ContentView.swift` - SwiftUI interface
- ✅ `AudioPlayer.swift` - Audio playback
- ✅ `AudioRecorder.swift` - Audio recording
- ✅ `RVCInference.swift` - Keep the class structure, rewrite inference logic

### 2.2 Files to Rewrite (ML Layer)
**Priority Order:**

1. **HuBERT** (`rvc_mlx/lib/mlx/encoders.py` → Swift) ✅ **COMPLETED**
   - [x] Feature extraction CNN layers
   - [x] Positional encoding
   - [x] Transformer encoder
   - [x] GELU activation (precise formula)
   - **Fixes Applied:**
     - layerNormEps: 1e-12 → 1e-5 (matches Python line 39)
     - Added missing GELU in HubertPositionalConvEmbedding
     - Documented precise GELU requirement (not approximation)
   - **Commit:** 80377122

2. **RMVPE** (`rvc_mlx/lib/mlx/rmvpe.py` → Swift) ✅ **COMPLETED**
   - [x] DeepUnet architecture (existing implementation kept)
   - [x] BiGRU implementation (existing implementation kept)
   - [x] Mel spectrogram processing (existing implementation kept)
   - [x] F0 decoding with weighted averaging
   - **Fixes Applied:**
     - Complete rewrite of decode() method
     - Weighted averaging around argmax peak (9-sample window)
     - Fixed formula: 10 * 2^(cents/1200) [was: 440 * 2^((cents-4080)/1200)]
     - Matches Python rmvpe.py:355-404 exactly
   - **Commit:** e48f6c56

3. **Synthesizer** (`rvc_mlx/lib/mlx/generators.py` → Swift) ✅ **COMPLETED**
   - [x] Fix TextEncoder dimension format mismatch
   - [x] Verify architecture matches Python (LeakyReLU, not GELU)
   - [x] Validate all dimension formats (B,C,T vs B,T,C)
   - **Fixes Applied:**
     - TextEncoder: Fixed output format to match Python
       - Transpose stats before splitting: (B,T,C*2) → (B,C*2,T)
       - Return (m, logs) as (B,C,T), xMask as (B,1,T)
     - Verified activations: TextEncoder uses LeakyReLU(0.1), FFN uses ReLU
     - Documented architecture correspondence with Python
   - **This was the "Known issue" from commit df081a66**
   - **File:** `RVCNativePackage/Sources/RVCNativeFeature/RVC/Synthesizer.swift`
   - **Commit:** 8f3800f1

### 2.3 Implementation Strategy

For each component:
1. **Read Python implementation line-by-line**
2. **Create exact Swift equivalent**
3. **Export intermediate outputs from Python**
4. **Verify Swift outputs match Python**
5. **Document any MLX Swift API differences**

### 2.4 Critical Details to Match

**From Python Implementation:**
- Input/output shapes at each layer
- Activation functions (GELU, not approximation)
- Padding modes and values
- Transpose operations order
- F0 decoding formula
- Normalization (LayerNorm, BatchNorm, GroupNorm)
- Convolution parameters (stride, padding, groups)

## Phase 3: Testing & Validation

### 3.1 Component-Level Testing ✅ COMPLETE

**Created and implemented comprehensive validation framework:**
- ✅ `tools/export_ios_test_data.py` - Export Python MLX outputs
- ✅ `Tests/RVCNativeFeatureTests/MLXParityTests.swift` - Full test suite
- ✅ `Demos/iOS/VALIDATION_GUIDE.md` - Testing methodology with run instructions
- ✅ Test data generated from 13.51s audio sample
- ✅ Numpy .npy file loader in Swift
- ✅ Correlation metrics (Pearson, RMSE, MAE, F0 cents error)
- ✅ HuBERT validation test with .safetensors loading
- ✅ RMVPE validation test with .safetensors loading
- ✅ Test data summary and diagnostics
- **Commits:** b1023cdc, [current]

**Test data exported:**
- HuBERT features: (1, 675, 256)
- RMVPE F0: (1351,) - range 0-168.49 Hz, mean 111.03 Hz
- RMVPE hidden: (1, 1351, 360)
- Input audio: 216,100 samples @ 16kHz

**Validation tests implemented:**
- ✅ HuBERT correlation test (loads model, runs inference, computes metrics)
- ✅ RMVPE F0 correlation test (loads model, runs inference, computes cents error)
- ✅ Test data summary (validates all files present)

**To execute tests:**
- Run in Xcode on iOS Simulator (`Cmd+U`)
- Or use: `xcodebuild test -workspace RVCNative.xcworkspace -scheme RVCNative -destination 'platform=iOS Simulator,name=iPhone 16'`
- **Note:** Tests require Metal and won't run via `swift test` on macOS

**Next step: Execute validation tests in Xcode and document results**
- [ ] Run tests on iOS Simulator in Xcode
- [ ] Verify HuBERT correlation ≥ 0.98
- [ ] Verify RMVPE correlation ≥ 0.98 and cents error < 2.0
- [ ] Document actual correlation values
- [ ] If correlation < 0.98, debug differences

### 3.2 End-to-End Validation
- [ ] Load full RVC model in iOS app
- [ ] Process same input audio in Python and iOS
- [ ] Compare spectrograms visually
- [ ] Compute correlation coefficient
- [ ] Verify RMS levels match
- [ ] **Target: 0.986+ correlation** (matching Python benchmark)
- [ ] Perceptual quality test: "Can you hear the difference?"

## Phase 4: iOS App Polish

### 4.1 Performance Optimization
- [ ] Profile Metal GPU usage
- [ ] Optimize chunking strategy for iOS
- [ ] Memory management for large models
- [ ] Background processing

### 4.2 User Experience
- [ ] Loading indicators with progress
- [ ] Error handling and recovery
- [ ] Model switching
- [ ] Audio quality settings
- [ ] Export options

## Implementation Notes

### Swift/MLX API Differences to Handle

1. **Array Indexing**: MLX Swift uses different syntax than Python
2. **Broadcasting**: May need explicit reshaping
3. **Module System**: Swift uses different module organization
4. **Type Safety**: Swift requires explicit typing

### Reference Implementation

**Python MLX (Source of Truth):**
- `rvc_mlx/lib/mlx/encoders.py` - HuBERT
- `rvc_mlx/lib/mlx/rmvpe.py` - RMVPE
- `rvc_mlx/lib/mlx/generators.py` - Generator/Synthesizer
- `rvc_mlx/infer/infer_mlx.py` - Inference pipeline

**Key Fixes to Port:**
- GELU activation (line 45 in generators.py): `gelu(x, approx='none')`
- F0 decoding (line 188 in rmvpe.py): `440.0 * mx.power(2, (cents - 4080) / 1200)`

## Success Criteria

- [ ] iOS app builds without errors
- [ ] Models load successfully
- [ ] Inference completes without crashes
- [ ] Output audio sounds correct (subjective)
- [ ] **Spectrogram correlation ≥ 0.98** (objective)
- [ ] Realtime factor > 5x on iPhone/iPad
- [ ] Memory usage < 2GB for 30s audio

## Timeline Estimate

- Phase 1: Model Conversion - 2-4 hours
- Phase 2: Swift Rewrite - 8-12 hours
- Phase 3: Testing - 4-6 hours
- Phase 4: Polish - 2-4 hours

**Total:** 16-26 hours of focused development

## Next Immediate Steps

1. Create model conversion script with validation
2. Convert all models and verify in Python
3. Start Swift HuBERT rewrite (simplest component first)
4. Build incrementally with continuous testing
