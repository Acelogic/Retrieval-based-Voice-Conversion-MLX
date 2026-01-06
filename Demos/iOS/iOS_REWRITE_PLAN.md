# iOS RVC Implementation Rewrite Plan

**Date:** 2026-01-06
**Goal:** Create iOS/Swift MLX implementation with exact parity to Python MLX implementation (0.986 correlation)

## Phase 1: Model Conversion & Validation

### 1.1 Convert Models with Latest Fixes
- [ ] Create comprehensive model conversion script
  - Convert PyTorch HuBERT → MLX safetensors
  - Convert PyTorch RMVPE → MLX safetensors
  - Convert PyTorch RVC (Drake/Coder) → MLX safetensors
  - Apply all key remappings from Python implementation
  - Verify weight shapes and dtypes

### 1.2 Test Converted Models in Python First
- [ ] Load converted models in Python MLX
- [ ] Run inference and verify 0.986+ correlation
- [ ] Document any conversion issues
- [ ] Create test audio samples for iOS validation

## Phase 2: Swift Implementation Rewrite

### 2.1 Core Architecture (Match Python Exactly)

**Priority Order:**
1. **HuBERT** (`rvc_mlx/lib/mlx/encoders.py` → Swift)
   - Feature extraction CNN layers
   - Positional encoding
   - Transformer encoder
   - GELU activation (precise formula)

2. **RMVPE** (`rvc_mlx/lib/mlx/rmvpe.py` → Swift)
   - DeepUnet architecture
   - BiGRU implementation
   - Mel spectrogram processing
   - **F0 decoding: `440 * 2^((cents-4080)/1200)`** ✅ Already fixed

3. **Synthesizer** (`rvc_mlx/lib/mlx/generators.py` → Swift)
   - TextEncoder
   - ResidualCouplingBlock (Flow)
   - Generator (NSF-HiFiGAN)
   - Correct transpose/padding operations

### 2.2 Implementation Strategy

For each component:
1. **Read Python implementation line-by-line**
2. **Create exact Swift equivalent**
3. **Export intermediate outputs from Python**
4. **Verify Swift outputs match Python**
5. **Document any MLX Swift API differences**

### 2.3 Critical Details to Match

**From Python Implementation:**
- Input/output shapes at each layer
- Activation functions (GELU, not approximation)
- Padding modes and values
- Transpose operations order
- F0 decoding formula
- Normalization (LayerNorm, BatchNorm, GroupNorm)
- Convolution parameters (stride, padding, groups)

## Phase 3: Testing & Validation

### 3.1 Component-Level Testing
- [ ] HuBERT: Export Python features, compare with Swift
- [ ] RMVPE: Export Python F0, compare with Swift
- [ ] Generator: Export Python audio, compare with Swift
- [ ] Full pipeline: Achieve 0.98+ correlation iOS vs Python

### 3.2 End-to-End Validation
- [ ] Same input audio in Python and iOS
- [ ] Compare spectrograms
- [ ] Compute correlation
- [ ] Verify RMS levels
- [ ] Target: 0.986+ correlation (matching Python benchmark)

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
