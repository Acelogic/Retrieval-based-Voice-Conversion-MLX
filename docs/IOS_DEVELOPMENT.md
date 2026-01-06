# iOS Development - MLX Swift Port

**Status:** In Progress
**Last Updated:** 2026-01-06
**Objective:** Port MLX Python implementation to MLX Swift for native iOS inference

## Overview

After achieving Python MLX inference parity, the next phase is porting the implementation to MLX Swift for native iOS support. This document tracks iOS-specific development, issues, and solutions.

## Current Status

### Completed
- ✅ Python MLX inference parity achieved (correlation 0.999847)
- ✅ RMVPE optimization complete (0.8% voiced detection error)
- ✅ All Python MLX components working correctly
- ✅ Basic iOS app structure created
- ✅ Model conversion pipeline (PyTorch → MLX → Safetensors)

### In Progress
- ⏳ Swift implementation of RVC components
- ⏳ iOS audio quality verification
- ⏳ Performance optimization for mobile

## iOS Simulator Support & Limitations

### Critical "Revelations" regarding Simulator use:

1. **Stock Library Instability**: Using the *stock*, unmodified `mlx-swift` library on the simulator is highly unstable. The C++ backend frequently crashes with `nullptr` assertions in `std::string` constructors (e.g., in `device.cpp` or `metal.cpp`) because it tries to access Metal device properties that are invalid or null in the simulator environment.

2. **Force CPU**: While `MLX.Device.setDefault(device: Device.cpu)` *can* bypass some Metal crashes, the stock library's internal initialization sequences still often trigger the `nullptr` issues described above.

3. **Conclusion**: **Physical Device testing is mandatory.** The Simulator should only be used for basic UI layout checks or initial compilation verification. Any functional testing of the MLX inference pipeline must be done on a real device to avoid "fighting" the simulator's lack of proper Metal support and the stock library's fragility in that environment.

## Past Issues & Resolutions

### iOS Audio Inference - Scrambled Output (2026-01-05)

**Status**: ✅ **RESOLVED** (2026-01-05) - Fixed selective weight transposition

**Root Cause (CONFIRMED)**: **Incorrect Weight Transposition**
- The safetensors model file contains a **MIX** of PyTorch and MLX format weights
- Previous code was transposing ALL Conv1d weights unconditionally
- This incorrectly transposed weights that were already in MLX format
- Error: `[192, 1, 192]` (correct MLX) → `[192, 192, 1]` (broken)

**The Fix**:
- Implemented **selective transposition** based on weight key patterns
- ONLY transpose: `flow.*`, `dec.cond`, `dec.ups.*`, `dec.noise_convs`, `enc_p.proj`
- KEEP as-is: `enc_p.encoder.attn_*`, `enc_p.encoder.ffn_*`, `dec.conv_pre`, `dec.resblocks.*`

**Code Changes**:
- `RVCInference.swift:52-79` - Selective weight transposition logic

**Weight Format Issues Discovered**:

| Weight Type | Format in Safetensors | Notes |
|-------------|----------------------|-------|
| `flow.*` | PyTorch (out, in, kernel) | Needs transpose to MLX (out, kernel, in) |
| `dec.cond`, `dec.ups`, `dec.noise_convs` | PyTorch | Needs transpose |
| `enc_p.proj` | PyTorch | Needs transpose |
| `enc_p.encoder.attn_*` | MLX | Already correct |
| `enc_p.encoder.ffn_*` | MLX | Already correct |
| `dec.conv_pre`, `dec.resblocks` | MLX | Already correct |

**Auto-Detection Rule**: If `shape[2] < shape[1]` for 3D weight, transpose from PyTorch→MLX.

**Flow Layer Indexing Issue**:
- Model weights use indices `0, 2, 4, 6` (Flip modules at odd indices)
- Swift sequential creation uses `0, 1, 2, 3`
- Requires key remapping during loading

### HuBERT Dimension Mismatch (2026-01-05)

**Issue**: Channel mismatch between HuBERT output and TextEncoder input

**Debugging Process Used**:
1. ✅ Code analysis: Compared Python MLX vs Swift HuBERT implementations
2. ✅ Found Python applies `final_proj` (768→256) while Swift skipped it
3. ✅ Verified Python comment states features are 256-dim
4. ✅ Applied fix to enable `final_proj` and update `embeddingDim`

**Key Files Modified for Fix**:
- `HubertModel.swift:312` - Uncommented `final_proj` application
- `RVCInference.swift:80` - Changed `embeddingDim` from 768 → 256
- `Synthesizer.swift:389` - Updated documentation comment

### Silent Audio Output (2026-01-05)

**Status**: ✅ **RESOLVED**

**Issue**: Audio output was silent despite successful inference

**Root Cause**: Audio was being saved as Float32 PCM instead of Int16 PCM

**Fix**: Updated `AudioProcessor.swift` to save as **Int16 PCM**

### Model Conversion Completed (2026-01-05 22:21)

**Status**: ✅ **COMPLETE**

Converted actual user models from PyTorch to MLX-compatible safetensors:
- ✅ Coder999V2 (`CoderV2_250e_1000s.pth` → `coder.safetensors`)
- ✅ Slim Shady (`model.pth` → `slim_shady.safetensors`)
- ✅ Replaced bundled models in iOS app Assets folder
- Models location: `Demos/iOS/RVCNative/RVCNativePackage/Sources/RVCNativeFeature/Assets/`

### CRITICAL BUG FIX - Double Transposition (2026-01-05 22:25)

- ❌ **Root Cause**: The Python `convert.py` script already transposes ALL Conv1d weights to MLX format
- ❌ **Bug**: Swift code was doing selective transposition AGAIN, causing double-transposition
- ❌ **Result**: Weights for `flow.*`, `dec.cond`, `dec.ups.*`, etc. were transposed twice (MLX→PyTorch→MLX)
- ✅ **Fix**: Removed ALL transposition logic from `RVCInference.swift:52-79`
- ✅ **Reason**: Converted safetensors files already have weights in correct MLX format
- File: `RVCInference.swift:52-54` - Removed selective transposition, use weights directly

### CRITICAL BUG FIX #2 - Incomplete Conv1d Transposition (2026-01-05 22:30)

- ❌ **Root Cause**: Conversion script only transposed weights with "conv" in key name
- ❌ **Bug**: Some Conv1d layers (e.g., "proj") weren't transposed, causing shape mismatch errors
- ❌ **Error**: `input: (1,498,192) and weight: (384,192,1)` - weight still in PyTorch format
- ✅ **Fix**: Updated `convert.py:66-79` to transpose ALL 3D weights (except embeddings)
- ✅ **Logic**: Check `v.ndim == 3` instead of `"conv" in k`
- ✅ **Re-converted**: Both models re-converted and copied to iOS Assets (22:30)
- File: `rvc/lib/mlx/convert.py:66-79` - Transpose all 3D weights regardless of name

### Testing Status (2026-01-05 22:35)

- ✅ Rebuilt iOS app with fixed models
- ✅ Coder999V2 model: Inference completes successfully (no crashes!)
- ✅ Pipeline working: HuBERT → TextEncoder → Generator → Audio output
- ❌ Audio quality still poor ("messed up") - need to investigate

## Current Investigation: Audio Quality

### Possible Remaining Issues

1. **Python vs Swift Inference Comparison Needed**
   - Test SAME input audio with Python MLX reference implementation
   - Compare output waveforms to identify where divergence occurs
   - Use tensor dumps at each stage to find mismatch point

2. **Potential Issues to Investigate**:
   - **RMVPE Pitch Detection**: May be producing incorrect F0 values on device
   - **Chunking Artifacts**: Padding/cropping logic might introduce glitches
   - **Model Compatibility**: Converted models may have subtle weight issues
   - **NSF F0 Processing**: F0 contour calculation might differ from Python
   - **Input Audio Format**: Source audio quality/sample rate issues
   - **Generator Upsampling**: The 400x upsample (features→audio) might have bugs

3. **Next Debugging Steps**:
   - Run Python MLX inference on SAME input audio for comparison
   - Add tensor dumps to compare intermediate values (HuBERT features, F0, m_p, logs_p)
   - Test with known-good input audio (simple, clean speech)
   - Check RMVPE output on device vs Python
   - Verify Generator output matches Python reference

### Testing Needed

- ⏳ Compare iOS output with Python MLX output (same input audio)
- ⏳ Test with Slim Shady model
- ⏳ Verify RMVPE pitch detection accuracy on device
- ⏳ Test with multiple audio samples
- ⏳ Consider testing with fallback F0 (constant 200Hz) to isolate RMVPE
- ⏳ Verify all layer outputs match Python reference implementation

## Model Conversion Pipeline

### Python → MLX → Safetensors

The conversion process:

1. **PyTorch (.pth) → MLX (.npz)**
   ```bash
   python tools/convert_rvc_model.py input.pth output.npz
   ```
   - Transposes all Conv1d/Conv2d weights to MLX format
   - Fuses weight norm (weight_g + weight_v)
   - Handles LayerNorm gamma/beta parameters
   - Remaps key names for MLX module structure

2. **MLX (.npz) → Safetensors (.safetensors)**
   ```bash
   python tools/convert_mlx_to_safetensors.py output.npz output.safetensors
   ```
   - Converts to Swift-compatible format
   - All weights should already be in MLX format
   - No additional transposition needed in Swift

### Important Notes

- ✅ Conversion script handles ALL transpositions
- ✅ Swift code should use weights directly (no transposition)
- ✅ Use `convert.py` that transposes ALL 3D tensors, not just "conv" layers
- ❌ Never double-transpose weights

## Swift Implementation Status

### Components to Port

Based on Python MLX implementation:

1. **Core Modules** (`rvc_mlx/lib/mlx/`)
   - ✅ `modules.py` - Basic building blocks
   - ✅ `attentions.py` - Multi-head attention with relative position embeddings
   - ✅ `residuals.py` - ResidualCouplingBlock (flow)
   - ✅ `generators.py` - HiFiGANNSFGenerator
   - ✅ `encoders.py` - TextEncoder, PosteriorEncoder
   - ✅ `synthesizers.py` - Main Synthesizer

2. **Feature Extractors**
   - ✅ `hubert.py` - HuBERT content encoder
   - ✅ `rmvpe.py` - RMVPE pitch detection

3. **Pipeline**
   - ⏳ `pipeline_mlx.py` - Full inference pipeline
   - ⏳ Audio I/O handling
   - ⏳ Preprocessing/postprocessing

### Key Swift Considerations

1. **Weight Loading**: Use safetensors format, no transposition in Swift
2. **Dimension Ordering**: Ensure (B, T, C) format matches Python MLX
3. **Relative Position Embeddings**: Must load as direct attributes (no `.weight` suffix)
4. **LayerNorm**: Handle gamma/beta parameter names correctly
5. **GRU Implementation**: Use PyTorch-compatible formula if MLX Swift GRU differs

## Performance Targets

Based on Python MLX benchmarks:

- **RMVPE**: ~0.2s for 5s audio (target for iOS)
- **Full Pipeline**: ~2-3s for 5s audio (target for iOS)
- **Memory**: Optimize for mobile constraints

## References

- [INFERENCE_PARITY_ACHIEVED.md](INFERENCE_PARITY_ACHIEVED.md) - Python MLX parity metrics
- [RMVPE_OPTIMIZATION.md](RMVPE_OPTIMIZATION.md) - RMVPE debugging details
- [PYTORCH_MLX_DIFFERENCES.md](PYTORCH_MLX_DIFFERENCES.md) - PyTorch/MLX conversion guide

## Next Steps

1. ⏳ Compare iOS output with Python MLX reference (same input)
2. ⏳ Debug remaining audio quality issues
3. ⏳ Port latest Python fixes to Swift (relative embeddings, LayerNorm, etc.)
4. ⏳ Optimize performance for mobile devices
5. ⏳ Add comprehensive test suite
6. ⏳ Document Swift-specific implementation details

---

*Last Updated: 2026-01-06*
*Python MLX Status: ✅ Production Ready (0.999847 correlation)*
*iOS Swift Status: ⏳ In Progress (audio quality debugging)*
