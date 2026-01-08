# iOS Development - MLX Swift Port

**Status:** ✅ Complete (91.8% Spectrogram Correlation with Python MLX)
**Last Updated:** 2026-01-08

## Overview

The iOS implementation of RVC using Swift MLX has achieved production-ready status with excellent audio quality parity compared to the Python MLX reference implementation.

## Achievement Summary

| Model | Spectrogram Correlation | Status |
|-------|------------------------|--------|
| Drake | 92.9% | ✅ |
| Juice WRLD | 86.6% | ✅ |
| Eminem Modern | 94.4% | ✅ |
| Bob Marley | 93.5% | ✅ |
| Slim Shady | 91.9% | ✅ |
| **Average** | **91.8%** | ✅ |

## Completed Milestones

### Core ML Models
- ✅ **HuBERT**: Feature extraction with transformer encoder
- ✅ **RMVPE**: Pitch detection with CustomBatchNorm fix
- ✅ **TextEncoder**: Phone embedding and conditioning
- ✅ **ResidualCouplingBlock (Flow)**: Normalizing flow for voice conversion
- ✅ **NSF-HiFiGAN Generator**: Neural source-filter vocoder

### iOS App Features
- ✅ Model gallery with bundled voice models
- ✅ Audio file import (mp3, wav filtering)
- ✅ Microphone recording
- ✅ Native .pth → .safetensors conversion on-device
- ✅ Waveform visualization (original vs converted)
- ✅ Audio playback controls
- ✅ User model import from Files app

### Key Parity Fixes Applied
1. **Flow reverse pass order**: Critical fix - flip BEFORE flow in reverse mode
2. **CustomBatchNorm**: Properly loads running statistics for RMVPE
3. **Weight key remapping**: Comprehensive PyTorch → Swift module mapping
4. **Native ConvTransposed1d**: Using MLX Swift's native upsampling
5. **WaveNet architecture**: Single cond_layer matching Python MLX

## Key Technical Achievements

### 1. CustomBatchNorm Implementation
MLX Swift's built-in `BatchNorm` doesn't expose `runningMean`/`runningVar` via `parameters()`. Created `CustomBatchNorm` class that properly loads running statistics, eliminating NaN outputs.

### 2. Weight Key Remapping
Implemented comprehensive runtime key remapping to handle PyTorch → Swift module structure differences:
- `dec.ups.N` → `dec.up_N`
- `dec.resblocks.N.convs1.M` → `dec.resblock_N.c1_M`
- `dec.noise_convs.N` → `dec.noise_conv_N`
- `enc_p.encoder.attn_layers.N` → `enc_p.encoder.attn_N`
- `enc_p.encoder.norm_layers_1.N` → `enc_p.encoder.norm1_N`
- Flow index remapping: `{0,2,4,6}` → `{0,1,2,3}`
- LayerNorm: `.gamma` → `.weight`, `.beta` → `.bias`

### 3. Native ConvTransposed1d
Replaced manual implementation with MLX Swift's native `ConvTransposed1d` for upsampling (10x, 8x, 2x, 2x).

### 4. Flow Reverse Pass Fix
Critical fix: In reverse mode, flip BEFORE flow (not after). This single fix improved correlation from ~72% to ~92%.

## File Structure

```
Demos/iOS/RVCNative/
├── RVCNative.xcworkspace/         # Open this in Xcode
├── RVCNativePackage/
│   ├── Sources/RVCNativeFeature/
│   │   ├── RVC/
│   │   │   ├── RVCInference.swift      # Main pipeline, weight loading
│   │   │   ├── RVCModel.swift          # Generator, ResBlock
│   │   │   ├── Synthesizer.swift       # TextEncoder, Flow
│   │   │   ├── HuBERT.swift            # Feature extraction
│   │   │   ├── RMVPE.swift             # Pitch detection
│   │   │   ├── PthConverter.swift      # .pth → .safetensors
│   │   │   └── Transforms.swift        # STFT, mel spectrogram
│   │   └── Assets/
│   │       ├── hubert_base.safetensors
│   │       ├── rmvpe.safetensors
│   │       ├── Drake.safetensors
│   │       ├── slim_shady.safetensors
│   │       └── coder.safetensors
│   └── Tests/
└── Config/
```

## Building & Running

### Requirements
- Xcode 16+
- iOS 18.0+ deployment target
- Apple Silicon Mac (for development)

### Build Steps
1. Open `RVCNative.xcworkspace` in Xcode
2. Select target device/simulator
3. Build and run (Cmd+R)

### Running on Device
For best performance, deploy to a physical iOS device with Apple Silicon (A14+).

## Performance

- **Inference time**: ~2-3 seconds for 10s audio on iPhone 15 Pro
- **Memory usage**: ~2GB peak
- **Realtime factor**: ~5x realtime

## iOS Simulator Limitations

**Critical Note**: Physical device testing is mandatory for MLX inference testing.
- Simulator Metal support is incomplete
- Stock MLX Swift library may crash with nullptr assertions
- Use simulator only for basic UI layout verification

## Model Conversion Pipeline

### Python → MLX → Safetensors

1. **PyTorch (.pth) → MLX (.npz)**
   ```bash
   python tools/convert_rvc_model.py input.pth output.npz
   ```

2. **MLX (.npz) → Safetensors (.safetensors)**
   ```bash
   python tools/convert_npz_to_safetensors.py output.npz output.safetensors
   ```

Or use the all-in-one iOS conversion:
```bash
python tools/convert_models_for_ios.py \
    --model-path /path/to/model \
    --model-name "ModelName" \
    --output-dir Demos/iOS/RVCNative/.../Assets
```

## Related Documentation

- [AUDIO_QUALITY_FIX.md](../Demos/iOS/AUDIO_QUALITY_FIX.md) - Detailed fix documentation
- [BATCHNORM_FIX_VALIDATION.md](../Demos/iOS/BATCHNORM_FIX_VALIDATION.md) - BatchNorm validation results
- [MLX_PYTHON_SWIFT_DIFFERENCES.md](MLX_PYTHON_SWIFT_DIFFERENCES.md) - API differences guide
- [PYTORCH_MLX_SWIFT_DIFFERENCES.md](PYTORCH_MLX_SWIFT_DIFFERENCES.md) - Conversion guide
- [iOS_REWRITE_PLAN.md](../Demos/iOS/RVCNative/iOS_REWRITE_PLAN.md) - Implementation plan

## Historical Issues & Resolutions

<details>
<summary>Click to expand historical debugging notes</summary>

### iOS Audio Inference - Scrambled Output (2026-01-05)
**Status**: ✅ RESOLVED - Fixed selective weight transposition

### HuBERT Dimension Mismatch (2026-01-05)
**Status**: ✅ RESOLVED - Enabled final_proj layer

### Silent Audio Output (2026-01-05)
**Status**: ✅ RESOLVED - Fixed Int16 PCM output

### RMVPE NaN Outputs (2026-01-07)
**Status**: ✅ RESOLVED - CustomBatchNorm implementation

### Flow Reverse Pass Order (2026-01-07)
**Status**: ✅ RESOLVED - Flip before flow in reverse mode

</details>

## Conclusion

The iOS Swift MLX implementation is **production-ready** with:
- ✅ 91.8% average spectrogram correlation
- ✅ All components verified and tested
- ✅ Native on-device model conversion
- ✅ Full-featured iOS app with modern SwiftUI
- ✅ Comprehensive weight key remapping
- ✅ CustomBatchNorm for proper running stats

---

*Last Updated: 2026-01-08*
*iOS Swift Status: ✅ Production Ready (91.8% correlation)*
