# RVC MLX - Project Context for Claude

**Project:** Retrieval-based Voice Conversion (RVC) ported to Apple MLX
**Status:** Production Ready (Python MLX: 99.98% correlation, Swift MLX: 91.8% correlation)

## Project Overview

This project implements RVC voice conversion using Apple's MLX framework, with both:
- **Python MLX** (`rvc_mlx/`) - Desktop inference, 8.71x faster than PyTorch
- **Swift MLX** (`Demos/iOS/RVCNative/`) - Native iOS app

## Directory Structure

```
Retrieval-based-Voice-Conversion-MLX/
├── rvc_mlx/                    # Python MLX implementation
│   ├── lib/mlx/               # Core ML modules
│   │   ├── attentions.py      # Multi-head attention with relative position
│   │   ├── encoders.py        # TextEncoder, PosteriorEncoder
│   │   ├── generators.py      # HiFiGAN-NSF vocoder
│   │   ├── residuals.py       # ResidualCouplingBlock (Flow)
│   │   ├── rmvpe.py           # Pitch detection
│   │   └── synthesizers.py    # Main Synthesizer
│   └── infer/                 # Inference pipeline
├── tools/                      # Conversion & debugging scripts
│   ├── convert_rvc_model.py   # PyTorch → MLX conversion
│   ├── convert_models_for_ios.py  # Full iOS conversion
│   └── compare_rvc_full.py    # Parity verification
├── Demos/
│   ├── iOS/RVCNative/         # iOS Swift app
│   └── Mac/                   # Mac CLI demo
├── weights/                    # Converted model weights (.npz, .safetensors)
├── docs/                       # Technical documentation
└── benchmarks/                 # Performance tests
```

## Model Source of Truth

Original PyTorch models are at:
```
/Users/mcruz/Library/Application Support/Replay/com.replay.Replay/models/
```

---

# Python MLX - Critical Fixes (DO NOT REGRESS)

## 1. Dimension Ordering - (B, C, T) vs (B, T, C)

**Issue:** MLX Conv1d uses (Batch, Time, Channels) while PyTorch uses (Batch, Channels, Time)

**Fix:** Add transposes at module boundaries
```python
# TextEncoder output (rvc_mlx/lib/mlx/encoders.py:146-149)
m_p = m_p.transpose(0, 2, 1)  # (B, T, C) → (B, C, T)
logs_p = logs_p.transpose(0, 2, 1)

# Generator input/output (rvc_mlx/lib/mlx/generators.py)
x = x.transpose(0, 2, 1)  # Input: (B, C, T) → (B, T, C)
o = o.transpose(0, 2, 1)  # Output: (B, T, C) → (B, C, T)
```

## 2. LayerNorm gamma/beta Parameters

**Issue:** PyTorch RVC uses `.gamma` and `.beta` for LayerNorm (older convention)

**Fix:** Map during conversion (tools/convert_rvc_model.py)
```python
if key.endswith(".gamma"):
    new_key = key[:-6] + ".weight"
elif key.endswith(".beta"):
    new_key = key[:-5] + ".bias"
```

## 3. Relative Position Embeddings - DO NOT TRANSPOSE

**Issue:** Converter was transposing `emb_rel_k` and `emb_rel_v` as Conv weights

**Fix:** Skip transposition for embeddings (tools/convert_rvc_model.py:267-270)
```python
if "emb_rel" in base_key:
    pass  # Keep as-is, DO NOT transpose
```

## 4. Relative Position Embeddings - No .weight Suffix

**Issue:** Embeddings saved with `.weight` suffix but they're direct attributes

**Fix:** Don't add .weight suffix for emb_rel (tools/convert_rvc_model.py:258-263)
```python
if "emb_rel" in base_key:
    full_key = base_key  # No .weight suffix!
else:
    full_key = f"{base_key}.weight"
```

## 5. Flow Layer Index Mapping (CRITICAL)

**Issue:** PyTorch `ResidualCouplingBlock` interleaves Layer and Flip modules:
- Index 0: Layer, Index 1: Flip, Index 2: Layer, Index 3: Flip...
- MLX uses only Layer modules: 0, 1, 2, 3...

**Fix:** Map PyTorch index to MLX index `i // 2`
```python
# PyTorch flow.flows.0 → MLX flow.flow_0
# PyTorch flow.flows.2 → MLX flow.flow_1
# PyTorch flow.flows.4 → MLX flow.flow_2
mlx_idx = pytorch_idx // 2
```

## 6. RMVPE Shortcut Layer Bug (CRITICAL)

**Issue:** Extra BatchNorm in shortcut was causing signal explosion

**Wrong:**
```python
self.shortcut = nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size=1),
    nn.BatchNorm(out_channels)  # EXTRA - WRONG!
)
```

**Correct:**
```python
self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # No BatchNorm!
```

**File:** `rvc_mlx/lib/mlx/rmvpe.py:80-83`

## 7. RMVPE Custom PyTorchGRU

**Issue:** MLX's built-in GRU has different bias handling than PyTorch

**Fix:** Created custom `PyTorchGRU` class matching PyTorch formula exactly
```python
# PyTorch GRU formula:
r_t = σ(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)
z_t = σ(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)
n_t = tanh(W_in @ x_t + b_in + r_t * (W_hn @ h_{t-1} + b_hn))
h_t = (1 - z_t) * n_t + z_t * h_{t-1}
```

**File:** `rvc_mlx/lib/mlx/pytorch_gru.py`

## 8. RMVPE Reflect Padding

**Issue:** MLX was using constant padding, PyTorch uses reflect

**Fix:** Implement custom reflect padding
```python
# Reflect padding: mirror WITHOUT including edge value
# [1,2,3,4] + pad(2) = [1,2,3,4,3,2]
if pad_curr <= n_frames - 1:
    reflected = mel_np[:, -(pad_curr+1):-1][:, ::-1]
```

**File:** `rvc_mlx/lib/mlx/rmvpe.py:323-351`

---

# Swift MLX - Critical Fixes (DO NOT REGRESS)

## 1. Flow Reverse Pass Order (CRITICAL - 20% improvement)

**Issue:** Flip order differs between forward and reverse modes

**Wrong (72% correlation):**
```swift
// WRONG - flip after flow in reverse
for i in (0..<nFlows).reversed() {
    h = flows[i](h, xMask: xMask, g: g, reverse: true)
    h = h[0..., 0..., .stride(by: -1)]  // flip AFTER - WRONG!
}
```

**Correct (92% correlation):**
```swift
// CORRECT - flip BEFORE flow in reverse mode
for i in (0..<nFlows).reversed() {
    h = h[0..., 0..., .stride(by: -1)]  // flip BEFORE - CORRECT!
    h = flows[i](h, xMask: xMask, g: g, reverse: true)
}
```

**File:** `Demos/iOS/RVCNative/.../Synthesizer.swift`

## 2. CustomBatchNorm for RMVPE (CRITICAL - Fixes NaN)

**Issue:** MLX Swift's `BatchNorm` doesn't expose `runningMean`/`runningVar` via `parameters()`

**Symptoms:** Signal explosion (1e18), NaN outputs, all F0 = 0 Hz

**Fix:** Created `CustomBatchNorm` exposing running stats as properties
```swift
class CustomBatchNorm: Module {
    var runningMean: MLXArray  // Loadable via update(parameters:)
    var runningVar: MLXArray   // Loadable via update(parameters:)
    var weight: MLXArray
    var bias: MLXArray
}
```

**File:** `Demos/iOS/RVCNative/.../RMVPE.swift`

## 3. Named Properties vs Arrays (Weights won't load!)

**Issue:** MLX Swift `update(parameters:)` only works with named properties

**Wrong:**
```swift
var flows: [ResidualCouplingLayer] = []  // Weights WON'T LOAD!
```

**Correct:**
```swift
let flow_0: ResidualCouplingLayer
let flow_1: ResidualCouplingLayer
let flow_2: ResidualCouplingLayer
let flow_3: ResidualCouplingLayer
```

## 4. Weight Key Remapping (PthConverter.swift)

**Required remappings:**
```swift
// Decoder
"dec.noise_convs.N" → "dec.noise_conv_N"
"dec.ups.N" → "dec.up_N"
"dec.resblocks.N.convs1.M" → "dec.resblock_N.c1_M"
"dec.resblocks.N.convs2.M" → "dec.resblock_N.c2_M"

// Encoder
"enc_p.encoder.attn_layers.N" → "enc_p.encoder.attn_N"
"enc_p.encoder.norm_layers_1.N" → "enc_p.encoder.norm1_N"
"enc_p.encoder.ffn_layers.N" → "enc_p.encoder.ffn_N"

// Flow (skip Flip modules)
"flow.flows.0" → "flow.flow_0"
"flow.flows.2" → "flow.flow_1"
"flow.flows.4" → "flow.flow_2"

// LayerNorm
".gamma" → ".weight"
".beta" → ".bias"
```

## 5. Weight Transposition Rules

```swift
// 3D weights (Conv1d)
if k.contains(".up_") || k.contains(".ups.") {
    val = val.transposed(axes: [1, 2, 0])  // ConvTranspose
} else {
    val = val.transposed(axes: [0, 2, 1])  // Regular Conv
}

// 2D weights with "linear" in key
if k.contains("weight") && val.ndim == 2 && k.lowercased().contains("linear") {
    val = val.transposed()  // Linear: (Out, In) → (In, Out)
}
```

---

# Tensor Format Reference

| Framework | Conv1d Data | Conv1d Weight |
|-----------|-------------|---------------|
| PyTorch | (B, C, T) | (Out, In, K) |
| MLX Python | (B, T, C) | (Out, K, In) |
| MLX Swift | (B, T, C) | (Out, K, In) |

---

# Key Commands

## Run Comparative Benchmark
```bash
./run_comparative_benchmark.sh
```

## Convert Model for Python MLX
```bash
python3 tools/convert_rvc_model.py /path/to/model.pth weights/model.npz
```

## Convert Model for iOS
```bash
python3 tools/convert_models_for_ios.py \
    --model-path /path/to/model \
    --model-name "ModelName" \
    --output-dir Demos/iOS/RVCNative/.../Assets
```

## Build Mac CLI
```bash
swift build -c release --package-path Demos/Mac --product RVCNativeMac
```

## Build iOS App
```bash
cd Demos/iOS/RVCNative
xcodebuild -workspace RVCNative.xcworkspace -scheme RVCNative build
```

---

# Parity Results Achieved

## Python MLX vs PyTorch
- **Correlation:** 0.999847 (near perfect)
- **Speedup:** 8.71x faster than PyTorch MPS

## Swift MLX vs Python MLX
| Model | Correlation |
|-------|-------------|
| Drake | 92.9% |
| Juice WRLD | 86.6% |
| Eminem Modern | 94.4% |
| Bob Marley | 93.5% |
| Slim Shady | 91.9% |
| **Average** | **91.8%** |

---

# Documentation

- `docs/INFERENCE_PARITY_ACHIEVED.md` - Full parity achievement details
- `docs/RMVPE_OPTIMIZATION.md` - RMVPE debugging journey
- `docs/PYTORCH_MLX_DIFFERENCES.md` - PyTorch to Python MLX conversion
- `docs/MLX_PYTHON_SWIFT_DIFFERENCES.md` - Python MLX to Swift MLX
- `docs/IOS_DEVELOPMENT.md` - iOS implementation status
- `Demos/iOS/AUDIO_QUALITY_FIX.md` - Swift-specific fix history
