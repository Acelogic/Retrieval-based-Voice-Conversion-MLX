# iOS RVC Weight Conversion Guide

## Overview

The iOS RVC app requires model weights in `.safetensors` format with specific key naming conventions that match the Swift MLX model architecture.

---

## Weight Files Required

| Model | Source | Destination |
|-------|--------|-------------|
| HuBERT | `hubert_base.safetensors` | `Assets/hubert_base.safetensors` |
| RVC Synthesizer | RVC `.pth` → converted | `Assets/coder.safetensors` |
| RMVPE | `rmvpe_mlx.npz` | `Assets/rmvpe.safetensors` |

---

## RMVPE Weight Conversion

### Script
```bash
python3 tools/convert_rmvpe_weights.py
```

### Key Transformations

1. **Remove `fc.` prefix** from GRU and linear layer keys
2. **Convert snake_case to camelCase** for Swift array properties
3. **Keep PyTorch GRU weight names** (no format change needed)

### GRU Weight Mapping

| Python NPZ Key | Swift safetensors Key | Shape |
|----------------|----------------------|-------|
| `fc.bigru.forward_grus.0.weight_ih` | `bigru.forwardGRUs.0.weight_ih` | (768, 384) |
| `fc.bigru.forward_grus.0.weight_hh` | `bigru.forwardGRUs.0.weight_hh` | (768, 256) |
| `fc.bigru.forward_grus.0.bias_ih` | `bigru.forwardGRUs.0.bias_ih` | (768,) |
| `fc.bigru.forward_grus.0.bias_hh` | `bigru.forwardGRUs.0.bias_hh` | (768,) |

> **Note:** Swift uses custom `PyTorchGRU` class with the same weight names as Python.

---

## Why PyTorchGRU Instead of MLXNN GRU?

MLX Swift's built-in `GRU` uses different bias handling:

| | MLXNN GRU | PyTorchGRU |
|---|---|---|
| Input bias | `b` [768] for all gates | `bias_ih` [768] for all gates |
| Hidden bias | `bhn` [256] for "n" gate only | `bias_hh` [768] for all gates |

The PyTorch GRU formula applies `bias_hh` to ALL gates (r, z, n), not just the "n" gate. This difference caused completely wrong RMVPE output.

---

## Swift PyTorchGRU Implementation

```swift
class PyTorchGRU: Module {
    var weight_ih: MLXArray  // [3*H, D]
    var weight_hh: MLXArray  // [3*H, H]
    var bias_ih: MLXArray?   // [3*H]
    var bias_hh: MLXArray?   // [3*H]
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // ... PyTorch-compatible GRU computation
    }
}
```

---

## Verification

After conversion, verify weight keys:

```python
import mlx.core as mx

weights = dict(mx.load("Assets/rmvpe.safetensors"))

# Check GRU keys use camelCase and PyTorch names
for k in sorted(weights.keys()):
    if 'GRU' in k:
        print(f"  {k}: {weights[k].shape}")

# Expected output:
#   bigru.backwardGRUs.0.bias_hh: (768,)
#   bigru.backwardGRUs.0.bias_ih: (768,)
#   bigru.backwardGRUs.0.weight_hh: (768, 256)
#   bigru.backwardGRUs.0.weight_ih: (768, 384)
#   bigru.forwardGRUs.0.bias_hh: (768,)
#   bigru.forwardGRUs.0.bias_ih: (768,)
#   bigru.forwardGRUs.0.weight_hh: (768, 256)
#   bigru.forwardGRUs.0.weight_ih: (768, 384)
```

---

## Common Issues

### 1. 100% Voiced Frames (maxx all high)
**Cause:** MLXNN GRU with wrong bias format
**Fix:** Use PyTorchGRU implementation

### 2. 0% Voiced Frames (maxx all low)
**Cause:** GRU weights not loading (key mismatch)
**Fix:** Use camelCase for array properties (`forwardGRUs` not `forward_grus`)

### 3. ~5% Voiced Frames (maxx slightly above threshold)
**Cause:** GRU working but upstream layers (UNet/CNN) may have issues
**Status:** Currently investigating
