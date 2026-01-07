# Python MLX vs Swift MLX Implementation Differences

## Overview

This document outlines the key differences between Python MLX and Swift MLX that were critical for achieving inference parity in the RVC port. While both use the same underlying MLX framework, the Swift bindings have different conventions and APIs.

## 1. Module Parameter Registration

### Python MLX
```python
# Lists assigned to self are NOT automatically tracked
self.flows = [ResidualCouplingLayer(...) for _ in range(n_flows)]

# Must explicitly register with setattr
for i, f in enumerate(self.flows):
    setattr(self, f"flow_{i}", f)
```

### Swift MLX
```swift
// Named properties ARE automatically tracked
let flow_0: ResidualCouplingLayer
let flow_1: ResidualCouplingLayer
let flow_2: ResidualCouplingLayer
let flow_3: ResidualCouplingLayer

// Arrays are NOT tracked for weight loading
var flows: [ResidualCouplingLayer] = []  // Won't load weights!
```

**Critical Insight**: In Swift MLX, you MUST use named properties (not arrays) for modules if you want weights to load via `update(parameters:)`. The weight key paths must exactly match the property names.

## 2. Weight Key Path Matching

### Python MLX
Weight keys like `flow.flow_0.enc.cond_layer.weight` map to:
```python
self.flow_0 = ResidualCouplingLayer(...)  # setattr
self.flow_0.enc = WaveNet(...)
self.flow_0.enc.cond_layer = Conv1d(...)
```

### Swift MLX
Must mirror the exact structure:
```swift
class ResidualCouplingBlock: Module {
    let flow_0: ResidualCouplingLayer  // Property name = "flow_0"
    let flow_1: ResidualCouplingLayer
    // ...
}

class ResidualCouplingLayer: Module {
    let enc: WaveNet  // Property name = "enc"
    // ...
}

class WaveNet: Module {
    let cond_layer: Conv1d?  // Property name = "cond_layer"
    // ...
}
```

**Key Point**: Weight file keys must exactly match the Swift property path: `flow.flow_0.enc.cond_layer.weight`

## 3. Conv1d Layer Architecture

### Python MLX WaveNet
```python
# Single cond_layer at WaveNet level
self.cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)

# Per-layer slicing during forward pass
for i in range(self.n_layers):
    g_l = g[:, :, i * 2 * hidden : (i + 1) * 2 * hidden]
```

### Swift MLX WaveNet
```swift
// Must match Python structure exactly
let cond_layer: MLXNN.Conv1d?  // Single layer: outputs 2 * hidden * n_layers

// Per-layer slicing in callAsFunction
for i in 0..<nLayers {
    let startCh = i * 2 * hiddenChannels
    let endCh = (i + 1) * 2 * hiddenChannels
    let gSlice = gCond[0..., 0..., startCh..<endCh]
}
```

## 4. Array Slicing Syntax

### Python MLX
```python
# Reverse slice on last axis
x = x[:, :, ::-1]

# Range slice
g_l = g[:, :, start:end]
```

### Swift MLX
```swift
// Reverse slice on last axis
x = x[0..., 0..., .stride(by: -1)]

// Range slice
let gSlice = g[0..., 0..., start..<end]
```

## 5. Flow Reverse Pass Order (CRITICAL!)

### Python MLX
```python
def __call__(self, x, x_mask, g=None, reverse=False):
    if not reverse:
        for i in range(self.n_flows):
            flow = getattr(self, f"flow_{i}")
            x, _ = flow(x, x_mask, g=g, reverse=False)
            x = x[:, :, ::-1]  # Flip AFTER flow
    else:
        for i in reversed(range(self.n_flows)):
            flow = getattr(self, f"flow_{i}")
            x = x[:, :, ::-1]  # Flip BEFORE flow!
            x, _ = flow(x, x_mask, g=g, reverse=True)
    return x
```

### Swift MLX
```swift
func callAsFunction(_ x: MLXArray, xMask: MLXArray, g: MLXArray?, reverse: Bool = false) -> MLXArray {
    var h = x

    if !reverse {
        // Forward: flow then flip
        for i in 0..<nFlows {
            h = flows[i](h, xMask: xMask, g: g, reverse: false)
            h = h[0..., 0..., .stride(by: -1)]
        }
    } else {
        // Reverse: flip then flow (CRITICAL: different order!)
        for i in (0..<nFlows).reversed() {
            h = h[0..., 0..., .stride(by: -1)]  // Flip FIRST
            h = flows[i](h, xMask: xMask, g: g, reverse: true)
        }
    }
    return h
}
```

**This is the single most important fix for parity!** Getting the flip order wrong in reverse mode drops correlation from ~92% to ~72%.

## 6. Optional Type Handling

### Python MLX
```python
if g is not None:
    g = self.cond_layer(g)
```

### Swift MLX
```swift
var gCond: MLXArray? = nil
if let g = g, let condLayer = cond_layer {
    gCond = condLayer(g)
}
```

## 7. Module Initialization

### Python MLX
```python
class WaveNet(nn.Module):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels):
        super().__init__()
        # Properties assigned after super().__init__()
        self.cond_layer = nn.Conv1d(...)
```

### Swift MLX
```swift
class WaveNet: Module {
    let cond_layer: MLXNN.Conv1d?

    init(hiddenChannels: Int, kernelSize: Int, dilationRate: Int, nLayers: Int, ginChannels: Int) {
        // Properties MUST be assigned BEFORE super.init()
        self.cond_layer = ginChannels != 0 ? MLXNN.Conv1d(...) : nil
        super.init()
    }
}
```

## 8. Last Layer Special Cases

### Python MLX WaveNet
```python
# Last res_skip_layer has different output channels
for i in range(n_layers):
    res_skip_channels = hidden_channels if i == n_layers - 1 else 2 * hidden_channels
    self.res_skip_layers.append(nn.Conv1d(hidden_channels, res_skip_channels, 1))
```

### Swift MLX WaveNet
```swift
// Must explicitly handle last layer
let res_skip_layer_0 = MLXNN.Conv1d(..., outputChannels: 2 * hiddenChannels, ...)
let res_skip_layer_1 = MLXNN.Conv1d(..., outputChannels: 2 * hiddenChannels, ...)
let res_skip_layer_2 = MLXNN.Conv1d(..., outputChannels: hiddenChannels, ...)  // Last layer: different!
```

## 9. Tensor Format Consistency

Both Python MLX and Swift MLX use channels-last format `(B, T, C)` for 1D convolutions, but be careful:

### Python MLX
```python
# Explicit about format in comments
# x: (N, L, C) -> (Batch, Length, Channels)
```

### Swift MLX
```swift
// Same format, but Swift uses explicit axis indices
// x: [B, T, C] = [Batch, Time, Channels]
// Axis 0 = Batch, Axis 1 = Time, Axis 2 = Channels
```

## 10. Debug Output

### Python MLX
```python
print(f"DEBUG: shape={x.shape}, min={x.min().item()}, max={x.max().item()}")
```

### Swift MLX
```swift
print("DEBUG: shape=\(x.shape), min=\(x.min().item(Float.self)), max=\(x.max().item(Float.self))")
```

## Summary Table

| Feature | Python MLX | Swift MLX |
|---------|------------|-----------|
| Module registration | `setattr(self, name, layer)` | Named properties only |
| Array slicing | `x[:, :, ::-1]` | `x[0..., 0..., .stride(by: -1)]` |
| Range slicing | `x[:, :, a:b]` | `x[0..., 0..., a..<b]` |
| Optional check | `if x is not None:` | `if let x = x { }` |
| Super init order | After property assignment | Before property assignment |
| Weight path | Exact match via setattr | Exact match via property names |

## Common Pitfalls

1. **Using arrays for modules** - Weights won't load; use named properties
2. **Wrong flip order in reverse pass** - Forward: flow→flip, Reverse: flip→flow
3. **Missing last layer special case** - res_skip_layer output channels differ
4. **Assuming Python property names** - Swift requires explicit properties, not dynamic setattr
5. **Not matching weight key paths exactly** - Weight file keys must match Swift property hierarchy

## Parity Results After Fixes

| Model | Before Fixes | After Fixes |
|-------|--------------|-------------|
| Drake | 78.0% | 92.9% |
| Juice WRLD | 58.3% | 86.6% |
| Eminem Modern | 82.6% | 94.4% |
| Bob Marley | 71.3% | 93.5% |
| Slim Shady | 71.0% | 91.9% |
| **Average** | **72.6%** | **91.8%** |
