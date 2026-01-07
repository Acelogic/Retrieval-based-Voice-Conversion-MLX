# PyTorch vs MLX Swift Implementation Differences

## Overview

This document outlines the key differences between PyTorch and MLX Swift that are critical for porting neural network models. These differences span tensor formats, weight layouts, APIs, and architectural patterns.

## 1. Tensor Dimension Ordering

### Conv1d Data Format
| Framework | Format | Description |
|-----------|--------|-------------|
| PyTorch | `(B, C, T)` | Batch, Channels, Time |
| MLX Swift | `(B, T, C)` | Batch, Time, Channels |

**Impact**: Requires transposes at module boundaries.

```swift
// PyTorch format to MLX Swift format
let mlxInput = pytorchInput.transposed(0, 2, 1)  // (B, C, T) → (B, T, C)

// MLX Swift format to PyTorch format
let pytorchOutput = mlxOutput.transposed(0, 2, 1)  // (B, T, C) → (B, C, T)
```

### Conv2d Data Format
| Framework | Format |
|-----------|--------|
| PyTorch | `(B, C, H, W)` |
| MLX Swift | `(B, H, W, C)` |

## 2. Weight Shapes

### Conv1d Weights
| Framework | Shape |
|-----------|-------|
| PyTorch | `(Out_Channels, In_Channels, Kernel_Size)` |
| MLX Swift | `(Out_Channels, Kernel_Size, In_Channels)` |

**Conversion**:
```python
# Python conversion script
mlx_weight = pytorch_weight.transpose(0, 2, 1)
```

```swift
// Swift conversion (if needed at runtime)
let mlxWeight = pytorchWeight.transposed(axes: [0, 2, 1])
```

### ConvTranspose1d Weights
| Framework | Shape |
|-----------|-------|
| PyTorch | `(In_Channels, Out_Channels, Kernel_Size)` |
| MLX Swift | `(Out_Channels, Kernel_Size, In_Channels)` |

**Conversion**:
```python
mlx_weight = pytorch_weight.transpose(1, 2, 0)
```

### Conv2d Weights
| Framework | Shape |
|-----------|-------|
| PyTorch | `(Out_C, In_C, H, W)` |
| MLX Swift | `(Out_C, H, W, In_C)` |

**Conversion**:
```python
mlx_weight = pytorch_weight.transpose(0, 2, 3, 1)
```

### Linear/Embedding Weights
Both frameworks use the same shape - no conversion needed:
- Linear: `(Out_Features, In_Features)`
- Embedding: `(Num_Embeddings, Embedding_Dim)`

## 3. Weight Normalization

### PyTorch
```python
# Built-in support
torch.nn.utils.weight_norm(layer)
# Stores as weight_g and weight_v parameters
```

### MLX Swift
```swift
// No built-in weight_norm - must fuse during conversion
// weight = weight_g * (weight_v / ||weight_v||)

if let weightG = params["weight_g"], let weightV = params["weight_v"] {
    let vSqr = weightV * weightV
    let vNorm = sqrt(vSqr.sum(axes: [1, 2], keepDims: true) + 1e-12)
    let fusedWeight = weightG * (weightV / vNorm)
    params["weight"] = fusedWeight
}
```

## 4. Module Definition

### PyTorch
```python
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel)

    def forward(self, x):
        return self.conv(x)
```

### MLX Swift
```swift
class MyModule: Module {
    let conv: MLXNN.Conv1d

    init(inCh: Int, outCh: Int, kernel: Int) {
        self.conv = MLXNN.Conv1d(inputChannels: inCh, outputChannels: outCh, kernelSize: kernel)
        super.init()  // MUST call after property initialization
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return conv(x)
    }
}
```

**Key Differences**:
1. Swift uses `let` properties (immutable after init)
2. `super.init()` called AFTER property assignment in Swift
3. Swift uses `callAsFunction` instead of `forward`

## 5. Parameter Access and Loading

### PyTorch
```python
# Save
torch.save(model.state_dict(), "model.pth")

# Load
state_dict = torch.load("model.pth")
model.load_state_dict(state_dict, strict=False)

# Access parameters
for name, param in model.named_parameters():
    print(name, param.shape)
```

### MLX Swift
```swift
// Load from safetensors
let weights = try MLX.loadArrays(url: url)

// Update model with weights
model.update(parameters: ModuleParameters.unflattened(weights))

// Set eval mode
model.train(false)

// Access parameters
let params = model.parameters()
```

## 6. Array Operations

### Slicing
```python
# PyTorch
x[:, :, ::-1]  # Reverse last dimension
x[:, :, a:b]   # Range slice
```

```swift
// MLX Swift
x[0..., 0..., .stride(by: -1)]  // Reverse last dimension
x[0..., 0..., a..<b]             // Range slice
```

### Padding
```python
# PyTorch
F.pad(x, (left, right))  # 1D padding
F.pad(x, (l, r, t, b))   # 2D padding
```

```swift
// MLX Swift - must specify all dimensions
MLX.padded(x, widths: [IntOrPair((0, 0)), IntOrPair((left, right)), IntOrPair((0, 0))])
```

### Concatenation
```python
# PyTorch
torch.cat([x0, x1], dim=2)
```

```swift
// MLX Swift
MLX.concatenated([x0, x1], axis: 2)
```

## 7. Activation Functions

### PyTorch
```python
F.leaky_relu(x, negative_slope=0.1)
F.gelu(x)
torch.sigmoid(x)
torch.tanh(x)
```

### MLX Swift
```swift
leakyRelu(x, negativeSlope: 0.1)
gelu(x, approximate: .none)  // Note: .none for exact match
sigmoid(x)
tanh(x)
```

## 8. BatchNorm Parameters

### PyTorch
```python
# Running statistics
bn.running_mean
bn.running_var
bn.num_batches_tracked
```

### MLX Swift
```swift
// Camel case naming
bn.runningMean
bn.runningVar
// num_batches_tracked often skipped in inference
```

**Key Mapping in Conversion**:
```swift
key = key.replacingOccurrences(of: ".running_mean", with: ".runningMean")
key = key.replacingOccurrences(of: ".running_var", with: ".runningVar")
```

## 9. LayerNorm Parameters

### PyTorch (older/custom)
```python
# Some models use gamma/beta
ln.gamma  # Scale
ln.beta   # Bias
```

### MLX Swift
```swift
ln.weight  // Scale
ln.bias    // Bias
```

**Key Mapping**:
```swift
key = key.replacingOccurrences(of: ".gamma", with: ".weight")
key = key.replacingOccurrences(of: ".beta", with: ".bias")
```

## 10. ModuleList vs Named Properties

### PyTorch
```python
# ModuleList automatically registers
self.flows = nn.ModuleList([
    ResidualCouplingLayer(...) for _ in range(n_flows)
])
# Access: self.flows[0], self.flows[1], ...
# Weight keys: flow.flows.0.*, flow.flows.1.*, ...
```

### MLX Swift
```swift
// Must use named properties for weight loading
let flow_0: ResidualCouplingLayer
let flow_1: ResidualCouplingLayer
let flow_2: ResidualCouplingLayer
let flow_3: ResidualCouplingLayer

// Arrays don't work for weight loading!
// var flows: [ResidualCouplingLayer] = []  // WRONG
```

## 11. Device Management

### PyTorch
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
x = x.to(device)
```

### MLX Swift
```swift
// Automatic - no explicit device management needed
// MLX uses unified memory on Apple Silicon
// Optionally set default device:
MLX.Device.setDefault(device: Device.gpu)
```

## 12. Gradient Control

### PyTorch
```python
with torch.no_grad():
    output = model(x)
```

### MLX Swift
```swift
// No gradients by default in inference mode
// Just call the model directly
let output = model(x)

// Ensure computation is complete
MLX.eval(output)
```

## 13. Random Number Generation

### PyTorch
```python
torch.randn(shape)
torch.normal(mean, std, size)
```

### MLX Swift
```swift
MLXRandom.normal(shape)
MLXRandom.normal(shape, mean: 0.0, std: 1.0)
```

## Summary Table

| Feature | PyTorch | MLX Swift |
|---------|---------|-----------|
| Conv1d data | (B, C, T) | (B, T, C) |
| Conv1d weight | (Out, In, K) | (Out, K, In) |
| Super init | Before properties | After properties |
| Forward method | `forward(self, x)` | `callAsFunction(_ x:)` |
| Param access | `state_dict()` | `parameters()` |
| Weight loading | `load_state_dict()` | `update(parameters:)` |
| Softmax axis | `dim=-1` | `axis: -1` |
| Device | Explicit `.to(device)` | Automatic |
| No gradients | `torch.no_grad()` | Default behavior |
| ModuleList | Supported | Use named properties |
| Weight norm | Built-in | Fuse manually |

## Common Pitfalls

1. **Forgetting weight transposition** - Conv weights have different layouts
2. **Using arrays instead of named properties** - Weights won't load
3. **Wrong super.init() order** - Swift requires properties initialized first
4. **Missing gamma/beta → weight/bias mapping** - LayerNorm will use defaults
5. **Not fusing weight normalization** - Model will have wrong weights
6. **Assuming same padding API** - MLX requires explicit pad_width for all dims
7. **Mixing tensor formats** - Keep track of (B,C,T) vs (B,T,C) throughout

## Conversion Checklist

- [ ] Transpose all Conv1d weights: `(Out, In, K)` → `(Out, K, In)`
- [ ] Transpose all ConvTranspose1d weights: `(In, Out, K)` → `(Out, K, In)`
- [ ] Fuse weight_g/weight_v into single weight tensors
- [ ] Map running_mean → runningMean, running_var → runningVar
- [ ] Map gamma → weight, beta → bias for LayerNorm
- [ ] Convert ModuleList to named properties
- [ ] Add tensor format transposes at module boundaries
- [ ] Set model to eval mode: `model.train(false)`
- [ ] Call `MLX.eval()` after operations to ensure completion
