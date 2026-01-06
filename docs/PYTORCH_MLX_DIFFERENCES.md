# PyTorch vs MLX Implementation Differences

## Overview

This document outlines the key differences between PyTorch and MLX that were critical for achieving inference parity in the RVC port.

## 1. Tensor Dimension Ordering

### Conv1d
**PyTorch:** `(Batch, Channels, Length)`
**MLX:** `(Batch, Length, Channels)`

**Impact:** Requires transposes at module boundaries or consistent use of one convention throughout.

**Our Approach:**
- Internal MLX modules use MLX format `(B, T, C)`
- TextEncoder output transposes to PyTorch format `(B, C, T)` for compatibility
- Generator transposes input from `(B, C, T)` to `(B, T, C)`, then back for output

### Conv2d
**PyTorch:** `(Batch, Channels, Height, Width)`
**MLX:** `(Batch, Height, Width, Channels)`

### ConvTranspose1d
**PyTorch:** Input/Output use `(Batch, Channels, Length)`
**MLX:** Input/Output use `(Batch, Length, Channels)`

## 2. Weight Shapes

### Conv1d Weights
**PyTorch:** `(Out_Channels, In_Channels, Kernel_Size)`
**MLX:** `(Out_Channels, Kernel_Size, In_Channels)`

**Conversion:** `pytorch_weight.transpose(0, 2, 1)`

### ConvTranspose1d Weights
**PyTorch:** `(In_Channels, Out_Channels, Kernel_Size)`
**MLX:** `(Out_Channels, Kernel_Size, In_Channels)`

**Conversion:** `pytorch_weight.transpose(1, 2, 0)`

### Conv2d Weights
**PyTorch:** `(Out_Channels, In_Channels, Height, Width)`
**MLX:** `(Out_Channels, Height, Width, In_Channels)`

**Conversion:** `pytorch_weight.transpose(0, 2, 3, 1)`

### Linear Weights
**Both:** `(Out_Features, In_Features)` - No change needed

### Embedding Weights
**Both:** `(Num_Embeddings, Embedding_Dim)` - No change needed

### LayerNorm Parameters
**PyTorch (newer):** `.weight` and `.bias`
**PyTorch (older):** `.gamma` and `.beta`
**MLX:** `.weight` and `.bias`

**Conversion:** Map `.gamma` → `.weight` and `.beta` → `.bias`

## 3. Weight Normalization

### PyTorch
```python
# Stores as two parameters: weight_g, weight_v
# Runtime: w = weight_g * (weight_v / ||weight_v||)
torch.nn.utils.weight_norm(layer)
```

### MLX
```python
# No built-in weight_norm
# Must fuse during conversion: w = g * (v / ||v||)
```

**Conversion Logic:**
```python
if "weight_g" in params and "weight_v" in params:
    v = params["weight_v"]
    g = params["weight_g"]

    # Compute norm along appropriate dimensions
    if v.ndim == 3:  # Conv1d
        norm_v = np.linalg.norm(v, axis=(1, 2), keepdims=True)
    elif v.ndim == 4:  # Conv2d
        norm_v = np.linalg.norm(v, axis=(1, 2, 3), keepdims=True)
    elif v.ndim == 2:  # Linear
        norm_v = np.linalg.norm(v, axis=1, keepdims=True)

    final_weight = v * (g / (norm_v + 1e-8))
```

## 4. Padding API

### PyTorch
```python
# Pad last dimension with (1, 2)
x = F.pad(x, (1, 2))

# Complex padding
x = F.pad(x, (left, right, top, bottom))
```

### MLX
```python
# Must specify all dimensions
x = mx.pad(x, pad_width=[(0, 0), (0, 0), (1, 2)])

# pad_width is list of (before, after) tuples for each dimension
```

## 5. Array Slicing

### Both Frameworks
Array slicing is similar, but MLX doesn't support in-place operations:

**PyTorch:**
```python
x[:, :, :, 1:]  # Returns view
x[:, :, :length, length-1:]  # Returns view
```

**MLX:**
```python
x[:, :, :, 1:]  # Returns new array
x[:, :, :length, length-1:]  # Returns new array
```

## 6. Random Number Generation

### PyTorch
```python
torch.randn(shape)
torch.normal(mean, std, size)
```

### MLX
```python
mx.random.normal(shape)
mx.random.normal(shape, mean=0.0, scale=1.0)
```

## 7. Gradient Computation

### PyTorch
```python
with torch.no_grad():
    # Disable gradients
    output = model(x)
```

### MLX
```python
# No gradients by default in inference mode
# Use mx.eval() to ensure computation is complete
mx.eval(model.parameters())
output = model(x)
```

## 8. Module Parameter Registration

### PyTorch
```python
# Automatically tracks nn.Parameter
self.weight = nn.Parameter(torch.randn(10, 10))

# Or register buffer
self.register_buffer('running_mean', torch.zeros(10))
```

### MLX
```python
# Automatically tracks mx.array assigned in __init__
self.weight = mx.random.normal((10, 10))

# For non-trainable parameters, same approach
self.running_mean = mx.zeros((10,))
```

**Important for relative embeddings:**
```python
# These are tracked as parameters automatically
self.emb_rel_k = mx.random.normal(shape) * stddev
self.emb_rel_v = mx.random.normal(shape) * stddev
```

## 9. Module State Dict / Weight Loading

### PyTorch
```python
# Save
torch.save(model.state_dict(), "model.pth")

# Load
state_dict = torch.load("model.pth")
model.load_state_dict(state_dict, strict=False)
```

### MLX
```python
# Save
weights = dict(model.parameters())
mx.savez("model.npz", **weights)

# Load (method 1: direct from file)
model.load_weights("model.npz", strict=False)

# Load (method 2: from dict)
weights = mx.load("model.npz")
model.load_weights(list(weights.items()), strict=False)
```

**Key Difference:** MLX's `load_weights` matches parameter paths in the module tree. Parameters must be assigned as `self.param_name` in `__init__` to be discoverable.

## 10. Attention Mask Convention

### PyTorch (common convention)
```python
# 0 = masked (ignore), 1 = attend
mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
scores = scores + (1.0 - mask) * -1e4
```

### MLX (our implementation)
```python
# Same convention: 0 = masked, 1 = attend
mask = mask[:, None, None, :]  # (B, 1, 1, T)
scores = mx.where(mask == 0, -1e4, scores)
```

## 11. Matrix Multiplication

### PyTorch
```python
# @ operator or torch.matmul
output = query @ key.transpose(-2, -1)
output = torch.matmul(attn, value)
```

### MLX
```python
# @ operator or mx.matmul
output = query @ key.transpose(0, 1, 3, 2)
output = mx.matmul(attn, value)
```

**Note:** Transpose dimensions are specified explicitly in MLX, no negative indexing.

## 12. Softmax

### PyTorch
```python
attn = F.softmax(scores, dim=-1)
attn = torch.softmax(scores, dim=-1)
```

### MLX
```python
attn = mx.softmax(scores, axis=-1)
```

**Note:** `dim` → `axis` parameter name change.

## 13. Type Casting

### PyTorch
```python
x = x.to(torch.float32)
x = x.long()
x = x.type(torch.int64)
```

### MLX
```python
x = x.astype(mx.float32)
x = x.astype(mx.int64)
```

## 14. Device Management

### PyTorch
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
x = x.to(device)
```

### MLX
```python
# No explicit device management
# MLX automatically uses GPU if available on Apple Silicon
# All operations run on unified memory architecture
```

## 15. Half Precision

### PyTorch
```python
model = model.half()  # Convert to fp16
x = x.half()
```

### MLX
```python
# MLX handles precision automatically based on hardware
# Can explicitly cast if needed:
x = x.astype(mx.float16)
```

## Summary Table

| Feature | PyTorch | MLX |
|---------|---------|-----|
| Conv1d data format | (B, C, T) | (B, T, C) |
| Conv1d weight format | (Out, In, K) | (Out, K, In) |
| ConvTranspose1d weight | (In, Out, K) | (Out, K, In) |
| Conv2d data format | (B, C, H, W) | (B, H, W, C) |
| Padding API | `F.pad(x, tuple)` | `mx.pad(x, pad_width=list)` |
| Softmax param | `dim` | `axis` |
| Device management | Explicit | Automatic |
| Gradient mode | `torch.no_grad()` | No gradients by default |
| Weight norm | Built-in | Must fuse manually |
| Module loading | `load_state_dict()` | `load_weights()` |

## Best Practices for Porting

1. **Use consistent dimension ordering** within modules to avoid confusion
2. **Add transposes at module boundaries** where interfacing with PyTorch format
3. **Fuse weight norm during conversion**, don't try to preserve separate parameters
4. **Test layer-by-layer** to catch dimension mismatches early
5. **Pay special attention to parameter naming** - embeddings vs weights
6. **Verify shapes at every step** in complex operations like attention
7. **Use MLX's automatic device management** - don't try to replicate PyTorch's device logic
8. **Check pad_width carefully** when porting padding operations
9. **Map axis/dim parameters** when porting operations
10. **Test with random inputs first** before using real weights

## Common Pitfalls

1. Forgetting to transpose Conv weights during conversion
2. Not handling gamma/beta LayerNorm parameters
3. Transposing embedding parameters that shouldn't be transposed
4. Using wrong pad_width dimensions
5. Assuming PyTorch state_dict keys will match MLX module paths
6. Not calling `mx.eval()` after loading weights
7. Mixing (B, C, T) and (B, T, C) formats within a module
8. Incorrect reshape dimensions in complex indexing operations
9. Not testing with actual weights (only random initialization)
10. Assuming numerical perfect equality (some minor differences are expected)

## Acceptable Numerical Differences

When porting, expect small numerical differences due to:
- Different floating-point operation orders
- Different random number generators
- Hardware-specific optimizations
- Precision handling differences

**Acceptable ranges:**
- Activations: RMSE < 0.01, max diff < 0.1
- Final output: Correlation > 0.99, RMSE < 0.01
- Weights: Should match exactly (diff < 1e-6) after conversion

**Red flags:**
- Correlation < 0.9
- RMSE > 0.1
- Output range significantly different
- NaN or Inf values
