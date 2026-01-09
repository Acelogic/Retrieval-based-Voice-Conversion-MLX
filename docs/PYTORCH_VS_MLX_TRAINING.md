# PyTorch vs MLX Training Pipeline

A practical guide to the differences encountered when porting RVC training from PyTorch to MLX.

---

## 1. Gradient Computation

### PyTorch
```python
# Automatic backward pass
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

### MLX
```python
# Explicit value_and_grad pattern
loss_fn = nn.value_and_grad(model, compute_loss)
(loss, aux), grads = loss_fn(model)
optimizer.update(model, grads)
mx.eval(model.parameters(), optimizer.state)  # Trigger computation
```

**Key Differences:**
- MLX uses lazy evaluation - nothing computes until `mx.eval()` is called
- Gradients are returned as a nested dict matching model structure
- No need for `zero_grad()` - each `value_and_grad` call produces fresh gradients
- Must explicitly evaluate after updates to trigger computation

---

## 2. Tensor Dimension Ordering

### PyTorch
```python
# Conv1d: (Batch, Channels, Time)
audio = torch.randn(4, 1, 16000)  # (B, C, T)
conv = nn.Conv1d(1, 64, kernel_size=7)
out = conv(audio)  # (B, 64, T')
```

### MLX
```python
# Conv1d: (Batch, Time, Channels)
audio = mx.random.normal((4, 16000, 1))  # (B, T, C)
conv = nn.Conv1d(1, 64, kernel_size=7)
out = conv(audio)  # (B, T', 64)
```

**Implications:**
- Transpose at module boundaries when porting PyTorch code
- Slicing operations need different axis indices
- Concatenation along channel dimension differs

```python
# PyTorch: concat along dim 1 (channels)
torch.cat([a, b], dim=1)

# MLX: concat along dim 2 (channels in last position)
mx.concatenate([a, b], axis=2)
```

---

## 3. Weight Shapes

### Conv1d Weights

| Framework | Weight Shape |
|-----------|--------------|
| PyTorch | `(out_channels, in_channels, kernel_size)` |
| MLX | `(out_channels, kernel_size, in_channels)` |

### Linear Weights

| Framework | Weight Shape |
|-----------|--------------|
| PyTorch | `(out_features, in_features)` |
| MLX | `(in_features, out_features)` |

**Weight Conversion:**
```python
# Conv1d: transpose last two dims
mlx_weight = pytorch_weight.transpose(0, 2, 1)

# Linear: transpose
mlx_weight = pytorch_weight.T
```

---

## 4. Gradient Clipping

### PyTorch
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### MLX
```python
from mlx.optimizers import clip_grad_norm
grads, norm = clip_grad_norm(grads, max_norm=1.0)
```

**Critical Issue:** MLX's `clip_grad_norm` doesn't handle `inf` values well:
- When any gradient is `inf`, norm becomes `inf`
- `clip_coef = max_norm / inf = 0`
- All gradients get zeroed, causing NaN after update

**Solution:** Sanitize gradients before clipping:
```python
def sanitize_gradients(grads, max_value=1e3):
    """Replace inf/nan with finite values before clipping."""
    def sanitize(g):
        if isinstance(g, dict):
            return {k: sanitize(v) for k, v in g.items()}
        elif hasattr(g, 'shape'):
            g = mx.where(mx.isnan(g), mx.zeros_like(g), g)
            g = mx.where(mx.isinf(g), mx.full(g.shape, max_value) * mx.sign(g), g)
            return mx.clip(g, -max_value, max_value)
        return g
    return sanitize(grads)
```

---

## 5. In-Place Operations

### PyTorch
```python
# In-place operations common
x += 1
x.mul_(2)
x[:, 0] = 0
```

### MLX
```python
# No in-place operations - always creates new array
x = x + 1
x = x * 2
# Slice assignment not supported - use mx.where or reconstruct
```

**Implications:**
- Memory usage patterns differ
- Some PyTorch patterns need restructuring

---

## 6. Random Number Generation

### PyTorch
```python
torch.manual_seed(42)
torch.randn(shape)
torch.randint(0, 10, shape)
```

### MLX
```python
mx.random.seed(42)
mx.random.normal(shape)
mx.random.randint(0, 10, shape)
```

**Key Difference:** MLX random functions return arrays directly, no need for `.to(device)`.

---

## 7. Device Management

### PyTorch
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data = data.to(device)
```

### MLX
```python
# No explicit device management needed
# MLX automatically uses Metal GPU on Apple Silicon
# Arrays are created on the default device
```

**Benefit:** Simpler code, no `.to(device)` calls everywhere.

---

## 8. Model Freezing

### PyTorch
```python
for param in model.encoder.parameters():
    param.requires_grad = False
```

### MLX
```python
model.encoder.freeze()
# Or selectively:
model.encoder.freeze(recurse=True, keys=None)
```

**Note:** MLX's `freeze()` method is cleaner but behaves slightly differently - it marks the module as frozen rather than individual parameters.

---

## 9. Optimizer State

### PyTorch
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
# State is internal to optimizer
optimizer.state_dict()  # For checkpointing
```

### MLX
```python
optimizer = optim.AdamW(learning_rate=1e-4)
optimizer.update(model, grads)
# State must be explicitly evaluated
mx.eval(optimizer.state)
```

**Key Differences:**
- MLX optimizers don't take parameters at construction
- Must pass model and grads to `update()`
- State is a public attribute that must be evaluated

---

## 10. Loss Function Returns

### PyTorch
```python
# Common pattern: return multiple values
def compute_loss(model, batch):
    ...
    return loss, (aux1, aux2, aux3)

loss, aux = compute_loss(model, batch)
loss.backward()
```

### MLX
```python
# Same pattern works, but with value_and_grad
def compute_loss(model, batch):
    ...
    return loss, (aux1, aux2, aux3)

loss_fn = nn.value_and_grad(model, lambda m: compute_loss(m, batch))
(loss, aux), grads = loss_fn(model)
```

**Note:** Auxiliary returns don't get gradients computed - only the first return value does.

---

## 11. Spectral Operations (FFT/STFT)

### PyTorch
```python
spec = torch.stft(audio, n_fft=2048, hop_length=512,
                  win_length=2048, window=hann_window,
                  return_complex=True)
mag = spec.abs()
```

### MLX
```python
# MLX doesn't have built-in STFT - must implement manually
def stft(x, n_fft, hop_length, win_length, window):
    # Frame the signal
    # Apply window
    # Compute FFT per frame
    frames = frame_signal(x, win_length, hop_length)
    windowed = frames * window
    spec = mx.fft.rfft(windowed, n=n_fft)
    return spec
```

**Implication:** STFT must be implemented from scratch or use a library. See `rvc_mlx/train/mel_processing.py`.

---

## 12. DataLoader Patterns

### PyTorch
```python
from torch.utils.data import DataLoader, Dataset

dataset = MyDataset(...)
loader = DataLoader(dataset, batch_size=4, shuffle=True,
                    num_workers=4, pin_memory=True)

for batch in loader:
    batch = {k: v.to(device) for k, v in batch.items()}
```

### MLX
```python
# Custom DataLoader implementation
class DataLoader:
    def __iter__(self):
        for batch in self._get_batches():
            # Convert numpy to mx.array
            yield {k: mx.array(v) for k, v in batch.items()}
```

**Key Differences:**
- No built-in DataLoader in MLX
- No automatic multiprocessing (must implement if needed)
- No `pin_memory` concept (Metal has unified memory)

---

## 13. Checkpointing

### PyTorch
```python
# Save
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
}, 'checkpoint.pt')

# Load
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model'])
```

### MLX
```python
# Save weights (must flatten nested dicts)
weights = flatten_dict(model.parameters())
mx.savez('model.npz', **weights)

# Save training state separately (as JSON)
state = {'epoch': epoch, 'loss': loss}
json.dump(state, open('state.json', 'w'))

# Load
weights = dict(mx.load('model.npz'))
model.load_weights(list(weights.items()))
```

**Key Differences:**
- MLX uses `.npz` format (numpy compatible)
- Optimizer state saved separately
- Must handle nested parameter dicts manually

---

## 14. Debugging NaN/Inf

### PyTorch
```python
torch.autograd.set_detect_anomaly(True)
# Will print stack trace when NaN is produced
```

### MLX
```python
# No built-in anomaly detection
# Must manually check:
def check_tensor(x, name):
    mx.eval(x)
    if mx.isnan(x).any().item():
        print(f"NaN in {name}")
    if mx.isinf(x).any().item():
        print(f"Inf in {name}")
```

**Recommendation:** Add explicit checks at key points during debugging.

---

## 15. Memory Management

### PyTorch
```python
# Explicit cache clearing
torch.cuda.empty_cache()

# Gradient checkpointing for large models
from torch.utils.checkpoint import checkpoint
```

### MLX
```python
# Clear Metal cache
mx.metal.clear_cache()

# Delete intermediate tensors
del intermediate_tensor

# No built-in gradient checkpointing
```

**Note:** MLX's lazy evaluation helps with memory - intermediate results aren't computed until needed.

---

## 16. Mixed Precision Training

### PyTorch
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = model(batch)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### MLX
```python
# MLX automatically uses appropriate precision
# float16 operations are native on Apple Silicon
# No explicit mixed precision API needed

# Can explicitly cast if needed:
x = x.astype(mx.float16)
```

---

## Summary Table

| Aspect | PyTorch | MLX |
|--------|---------|-----|
| Gradient computation | `loss.backward()` | `nn.value_and_grad()` |
| Evaluation | Eager by default | Lazy, needs `mx.eval()` |
| Tensor format | `(B, C, T)` | `(B, T, C)` |
| Device management | Explicit `.to(device)` | Automatic |
| In-place ops | Supported | Not supported |
| DataLoader | Built-in, multiprocess | Must implement |
| STFT | `torch.stft()` | Must implement |
| Anomaly detection | Built-in | Manual checks |
| Mixed precision | `autocast` | Automatic |

---

## Porting Checklist

When porting PyTorch training to MLX:

- [ ] Replace `backward()` with `value_and_grad()`
- [ ] Add `mx.eval()` calls after optimizer updates
- [ ] Transpose tensor dimensions at boundaries
- [ ] Transpose weight shapes during conversion
- [ ] Implement custom DataLoader
- [ ] Implement STFT if needed
- [ ] Add gradient sanitization before clipping
- [ ] Remove in-place operations
- [ ] Remove `.to(device)` calls
- [ ] Add manual NaN/Inf checks for debugging
