# MLX Training Stability Fixes

This document explains the issues encountered while implementing RVC training in MLX and the fixes applied to achieve stable training.

## Overview

Training RVC models involves a GAN-style setup with:
- **Generator (Synthesizer)**: Converts phone embeddings + pitch to audio
- **Discriminator (MultiPeriodDiscriminator)**: Distinguishes real vs generated audio

The main challenge was preventing NaN values from appearing during training, which would cause the model to diverge.

---

## Issue 1: Gradient Overflow Causing NaN

### Problem

During adversarial training, gradients could contain `inf` values (from KL divergence or other computations). MLX's `clip_grad_norm` computes:

```python
total_norm = sqrt(sum(g^2))
clip_coef = max_norm / total_norm
scaled_grads = grads * clip_coef
```

When any gradient contains `inf`:
1. `total_norm = inf`
2. `clip_coef = max_norm / inf = 0`
3. All gradients get multiplied by 0
4. After optimizer update with zeroed gradients + weight decay, parameters become NaN

### Symptoms

```
Step 1/10
  Loss: total=45.802 (gen=9.005, fm=1.660, mel=34.300, kl=0.837)
  Grad norm: inf
  AFTER UPDATE: NaN in generator.emb_g.weight
```

The gradient norm was `inf` even though individual gradient arrays showed no NaN or Inf when checked element-wise. This was because extremely large (but finite) values overflowed when squared and summed.

### Solution

Created `sanitize_gradients()` function that runs before gradient clipping:

```python
def sanitize_gradients(grads: Dict, max_grad_value: float = 1e3) -> Tuple[Dict, bool]:
    """
    Sanitize gradients by replacing NaN/Inf with finite values.
    """
    def sanitize(g):
        if isinstance(g, dict):
            return {k: sanitize(v) for k, v in g.items()}
        elif hasattr(g, 'shape'):
            mx.eval(g)
            # Replace nan with 0, inf with max_value
            g_safe = mx.where(mx.isnan(g), mx.zeros_like(g), g)
            g_safe = mx.where(mx.isinf(g) & (g > 0), mx.full(g.shape, max_grad_value), g_safe)
            g_safe = mx.where(mx.isinf(g) & (g < 0), mx.full(g.shape, -max_grad_value), g_safe)
            # Clamp extreme values
            g_clamped = mx.clip(g_safe, -max_grad_value, max_grad_value)
            return g_clamped
        return g
    return sanitize(grads), had_issues
```

This ensures `clip_grad_norm` receives finite gradients and can properly scale them.

**File:** `rvc_mlx/train/trainer.py`

---

## Issue 2: Dimension Ordering in slice_segments

### Problem

The `slice_segments` function in `commons.py` assumes MLX format `(B, T, C)` by default, but the Synthesizer's internal tensors use PyTorch format `(B, C, T)` for compatibility with the flow and decoder.

### Symptoms

```
ValueError: Cannot reshape array of size 0 into shape (2, 200)
```

Slicing along the wrong dimension resulted in empty or incorrectly shaped tensors.

### Solution

Added `time_first` parameter to `slice_segments` and `rand_slice_segments`:

```python
def slice_segments(x: mx.array, ids_str: mx.array, segment_size: int, time_first: bool = True):
    """
    Args:
        time_first: If True, assumes (B, T, C) MLX format. If False, assumes (B, C, T) PyTorch format.
    """
    if time_first:
        # MLX format: (B, T, C) - slice along dim 1
        segment = x[i, start_idx:start_idx + segment_size, :]
    else:
        # PyTorch format: (B, C, T) - slice along dim 2
        segment = x[i, :, start_idx:start_idx + segment_size]
```

In `synthesizers.py`, calls were updated to use `time_first=False`:

```python
z_slice, ids_slice = rand_slice_segments(z, z_lengths, segment_size_frames, time_first=False)
```

**Files:** `rvc_mlx/lib/mlx/commons.py`, `rvc_mlx/lib/mlx/synthesizers.py`

---

## Issue 3: generator_loss Return Type

### Problem

The `generator_loss` function returns a scalar `mx.array`, but code was trying to unpack it as a tuple:

```python
loss_gen, _ = generator_loss(y_d_gs)  # WRONG - IndexError: vector
```

### Solution

```python
loss_gen = generator_loss(y_d_gs)  # Correct - returns scalar
```

**File:** `rvc_mlx/train/losses.py` returns scalar, callers updated accordingly.

---

## Issue 4: KL Loss Scale

### Problem

The KL divergence loss could produce very large gradients when `z_p` values had a wide range (e.g., -25 to +24). This contributed to gradient explosion.

### Solution

Scale KL loss by a factor (0.01-1.0 depending on training stage):

```python
loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, x_mask) * KL_SCALE
```

In the trainer config:
```python
c_kl: float = 1.0  # KL loss weight (can reduce if gradients explode)
```

**File:** `rvc_mlx/train/trainer.py`

---

## Issue 5: Encoder Freezing

### Problem

When fine-tuning from pretrained weights, training the text encoder (`enc_p`) can destabilize the model since it's already learned good representations.

### Solution

Freeze the encoder during fine-tuning:

```python
if config.freeze_encoder and config.is_finetuning:
    if hasattr(net_g, 'enc_p'):
        net_g.enc_p.freeze()
        print("Frozen: enc_p (TextEncoder)")
```

**File:** `rvc_mlx/train/trainer.py`

---

## Issue 6: Discriminator Learning Rate

### Problem

The discriminator can learn faster than the generator, causing mode collapse where the generator can't fool the discriminator.

### Solution

Use a lower learning rate for the discriminator:

```python
d_lr_scale: float = 0.2  # Discriminator LR = learning_rate * d_lr_scale
```

Also skip discriminator updates when it's winning too easily:

```python
d_loss_threshold: float = 1.0  # Skip D update if loss < threshold
```

**File:** `rvc_mlx/train/trainer.py`

---

## Training Configuration

Final working configuration:

```python
@dataclass
class TrainingConfig:
    # Training
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.8, 0.99)
    eps: float = 1e-9
    lr_decay: float = 0.999875

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Loss weights
    c_mel: float = 45.0
    c_kl: float = 1.0

    # Discriminator balance
    d_lr_scale: float = 0.2
    d_loss_threshold: float = 1.0

    # Fine-tuning
    freeze_encoder: bool = True
```

---

## Verification

### Test 1: Non-Adversarial Training (mel + kl only)

```bash
python -c "
from rvc_mlx.train.trainer import RVCTrainer, TrainingConfig
config = TrainingConfig(disable_adversarial=True, epochs=2)
# ... setup and train
"
```

Result: 2 epochs completed without NaN, loss decreased from 425 to 285.

### Test 2: Full Adversarial Training

```bash
python debug_nan_step2.py
```

Result: 50 steps completed without NaN. Losses volatile early but numerically stable.

---

## Key Takeaways

1. **Always sanitize gradients before clipping** - MLX's `clip_grad_norm` doesn't handle inf values gracefully.

2. **Be explicit about tensor formats** - MLX uses `(B, T, C)` while PyTorch uses `(B, C, T)`. Add parameters to clarify which format functions expect.

3. **Balance GAN training carefully** - Use lower discriminator learning rate and skip updates when D is winning.

4. **Freeze pretrained components** - When fine-tuning, freeze components that already have good representations.

5. **Evaluate immediately after updates** - Call `mx.eval()` after optimizer updates to catch NaN early and prevent accumulation.

---

## Files Modified

| File | Changes |
|------|---------|
| `rvc_mlx/train/trainer.py` | Added `sanitize_gradients()`, encoder freezing, D learning rate scaling |
| `rvc_mlx/lib/mlx/commons.py` | Added `time_first` parameter to slice functions |
| `rvc_mlx/lib/mlx/synthesizers.py` | Updated slice calls to use `time_first=False` |
| `rvc_mlx/train/losses.py` | Verified return types |
| `rvc_mlx/train/discriminators.py` | Added grouped convolution support |

---

## References

- Original RVC training: `rvc/train/train.py`
- PyTorch discriminator: `rvc/lib/algorithm/discriminators.py`
- MLX gradient clipping: `mlx.optimizers.clip_grad_norm`
