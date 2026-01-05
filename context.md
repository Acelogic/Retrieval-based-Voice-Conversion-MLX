# Project Context & Session Summary

**Date:** 2026-01-05
**Objective:** Add native Apple Silicon (MLX) inference support to RVC CLI.

## Accomplishments

### MLX Pipeline (`--backend mlx`) ✅ COMPLETE
1.  **Core Components** in `rvc/lib/mlx/`:
    *   `modules.py`: WaveNet
    *   `attentions.py`: MultiHeadAttention, FFN
    *   `residuals.py`: ResBlock, ResidualCouplingBlock
    *   `generators.py`: HiFiGANNSFGenerator, SineGenerator
    *   `encoders.py`: TextEncoder, PosteriorEncoder
    *   `synthesizers.py`: Synthesizer
    *   `hubert.py`: Full HuBERT encoder
    *   `rmvpe.py`: E2E pitch detection with DeepUnet

2.  **Weight Converters**:
    *   `convert.py`: RVC Synthesizer weights
    *   `convert_hubert.py`: HuBERT embedder weights
    *   `convert_rmvpe.py`: RMVPE pitch predictor weights

3.  **Custom Implementations** (MLX lacks native support):
    *   `BiGRU`: Bidirectional GRU wrapper
    *   `ConvTranspose1d` / `ConvTranspose2d`: Zero-insertion + convolution

4.  **Performance**: ~2.97s inference on Apple Silicon (comparable to PyTorch MPS)

## Critical "Tidbits" for Future Sessions

### 1. Model Locations
> **`/Users/mcruz/Library/Application Support/Replay/com.replay.Replay/models`**

### 2. Environment Variables
*   **`export OMP_NUM_THREADS=1`**: MANDATORY on macOS to prevent `faiss` segfault.

### 3. Runtime Environment
*   **Conda Environment**: `conda run -n rvc python rvc_cli.py ...`

### 4. Weight Conversion Commands
```bash
# Convert Hubert weights (one-time)
python rvc/lib/mlx/convert_hubert.py

# Convert RMVPE weights (one-time)
python rvc/lib/mlx/convert_rmvpe.py
```

### 5. Backend Selection
| Backend | Description |
|---------|-------------|
| `torch` | Pure PyTorch with MPS (default) |
| `mlx` | Full MLX inference (Hubert, RMVPE, Synthesizer) |

### 6. Implementation Details
*   **Data Layout**: MLX uses `(N, L, C)` (Channels Last).
*   **GRU Bias**: MLX GRU has `b` (3*H) and `bhn` (H). PyTorch `bias_hh` sliced for `bhn`.

## Next Steps
*   **Numerical Validation**: Compare output quality between backends.
*   **Optimization**: Profile and optimize MLX kernels if needed.
