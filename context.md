# Project Context & Session Summary

**Date:** 2026-01-05
**Objective:** Add native Apple Silicon (MLX) inference support to RVC CLI.

## Accomplishments

### MLX Pipeline (`--backend mlx`) ✅ COMPLETE
1.  **Core Components** in `rvc/lib/mlx/`:
    *   `modules.py`, `attentions.py`, `residuals.py`, `generators.py`, `encoders.py`, `synthesizers.py`
    *   `hubert.py`: Full HuBERT encoder
    *   `rmvpe.py`: E2E pitch detection with DeepUnet + **GPU-native mel spectrogram**

2.  **Weight Converters**: `convert.py`, `convert_hubert.py`, `convert_rmvpe.py`

3.  **Custom Implementations**: `BiGRU`, `ConvTranspose1d`, `ConvTranspose2d`, **MLX FFT mel spectrogram**

4.  **Performance**: MLX **0.5% FASTER** than PyTorch (3.12s vs 3.14s)

## Key Optimization: MLX-Native Mel Spectrogram
Replaced librosa CPU-based mel spectrogram (645ms first call) with GPU-accelerated implementation using:
- `mx.fft.rfft` for Fast Fourier Transform
- Pre-computed mel filterbank matrix
- Hann window

## Critical "Tidbits"

### Model Locations
> **`/Users/mcruz/Library/Application Support/Replay/com.replay.Replay/models`**

### Environment Variables
*   **`export OMP_NUM_THREADS=1`**: MANDATORY to prevent faiss segfault.

### Runtime Environment
*   **Conda Environment**: `conda run -n rvc python rvc_cli.py ...`

### Backend Selection
| Backend | Description | Performance |
|---------|-------------|-------------|
| `torch` | PyTorch with MPS | 3.14s |
| `mlx` | Full MLX inference | **3.12s** (-0.5%) |
