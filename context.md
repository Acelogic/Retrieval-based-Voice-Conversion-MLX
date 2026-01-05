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
| `torch` | PyTorch with MPS | 2.81s |
| `mlx` | Full MLX inference | **2.91s** |

## 🚀 TODO / Future Optimizations

### 1. Batch Processing in RMVPE mel2hidden
- [ ] Optimize `mel2hidden` to process the mel spectrogram in chunks for better GPU cache utilization and throughput.

### 2. Fused Operations in Hubert
- [ ] Profile and possibly fuse transformer blocks (Q/K/V projections, softmax, output projection) in Hubert using `mx.compile` more strategically or custom kernels.

### 3. Cache Warmup on Model Load
- [ ] Run a single dummy inference iteration immediately after loading models to trigger all MLX kernel compilation (JIT). This shifts the one-time "first run" penalty to the startup phase.

### 4. Proper End-to-End float16 Support
Currently, float16 caused a slowdown because of constant casting between float32 (audio/mel) and float16 (model). To fix:
- [ ] Convert input audio to `float16` immediately after loading.
- [ ] Update `mel_spectrogram` to output `float16`.
- [ ] Implement `tree_map` to cast all model parameters to `float16` at load time.
- [ ] Ensure the entire pipeline operates in `float16`, only casting back to `float32` for final storage.

### 5. Streaming Synthesis
- [ ] Implement overlapping chunk processing for the Synthesizer to reduce peak memory usage and potentially enable real-time/streaming output.

### 6. Remove librosa Dependency Entirely
- [x] Replaced librosa `to_mono` and `resample` with `scipy`/`numpy` in `load_audio_infer`.
- [ ] Investigate moving audio loading entirely to a more lightweight solution if `soundfile`/`scipy` overhead is still noticeable.

### 7. Custom Metal Kernels
- [ ] For absolute peak performance, write optimized Metal shaders for the most compute-intensive operations if `mx.compile` isn't sufficient.

### 8. Quantization (INT8/INT4)
- [ ] Explore `mlx.nn.QuantizedLinear` for the Synthesizer model to reduce memory bandwidth requirements.
