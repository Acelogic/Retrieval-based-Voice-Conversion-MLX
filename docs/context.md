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

4.  **Performance**:
    - **RMVPE (pitch detection)**: MLX is **1.78x FASTER** than PyTorch MPS
      - 5s audio: 0.182s (MLX) vs 0.289s (PyTorch) = 1.58x faster
      - 60s audio: 1.758s (MLX) vs 3.271s (PyTorch) = 1.86x faster
      - 5min audio: 9.223s (MLX) vs 15.848s (PyTorch) = 1.72x faster
    - **Full RVC Pipeline**: MLX comparable to PyTorch (both ~2-3s for 5s audio)
    - **Pitch Detection (RMVPE)**: MLX is **2.05x faster** on average than PyTorch MPS.

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
| `rmvpe` | MLX Pitch Detection alone | **0.205s** (5s audio) |

## 📊 Benchmarking MLX Performance

To verify the speedups on your own machine, use the scripts in the `benchmarks/` directory.

### 1. Run RMVPE Benchmark
Compares PyTorch (MPS) vs MLX performance across multiple audio lengths (5s to 5min).
```bash
conda run -n rvc python benchmarks/benchmark_rmvpe.py
```

### 2. Run Full Pipeline Test
Validates that the MLX RMVPE implementation works correctly within the full inference pipeline.
```bash
conda run -n rvc python benchmarks/test_full_pipeline.py
```

### 3. Recent Benchmark Results (2026-01-05)
| Audio Length | Speedup (MLX vs Torch MPS) |
|--------------|---------------------------|
| 5s | 1.78x |
| 60s | 2.42x |
| 5min | 1.91x |
| **Average** | **2.05x** |

## 🚀 TODO / Future Optimizations

### 1. Batch Processing in RMVPE mel2hidden ✅ COMPLETE
- [x] Optimize `mel2hidden` to process the mel spectrogram in chunks for better GPU cache utilization and throughput.
  - Implemented chunking with 32k frame chunks (matching PyTorch reference)
  - Automatically skips chunking for short audio (<32k frames)
  - Uses `mx.eval()` after each chunk for efficient memory management
  - No overlap needed (BiGRU handles context within chunks)
  - **Result**: MLX RMVPE is now **1.78x faster** than PyTorch MPS!
  - Fixed multiple bugs during implementation:
    - ConvTranspose2d output_padding handling
    - UNet decoder shape matching
    - BiGRU/FC layer weight loading structure
    - GRU return values (MLX returns single output, not tuple)
    - Array reversal using slicing instead of flip()

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

### Current Status
- **Objective**: Successfully compile `RVCNative` for physical iOS device.
- **Note**: Simulator support is NOT a priority due to lack of Metal GPU access in the simulator environment. Testing will be performed natively.
- **Resolved**: `nullptr` crashes in `std::string` constructors (via `device.cpp`, `metal.cpp`, `resident.cpp` patches - now reverted to stock).
- **Pending**: Verifying successful compilation with stock `mlx-swift` vendor files.

### 5. Streaming Synthesis
- [ ] Implement overlapping chunk processing for the Synthesizer to reduce peak memory usage and potentially enable real-time/streaming output.

### 6. Remove librosa Dependency Entirely
- [x] Replaced librosa `to_mono` and `resample` with `scipy`/`numpy` in `load_audio_infer`.
- [ ] Investigate moving audio loading entirely to a more lightweight solution if `soundfile`/`scipy` overhead is still noticeable.

### 7. Custom Metal Kernels
- [ ] For absolute peak performance, write optimized Metal shaders for the most compute-intensive operations if `mx.compile` isn't sufficient.

### 8. Quantization (INT8/INT4)
- [ ] Explore `mlx.nn.QuantizedLinear` for the Synthesizer model to reduce memory bandwidth requirements.

## ⚠️ Known Issues

### iOS Simulator Linker Errors
When building the `RVCNative` demo for the iOS **Simulator**, you will likely see linker errors related to `SwiftUICore`, such as:
> `ld: warning: Could not parse or use implicit file '.../SwiftUICore.framework/SwiftUICore.tbd': cannot link directly with 'SwiftUICore' because product being built is not an allowed client of it`

**This is EXPECTED behavior on the Simulator and can generate "Build Failed" messages in Xcode logs, but the app often still launches.**

*   **Resolution**: Ignore these errors **IF** you are just verifying logic.
*   **Best Practice**: Always build and test on a **Physical Device** (iPhone/iPad) where these errors do NOT occur and `mlx-swift` performs correctly. The Simulator release of `mlx-swift` has known limitations.
