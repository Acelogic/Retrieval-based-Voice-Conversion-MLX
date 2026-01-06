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
- **Resolved**:
    - **Compilation**: Successfully building with stock `mlx-swift` submodules.
    - **Channel Mismatch**: Fixed `HubertModel` to return 768-dim features and removed incorrect transpose in `RVCInference.swift`.
    - **Silent Audio**: Fixed by updating `AudioProcessor.swift` to save as **Int16 PCM** (was Float32).
- **Pending**: Full end-to-end verification on a physical iOS Device.

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

### iOS Simulator Support & Limitations
**Critical "Revelations" regarding Simulator use:**
1.  **Stock Library Instability**: Using the *stock*, unmodified `mlx-swift` library on the simulator is highly unstable. The C++ backend frequently crashes with `nullptr` assertions in `std::string` constructors (e.g., in `device.cpp` or `metal.cpp`) because it tries to access Metal device properties that are invalid or null in the simulator environment.
2.  **Force CPU**: While `MLX.Device.setDefault(device: Device.cpu)` *can* bypass some Metal crashes, the stock library's internal initialization sequences still often trigger the `nullptr` issues described above.
3.  **Conclusion**: **Physical Device testing is mandatory.** The Simulator should only be used for basic UI layout checks or initial compilation verification. Any functional testing of the MLX inference pipeline must be done on a real device to avoid "fighting" the simulator's lack of proper Metal support and the stock library's fragility in that environment.

### iOS Audio Inference - Scrambled Output (2026-01-05)

**Status**: 🔴 UNRESOLVED - Audio output remains distorted/scrambled on physical device.

**Root Cause Identified**: The Swift implementation was missing critical components:
1. **Missing TextEncoder (`enc_p`)**: Swift Generator expected 768-dim input, but model weights expect 192-dim (transformed by `enc_p.emb_phone`)
2. **Missing Flow Module**: Voice conversion requires `ResidualCouplingBlock` for proper feature transformation

**Weight Format Issues Discovered**:

| Weight Type | Format in Safetensors | Notes |
|-------------|----------------------|-------|
| `flow.*` | PyTorch (out, in, kernel) | Needs transpose to MLX (out, kernel, in) |
| `dec.cond`, `dec.ups`, `dec.noise_convs` | PyTorch | Needs transpose |
| `enc_p.proj` | PyTorch | Needs transpose |
| `enc_p.encoder.attn_*` | MLX | Already correct |
| `enc_p.encoder.ffn_*` | MLX | Already correct |
| `dec.conv_pre`, `dec.resblocks` | MLX | Already correct |

**Auto-Detection Rule**: If `shape[2] < shape[1]` for 3D weight, transpose from PyTorch→MLX.

**Flow Layer Indexing Issue**:
- Model weights use indices `0, 2, 4, 6` (Flip modules at odd indices)
- Swift sequential creation uses `0, 1, 2, 3`
- Requires key remapping during loading

**Attempted Fixes (All Still Producing Scrambled Audio)**:
1. ✅ Feature upsampling interpolation (broadcast → linear interp)
2. ✅ Voiced mask axis fix in SineGenerator (axis 2 → axis 1)
3. ✅ Full Synthesizer implementation (TextEncoder, Flow, Generator)
4. ✅ Conv1d weight auto-transposition based on shape
5. ✅ Flow layer key remapping (0,2,4,6 → 0,1,2,3)
6. ✅ Simplified pipeline (just Linear projection + Generator)

**Recommended Debugging Approach**:
1. **Verify Python MLX first**: Run CLI inference to confirm Python implementation works
2. **Tensor comparison**: Save intermediate tensors (HuBERT output, F0, etc.) from both Python and Swift, compare at each stage
3. **Isolate component**: Test each component (HuBERT, F0, Generator) independently
4. **Check weight loading**: Print loaded weight shapes vs expected shapes to verify mapping

**Key Files Modified**:
- `RVCInference.swift` - Main inference orchestration  
- `RVCModel.swift` - Generator with Conv1d/ConvTranspose1d
- `Synthesizer.swift` - Full TextEncoder, Flow, ResidualCouplingBlock (created but unused)

**Possible Remaining Issues**:
- HuBERT output format differences (transposition?)
- F0 scaling or format mismatch
- Subtle differences in Conv1d padding between Python MLX and Swift MLX
- Weight loading not matching expected module hierarchy

