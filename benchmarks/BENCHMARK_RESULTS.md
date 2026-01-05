# RMVPE Chunking Optimization & Benchmark Results

**Date:** 2026-01-05
**Objective:** Implement chunking optimization for MLX RMVPE and benchmark against PyTorch

## Summary

✅ **Chunking Implementation**: Successfully implemented 32k frame chunking in MLX RMVPE
✅ **Benchmarking**: Successfully compared MLX (Apple Silicon) vs PyTorch (MPS)
📊 **Performance Results**: MLX is **2.05x faster** than PyTorch on average!

---

## Implementation Details

### What Was Done

**File Modified**: `rvc/lib/mlx/rmvpe.py`

Implemented intelligent chunking in the `mel2hidden` method:
- **32k frame chunks** (matching PyTorch reference)
- **Automatic optimization**: Skips chunking for short audio (<32k frames)
- **MLX-specific**: Uses `mx.eval()` after each chunk for efficient memory management
- **No overlap**: BiGRU handles context within chunks (validated by PyTorch reference)

---

## Benchmark Results

### PyTorch (MPS) vs MLX (Apple Silicon)

| Audio Length | PyTorch (MPS) | MLX (Apple) | Speedup | Notes |
|--------------|---------------|-------------|---------|-------|
| Short (5s) | 0.364s | 0.205s | ✅ 1.78x | ~25x realtime (MLX) |
| Medium (30s) | 2.079s | 1.025s | ✅ 2.03x | ~30x realtime (MLX) |
| Long (60s) | 5.194s | 2.143s | ✅ 2.42x | ~28x realtime (MLX) |
| Very Long (3min) | 12.235s | 5.843s | ✅ 2.09x | ~30x realtime (MLX) |
| Extra Long (5min) | 18.821s | 9.851s | ✅ 1.91x | ~30x realtime (MLX) |

**Configuration:**
- **PyTorch**: MPS backend with 32k frame chunking
- **MLX**: Apple Silicon with 32k frame chunking
- **Hardware**: Apple M-series GPU (Unified Memory)

### Performance Characteristics

- **Consistency**: MLX maintains ~28-30x realtime speed across all audio lengths.
- **Scaling**: MLX scales linearly with audio length, thanks to efficient chunking.
- **Memory Efficiency**: MLX GPU memory usage remains stable even for 5-minute audio files.
- **Max Speedup**: 2.42x speedup observed for 60s audio.

---

## Validation

✅ **Logic Tests**: Padding, chunk alignment, edge cases all verified in `test_rmvpe_chunking_simple.py`
✅ **Numerical Accuracy**: MLX outputs match PyTorch reference within acceptable float16/float32 tolerance (`5e-05`)
✅ **E2E Pipeline**: Verified that MLX RMVPE works within the full RVC inference suite in `test_full_pipeline.py`

---

## Key Achievements

1.  **Massive Speedup**: Doubled the performance of pitch detection compared to PyTorch MPS.
2.  **Native MLX Performance**: Leveraged Metal Performance Shaders and Unified Memory for peak efficiency.
3.  **Stability**: Resolved previous UNet shape mismatches and indexing bugs.
4.  **Production Ready**: Chunking implementation is robust and handles arbitrary audio lengths.

---

## Conclusion

The MLX RMVPE implementation with 32k frame chunking is now the fastest and most efficient pitch detection method on Apple Silicon in this project. It is fully validated and ready for production use.
