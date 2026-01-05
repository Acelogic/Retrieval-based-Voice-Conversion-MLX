# Benchmarks and Testing Scripts

This folder contains various test and benchmark scripts used to validate and measure the performance of the MLX RMVPE implementation.

## Benchmark Scripts

### `benchmark_rmvpe.py`
Comprehensive benchmark comparing PyTorch (MPS) vs MLX RMVPE performance across various audio lengths (5s, 30s, 60s, 3min, 5min).

**Usage:**
```bash
export OMP_NUM_THREADS=1
python benchmarks/benchmark_rmvpe.py
```

**Results**: MLX is 1.78x faster on average (1.58x-1.88x speedup range)

## Test Scripts

### `test_full_pipeline.py`
End-to-end testing suite that validates:
- PyTorch RMVPE (baseline)
- MLX RMVPE (optimized)
- Full RVC inference pipeline

### `test_rmvpe_chunking.py` & `test_rmvpe_chunking_simple.py`
Validates chunking implementation for processing audio in 32k frame chunks.

## Debug Scripts

### `debug_with_weights.py`
Tests E2E model with actual loaded weights to validate architecture and inference.

### `debug_unet_channels.py`
Traces UNet channel dimensions through the network to debug shape mismatches.

## Running Tests

All tests should be run with the environment variable set:
```bash
export OMP_NUM_THREADS=1
python benchmarks/<script_name>.py
```

## Documentation

See the `docs/` folder for detailed implementation notes and bug reports.
