# Development Tools

This folder contains utility scripts for development, debugging, and inspection of the MLX implementation.

## Scripts

### `check_mlx_ops.py`
Checks for the availability of specific MLX operations in the installed MLX version.

**Usage:**
```bash
python tools/check_mlx_ops.py
```

**Purpose:**
- Verifies if `mx.conv_transpose2d` exists
- Checks if `nn.ConvTranspose2d` is available
- Useful for debugging version compatibility issues

### `inspect_hubert_keys.py`
Inspects and compares HuBERT model parameters with converted weight files.

**Usage:**
```bash
python tools/inspect_hubert_keys.py
```

**Purpose:**
- Lists model parameters from the HubertModel class
- Lists weight keys from the converted `hubert_mlx.npz` file
- Compares model structure vs weight file structure
- Identifies missing or extra parameters
- Useful for debugging weight loading issues

### `test_mel.py`
Test script for verifying Mel Spectrogram computation between Python and Swift.

**Usage:**
```bash
python tools/test_mel.py
```

**Purpose:**
- Compares Mel filterbank construction
- Validates STFT computation
- Checks log-mel spectrogram output
- Useful for debugging RMVPE preprocessing differences

### `test_bn.swift`
Test script for examining BatchNorm parameter exposure in MLX Swift.

**Purpose:**
- Tests if BatchNorm exposes running statistics via `parameters()`
- Verifies parameter loading behavior
- Standalone Swift test for BatchNorm debugging

### `test_conv_trans.swift`
Test script for ConvTranspose2d weight layout verification.

**Purpose:**
- Tests ConvTranspose2d weight shape expectations
- Verifies weight transposition logic
- Quick test for decoder weight issues

### `test_stride.swift`
Test script for verifying stride and convolution output shapes.

**Purpose:**
- Tests Conv2d and pooling stride behavior
- Validates output shape calculations
- Debugging spatial dimension mismatches

## When to Use These Tools

- **After MLX updates**: Run `check_mlx_ops.py` to verify API compatibility
- **Weight conversion issues**: Use `inspect_hubert_keys.py` to debug parameter mismatches
- **Model architecture changes**: Verify that model structure matches weight files
- **RMVPE debugging**: Use `test_mel.py` to verify preprocessing parity
- **BatchNorm issues**: Use `test_bn.swift` to check running stats loading
- **Shape mismatches**: Use `test_stride.swift` or `test_conv_trans.swift` to isolate issues

## Related Folders

- `benchmarks/` - Performance testing and validation scripts
- `rvc/lib/mlx/` - Weight conversion scripts (`convert.py`, `convert_hubert.py`, `convert_rmvpe.py`)
