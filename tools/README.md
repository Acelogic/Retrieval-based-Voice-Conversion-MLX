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

## When to Use These Tools

- **After MLX updates**: Run `check_mlx_ops.py` to verify API compatibility
- **Weight conversion issues**: Use `inspect_hubert_keys.py` to debug parameter mismatches
- **Model architecture changes**: Verify that model structure matches weight files

## Related Folders

- `benchmarks/` - Performance testing and validation scripts
- `rvc/lib/mlx/` - Weight conversion scripts (`convert.py`, `convert_hubert.py`, `convert_rmvpe.py`)
