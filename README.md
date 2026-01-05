# Retrieval-based Voice Conversion (RVC) CLI

A stripped-down, command-line interface version of the Retrieval-based Voice Conversion (RVC) tool, optimized for performance and ease of integration.

## Features

- **CLI-Only**: No WebUI overhead (Gradio removed).
- **Core ML Functionality**: Supports core RVC features including Inference, Training, and Preprocessing.
- **Apple Silicon Native**: Full MLX inference support for M-series Macs.
- **Lightweight**: Minimized dependencies for easier deployment.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Acelogic/Retrieval-based-Voice-Conversion-MLX.git
    cd Retrieval-based-Voice-Conversion-MLX
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

    *Note: Ensure you have `ffmpeg` installed on your system if not using the python wrapper.*

## Usage

The main entry point is `rvc_cli.py`. You can see all available commands by running:

```bash
python rvc_cli.py --help
```

### Common Commands

**Inference:**
```bash
python rvc_cli.py infer --input_path <audio_file> --output_path <output_file> --pth_path <path_to_pth> --index_path <path_to_index>
```

**Training:**
```bash
python rvc_cli.py train --model_name <name> --total_epoch 100 ...
```

## Apple Silicon (MLX) Acceleration

This fork includes native Apple Silicon acceleration using the [MLX](https://github.com/ml-explore/mlx) framework.

### Backend Options

| Backend | Description |
|---------|-------------|
| `torch` | Pure PyTorch with MPS acceleration (default) |
| `mlx` | Full MLX: All inference runs natively on Apple Silicon GPU |

### Usage

```bash
# Standard PyTorch (MPS)
python rvc_cli.py infer --input_path audio.wav --output_path out.wav --pth_path model.pth --index_path model.index

# MLX (Apple Silicon native - up to 1.88x faster on RMVPE!)
python rvc_cli.py infer ... --backend mlx
```

> **Note**: On macOS, set `export OMP_NUM_THREADS=1` to prevent faiss-related crashes.

### Performance Benchmarks

#### Full RVC Pipeline
Tested on Apple Silicon (M-series) with a ~13s audio file:

| Backend | Time | vs PyTorch |
|---------|------|------------|
| `torch` (MPS) | 3.14s | baseline |
| `mlx` | **3.12s** | **comparable** |

#### RMVPE Pitch Detection (Optimized with Chunking)
Tested on Apple Silicon with various audio lengths:

| Audio Length | PyTorch (MPS) | MLX (Apple) | Speedup |
|--------------|---------------|-------------|---------|
| 5s           | 0.289s        | 0.182s      | **1.58x faster** |
| 30s          | 1.627s        | 0.864s      | **1.88x faster** |
| 60s          | 3.271s        | 1.758s      | **1.86x faster** |
| 3min         | 9.595s        | 5.178s      | **1.85x faster** |
| 5min         | 15.848s       | 9.223s      | **1.72x faster** |

**Average Speedup: 1.78x** - MLX RMVPE with chunking optimization is significantly faster than PyTorch MPS!

Both backends produce equivalent audio quality. The MLX backend eliminates PyTorch dependency overhead for deployment.

> **Note**: See `docs/` folder for detailed benchmarks and implementation notes.

### Weight Conversion (One-time setup for `mlx`)

Before using the MLX backend for the first time, convert the embedder weights:

```bash
# Convert Hubert embedder weights
python rvc/lib/mlx/convert_hubert.py

# Convert RMVPE pitch predictor weights
python rvc/lib/mlx/convert_rmvpe.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This project is based on the [Applio](https://github.com/IAHispano/Applio) repository but has been stripped down to its core CLI components. All original credits go to the respective authors.
