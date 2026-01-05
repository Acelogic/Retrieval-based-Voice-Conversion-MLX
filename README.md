# Retrieval-based Voice Conversion (RVC) CLI

A stripped-down, command-line interface version of the Retrieval-based Voice Conversion (RVC) tool, optimized for performance and ease of integration.

## Features

- **CLI-Only**: No WebUI overhead (Gradio removed).
- **Core ML Functionality**: Supports core RVC features including Inference, Training, and Preprocessing.
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
python rvc_cli.py infer --model_path <path_to_pth> --input_path <audio_file> --output_path <output_file> --index_path <path_to_index>
```

**Training:**
```bash
python rvc_cli.py train --model_name <name> --total_epoch 100 ...
```

**(Add more usage examples as you explore the CLI options)**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This project is based on the [Applio](https://github.com/IAHispano/Applio) repository but has been stripped down to its core CLI components. All original credits go to the respective authors.
