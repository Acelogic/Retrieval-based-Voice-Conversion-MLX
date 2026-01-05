# Project Context & Session Summary

**Date:** 2026-01-05
**Objective:** Add native Apple Silicon (MLX) inference support to RVC CLI.

## Accomplishments
1.  **MLX Core Integration**:
    *   Added `mlx` dependency for macOS.
    *   Created `rvc/lib/mlx/` package containing ported modules:
        *   `modules.py`: WaveNet
        *   `attentions.py`: MultiHeadAttention, FFN
        *   `residuals.py`: ResBlock, ResidualCouplingBlock
        *   `generators.py`: HiFiGANNSFGenerator, SineGenerator
        *   `encoders.py`: TextEncoder, PosteriorEncoder
        *   `synthesizers.py`: Synthesizer (The main generator model)
    *   **Architecture Choice**: Adopted a **Hybrid Pipeline**. We rely on the existing PyTorch implementation for complex Feature Extraction (Hubert, RMVPE) to ensure compatibility and stability, and use MLX solely for the computationally expensive HiFiGAN synthesis step.

2.  **Inference Pipeline**:
    *   Implemented `VoiceConverterMLX` and `PipelineMLX` in `rvc/infer/infer_mlx.py`.
    *   Implemented on-the-fly weight conversion in `rvc/lib/mlx/convert.py` which loads a standard RVC `.pth`, fuses `weight_norm` layers, and transposes weights to match MLX's (N, L, C) layout.

3.  **CLI Integration**:
    *   Modified `rvc_cli.py` to accept `--backend mlx`.
    *   Standard usage: `python rvc_cli.py infer ... --backend mlx`.

## Critical "Tidbits" for Future Sessions

### 1. Model Locations
The user's test models are located at:
> **`/Users/mcruz/Library/Application Support/Replay/com.replay.Replay/models`**

You should verify availability of models here before running tests.

### 2. Environment Variables
*   **`export OMP_NUM_THREADS=1`**: This is **MANDATORY** on macOS to prevent `faiss` from crashing the process with a segmentation fault.

### 5. Runtime Environment
*   **Conda Environment**: All commands must be run within the `rvc` Conda environment.
    *   Example: `conda run -n rvc python rvc_cli.py ...` or `source activate rvc` before running.

### 3. Model Compatibility
*   **Config Required**: The MLX converter expects the `.pth` file to contain a `config` key (list of hyperparameters) alongside the `weight` key.
*   **No Pretrained-Only**: Raw training checkpoints (like `f0G40k.pth`) often lack the `config` key and will fail to load in the current MLX implementation. Use fully trained/exported RVC models.

### 4. Implementation Details
*   **Data Layout**: PyTorch uses `(N, C, L)` (Channels First). MLX components were ported to use `(N, L, C)` (Channels Last) which is more native to MLX/Transformers. The converter handles this transposition.
*   **Missing Layers**: `mlx.nn` does not yet have a `ConvTranspose1d` layer. We implemented a custom `ConvTranspose1d` in `rvc/lib/mlx/generators.py` using an upsample-and-convolve approach.
*   **Weight Transposition**: 
    *   Regular Conv1d: PyTorch `(Out, In, K)` -> MLX `(Out, K, In)`. Transpose `(0, 2, 1)`.
    *   ConvTranspose1d: PyTorch `(In, Out, K)` -> MLX `(Out, K, In)` (effectively). Transpose `(1, 2, 0)`.
*   **Performance**: The current implementation converts weights *every time* inference is run. For production, we should implement a mechanism to save/load converted `.npz` or `.safetensors` MLX weights.

## Next Steps
*   **Final Verification**: Run a full end-to-end test using a model from the Replay directory.
*   **Optimization**: Cache converted MLX weights to disk.
*   **Benchmarks**: Compare MPS (PyTorch) vs MLX performance.
