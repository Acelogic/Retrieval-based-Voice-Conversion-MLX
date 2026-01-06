#!/usr/bin/env python3
"""
Export intermediate outputs from Python MLX RVC for iOS validation

This script processes a test audio file and exports intermediate outputs
from each component (HuBERT, RMVPE, TextEncoder, Generator) to allow
direct comparison with iOS Swift implementation.

Usage:
    python tools/export_ios_test_data.py --audio path/to/test.wav --output-dir ios_test_data/
"""

import argparse
import numpy as np
import mlx.core as mx
from pathlib import Path
import json
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rvc_mlx.infer.infer_mlx import RVC_MLX
from rvc_mlx.lib.mlx.hubert import HubertModel, HubertConfig
from rvc_mlx.lib.mlx.rmvpe import RMVPE0Predictor
import soundfile as sf


def export_array(array, name: str, output_dir: Path):
    """Export MLX array to numpy file"""
    if isinstance(array, mx.array):
        array = np.array(array)
    np_path = output_dir / f"{name}.npy"
    np.save(np_path, array)
    print(f"  Exported {name}: shape={array.shape}, dtype={array.dtype}")
    return array


def export_test_data(audio_path: Path, output_dir: Path, model_path: Path):
    """Export all intermediate outputs for testing"""

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Loading test audio...")
    print(f"{'='*60}")

    # Load audio
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio[:, 0]  # Take first channel if stereo

    # Resample to 16kHz if needed
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    print(f"Audio: {len(audio)} samples, {sr}Hz, duration={len(audio)/sr:.2f}s")
    export_array(audio, "input_audio", output_dir)

    # Initialize RVC inference
    print(f"\n{'='*60}")
    print("Initializing RVC components...")
    print(f"{'='*60}")

    # Initialize HuBERT
    hubert_config = HubertConfig()
    hubert_model = HubertModel(config=hubert_config)

    # Load HuBERT weights
    hubert_weights_path = Path("rvc_mlx/models/hubert_base.safetensors")
    if hubert_weights_path.exists():
        print(f"Loading HuBERT from: {hubert_weights_path}")
        hubert_weights = mx.load(str(hubert_weights_path))
        hubert_model.load_weights(list(hubert_weights.items()))
    else:
        print(f"Warning: HuBERT weights not found at {hubert_weights_path}")

    # Initialize RMVPE
    rmvpe = RMVPE0Predictor()
    # RMVPE weights are loaded in the constructor if available

    # Convert to MLX array
    audio_mx = mx.array(audio)

    print(f"\n{'='*60}")
    print("1. HuBERT Feature Extraction")
    print(f"{'='*60}")

    # HuBERT expects (B, T) format
    audio_batch = audio_mx[None, :]  # (1, T)
    hubert_features = hubert_model(audio_batch)  # (1, T', 256)

    print(f"Input shape: {audio_batch.shape}")
    print(f"Output shape: {hubert_features.shape}")
    export_array(hubert_features, "hubert_features", output_dir)

    print(f"\n{'='*60}")
    print("2. RMVPE Pitch Extraction")
    print(f"{'='*60}")

    # RMVPE
    f0 = rmvpe.infer_from_audio(audio_mx, thred=0.03)
    print(f"F0 shape: {f0.shape}")
    f0_np = np.array(f0)
    print(f"F0 range: [{np.min(f0_np):.2f}, {np.max(f0_np):.2f}] Hz")
    voiced_f0 = f0_np[f0_np > 0]
    if len(voiced_f0) > 0:
        print(f"F0 mean: {np.mean(voiced_f0):.2f} Hz (voiced frames)")
    export_array(f0, "rmvpe_f0", output_dir)

    # Also export RMVPE hidden states for deeper debugging
    mel = rmvpe.mel_spectrogram(audio_mx)
    hidden = rmvpe.mel2hidden(mel)
    export_array(hidden, "rmvpe_hidden", output_dir)

    print(f"\n{'='*60}")
    print("3. Summary")
    print(f"{'='*60}")

    print(f"\nComponent outputs exported:")
    print(f"  - HuBERT features: {hubert_features.shape}")
    print(f"  - RMVPE F0: {f0.shape}")
    print(f"  - RMVPE hidden: {hidden.shape}")

    # Note: Full pipeline and TextEncoder require model to be loaded
    # For now, we focus on validating individual components

    # Export metadata
    metadata = {
        "audio_path": str(audio_path),
        "sample_rate": sr,
        "duration_seconds": len(audio) / sr,
        "hubert_output_shape": list(hubert_features.shape),
        "f0_shape": list(f0.shape),
        "model_path": str(model_path) if model_path else None,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print("✅ Export Complete!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"\nFiles exported:")
    for f in sorted(output_dir.glob("*.npy")):
        print(f"  - {f.name}")


def main():
    parser = argparse.ArgumentParser(description='Export RVC intermediate outputs for iOS testing')
    parser.add_argument('--audio', type=str, required=True,
                        help='Path to test audio file (.wav)')
    parser.add_argument('--output-dir', type=str, default='ios_test_data',
                        help='Output directory for test data')
    parser.add_argument('--model-path', type=str,
                        help='Path to RVC model file (.safetensors or .pth)')

    args = parser.parse_args()

    audio_path = Path(args.audio)
    output_dir = Path(args.output_dir)
    model_path = Path(args.model_path) if args.model_path else None

    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    export_test_data(audio_path, output_dir, model_path)


if __name__ == '__main__':
    main()
