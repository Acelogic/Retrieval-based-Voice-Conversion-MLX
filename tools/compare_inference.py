#!/usr/bin/env python3
"""
Deterministic comparison of rvc-mlx vs rvc (PyTorch) inference outputs.

This script runs inference on both implementations with identical inputs and settings,
then compares the resulting waveforms using multiple metrics:
- Mean Squared Error (MSE)
- Cross-correlation
- Sample-by-sample difference histogram
- Spectral analysis (MFCC distance)

Usage:
    python tools/compare_inference.py \
        --input test-audio/coder_audio_stock.wav \
        --pth-model /path/to/model.pth \
        --mlx-model weights/model.npz \
        --seed 42
"""

import os
import sys
import argparse
import numpy as np
import soundfile as sf
import librosa

# Add project root to path
sys.path.insert(0, os.getcwd())

def set_seeds(seed):
    """Set random seeds for reproducibility in both PyTorch and MLX."""
    np.random.seed(seed)
    
    # Set PyTorch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    # Set MLX seed if available
    try:
        import mlx.core as mx
        # MLX uses global random state seeding
        mx.random.seed(seed)
    except ImportError:
        pass


def run_pytorch_inference(pth_model_path, audio_input, pitch=0, index_path="", index_rate=0.0, protect=0.5):
    """Run inference using PyTorch RVC implementation."""
    import torch
    
    from rvc.infer.infer import VoiceConverter
    
    print("\n=== Running PyTorch RVC inference ===")
    
    converter = VoiceConverter()
    
    # Create temp output file
    output_path = "/tmp/pytorch_rvc_output.wav"
    
    converter.convert_audio(
        audio_input_path=audio_input,
        audio_output_path=output_path,
        model_path=pth_model_path,
        index_path=index_path,
        pitch=pitch,
        f0_method="rmvpe",
        index_rate=index_rate,
        volume_envelope=1.0,
        protect=protect,
        hop_length=128,
        split_audio=False,
        f0_autotune=False,
        embedder_model="contentvec",
        export_format="WAV",
    )
    
    audio, sr = sf.read(output_path)
    return audio, sr


def run_mlx_inference(mlx_model_path, audio_input, pitch=0, index_path="", index_rate=0.0, protect=0.5):
    """Run inference using MLX RVC implementation."""
    print("\n=== Running MLX RVC inference ===")
    
    from rvc_mlx.infer.infer_mlx import RVC_MLX
    
    rvc = RVC_MLX(mlx_model_path)
    
    output_path = "/tmp/mlx_rvc_output.wav"
    
    rvc.infer(
        audio_input=audio_input,
        audio_output=output_path,
        pitch=pitch,
        f0_method="rmvpe",
        index_path=index_path if index_path else None,
        index_rate=index_rate,
        volume_envelope=1.0,
        protect=protect,
    )
    
    audio, sr = sf.read(output_path)
    return audio, sr


def compute_metrics(audio1, audio2, sr1, sr2, name1="Audio 1", name2="Audio 2"):
    """Compute comparison metrics between two audio signals."""
    print(f"\n=== Comparison: {name1} vs {name2} ===")
    
    # Check sample rates
    if sr1 != sr2:
        print(f"WARNING: Sample rates differ! {sr1} vs {sr2}")
        # Resample to match
        if sr1 > sr2:
            audio1 = librosa.resample(audio1, orig_sr=sr1, target_sr=sr2)
            sr1 = sr2
        else:
            audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=sr1)
            sr2 = sr1
    
    # Ensure same length
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]
    
    print(f"Sample rate: {sr1} Hz")
    print(f"Duration: {min_len / sr1:.2f} seconds ({min_len} samples)")
    
    # Basic stats
    print(f"\n{name1}:")
    print(f"  Mean: {np.mean(audio1):.6f}, Std: {np.std(audio1):.6f}, Max: {np.max(np.abs(audio1)):.6f}")
    print(f"{name2}:")
    print(f"  Mean: {np.mean(audio2):.6f}, Std: {np.std(audio2):.6f}, Max: {np.max(np.abs(audio2)):.6f}")
    
    # MSE
    mse = np.mean((audio1 - audio2) ** 2)
    print(f"\nMean Squared Error: {mse:.10f}")
    
    # RMSE in dB
    rmse = np.sqrt(mse)
    if rmse > 0:
        rmse_db = 20 * np.log10(rmse)
        print(f"RMSE: {rmse:.6f} ({rmse_db:.1f} dB)")
    
    # Cross-correlation
    correlation = np.corrcoef(audio1, audio2)[0, 1]
    print(f"Pearson Correlation: {correlation:.6f}")
    
    # Signal-to-Noise Ratio (treating difference as noise)
    signal_power = np.mean(audio1 ** 2)
    noise_power = mse
    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
        print(f"SNR (signal vs difference): {snr:.1f} dB")
    
    # MFCC distance
    try:
        mfcc1 = librosa.feature.mfcc(y=audio1, sr=sr1, n_mfcc=13)
        mfcc2 = librosa.feature.mfcc(y=audio2, sr=sr1, n_mfcc=13)
        # Ensure same shape
        min_frames = min(mfcc1.shape[1], mfcc2.shape[1])
        mfcc1 = mfcc1[:, :min_frames]
        mfcc2 = mfcc2[:, :min_frames]
        mfcc_dist = np.mean(np.sqrt(np.sum((mfcc1 - mfcc2) ** 2, axis=0)))
        print(f"Mean MFCC Euclidean Distance: {mfcc_dist:.4f}")
    except Exception as e:
        print(f"MFCC computation failed: {e}")
    
    # Difference analysis
    diff = audio1 - audio2
    print(f"\nDifference Statistics:")
    print(f"  Max absolute diff: {np.max(np.abs(diff)):.6f}")
    print(f"  Mean absolute diff: {np.mean(np.abs(diff)):.6f}")
    print(f"  Std of diff: {np.std(diff):.6f}")
    
    return {
        "mse": mse,
        "correlation": correlation,
        "snr_db": snr if noise_power > 0 else float('inf'),
        "mfcc_distance": mfcc_dist if 'mfcc_dist' in dir() else None,
    }


def save_difference_audio(audio1, audio2, sr, output_path):
    """Save the difference between two audio signals as an audio file."""
    min_len = min(len(audio1), len(audio2))
    diff = audio1[:min_len] - audio2[:min_len]
    
    # Normalize for audibility
    max_val = np.max(np.abs(diff))
    if max_val > 0:
        diff_normalized = diff / max_val * 0.9
    else:
        diff_normalized = diff
    
    sf.write(output_path, diff_normalized, sr)
    print(f"Saved difference audio to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare rvc-mlx vs PyTorch RVC inference")
    parser.add_argument("--input", type=str, required=True, help="Input audio file")
    parser.add_argument("--pth-model", type=str, help="Path to PyTorch model (.pth)")
    parser.add_argument("--mlx-model", type=str, help="Path to MLX model (.npz)")
    parser.add_argument("--pitch", type=int, default=0, help="Pitch shift in semitones")
    parser.add_argument("--index", type=str, default="", help="Path to index file")
    parser.add_argument("--index-rate", type=float, default=0.0, help="Index rate (0-1)")
    parser.add_argument("--protect", type=float, default=0.5, help="Protection level (0-0.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save-outputs", action="store_true", help="Save both outputs to current dir")
    parser.add_argument("--mode", choices=["both", "mlx-only", "pytorch-only"], default="both",
                        help="Which implementations to run")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    print(f"Setting random seed: {args.seed}")
    set_seeds(args.seed)
    
    pytorch_audio = None
    mlx_audio = None
    sr = None
    
    if args.mode in ["both", "pytorch-only"] and args.pth_model:
        if not os.path.exists(args.pth_model):
            print(f"Error: PyTorch model not found: {args.pth_model}")
        else:
            set_seeds(args.seed)  # Reset seed before each inference
            pytorch_audio, sr = run_pytorch_inference(
                args.pth_model, args.input, args.pitch, args.index, args.index_rate, args.protect
            )
            if args.save_outputs:
                os.makedirs("test-audio", exist_ok=True)
                out_path = os.path.join("test-audio", "pytorch_output.wav")
                sf.write(out_path, pytorch_audio, sr)
                print(f"Saved: {out_path}")
    
    if args.mode in ["both", "mlx-only"] and args.mlx_model:
        if not os.path.exists(args.mlx_model):
            print(f"Error: MLX model not found: {args.mlx_model}")
        else:
            set_seeds(args.seed)  # Reset seed before each inference
            mlx_audio, sr_mlx = run_mlx_inference(
                args.mlx_model, args.input, args.pitch, args.index, args.index_rate, args.protect
            )
            if sr is None:
                sr = sr_mlx
            if args.save_outputs:
                os.makedirs("test-audio", exist_ok=True)
                out_path = os.path.join("test-audio", "mlx_output.wav")
                sf.write(out_path, mlx_audio, sr_mlx)
                print(f"Saved: {out_path}")
    
    # Compare if both are available
    if pytorch_audio is not None and mlx_audio is not None:
        metrics = compute_metrics(pytorch_audio, mlx_audio, sr, sr_mlx, "PyTorch RVC", "MLX RVC")
        
        # Save difference
        os.makedirs("test-audio", exist_ok=True)
        diff_path = os.path.join("test-audio", "difference.wav")
        save_difference_audio(pytorch_audio, mlx_audio, sr, diff_path)
        
        # Summary
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        if metrics["correlation"] > 0.99:
            print("✅ EXCELLENT: Outputs are highly similar (correlation > 0.99)")
        elif metrics["correlation"] > 0.95:
            print("⚠️  GOOD: Outputs are similar but have some differences (correlation > 0.95)")
        elif metrics["correlation"] > 0.8:
            print("⚠️  MODERATE: Noticeable differences exist (correlation > 0.8)")
        else:
            print("❌ POOR: Significant differences between outputs")
    
    elif pytorch_audio is not None:
        print("\nOnly PyTorch output available. Run with --mlx-model to compare.")
    elif mlx_audio is not None:
        print("\nOnly MLX output available. Run with --pth-model to compare.")
    else:
        print("\nNo outputs generated. Check model paths.")


if __name__ == "__main__":
    main()
