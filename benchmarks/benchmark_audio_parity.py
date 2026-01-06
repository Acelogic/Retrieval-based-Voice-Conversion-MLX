#!/usr/bin/env python3
"""
Audio Inference Parity Benchmark

Compares PyTorch vs MLX RVC implementations with real audio files.
Tests the complete inference pipeline:
- Audio loading and preprocessing
- HuBERT feature extraction
- RMVPE pitch detection
- RVC synthesis (TextEncoder + Generator)
- Audio output

Measures both performance and numerical accuracy.
"""

import os
import sys
import time
import argparse
import numpy as np
import soundfile as sf
import hashlib
from pathlib import Path

# Set required environment variable
os.environ["OMP_NUM_THREADS"] = "1"

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def load_audio(file_path, sr=16000):
    """Load audio file and resample if needed."""
    import librosa

    try:
        audio, samplerate = sf.read(file_path)
    except Exception as e:
        print(f"Error loading audio {file_path}: {e}")
        return None

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    if samplerate != sr:
        audio = librosa.resample(audio, orig_sr=samplerate, target_sr=sr)

    return audio.astype(np.float32)


def benchmark_pytorch_audio(audio_path, model_path, num_runs=3, warmup=True, save_output=None):
    """Benchmark PyTorch RVC inference with real audio."""
    import torch
    sys.path.insert(0, str(Path(project_root) / "rvc"))

    from rvc.infer.pipeline import Pipeline as VC
    from rvc.lib.utils import load_embedding
    from rvc.lib.algorithm.synthesizers import Synthesizer
    from rvc.configs.config import Config

    print("\n" + "="*60)
    print("PYTORCH RVC AUDIO INFERENCE")
    print("="*60)
    print(f"Audio: {audio_path}")
    print(f"Model: {model_path}")

    try:
        # Load audio
        print("\nLoading audio...")
        audio = load_audio(audio_path, sr=16000)
        if audio is None:
            return None
        print(f"  Audio length: {len(audio)/16000:.2f}s")

        # Initialize config
        config = Config()

        # Load HuBERT model
        print("\nLoading HuBERT model...")
        embedder_model = "contentvec"
        hubert_model = load_embedding(embedder_model, None)
        hubert_model = hubert_model.to(config.device).float()
        hubert_model.eval()

        # Load RVC model
        print("\nLoading RVC model...")
        cpt = torch.load(model_path, map_location="cpu", weights_only=True)
        tgt_sr = cpt["config"][-1]

        # Get model config
        model_config = cpt["config"]
        kwargs = {
            "spec_channels": model_config[0],
            "segment_size": model_config[1],
            "inter_channels": model_config[2],
            "hidden_channels": model_config[3],
            "filter_channels": model_config[4],
            "n_heads": model_config[5],
            "n_layers": model_config[6],
            "kernel_size": model_config[7],
            "p_dropout": model_config[8],
            "resblock": model_config[9],
            "resblock_kernel_sizes": model_config[10],
            "resblock_dilation_sizes": model_config[11],
            "upsample_rates": model_config[12],
            "upsample_initial_channel": model_config[13],
            "upsample_kernel_sizes": model_config[14],
            "spk_embed_dim": model_config[15],
            "gin_channels": model_config[16],
            "sr": model_config[17],
            "use_f0": True,
        }

        # Initialize synthesizer
        net_g = Synthesizer(**kwargs)
        net_g.load_state_dict(cpt["weight"], strict=False)
        net_g.eval()
        if hasattr(net_g, 'remove_weight_norm'):
            net_g.remove_weight_norm()
        net_g = net_g.to(config.device)

        # Initialize pipeline
        print("\nInitializing pipeline...")
        vc = VC(tgt_sr, config)

        # Warmup
        if warmup:
            print("\nWarming up...")
            _ = vc.pipeline(
                model=hubert_model,
                net_g=net_g,
                sid=0,
                audio=audio[:16000*2],  # 2 second warmup
                pitch=0,
                f0_method="rmvpe",
                file_index="",
                index_rate=0.0,
                pitch_guidance=True,
                volume_envelope=1.0,
                version="v2",
                protect=0.33,
                f0_autotune=False,
                f0_autotune_strength=1.0,
                proposed_pitch=False,
                proposed_pitch_threshold=155.0,
            )
            if torch.backends.mps.is_available():
                torch.mps.synchronize()

        # Benchmark runs
        print(f"\nRunning {num_runs} benchmark iterations...")
        times = []
        outputs = []

        for i in range(num_runs):
            start = time.perf_counter()

            output_audio = vc.pipeline(
                model=hubert_model,
                net_g=net_g,
                sid=0,
                audio=audio,
                pitch=0,
                f0_method="rmvpe",
                file_index="",
                index_rate=0.0,
                pitch_guidance=True,
                volume_envelope=1.0,
                version="v2",
                protect=0.33,
                f0_autotune=False,
                f0_autotune_strength=1.0,
                proposed_pitch=False,
                proposed_pitch_threshold=155.0,
            )

            if torch.backends.mps.is_available():
                torch.mps.synchronize()

            elapsed = time.perf_counter() - start
            times.append(elapsed)
            outputs.append(output_audio)

            print(f"  Run {i+1}: {elapsed:.4f}s")

        median_time = np.median(times)
        final_output = outputs[0]  # Use first output for comparison

        print(f"\nResults:")
        print(f"  Median time: {median_time:.4f}s")
        print(f"  Min time: {min(times):.4f}s")
        print(f"  Max time: {max(times):.4f}s")
        print(f"  Output shape: {final_output.shape}")
        print(f"  Output length: {len(final_output)/tgt_sr:.2f}s @ {tgt_sr}Hz")

        # Save output if requested
        if save_output:
            sf.write(save_output, final_output, tgt_sr)
            print(f"\n  Saved output to: {save_output}")

        return {
            "median_time": median_time,
            "times": times,
            "output": final_output,
            "sample_rate": tgt_sr,
            "input_audio": audio,
            "sha256": hashlib.sha256(final_output.tobytes()).hexdigest(),
            "model_path": model_path,
            "audio_path": audio_path
        }

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None


def benchmark_mlx_audio(audio_path, model_path, num_runs=3, warmup=True, save_output=None):
    """Benchmark MLX RVC inference with real audio."""
    import mlx.core as mx
    from rvc_mlx.infer.infer_mlx import RVC_MLX

    print("\n" + "="*60)
    print("MLX RVC AUDIO INFERENCE")
    print("="*60)
    print(f"Audio: {audio_path}")
    print(f"Model: {model_path}")

    try:
        # Load audio
        print("\nLoading audio...")
        audio = load_audio(audio_path, sr=16000)
        if audio is None:
            return None
        print(f"  Audio length: {len(audio)/16000:.2f}s")

        # Initialize RVC_MLX
        print("\nLoading MLX RVC model...")
        rvc = RVC_MLX(model_path)
        tgt_sr = rvc.tgt_sr

        print(f"  Target sample rate: {tgt_sr}Hz")

        # Warmup
        if warmup:
            print("\nWarming up...")
            _ = rvc.pipeline.pipeline(
                rvc.hubert_model,
                rvc.net_g,
                0,  # sid
                audio[:16000*2],  # 2 second warmup
                0,  # pitch
                "rmvpe",
                None,  # file_index
                0.0,  # index_rate
                True,  # pitch_guidance
                1.0,  # volume_envelope
                "v2",
                0.33,  # protect
                False,  # f0_autotune
                1.0,  # f0_autotune_strength
                False,  # proposed_pitch
                155.0,  # proposed_pitch_threshold
            )
            mx.eval(mx.zeros(1))  # Synchronize

        # Benchmark runs
        print(f"\nRunning {num_runs} benchmark iterations...")
        times = []
        outputs = []

        for i in range(num_runs):
            start = time.perf_counter()

            output_audio = rvc.pipeline.pipeline(
                rvc.hubert_model,
                rvc.net_g,
                0,  # sid
                audio,
                0,  # pitch
                "rmvpe",
                None,  # file_index
                0.0,  # index_rate
                True,  # pitch_guidance
                1.0,  # volume_envelope
                "v2",
                0.33,  # protect
                False,  # f0_autotune
                1.0,  # f0_autotune_strength
                False,  # proposed_pitch
                155.0,  # proposed_pitch_threshold
            )

            mx.eval(mx.zeros(1))  # Synchronize

            elapsed = time.perf_counter() - start
            times.append(elapsed)
            outputs.append(output_audio)

            print(f"  Run {i+1}: {elapsed:.4f}s")

        median_time = np.median(times)
        final_output = outputs[0]  # Use first output for comparison

        print(f"\nResults:")
        print(f"  Median time: {median_time:.4f}s")
        print(f"  Min time: {min(times):.4f}s")
        print(f"  Max time: {max(times):.4f}s")
        print(f"  Output shape: {final_output.shape}")
        print(f"  Output length: {len(final_output)/tgt_sr:.2f}s @ {tgt_sr}Hz")

        # Save output if requested
        if save_output:
            sf.write(save_output, final_output, tgt_sr)
            print(f"\n  Saved output to: {save_output}")

        return {
            "median_time": median_time,
            "times": times,
            "output": final_output,
            "sample_rate": tgt_sr,
            "input_audio": audio,
            "sha256": hashlib.sha256(final_output.tobytes()).hexdigest(),
            "model_path": model_path,
            "audio_path": audio_path
        }

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_audio_results(pt_result, mlx_result, plot_path=None):
    """Compare PyTorch and MLX audio outputs."""
    print("\n" + "="*60)
    print("AUDIO INFERENCE COMPARISON")
    print("="*60)

    if pt_result is None or mlx_result is None:
        print("\n❌ Cannot compare - one or both benchmarks failed")
        return

    # Performance comparison
    pt_time = pt_result["median_time"]
    mlx_time = mlx_result["median_time"]
    speedup = pt_time / mlx_time

    print(f"\n📊 Performance:")
    print(f"  PyTorch (MPS):  {pt_time:.4f}s")
    print(f"  MLX:            {mlx_time:.4f}s")
    print(f"  Speedup:        {speedup:.2f}x {'(MLX faster)' if speedup > 1 else '(PyTorch faster)'}")

    # SHA256 Comparison
    print(f"\n🔐 SHA256 Checksums:")
    print(f"  PyTorch: {pt_result['sha256']}")
    print(f"  MLX:     {mlx_result['sha256']}")
    if pt_result['sha256'] == mlx_result['sha256']:
        print("  ⚠️  IDENTICAL HASHES (Suspicious if distinct frameworks)")
    else:
        print("  ✅ Hashes distinct (Expected from different frameworks)")

    # Audio comparison
    pt_audio = pt_result["output"]
    mlx_audio = mlx_result["output"]

    print(f"\n📈 Audio Comparison:")
    print(f"  PyTorch output: {pt_audio.shape} @ {pt_result['sample_rate']}Hz")
    print(f"  MLX output:     {mlx_audio.shape} @ {mlx_result['sample_rate']}Hz")

    # Handle length differences (trim to shorter)
    min_len = min(len(pt_audio), len(mlx_audio))
    if len(pt_audio) != len(mlx_audio):
        print(f"  ⚠️  Length mismatch - comparing first {min_len} samples")
        pt_audio = pt_audio[:min_len]
        mlx_audio = mlx_audio[:min_len]

    # Numerical comparison
    diff = np.abs(pt_audio - mlx_audio)
    max_diff = diff.max()
    mean_diff = diff.mean()
    rmse = np.sqrt(np.mean(diff**2))

    # Correlation
    correlation = np.corrcoef(pt_audio, mlx_audio)[0, 1]

    print(f"\n📐 Numerical Accuracy:")
    print(f"  Max difference:  {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    print(f"  RMSE:            {rmse:.6f}")
    print(f"  Waveform Corr:   {correlation:.6f}")

    # Spectrogram Correlation (Perceptual)
    spec_corr = 0.0
    try:
        import librosa
        
        # Compute STFT
        n_fft = 1024
        hop_length = 256
        # Use abs() to get magnitude, discarding phase
        S_pt = np.abs(librosa.stft(pt_audio, n_fft=n_fft, hop_length=hop_length))
        S_mlx = np.abs(librosa.stft(mlx_audio, n_fft=n_fft, hop_length=hop_length))
        
        # Log-Mel Spectrogram (better for perception)
        # Assuming 40k or 48k sample rate
        sr = pt_result['sample_rate']
        mel_pt = librosa.feature.melspectrogram(S=S_pt**2, sr=sr, n_mels=80)
        mel_mlx = librosa.feature.melspectrogram(S=S_mlx**2, sr=sr, n_mels=80)
        
        log_mel_pt = librosa.power_to_db(mel_pt, ref=np.max)
        log_mel_mlx = librosa.power_to_db(mel_mlx, ref=np.max)
        
        # Flatten and correlate
        spec_corr = np.corrcoef(log_mel_pt.flatten(), log_mel_mlx.flatten())[0, 1]
        
        print(f"  Spectrogram Corr:{spec_corr:.6f} (Perceptual similarity, ignores phase)")
        
    except ImportError:
        print("  Spectrogram Corr: Skipped (librosa not installed)")

    # Audio quality metrics
    pt_rms = np.sqrt(np.mean(pt_audio**2))
    mlx_rms = np.sqrt(np.mean(mlx_audio**2))

    print(f"\n🔊 Audio Characteristics:")
    print(f"  PyTorch RMS:  {pt_rms:.6f}")
    print(f"  MLX RMS:      {mlx_rms:.6f}")
    print(f"  RMS ratio:    {mlx_rms/pt_rms:.6f}")

    # Overall status
    print(f"\n📋 Overall Status:")
    if spec_corr > 0.95:
        print(f"  ✅ Perceptually Identical (Spectrogram Corr > 0.95)")
        print(f"  ℹ️  Waveform Corr {correlation:.4f} may be low due to phase drift (expected).")
    elif correlation > 0.95:
        print(f"  ✅ Exact Match (Waveform Correlation > 0.95)")
    else:
        print(f"  ❌ Poor match (Spectrogram Corr < 0.95)")

    print("\n" + "="*60)
    
    if plot_path:
        try:
            import matplotlib.pyplot as plt
            import librosa.display
            
            print(f"\nGeneratng analysis plot: {plot_path}")
            
            # Data
            y_pt = pt_audio
            y_mlx = mlx_audio
            sr = pt_result['sample_rate']
            
            plt.figure(figsize=(16, 14))
            
            # Title with Metadata
            pt_model_name = Path(pt_result['model_path']).name
            mlx_model_name = Path(mlx_result['model_path']).name
            audio_name = Path(pt_result['audio_path']).name
            
            # Try to get more metadata if possible
            pt_meta = ""
            pt_dir = Path(pt_result['model_path']).parent
            # Check for metadata like 'params.json' or just use directory name as hint
            if pt_dir.name != "models":
                 pt_meta = f" (Source: {pt_dir.name})"

            title_text = (
                f"RVC Parity Benchmark | Audio: {audio_name}\n"
                f"Models: PT({pt_model_name}{pt_meta}) vs MLX({mlx_model_name})\n"
                f"PT SHA256: {pt_result['sha256']}\n"
                f"MLX SHA256: {mlx_result['sha256']}"
            )
            
            plt.suptitle(title_text, fontsize=12, fontfamily='monospace')
            
            # 1. Spectrograms (Difference?)
            # Let's plot both side by side or Top/Bottom
            plt.subplot(3, 1, 1)
            D_pt = librosa.amplitude_to_db(np.abs(librosa.stft(y_pt)), ref=np.max)
            D_mlx = librosa.amplitude_to_db(np.abs(librosa.stft(y_mlx)), ref=np.max)
            
            # Diff spectrogram
            # Resize if needed? We already trimmed.
            min_shapes = min(D_pt.shape[1], D_mlx.shape[1])
            D_pt = D_pt[:, :min_shapes]
            D_mlx = D_mlx[:, :min_shapes]
            
            librosa.display.specshow(D_pt, sr=sr, x_axis='time', y_axis='hz')
            plt.title('PyTorch Spectrogram')
            plt.colorbar(format='%+2.0f dB')
            
            # 2. Waveform Overlay
            plt.subplot(3, 1, 2)
            librosa.display.waveshow(y_pt, sr=sr, alpha=0.5, label='PyTorch', color='b')
            librosa.display.waveshow(y_mlx, sr=sr, alpha=0.5, label='MLX', color='r')
            plt.title('Waveform Overlay')
            plt.legend(loc='upper right')
            
            # 3. Spectral Features
            plt.subplot(3, 1, 3)
            cent_pt = librosa.feature.spectral_centroid(y=y_pt, sr=sr)[0]
            cent_mlx = librosa.feature.spectral_centroid(y=y_mlx, sr=sr)[0]
            times = librosa.times_like(cent_pt)
            
            plt.plot(times, cent_pt, label='PT Centroids', color='b')
            plt.plot(times, cent_mlx, label='MLX Centroids', color='r', linestyle='--')
            plt.title('Spectral Centroid Comparison')
            plt.legend(loc='upper right')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.90]) # Adjust for suptitle
            plt.savefig(plot_path)
            print(f"  ✅ Saved plot to {plot_path}")
            
        except Exception as e:
            print(f"  ❌ Failed to generate plot: {e}")




def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RVC audio inference parity (PyTorch vs MLX)"
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="test-audio/coder_audio_stock.wav",
        help="Path to input audio file",
    )
    parser.add_argument(
        "--pt-model",
        type=str,
        default="/Users/mcruz/Library/Application Support/Replay/com.replay.Replay/models/Drake/model.pth",
        help="Path to PyTorch model (.pth)",
    )
    parser.add_argument(
        "--mlx-model",
        type=str,
        default="rvc_mlx/models/checkpoints/Drake.npz",
        help="Path to MLX model (.npz)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of benchmark runs",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warmup run",
    )
    parser.add_argument(
        "--pytorch-only",
        action="store_true",
        help="Only run PyTorch benchmark",
    )
    parser.add_argument(
        "--mlx-only",
        action="store_true",
        help="Only run MLX benchmark",
    )
    parser.add_argument(
        "--save-outputs",
        action="store_true",
        help="Save output audio files for manual comparison",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp",
        help="Directory to save output audio files",
    )
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="Generate and save analysis plot (spectrogram/waveform/features)",
    )

    args = parser.parse_args()

    print("="*60)
    print("RVC AUDIO INFERENCE PARITY BENCHMARK")
    print("="*60)
    print(f"Configuration:")
    print(f"  Input audio: {args.audio}")
    print(f"  Benchmark runs: {args.runs}")
    print(f"  Warmup: {'Yes' if not args.no_warmup else 'No'}")
    print(f"  Save outputs: {'Yes' if args.save_outputs else 'No'}")
    print(f"  Save plot: {'Yes' if args.save_plot else 'No'}")

    # Prepare output paths
    pt_output = None
    mlx_output = None
    plot_output = None
    
    if args.save_outputs or args.save_plot:
        os.makedirs(args.output_dir, exist_ok=True)
        
    if args.save_outputs:
        pt_output = os.path.join(args.output_dir, "pytorch_output.wav")
        mlx_output = os.path.join(args.output_dir, "mlx_output.wav")
        
    if args.save_plot:
        plot_output = os.path.join(args.output_dir, "benchmark_plot.png")

    # Run benchmarks
    pt_result = None
    mlx_result = None

    if not args.mlx_only:
        pt_result = benchmark_pytorch_audio(
            args.audio,
            args.pt_model,
            num_runs=args.runs,
            warmup=not args.no_warmup,
            save_output=pt_output,
        )

    if not args.pytorch_only:
        mlx_result = benchmark_mlx_audio(
            args.audio,
            args.mlx_model,
            num_runs=args.runs,
            warmup=not args.no_warmup,
            save_output=mlx_output,
        )

    # Compare results
    if pt_result and mlx_result:
        compare_audio_results(pt_result, mlx_result, plot_path=plot_output)

    print("\n✅ Benchmark complete!")

    if args.save_outputs and pt_output and mlx_output:
        print(f"\n📁 Saved outputs:")
        print(f"  PyTorch: {pt_output}")
        print(f"  MLX:     {mlx_output}")
        print(f"\n  Compare them with: diff <(xxd {pt_output}) <(xxd {mlx_output}) | head")


if __name__ == "__main__":
    main()
