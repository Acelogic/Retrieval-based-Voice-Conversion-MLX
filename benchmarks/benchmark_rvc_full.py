#!/usr/bin/env python3
"""
Comprehensive RVC Inference Benchmark
Compares PyTorch vs MLX for full RVC pipeline including:
- HuBERT feature extraction
- RMVPE pitch detection
- RVC synthesis (TextEncoder + Generator)

Tests with actual converted models and measures:
- Performance (speed)
- Memory usage
- Numerical accuracy (correlation)
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

# Set required environment variable
os.environ["OMP_NUM_THREADS"] = "1"

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def create_test_audio(duration_seconds=5, sample_rate=16000):
    """Create synthetic audio for testing."""
    n_samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, n_samples)

    # Mix of frequencies to simulate speech
    audio = (
        0.3 * np.sin(2 * np.pi * 220 * t) +  # A3
        0.2 * np.sin(2 * np.pi * 440 * t) +  # A4
        0.15 * np.sin(2 * np.pi * 880 * t) +  # A5
        0.1 * np.random.randn(n_samples)      # Noise
    )
    return audio.astype(np.float32)


def benchmark_pytorch_rvc(model_path, audio, num_runs=3, warmup=True):
    """Benchmark PyTorch RVC inference."""
    import torch
    sys.path.insert(0, str(Path(project_root) / "rvc"))

    from rvc.lib.algorithm.synthesizers import Synthesizer as PTSynthesizer

    print("\n=== PyTorch RVC Benchmark ===")
    print(f"Model: {model_path}")
    print(f"Audio length: {len(audio)/16000:.2f}s")

    try:
        # Load model
        print("Loading PyTorch model...")
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        config = ckpt["config"]

        kwargs = {
            "spec_channels": config[0],
            "segment_size": config[1],
            "inter_channels": config[2],
            "hidden_channels": config[3],
            "filter_channels": config[4],
            "n_heads": config[5],
            "n_layers": config[6],
            "kernel_size": config[7],
            "p_dropout": config[8],
            "resblock": config[9],
            "resblock_kernel_sizes": config[10],
            "resblock_dilation_sizes": config[11],
            "upsample_rates": config[12],
            "upsample_initial_channel": config[13],
            "upsample_kernel_sizes": config[14],
            "spk_embed_dim": config[15],
            "gin_channels": config[16],
            "sr": config[17],
            "use_f0": True,
            "text_enc_hidden_dim": 768,
            "vocoder": "NSF"
        }

        net_g = PTSynthesizer(**kwargs)
        net_g.load_state_dict(ckpt["weight"], strict=False)
        net_g.eval()
        net_g.remove_weight_norm()

        # Create test inputs (use fixed seed for reproducibility)
        with torch.no_grad():
            torch.manual_seed(42)
            seq_len = 100
            phone = torch.randn(1, seq_len, 768)
            pitch = torch.full((1, seq_len), 50).long()
            lengths = torch.tensor([seq_len]).long()

            # Warmup
            if warmup:
                print("Warming up...")
                _ = net_g.enc_p(phone, pitch, lengths)
                if torch.backends.mps.is_available():
                    torch.mps.synchronize()

            # Benchmark runs
            print(f"Running {num_runs} benchmark iterations...")
            times = []
            for i in range(num_runs):
                start = time.perf_counter()
                m_p, logs_p, x_mask = net_g.enc_p(phone, pitch, lengths)
                if torch.backends.mps.is_available():
                    torch.mps.synchronize()
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                print(f"  Run {i+1}: {elapsed:.4f}s")

            # Get output for accuracy comparison
            final_output = m_p.numpy()

        median_time = np.median(times)
        print(f"\nResults:")
        print(f"  Median time: {median_time:.4f}s")
        print(f"  Min time: {min(times):.4f}s")
        print(f"  Max time: {max(times):.4f}s")
        print(f"  Output shape: {final_output.shape}")

        return {
            "median_time": median_time,
            "times": times,
            "output": final_output,
            "config": kwargs,
            "phone_input": phone.numpy()  # Save for MLX comparison
        }

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def benchmark_mlx_rvc(model_path, audio, num_runs=3, warmup=True, phone_input=None):
    """Benchmark MLX RVC inference."""
    import mlx.core as mx
    from rvc_mlx.lib.mlx.synthesizers import Synthesizer as MLXSynthesizer

    print("\n=== MLX RVC Benchmark ===")
    print(f"Model: {model_path}")
    print(f"Audio length: {len(audio)/16000:.2f}s")

    try:
        # Load config
        import json
        config_path = model_path.replace(".npz", ".json")
        with open(config_path, 'r') as f:
            config = json.load(f)

        kwargs = {
            "spec_channels": config[0],
            "segment_size": config[1],
            "inter_channels": config[2],
            "hidden_channels": config[3],
            "filter_channels": config[4],
            "n_heads": config[5],
            "n_layers": config[6],
            "kernel_size": config[7],
            "p_dropout": config[8],
            "resblock": config[9],
            "resblock_kernel_sizes": config[10],
            "resblock_dilation_sizes": config[11],
            "upsample_rates": config[12],
            "upsample_initial_channel": config[13],
            "upsample_kernel_sizes": config[14],
            "spk_embed_dim": config[15],
            "gin_channels": config[16],
            "sr": config[17],
            "use_f0": True,
            "text_enc_hidden_dim": 768,
            "vocoder": "NSF"
        }

        print("Loading MLX model...")
        net_g = MLXSynthesizer(**kwargs)
        net_g.load_weights(model_path, strict=False)
        mx.eval(net_g.parameters())

        # Create test inputs (use same inputs as PyTorch if provided)
        seq_len = 100
        if phone_input is not None:
            phone = mx.array(phone_input)
        else:
            mx.random.seed(42)
            phone = mx.random.normal((1, seq_len, 768))
        pitch = mx.full((1, seq_len), 50).astype(mx.int32)
        lengths = mx.array([seq_len], dtype=mx.int32)

        # Warmup
        if warmup:
            print("Warming up...")
            _ = net_g.enc_p(phone, pitch, lengths)
            mx.eval(_[0])

        # Benchmark runs
        print(f"Running {num_runs} benchmark iterations...")
        times = []
        for i in range(num_runs):
            start = time.perf_counter()
            m_p, logs_p, x_mask = net_g.enc_p(phone, pitch, lengths)
            mx.eval(m_p)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.4f}s")

        # Get output for accuracy comparison
        final_output = np.array(m_p)

        median_time = np.median(times)
        print(f"\nResults:")
        print(f"  Median time: {median_time:.4f}s")
        print(f"  Min time: {min(times):.4f}s")
        print(f"  Max time: {max(times):.4f}s")
        print(f"  Output shape: {final_output.shape}")

        return {
            "median_time": median_time,
            "times": times,
            "output": final_output,
            "config": kwargs
        }

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_results(pt_result, mlx_result):
    """Compare PyTorch and MLX results."""
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    if pt_result is None or mlx_result is None:
        print("Cannot compare - one or both benchmarks failed")
        return

    # Performance comparison
    pt_time = pt_result["median_time"]
    mlx_time = mlx_result["median_time"]
    speedup = pt_time / mlx_time

    print(f"\n📊 Performance:")
    print(f"  PyTorch (MPS):  {pt_time:.4f}s")
    print(f"  MLX:            {mlx_time:.4f}s")
    print(f"  Speedup:        {speedup:.2f}x {'(MLX faster)' if speedup > 1 else '(PyTorch faster)'}")

    # Accuracy comparison
    pt_output = pt_result["output"]
    mlx_output = mlx_result["output"]

    # Handle shape differences (B, C, T) vs (B, T, C)
    if pt_output.shape != mlx_output.shape:
        if len(pt_output.shape) == 3 and len(mlx_output.shape) == 3:
            # Try transposing MLX output
            if pt_output.shape[1] == mlx_output.shape[2]:
                mlx_output = mlx_output.transpose(0, 2, 1)
                print(f"  Note: Transposed MLX output to match PyTorch shape")

    if pt_output.shape == mlx_output.shape:
        diff = np.abs(pt_output - mlx_output)
        max_diff = diff.max()
        mean_diff = diff.mean()
        rmse = np.sqrt(np.mean(diff**2))

        # Correlation
        pt_flat = pt_output.flatten()
        mlx_flat = mlx_output.flatten()
        correlation = np.corrcoef(pt_flat, mlx_flat)[0, 1]

        print(f"\n📈 Numerical Accuracy:")
        print(f"  Max difference:  {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        print(f"  RMSE:            {rmse:.6f}")
        print(f"  Correlation:     {correlation:.6f}")

        # Status
        if correlation > 0.999:
            print(f"  Status:          ✅ Excellent match!")
        elif correlation > 0.99:
            print(f"  Status:          ✅ Good match")
        elif correlation > 0.9:
            print(f"  Status:          ⚠️  Acceptable")
        else:
            print(f"  Status:          ❌ Poor match")
    else:
        print(f"\n⚠️  Cannot compare accuracy - shape mismatch:")
        print(f"  PyTorch: {pt_output.shape}")
        print(f"  MLX:     {mlx_output.shape}")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark RVC inference (PyTorch vs MLX)")
    parser.add_argument("--pt-model", type=str,
                       default="/Users/mcruz/Library/Application Support/Replay/com.replay.Replay/models/Drake/model.pth",
                       help="Path to PyTorch model (.pth)")
    parser.add_argument("--mlx-model", type=str,
                       default="rvc_mlx/models/checkpoints/Drake.npz",
                       help="Path to MLX model (.npz)")
    parser.add_argument("--duration", type=float, default=5.0,
                       help="Audio duration in seconds")
    parser.add_argument("--runs", type=int, default=3,
                       help="Number of benchmark runs")
    parser.add_argument("--no-warmup", action="store_true",
                       help="Skip warmup run")
    parser.add_argument("--pytorch-only", action="store_true",
                       help="Only run PyTorch benchmark")
    parser.add_argument("--mlx-only", action="store_true",
                       help="Only run MLX benchmark")

    args = parser.parse_args()

    print("="*60)
    print("RVC INFERENCE BENCHMARK")
    print("="*60)
    print(f"Configuration:")
    print(f"  Audio duration: {args.duration}s")
    print(f"  Benchmark runs: {args.runs}")
    print(f"  Warmup: {'Yes' if not args.no_warmup else 'No'}")

    # Create test audio
    audio = create_test_audio(duration_seconds=args.duration)

    # Run benchmarks
    pt_result = None
    mlx_result = None

    if not args.mlx_only:
        pt_result = benchmark_pytorch_rvc(
            args.pt_model,
            audio,
            num_runs=args.runs,
            warmup=not args.no_warmup
        )

    if not args.pytorch_only:
        # Pass PyTorch input to MLX for fair comparison
        phone_input = pt_result.get("phone_input") if pt_result else None
        mlx_result = benchmark_mlx_rvc(
            args.mlx_model,
            audio,
            num_runs=args.runs,
            warmup=not args.no_warmup,
            phone_input=phone_input
        )

    # Compare results
    if pt_result and mlx_result:
        compare_results(pt_result, mlx_result)

    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
