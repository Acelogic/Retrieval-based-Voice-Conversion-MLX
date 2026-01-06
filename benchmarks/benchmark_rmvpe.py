#!/usr/bin/env python3
"""
Benchmark script comparing PyTorch vs MLX RMVPE implementations.
Tests the impact of chunking optimization on performance.
"""

import os
import sys
import time
import numpy as np
import argparse
from pathlib import Path

# Set environment variable before imports
os.environ["OMP_NUM_THREADS"] = "1"

# Add project root to sys.path to allow imports when run from root
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def create_synthetic_audio(duration_seconds, sample_rate=16000):
    """Create synthetic audio for testing."""
    n_samples = int(duration_seconds * sample_rate)
    # Create audio with some frequency content (sine waves + noise)
    t = np.linspace(0, duration_seconds, n_samples)
    audio = (
        0.3 * np.sin(2 * np.pi * 440 * t)  # A4 note
        + 0.2 * np.sin(2 * np.pi * 880 * t)  # A5 note
        + 0.1 * np.random.randn(n_samples)  # Noise
    )
    return audio.astype(np.float32)


def benchmark_torch_rmvpe(audio, warmup=True, num_runs=3):
    """Benchmark PyTorch RMVPE implementation."""
    try:
        import torch
        from rvc.lib.predictors.RMVPE import RMVPE0Predictor as TorchRMVPE

        model_file = "rvc/models/predictors/rmvpe.pt"

        if not os.path.exists(model_file):
            return None, f"Model not found at {model_file}"

        # Initialize predictor
        predictor = TorchRMVPE(model_path=model_file, device="mps")

        # Warmup run
        if warmup:
            _ = predictor.infer_from_audio(audio[:16000], thred=0.03)
            if torch.backends.mps.is_available():
                torch.mps.synchronize()

        # Multiple benchmark runs
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            f0 = predictor.infer_from_audio(audio, thred=0.03)
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            elapsed_time = time.perf_counter() - start_time
            times.append(elapsed_time)

        # Return median time
        return np.median(times), None

    except Exception as e:
        return None, str(e)


def benchmark_mlx_rmvpe(audio, warmup=True, num_runs=3):
    """Benchmark MLX RMVPE implementation."""
    try:
        import mlx.core as mx
        from rvc.lib.mlx.rmvpe import RMVPE0Predictor as MLXRMVPE

        weights_file = "rvc/models/predictors/rmvpe_mlx.npz"

        if not os.path.exists(weights_file):
            return None, f"Model not found at {weights_file}"

        # Initialize predictor
        predictor = MLXRMVPE(weights_path=weights_file)

        # Warmup run
        if warmup:
            _ = predictor.infer_from_audio(audio[:16000], thred=0.03)
            mx.eval(mx.zeros(1))  # Synchronize

        # Multiple benchmark runs
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            f0 = predictor.infer_from_audio(audio, thred=0.03)
            mx.eval(mx.zeros(1))  # Synchronize
            elapsed_time = time.perf_counter() - start_time
            times.append(elapsed_time)

        # Return median time
        return np.median(times), None

    except Exception as e:
        return None, str(e)


def run_benchmark_suite(quick=False):
    """Run comprehensive benchmark suite."""
    print("=" * 80)
    print("RMVPE Benchmark: PyTorch (MPS) vs MLX (Apple Silicon)")
    print("=" * 80)
    print()
    print("Configuration:")
    print("  - PyTorch: MPS backend with 32k frame chunking")
    print("  - MLX: Apple Silicon with 32k frame chunking (NEW)")
    print("  - Warmup: Yes (1 run on 1s audio)")
    print("  - Runs per test: 3 (median reported)")
    print()

    # Test configurations
    test_configs = [
        {"name": "Short (5s)", "duration": 5},
        {"name": "Medium (30s)", "duration": 30},
        {"name": "Long (60s)", "duration": 60},
        {"name": "Very Long (3min)", "duration": 180},
        {"name": "Extra Long (5min)", "duration": 300},
    ]

    if quick:
        print("🚀 Running in QUICK mode (only Short and Medium tests)")
        test_configs = test_configs[:2]

    print(
        f"{'Audio Length':<20} {'PyTorch (MPS)':<18} {'MLX (Apple)':<18} {'Speedup':<20}"
    )
    print("=" * 80)

    results = []

    for test_config in test_configs:
        duration = test_config["duration"]
        name = test_config["name"]

        # Create synthetic audio
        audio = create_synthetic_audio(duration)

        # Benchmark PyTorch
        torch_time, torch_error = benchmark_torch_rmvpe(audio, warmup=True, num_runs=3)

        # Benchmark MLX
        mlx_time, mlx_error = benchmark_mlx_rmvpe(audio, warmup=True, num_runs=3)

        # Display results
        torch_str = f"{torch_time:.3f}s" if torch_time else "ERROR"
        mlx_str = f"{mlx_time:.3f}s" if mlx_time else "ERROR"

        if torch_time and mlx_time:
            speedup = torch_time / mlx_time
            if speedup >= 1.0:
                speedup_str = f"✅ {speedup:.2f}x faster"
            else:
                slowdown = 1 / speedup
                speedup_str = f"⚠️  {slowdown:.2f}x slower"

            results.append(
                {
                    "name": name,
                    "duration": duration,
                    "torch": torch_time,
                    "mlx": mlx_time,
                    "speedup": speedup,
                }
            )
        else:
            speedup_str = "N/A"

        print(f"{name:<20} {torch_str:<18} {mlx_str:<18} {speedup_str:<20}")

    # Summary statistics
    if results:
        print()
        print("=" * 80)
        print("Summary Statistics")
        print("=" * 80)

        speedups = [r["speedup"] for r in results]
        avg_speedup = np.mean(speedups)
        min_speedup = np.min(speedups)
        max_speedup = np.max(speedups)

        print(f"  Average Speedup: {avg_speedup:.2f}x")
        print(
            f"  Min Speedup:     {min_speedup:.2f}x ({[r['name'] for r in results if r['speedup'] == min_speedup][0]})"
        )
        print(
            f"  Max Speedup:     {max_speedup:.2f}x ({[r['name'] for r in results if r['speedup'] == max_speedup][0]})"
        )
        print()

        if avg_speedup >= 1.0:
            print(f"  🎉 MLX is {avg_speedup:.2f}x faster than PyTorch on average!")
        else:
            print(f"  ⚠️  MLX is {1/avg_speedup:.2f}x slower than PyTorch on average")

        # Analyze impact by audio length
        print()
        print("=" * 80)
        print("Performance vs Audio Length Analysis")
        print("=" * 80)
        print()

        for result in results:
            improvement = (result["speedup"] - 1) * 100
            if result["speedup"] >= 1.0:
                print(f"  {result['name']:<20} MLX is {improvement:>5.1f}% faster")
            else:
                print(f"  {result['name']:<20} MLX is {-improvement:>5.1f}% slower")

        # Check if chunking helps more with longer audio
        if len(results) >= 2:
            short_speedup = results[0]["speedup"]
            long_speedup = results[-1]["speedup"]
            if long_speedup > short_speedup:
                diff = (long_speedup - short_speedup) * 100
                print(
                    f"\n  💡 Chunking optimization improves performance by {diff:.1f}% on longer audio"
                )

    print()
    print("=" * 80)
    print("Note: MLX implementation now uses 32k frame chunking for better GPU cache")
    print("      utilization and memory efficiency on long audio files.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark RMVPE implementations")
    parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmark (shorter tests)"
    )
    args = parser.parse_args()

    run_benchmark_suite(quick=args.quick)


if __name__ == "__main__":
    main()
