"""
PyTorch vs MLX Training Performance Comparison

Compares training performance between PyTorch and MLX implementations.
"""

import os
import sys
import time
import argparse
import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


@dataclass
class ComparisonResult:
    """Results from a PyTorch vs MLX comparison."""
    batch_size: int
    num_steps: int

    # PyTorch metrics
    pytorch_samples_per_second: float
    pytorch_time_per_step_ms: float
    pytorch_peak_memory_mb: float

    # MLX metrics
    mlx_samples_per_second: float
    mlx_time_per_step_ms: float
    mlx_peak_memory_mb: float

    # Comparison metrics
    speedup: float  # MLX samples/s / PyTorch samples/s
    memory_ratio: float  # PyTorch memory / MLX memory

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def benchmark_pytorch_training(batch_size: int, num_steps: int, warmup_steps: int = 5) -> Dict[str, float]:
    """
    Benchmark PyTorch training step.

    Returns dict with samples_per_second, time_per_step_ms, peak_memory_mb
    """
    try:
        import torch
        from rvc.lib.algorithm.synthesizers import Synthesizer as PyTorchSynthesizer
    except ImportError:
        print("PyTorch or RVC not available. Skipping PyTorch benchmark.")
        return None

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  PyTorch device: {device}")

    # Create model
    model = PyTorchSynthesizer(
        spec_channels=1025,
        segment_size=32000,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.0,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[10, 10, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16, 4, 4],
        spk_embed_dim=256,
        gin_channels=256,
        sr=40000,
        use_f0=True,
    ).to(device)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Create synthetic batch
    def create_batch():
        phone_length = 256
        spec_length = 512
        return {
            "phone": torch.randn(batch_size, phone_length, 768, device=device),
            "phone_lengths": torch.full((batch_size,), phone_length, device=device),
            "pitch": torch.randn(batch_size, phone_length, device=device),
            "pitchf": torch.rand(batch_size, phone_length, device=device) * 400 + 50,
            "spec": torch.randn(batch_size, 1025, spec_length, device=device),
            "spec_lengths": torch.full((batch_size,), spec_length, device=device),
            "wave": torch.randn(batch_size, 32000, 1, device=device),
            "sid": torch.zeros(batch_size, dtype=torch.long, device=device),
        }

    # Warmup
    print(f"    Warming up ({warmup_steps} steps)...")
    for _ in range(warmup_steps):
        batch = create_batch()
        optimizer.zero_grad()
        try:
            output = model(
                batch["phone"],
                batch["phone_lengths"],
                batch["pitch"],
                batch["pitchf"],
                batch["spec"],
                batch["spec_lengths"],
                batch["sid"],
            )
            if isinstance(output, tuple):
                loss = output[0].mean()
            else:
                loss = output.mean()
            loss.backward()
            optimizer.step()
        except Exception as e:
            print(f"    Warning: PyTorch forward failed: {e}")
            return None

    if device == "mps":
        torch.mps.synchronize()

    # Reset memory stats
    if device == "mps":
        torch.mps.reset_peak_memory_stats()
    elif device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Benchmark
    print(f"    Running benchmark ({num_steps} steps)...")
    total_samples = 0
    start_time = time.perf_counter()

    for step in range(num_steps):
        batch = create_batch()
        optimizer.zero_grad()
        output = model(
            batch["phone"],
            batch["phone_lengths"],
            batch["pitch"],
            batch["pitchf"],
            batch["spec"],
            batch["spec_lengths"],
            batch["sid"],
        )
        if isinstance(output, tuple):
            loss = output[0].mean()
        else:
            loss = output.mean()
        loss.backward()
        optimizer.step()
        total_samples += batch_size

    if device == "mps":
        torch.mps.synchronize()

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # Get memory
    if device == "mps":
        peak_memory = torch.mps.driver_allocated_memory() / (1024 * 1024)
    elif device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        peak_memory = 0.0

    return {
        "samples_per_second": total_samples / total_time,
        "time_per_step_ms": (total_time / num_steps) * 1000,
        "peak_memory_mb": peak_memory,
    }


def benchmark_mlx_training(batch_size: int, num_steps: int, warmup_steps: int = 5) -> Dict[str, float]:
    """
    Benchmark MLX training step.

    Returns dict with samples_per_second, time_per_step_ms, peak_memory_mb
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from rvc_mlx.lib.mlx.synthesizers import Synthesizer
    from rvc_mlx.train.losses import kl_loss

    print(f"  MLX device: Metal")

    # Create model
    model = Synthesizer(
        spec_channels=1025,
        segment_size=32000,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.0,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[10, 10, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16, 4, 4],
        spk_embed_dim=256,
        gin_channels=256,
        sr=40000,
        use_f0=True,
    )

    optimizer = optim.AdamW(learning_rate=1e-4)

    # Create synthetic batch
    def create_batch():
        phone_length = 256
        spec_length = 512
        return {
            "phone": mx.random.normal((batch_size, phone_length, 768)),
            "phone_lengths": mx.array([phone_length] * batch_size),
            "pitch": mx.random.normal((batch_size, phone_length)),
            "pitchf": mx.random.uniform(shape=(batch_size, phone_length)) * 400 + 50,
            "spec": mx.random.normal((batch_size, 1025, spec_length)),
            "spec_lengths": mx.array([spec_length] * batch_size),
            "wave": mx.random.normal((batch_size, 32000, 1)),
            "sid": mx.zeros((batch_size,), dtype=mx.int32),
        }

    def loss_fn(model, batch):
        output = model(
            batch["phone"],
            batch["phone_lengths"],
            batch["pitch"],
            batch["pitchf"],
            batch["spec"],
            batch["spec_lengths"],
            batch["sid"],
        )
        o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = output
        loss = kl_loss(z_p, logs_q, m_p, logs_p, y_mask)
        return loss, {}

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Warmup
    print(f"    Warming up ({warmup_steps} steps)...")
    for _ in range(warmup_steps):
        batch = create_batch()
        (loss, aux), grads = loss_and_grad(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    # Reset memory stats
    try:
        mx.metal.reset_peak_memory()
    except:
        pass

    # Benchmark
    print(f"    Running benchmark ({num_steps} steps)...")
    total_samples = 0
    start_time = time.perf_counter()

    for step in range(num_steps):
        batch = create_batch()
        (loss, aux), grads = loss_and_grad(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        total_samples += batch_size

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # Get memory
    try:
        peak_memory = mx.metal.get_peak_memory() / (1024 * 1024)
    except:
        peak_memory = mx.metal.get_active_memory() / (1024 * 1024)

    return {
        "samples_per_second": total_samples / total_time,
        "time_per_step_ms": (total_time / num_steps) * 1000,
        "peak_memory_mb": peak_memory,
    }


def run_comparison(
    batch_sizes: List[int],
    num_steps: int = 50,
    warmup_steps: int = 5,
) -> List[ComparisonResult]:
    """Run full comparison suite."""
    results = []

    print("\n" + "=" * 70)
    print("PyTorch vs MLX Training Performance Comparison")
    print("=" * 70)

    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        print("-" * 40)

        # Benchmark PyTorch
        print("  Running PyTorch benchmark...")
        pytorch_result = benchmark_pytorch_training(batch_size, num_steps, warmup_steps)

        # Benchmark MLX
        print("  Running MLX benchmark...")
        mlx_result = benchmark_mlx_training(batch_size, num_steps, warmup_steps)

        if pytorch_result and mlx_result:
            speedup = mlx_result["samples_per_second"] / pytorch_result["samples_per_second"]
            memory_ratio = pytorch_result["peak_memory_mb"] / max(mlx_result["peak_memory_mb"], 1)

            result = ComparisonResult(
                batch_size=batch_size,
                num_steps=num_steps,
                pytorch_samples_per_second=pytorch_result["samples_per_second"],
                pytorch_time_per_step_ms=pytorch_result["time_per_step_ms"],
                pytorch_peak_memory_mb=pytorch_result["peak_memory_mb"],
                mlx_samples_per_second=mlx_result["samples_per_second"],
                mlx_time_per_step_ms=mlx_result["time_per_step_ms"],
                mlx_peak_memory_mb=mlx_result["peak_memory_mb"],
                speedup=speedup,
                memory_ratio=memory_ratio,
            )
            results.append(result)

            print(f"\n  Results:")
            print(f"    PyTorch: {pytorch_result['samples_per_second']:.1f} samples/s, "
                  f"{pytorch_result['time_per_step_ms']:.1f} ms/step, "
                  f"{pytorch_result['peak_memory_mb']:.0f} MB")
            print(f"    MLX:     {mlx_result['samples_per_second']:.1f} samples/s, "
                  f"{mlx_result['time_per_step_ms']:.1f} ms/step, "
                  f"{mlx_result['peak_memory_mb']:.0f} MB")
            print(f"    Speedup: {speedup:.2f}x")
        elif mlx_result:
            print(f"\n  MLX only: {mlx_result['samples_per_second']:.1f} samples/s, "
                  f"{mlx_result['time_per_step_ms']:.1f} ms/step")

    return results


def print_comparison_summary(results: List[ComparisonResult]):
    """Print comparison summary table."""
    if not results:
        print("\nNo comparison results available.")
        return

    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)

    print(f"\n{'Batch':<8} {'PyTorch (s/s)':<14} {'MLX (s/s)':<14} {'Speedup':<10} {'Memory Ratio':<12}")
    print("-" * 58)

    for r in results:
        print(f"{r.batch_size:<8} {r.pytorch_samples_per_second:<14.1f} "
              f"{r.mlx_samples_per_second:<14.1f} {r.speedup:<10.2f}x "
              f"{r.memory_ratio:<12.2f}x")

    # Calculate averages
    avg_speedup = sum(r.speedup for r in results) / len(results)
    avg_memory_ratio = sum(r.memory_ratio for r in results) / len(results)

    print("-" * 58)
    print(f"{'Average':<8} {'':<14} {'':<14} {avg_speedup:<10.2f}x {avg_memory_ratio:<12.2f}x")

    print(f"\nMLX is on average {avg_speedup:.2f}x faster than PyTorch MPS")
    if avg_memory_ratio > 1:
        print(f"MLX uses {avg_memory_ratio:.2f}x less memory than PyTorch")


def save_comparison_results(results: List[ComparisonResult], output_path: str):
    """Save comparison results to JSON."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "comparison": "pytorch_mps_vs_mlx",
        "results": [r.to_dict() for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PyTorch vs MLX Training Comparison")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8",
                        help="Comma-separated batch sizes to test")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Number of steps per benchmark")
    parser.add_argument("--warmup-steps", type=int, default=5,
                        help="Number of warmup steps")
    parser.add_argument("--output", type=str, default="comparison_results.json",
                        help="Output file for results")

    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    results = run_comparison(
        batch_sizes=batch_sizes,
        num_steps=args.num_steps,
        warmup_steps=args.warmup_steps,
    )

    print_comparison_summary(results)
    save_comparison_results(results, args.output)


if __name__ == "__main__":
    main()
