"""
Training Benchmark Suite for RVC MLX

Measures training performance metrics:
- Samples per second (throughput)
- Time per epoch
- Peak memory usage
- GPU utilization
"""

import os
import sys
import time
import argparse
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import mlx.core as mx


@dataclass
class BenchmarkResult:
    """Results from a training benchmark run."""
    name: str
    batch_size: int
    num_steps: int
    total_time_seconds: float
    samples_per_second: float
    time_per_step_ms: float
    peak_memory_mb: float
    average_loss: float
    device: str = "mlx"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    batch_sizes: List[int] = None
    num_steps: int = 100
    warmup_steps: int = 10
    segment_size: int = 32000
    sample_rate: int = 40000
    pretrain_g_path: str = None  # Path to pretrained generator
    pretrain_d_path: str = None  # Path to pretrained discriminator
    use_real_weights: bool = False  # Use actual pretrained weights

    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4, 8, 16]

        # Default pretrain paths
        if self.pretrain_g_path is None:
            self.pretrain_g_path = f"rvc/models/pretraineds/hifi-gan/f0G{self.sample_rate // 1000}k.pth"
        if self.pretrain_d_path is None:
            self.pretrain_d_path = f"rvc/models/pretraineds/hifi-gan/f0D{self.sample_rate // 1000}k.pth"


def create_synthetic_batch(batch_size: int, config: BenchmarkConfig) -> Dict[str, mx.array]:
    """Create synthetic training batch for benchmarking."""
    # Typical dimensions for RVC training
    phone_length = 256  # HuBERT features
    spec_length = 512  # Spectrogram frames
    spec_channels = 1025  # Spectral bins

    batch = {
        "phone": mx.random.normal((batch_size, phone_length, 768)),  # HuBERT dim
        "phone_lengths": mx.array([phone_length] * batch_size),
        "pitch": mx.random.normal((batch_size, phone_length)),
        "pitchf": mx.random.uniform(shape=(batch_size, phone_length)) * 400 + 50,  # F0 range
        "spec": mx.random.normal((batch_size, spec_channels, spec_length)),
        "spec_lengths": mx.array([spec_length] * batch_size),
        "wave": mx.random.normal((batch_size, config.segment_size, 1)),
        "wave_lengths": mx.array([config.segment_size] * batch_size),
        "sid": mx.zeros((batch_size,), dtype=mx.int32),
    }

    return batch


def get_memory_usage() -> float:
    """Get current MLX memory usage in MB."""
    try:
        # MLX memory tracking
        return mx.metal.get_active_memory() / (1024 * 1024)
    except:
        return 0.0


def load_pretrained_weights(model, pretrain_path: str) -> bool:
    """
    Load pretrained weights into model.

    Args:
        model: MLX model to load weights into
        pretrain_path: Path to .pth or .npz pretrained weights

    Returns:
        True if weights were loaded successfully
    """
    import os

    if not os.path.exists(pretrain_path):
        print(f"  Pretrain not found: {pretrain_path}")
        return False

    try:
        if pretrain_path.endswith(".npz"):
            # Already converted MLX weights
            weights = dict(mx.load(pretrain_path))
            model.load_weights(list(weights.items()))
            print(f"  Loaded MLX weights from {pretrain_path}")
            return True
        elif pretrain_path.endswith(".pth"):
            # PyTorch weights - need conversion
            print(f"  Converting PyTorch weights from {pretrain_path}...")
            try:
                from tools.convert_rvc_model import convert_weights
                import tempfile

                # Convert to temporary npz
                with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
                    tmp_path = tmp.name

                convert_weights(pretrain_path, tmp_path)
                weights = dict(mx.load(tmp_path))
                model.load_weights(list(weights.items()))

                # Clean up
                os.unlink(tmp_path)
                if os.path.exists(tmp_path.replace(".npz", ".json")):
                    os.unlink(tmp_path.replace(".npz", ".json"))

                print(f"  Loaded and converted weights from {pretrain_path}")
                return True
            except ImportError:
                print(f"  Could not import converter. Run with --use-real-weights=false")
                return False
    except Exception as e:
        print(f"  Error loading weights: {e}")
        return False

    return False


def benchmark_forward_pass(batch_size: int, config: BenchmarkConfig) -> BenchmarkResult:
    """
    Benchmark generator forward pass only.
    """
    from rvc_mlx.lib.mlx.synthesizers import Synthesizer

    print(f"\n  Forward pass benchmark (batch_size={batch_size})...")

    # Create model with typical config
    model = Synthesizer(
        spec_channels=1025,
        segment_size=config.segment_size,
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
        sr=config.sample_rate,
        use_f0=True,
    )

    # Load pretrained weights if requested
    if config.use_real_weights and config.pretrain_g_path:
        loaded = load_pretrained_weights(model, config.pretrain_g_path)
        if loaded:
            print(f"    Using real pretrained weights")
        else:
            print(f"    Warning: Could not load weights, using random init")

    # Warmup
    print(f"    Warming up ({config.warmup_steps} steps)...")
    for _ in range(config.warmup_steps):
        batch = create_synthetic_batch(batch_size, config)
        _ = model(
            batch["phone"],
            batch["phone_lengths"],
            batch["pitch"],
            batch["pitchf"],
            batch["spec"],
            batch["spec_lengths"],
            batch["sid"],
        )
        mx.eval(model.parameters())

    # Clear memory stats
    try:
        mx.metal.reset_peak_memory()
    except:
        pass

    # Benchmark
    print(f"    Running benchmark ({config.num_steps} steps)...")
    total_samples = 0
    total_time = 0.0

    for step in range(config.num_steps):
        batch = create_synthetic_batch(batch_size, config)

        start_time = time.perf_counter()
        output = model(
            batch["phone"],
            batch["phone_lengths"],
            batch["pitch"],
            batch["pitchf"],
            batch["spec"],
            batch["spec_lengths"],
            batch["sid"],
        )
        mx.eval(output[0])  # Force evaluation
        end_time = time.perf_counter()

        total_time += (end_time - start_time)
        total_samples += batch_size

    # Get peak memory
    try:
        peak_memory = mx.metal.get_peak_memory() / (1024 * 1024)
    except:
        peak_memory = get_memory_usage()

    samples_per_second = total_samples / total_time
    time_per_step_ms = (total_time / config.num_steps) * 1000

    return BenchmarkResult(
        name="forward_pass",
        batch_size=batch_size,
        num_steps=config.num_steps,
        total_time_seconds=total_time,
        samples_per_second=samples_per_second,
        time_per_step_ms=time_per_step_ms,
        peak_memory_mb=peak_memory,
        average_loss=0.0,  # No loss in forward-only benchmark
    )


def benchmark_training_step(batch_size: int, config: BenchmarkConfig) -> BenchmarkResult:
    """
    Benchmark full training step (forward + backward + optimizer).
    """
    import mlx.nn as nn
    import mlx.optimizers as optim
    from rvc_mlx.lib.mlx.synthesizers import Synthesizer
    from rvc_mlx.train.discriminators import MultiPeriodDiscriminator
    from rvc_mlx.train.losses import generator_loss, discriminator_loss, kl_loss

    print(f"\n  Training step benchmark (batch_size={batch_size})...")

    # Create models
    model_g = Synthesizer(
        spec_channels=1025,
        segment_size=config.segment_size,
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
        sr=config.sample_rate,
        use_f0=True,
    )

    model_d = MultiPeriodDiscriminator()

    # Load pretrained weights if requested
    if config.use_real_weights:
        if config.pretrain_g_path:
            loaded_g = load_pretrained_weights(model_g, config.pretrain_g_path)
            if loaded_g:
                print(f"    Loaded generator pretrained weights")
        if config.pretrain_d_path:
            loaded_d = load_pretrained_weights(model_d, config.pretrain_d_path)
            if loaded_d:
                print(f"    Loaded discriminator pretrained weights")

    # Optimizers
    optimizer_g = optim.AdamW(learning_rate=1e-4)
    optimizer_d = optim.AdamW(learning_rate=1e-4)

    def loss_fn_g(model, batch):
        """Generator loss function."""
        o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = model(
            batch["phone"],
            batch["phone_lengths"],
            batch["pitch"],
            batch["pitchf"],
            batch["spec"],
            batch["spec_lengths"],
            batch["sid"],
        )

        # Simplified loss for benchmark
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, y_mask)
        return loss_kl, {"kl": loss_kl}

    # Create value_and_grad function
    loss_and_grad_g = nn.value_and_grad(model_g, loss_fn_g)

    # Warmup
    print(f"    Warming up ({config.warmup_steps} steps)...")
    for _ in range(config.warmup_steps):
        batch = create_synthetic_batch(batch_size, config)
        (loss, aux), grads = loss_and_grad_g(model_g, batch)
        optimizer_g.update(model_g, grads)
        mx.eval(model_g.parameters(), optimizer_g.state)

    # Clear memory stats
    try:
        mx.metal.reset_peak_memory()
    except:
        pass

    # Benchmark
    print(f"    Running benchmark ({config.num_steps} steps)...")
    total_samples = 0
    total_time = 0.0
    total_loss = 0.0

    for step in range(config.num_steps):
        batch = create_synthetic_batch(batch_size, config)

        start_time = time.perf_counter()
        (loss, aux), grads = loss_and_grad_g(model_g, batch)
        optimizer_g.update(model_g, grads)
        mx.eval(model_g.parameters(), optimizer_g.state)
        end_time = time.perf_counter()

        total_time += (end_time - start_time)
        total_samples += batch_size
        total_loss += float(loss.item())

    # Get peak memory
    try:
        peak_memory = mx.metal.get_peak_memory() / (1024 * 1024)
    except:
        peak_memory = get_memory_usage()

    samples_per_second = total_samples / total_time
    time_per_step_ms = (total_time / config.num_steps) * 1000
    average_loss = total_loss / config.num_steps

    return BenchmarkResult(
        name="training_step",
        batch_size=batch_size,
        num_steps=config.num_steps,
        total_time_seconds=total_time,
        samples_per_second=samples_per_second,
        time_per_step_ms=time_per_step_ms,
        peak_memory_mb=peak_memory,
        average_loss=average_loss,
    )


def run_benchmark_suite(config: BenchmarkConfig) -> List[BenchmarkResult]:
    """Run full benchmark suite across batch sizes."""
    results = []

    print("\n" + "=" * 60)
    print("RVC MLX Training Benchmark Suite")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Batch sizes: {config.batch_sizes}")
    print(f"  - Steps per benchmark: {config.num_steps}")
    print(f"  - Warmup steps: {config.warmup_steps}")
    print(f"  - Segment size: {config.segment_size}")
    print(f"  - Sample rate: {config.sample_rate}")

    print("\n" + "-" * 60)
    print("Forward Pass Benchmarks")
    print("-" * 60)

    for batch_size in config.batch_sizes:
        try:
            result = benchmark_forward_pass(batch_size, config)
            results.append(result)
            print(f"    Results: {result.samples_per_second:.1f} samples/sec, "
                  f"{result.time_per_step_ms:.1f} ms/step, "
                  f"{result.peak_memory_mb:.0f} MB peak")
        except Exception as e:
            print(f"    Failed for batch_size={batch_size}: {e}")

    print("\n" + "-" * 60)
    print("Training Step Benchmarks")
    print("-" * 60)

    for batch_size in config.batch_sizes:
        try:
            result = benchmark_training_step(batch_size, config)
            results.append(result)
            print(f"    Results: {result.samples_per_second:.1f} samples/sec, "
                  f"{result.time_per_step_ms:.1f} ms/step, "
                  f"{result.peak_memory_mb:.0f} MB peak, "
                  f"loss={result.average_loss:.4f}")
        except Exception as e:
            print(f"    Failed for batch_size={batch_size}: {e}")

    return results


def print_summary(results: List[BenchmarkResult]):
    """Print benchmark summary table."""
    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)

    # Group by benchmark type
    forward_results = [r for r in results if r.name == "forward_pass"]
    training_results = [r for r in results if r.name == "training_step"]

    if forward_results:
        print("\nForward Pass:")
        print(f"{'Batch':<8} {'Samples/s':<12} {'ms/step':<10} {'Memory (MB)':<12}")
        print("-" * 42)
        for r in forward_results:
            print(f"{r.batch_size:<8} {r.samples_per_second:<12.1f} {r.time_per_step_ms:<10.1f} {r.peak_memory_mb:<12.0f}")

    if training_results:
        print("\nTraining Step:")
        print(f"{'Batch':<8} {'Samples/s':<12} {'ms/step':<10} {'Memory (MB)':<12} {'Avg Loss':<10}")
        print("-" * 52)
        for r in training_results:
            print(f"{r.batch_size:<8} {r.samples_per_second:<12.1f} {r.time_per_step_ms:<10.1f} "
                  f"{r.peak_memory_mb:<12.0f} {r.average_loss:<10.4f}")


def save_results(results: List[BenchmarkResult], output_path: str):
    """Save benchmark results to JSON."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": "mlx",
        "results": [r.to_dict() for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="RVC MLX Training Benchmark")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8",
                        help="Comma-separated batch sizes to test")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Number of steps per benchmark")
    parser.add_argument("--warmup-steps", type=int, default=5,
                        help="Number of warmup steps")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                        help="Output file for results")
    parser.add_argument("--forward-only", action="store_true",
                        help="Only run forward pass benchmarks")
    parser.add_argument("--use-real-weights", action="store_true",
                        help="Use real pretrained weights instead of random init")
    parser.add_argument("--pretrain-g", type=str, default=None,
                        help="Path to pretrained generator (.pth or .npz)")
    parser.add_argument("--pretrain-d", type=str, default=None,
                        help="Path to pretrained discriminator (.pth or .npz)")
    parser.add_argument("--sample-rate", type=int, default=40000,
                        choices=[32000, 40000, 48000],
                        help="Sample rate for pretrains")

    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    config = BenchmarkConfig(
        batch_sizes=batch_sizes,
        num_steps=args.num_steps,
        warmup_steps=args.warmup_steps,
        sample_rate=args.sample_rate,
        use_real_weights=args.use_real_weights,
        pretrain_g_path=args.pretrain_g,
        pretrain_d_path=args.pretrain_d,
    )

    print("\nBenchmark Configuration:")
    print(f"  Use real weights: {config.use_real_weights}")
    if config.use_real_weights:
        print(f"  Generator pretrain: {config.pretrain_g_path}")
        print(f"  Discriminator pretrain: {config.pretrain_d_path}")

    results = run_benchmark_suite(config)
    print_summary(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
