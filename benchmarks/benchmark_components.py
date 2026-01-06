#!/usr/bin/env python3
"""
Individual Component Benchmarks
Tests each RVC component separately:
- TextEncoder
- Generator (HiFiGAN-NSF)
- RMVPE (pitch detection)
- HuBERT (feature extraction)

Measures performance and numerical accuracy for each component.
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


def benchmark_text_encoder(pt_model_path, mlx_model_path, num_runs=5):
    """Benchmark TextEncoder component."""
    print("\n" + "="*60)
    print("TEXT ENCODER BENCHMARK")
    print("="*60)

    import torch
    import mlx.core as mx
    sys.path.insert(0, str(Path(project_root) / "rvc"))

    from rvc.lib.algorithm.synthesizers import Synthesizer as PTSynth
    from rvc_mlx.lib.mlx.synthesizers import Synthesizer as MLXSynth

    # Load models
    print("Loading models...")
    pt_ckpt = torch.load(pt_model_path, map_location="cpu", weights_only=True)
    config = pt_ckpt["config"]

    kwargs = {
        "spec_channels": config[0], "segment_size": config[1],
        "inter_channels": config[2], "hidden_channels": config[3],
        "filter_channels": config[4], "n_heads": config[5],
        "n_layers": config[6], "kernel_size": config[7],
        "p_dropout": config[8], "resblock": config[9],
        "resblock_kernel_sizes": config[10],
        "resblock_dilation_sizes": config[11],
        "upsample_rates": config[12],
        "upsample_initial_channel": config[13],
        "upsample_kernel_sizes": config[14],
        "spk_embed_dim": config[15], "gin_channels": config[16],
        "sr": config[17], "use_f0": True,
        "text_enc_hidden_dim": 768, "vocoder": "NSF"
    }

    net_g_pt = PTSynth(**kwargs)
    net_g_pt.load_state_dict(pt_ckpt["weight"], strict=False)
    net_g_pt.eval()
    net_g_pt.remove_weight_norm()

    net_g_mlx = MLXSynth(**kwargs)
    net_g_mlx.load_weights(mlx_model_path, strict=False)
    mx.eval(net_g_mlx.parameters())

    # Create test inputs
    seq_len = 100
    phone_pt = torch.randn(1, seq_len, 768)
    pitch_pt = torch.full((1, seq_len), 50).long()
    lengths_pt = torch.tensor([seq_len]).long()

    phone_mlx = mx.array(phone_pt.numpy())
    pitch_mlx = mx.array(pitch_pt.numpy()).astype(mx.int32)
    lengths_mlx = mx.array([seq_len], dtype=mx.int32)

    # PyTorch benchmark
    print("\nPyTorch TextEncoder:")
    with torch.no_grad():
        # Warmup
        _ = net_g_pt.enc_p(phone_pt, pitch_pt, lengths_pt)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

        # Benchmark
        times = []
        for i in range(num_runs):
            start = time.perf_counter()
            m_p_pt, logs_p_pt, x_mask_pt = net_g_pt.enc_p(phone_pt, pitch_pt, lengths_pt)
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        pt_time = np.median(times)
        print(f"  Median time: {pt_time*1000:.2f}ms")
        pt_output = m_p_pt.numpy()

    # MLX benchmark
    print("\nMLX TextEncoder:")
    # Warmup
    _ = net_g_mlx.enc_p(phone_mlx, pitch_mlx, lengths_mlx)
    mx.eval(_[0])

    # Benchmark
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        m_p_mlx, logs_p_mlx, x_mask_mlx = net_g_mlx.enc_p(phone_mlx, pitch_mlx, lengths_mlx)
        mx.eval(m_p_mlx)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    mlx_time = np.median(times)
    print(f"  Median time: {mlx_time*1000:.2f}ms")
    mlx_output = np.array(m_p_mlx)

    # Compare
    print(f"\n📊 Performance:")
    print(f"  Speedup: {pt_time/mlx_time:.2f}x")

    # Accuracy
    diff = np.abs(pt_output - mlx_output)
    correlation = np.corrcoef(pt_output.flatten(), mlx_output.flatten())[0, 1]

    print(f"\n📈 Accuracy:")
    print(f"  Max diff: {diff.max():.6f}")
    print(f"  RMSE: {np.sqrt(np.mean(diff**2)):.6f}")
    print(f"  Correlation: {correlation:.6f}")
    print(f"  Status: {'✅ Match' if correlation > 0.999 else '⚠️ Check'}")

    return {"pt_time": pt_time, "mlx_time": mlx_time, "correlation": correlation}


def benchmark_generator(pt_model_path, mlx_model_path, num_runs=5):
    """Benchmark Generator component."""
    print("\n" + "="*60)
    print("GENERATOR BENCHMARK")
    print("="*60)

    import torch
    import mlx.core as mx
    sys.path.insert(0, str(Path(project_root) / "rvc"))

    from rvc.lib.algorithm.synthesizers import Synthesizer as PTSynth
    from rvc_mlx.lib.mlx.synthesizers import Synthesizer as MLXSynth

    # Load models (same as above)
    print("Loading models...")
    pt_ckpt = torch.load(pt_model_path, map_location="cpu", weights_only=True)
    config = pt_ckpt["config"]

    kwargs = {
        "spec_channels": config[0], "segment_size": config[1],
        "inter_channels": config[2], "hidden_channels": config[3],
        "filter_channels": config[4], "n_heads": config[5],
        "n_layers": config[6], "kernel_size": config[7],
        "p_dropout": config[8], "resblock": config[9],
        "resblock_kernel_sizes": config[10],
        "resblock_dilation_sizes": config[11],
        "upsample_rates": config[12],
        "upsample_initial_channel": config[13],
        "upsample_kernel_sizes": config[14],
        "spk_embed_dim": config[15], "gin_channels": config[16],
        "sr": config[17], "use_f0": True,
        "text_enc_hidden_dim": 768, "vocoder": "NSF"
    }

    net_g_pt = PTSynth(**kwargs)
    net_g_pt.load_state_dict(pt_ckpt["weight"], strict=False)
    net_g_pt.eval()
    net_g_pt.remove_weight_norm()

    net_g_mlx = MLXSynth(**kwargs)
    net_g_mlx.load_weights(mlx_model_path, strict=False)
    mx.eval(net_g_mlx.parameters())

    # Create test inputs (latent codes)
    seq_len = 100
    z_pt = torch.randn(1, config[2], seq_len)  # (B, C, T)
    f0_pt = torch.full((1, seq_len), 220.0)
    g_pt = torch.randn(1, 1, config[16])

    z_mlx = mx.array(z_pt.numpy())
    f0_mlx = mx.array(f0_pt.numpy())
    g_mlx = mx.array(g_pt.numpy())

    # PyTorch benchmark
    print("\nPyTorch Generator:")
    with torch.no_grad():
        # Warmup
        _ = net_g_pt.dec(z_pt, f0_pt, g_pt)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

        # Benchmark
        times = []
        for i in range(num_runs):
            start = time.perf_counter()
            output_pt = net_g_pt.dec(z_pt, f0_pt, g_pt)
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        pt_time = np.median(times)
        print(f"  Median time: {pt_time*1000:.2f}ms")
        pt_output = output_pt.numpy()

    # MLX benchmark
    print("\nMLX Generator:")
    # Warmup
    _ = net_g_mlx.dec(z_mlx, f0_mlx, g_mlx)
    mx.eval(_)

    # Benchmark
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        output_mlx = net_g_mlx.dec(z_mlx, f0_mlx, g_mlx)
        mx.eval(output_mlx)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    mlx_time = np.median(times)
    print(f"  Median time: {mlx_time*1000:.2f}ms")
    mlx_output = np.array(output_mlx)

    # Handle transpose if needed
    if pt_output.shape != mlx_output.shape:
        mlx_output = mlx_output.transpose(0, 2, 1)

    # Compare
    print(f"\n📊 Performance:")
    print(f"  Speedup: {pt_time/mlx_time:.2f}x")

    # Accuracy
    diff = np.abs(pt_output - mlx_output)
    correlation = np.corrcoef(pt_output.flatten(), mlx_output.flatten())[0, 1]

    print(f"\n📈 Accuracy:")
    print(f"  Max diff: {diff.max():.6f}")
    print(f"  RMSE: {np.sqrt(np.mean(diff**2)):.6f}")
    print(f"  Correlation: {correlation:.6f}")
    print(f"  Status: {'✅ Match' if correlation > 0.999 else '⚠️ Check'}")

    return {"pt_time": pt_time, "mlx_time": mlx_time, "correlation": correlation}


def benchmark_rmvpe(num_runs=5):
    """Benchmark RMVPE pitch detection."""
    print("\n" + "="*60)
    print("RMVPE BENCHMARK")
    print("="*60)

    import torch
    import mlx.core as mx

    # Create test audio
    duration = 5.0
    sr = 16000
    n_samples = int(duration * sr)
    t = np.linspace(0, duration, n_samples)
    audio = (
        0.3 * np.sin(2 * np.pi * 220 * t) +
        0.2 * np.sin(2 * np.pi * 440 * t) +
        0.1 * np.random.randn(n_samples)
    ).astype(np.float32)

    print(f"Audio: {duration}s @ {sr}Hz")

    try:
        # PyTorch RMVPE
        print("\nPyTorch RMVPE:")
        sys.path.insert(0, str(Path(project_root) / "rvc"))
        from rvc.lib.predictors.RMVPE import RMVPE0Predictor as PTRMVPE

        model_file = "rvc/models/predictors/rmvpe.pt"
        if not os.path.exists(model_file):
            print(f"  ⚠️ Model not found: {model_file}")
            pt_time = None
        else:
            predictor = PTRMVPE(model_path=model_file, device="mps")

            # Warmup
            _ = predictor.infer_from_audio(audio[:sr], thred=0.03)
            if torch.backends.mps.is_available():
                torch.mps.synchronize()

            # Benchmark
            times = []
            for i in range(num_runs):
                start = time.perf_counter()
                f0_pt = predictor.infer_from_audio(audio, thred=0.03)
                if torch.backends.mps.is_available():
                    torch.mps.synchronize()
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            pt_time = np.median(times)
            print(f"  Median time: {pt_time*1000:.2f}ms")

    except Exception as e:
        print(f"  ⚠️ Error: {e}")
        pt_time = None

    try:
        # MLX RMVPE
        print("\nMLX RMVPE:")
        from rvc_mlx.lib.mlx.rmvpe import RMVPE0Predictor as MLXRMVPE

        weights_file = "rvc_mlx/models/predictors/rmvpe_mlx.npz"
        if not os.path.exists(weights_file):
            print(f"  ⚠️ Model not found: {weights_file}")
            mlx_time = None
        else:
            predictor = MLXRMVPE(weights_path=weights_file)

            # Warmup
            _ = predictor.infer_from_audio(audio[:sr], thred=0.03)
            mx.eval(mx.zeros(1))

            # Benchmark
            times = []
            for i in range(num_runs):
                start = time.perf_counter()
                f0_mlx = predictor.infer_from_audio(audio, thred=0.03)
                mx.eval(mx.zeros(1))
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            mlx_time = np.median(times)
            print(f"  Median time: {mlx_time*1000:.2f}ms")

    except Exception as e:
        print(f"  ⚠️ Error: {e}")
        mlx_time = None

    # Compare
    if pt_time and mlx_time:
        print(f"\n📊 Performance:")
        print(f"  Speedup: {pt_time/mlx_time:.2f}x")
        print(f"  Realtime factor (MLX): {duration/mlx_time:.1f}x")

    return {"pt_time": pt_time, "mlx_time": mlx_time}


def main():
    parser = argparse.ArgumentParser(description="Benchmark individual RVC components")
    parser.add_argument("--pt-model", type=str,
                       default="/Users/mcruz/Library/Application Support/Replay/com.replay.Replay/models/Drake/model.pth",
                       help="Path to PyTorch model")
    parser.add_argument("--mlx-model", type=str,
                       default="rvc_mlx/models/checkpoints/Drake.npz",
                       help="Path to MLX model")
    parser.add_argument("--runs", type=int, default=5,
                       help="Number of benchmark runs per component")
    parser.add_argument("--component", type=str, choices=["all", "encoder", "generator", "rmvpe"],
                       default="all", help="Which component to benchmark")

    args = parser.parse_args()

    print("="*60)
    print("RVC COMPONENT BENCHMARKS")
    print("="*60)
    print(f"Runs per component: {args.runs}")

    results = {}

    if args.component in ["all", "encoder"]:
        results["encoder"] = benchmark_text_encoder(
            args.pt_model, args.mlx_model, args.runs
        )

    if args.component in ["all", "generator"]:
        results["generator"] = benchmark_generator(
            args.pt_model, args.mlx_model, args.runs
        )

    if args.component in ["all", "rmvpe"]:
        results["rmvpe"] = benchmark_rmvpe(args.runs)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for component, result in results.items():
        if result and "pt_time" in result and "mlx_time" in result:
            if result["pt_time"] and result["mlx_time"]:
                speedup = result["pt_time"] / result["mlx_time"]
                status = "MLX faster" if speedup > 1 else "PyTorch faster"
                print(f"\n{component.upper()}:")
                print(f"  PyTorch: {result['pt_time']*1000:.2f}ms")
                print(f"  MLX:     {result['mlx_time']*1000:.2f}ms")
                print(f"  Speedup: {speedup:.2f}x ({status})")
                if "correlation" in result:
                    print(f"  Correlation: {result['correlation']:.6f}")

    print("\n✅ Component benchmarks complete!")


if __name__ == "__main__":
    main()
