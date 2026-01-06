#!/usr/bin/env python3
"""
Check if LayerNorm weights and parameters match between PyTorch and MLX.
"""

import torch
import mlx.core as mx
import numpy as np

def main():
    # Load PyTorch model
    pt_path = "/Users/mcruz/Library/Application Support/Replay/com.replay.Replay/models/Drake/model.pth"
    print(f"Loading PyTorch model...")
    pt_ckpt = torch.load(pt_path, map_location="cpu")
    pt_weights = pt_ckpt["weight"]

    # Load MLX model
    mlx_path = "rvc_mlx/models/checkpoints/Drake.npz"
    print(f"Loading MLX model...")
    mlx_weights = mx.load(mlx_path)

    # Check first LayerNorm in encoder
    print("\n=== Checking enc_p.encoder.norm_layers_1.0 ===")

    pt_key_weight = "enc_p.encoder.norm_layers_1.0.weight"
    pt_key_bias = "enc_p.encoder.norm_layers_1.0.bias"
    mlx_key_weight = "enc_p.encoder.norm1_0.weight"
    mlx_key_bias = "enc_p.encoder.norm1_0.bias"

    if pt_key_weight in pt_weights:
        pt_w = pt_weights[pt_key_weight].numpy()
        print(f"PyTorch weight shape: {pt_w.shape}")
        print(f"PyTorch weight: mean={pt_w.mean():.6f}, std={pt_w.std():.6f}")
        print(f"PyTorch weight values: {pt_w[:10]}")
    else:
        print(f"❌ {pt_key_weight} not found!")

    if mlx_key_weight in mlx_weights:
        mlx_w = np.array(mlx_weights[mlx_key_weight])
        print(f"\nMLX weight shape: {mlx_w.shape}")
        print(f"MLX weight: mean={mlx_w.mean():.6f}, std={mlx_w.std():.6f}")
        print(f"MLX weight values: {mlx_w[:10]}")
    else:
        print(f"❌ {mlx_key_weight} not found!")

    if pt_key_weight in pt_weights and mlx_key_weight in mlx_weights:
        diff = np.abs(pt_w - mlx_w)
        print(f"\nWeight diff: max={diff.max():.10f}, mean={diff.mean():.10f}")
        if diff.max() < 1e-6:
            print("✅ Weights match!")
        else:
            print("❌ Weights don't match!")

    # Check bias
    print("\n--- Bias ---")
    if pt_key_bias in pt_weights:
        pt_b = pt_weights[pt_key_bias].numpy()
        print(f"PyTorch bias shape: {pt_b.shape}")
        print(f"PyTorch bias: mean={pt_b.mean():.6f}, std={pt_b.std():.6f}")
        print(f"PyTorch bias values: {pt_b[:10]}")
    else:
        print(f"❌ {pt_key_bias} not found!")

    if mlx_key_bias in mlx_weights:
        mlx_b = np.array(mlx_weights[mlx_key_bias])
        print(f"\nMLX bias shape: {mlx_b.shape}")
        print(f"MLX bias: mean={mlx_b.mean():.6f}, std={mlx_b.std():.6f}")
        print(f"MLX bias values: {mlx_b[:10]}")
    else:
        print(f"❌ {mlx_key_bias} not found!")

    if pt_key_bias in pt_weights and mlx_key_bias in mlx_weights:
        diff = np.abs(pt_b - mlx_b)
        print(f"\nBias diff: max={diff.max():.10f}, mean={diff.mean():.10f}")
        if diff.max() < 1e-6:
            print("✅ Bias matches!")
        else:
            print("❌ Bias doesn't match!")

    # Test LayerNorm directly
    print("\n\n=== Testing LayerNorm Directly ===")
    import sys
    sys.path.append(os.getcwd())
    sys.path.append("rvc")

    from rvc.lib.algorithm.synthesizers import Synthesizer as PTSynthesizer
    from rvc_mlx.lib.mlx.synthesizers import Synthesizer as MLXSynthesizer

    config = pt_ckpt["config"]
    pt_kwargs = {
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

    net_g_pt = PTSynthesizer(**pt_kwargs)
    net_g_pt.load_state_dict(pt_ckpt["weight"], strict=False)
    net_g_pt.eval()

    net_g_mlx = MLXSynthesizer(**pt_kwargs)
    net_g_mlx.load_weights(mlx_path, strict=False)

    # Create test input
    test_input_pt = torch.randn(1, 192, 100)  # (B, C, T)
    test_input_mlx = mx.array(test_input_pt.numpy()).transpose(0, 2, 1)  # (B, T, C)

    print(f"\nTest input shape: PT={test_input_pt.shape}, MLX={test_input_mlx.shape}")

    # Apply LayerNorm
    pt_norm = net_g_pt.enc_p.encoder.norm_layers_1[0]
    mlx_norm = getattr(net_g_mlx.enc_p.encoder, "norm1_0")

    with torch.no_grad():
        output_pt = pt_norm(test_input_pt)
        output_mlx = mlx_norm(test_input_mlx)

    print(f"\nLayerNorm output shapes: PT={output_pt.shape}, MLX={output_mlx.shape}")

    # Transpose MLX to match PyTorch
    output_mlx_np = np.array(output_mlx).transpose(0, 2, 1)
    output_pt_np = output_pt.numpy()

    print(f"\nPyTorch output: range=[{output_pt_np.min():.6f}, {output_pt_np.max():.6f}], mean={output_pt_np.mean():.6f}, std={output_pt_np.std():.6f}")
    print(f"MLX output:     range=[{output_mlx_np.min():.6f}, {output_mlx_np.max():.6f}], mean={output_mlx_np.mean():.6f}, std={output_mlx_np.std():.6f}")

    diff = np.abs(output_mlx_np - output_pt_np)
    print(f"\nDiff: max={diff.max():.6f}, mean={diff.mean():.6f}, RMSE={np.sqrt(np.mean(diff**2)):.6f}")

    if diff.max() < 0.01:
        print("✅ LayerNorm outputs match!")
    else:
        print("❌ LayerNorm outputs diverge!")

    # Check epsilon
    print(f"\n\nPyTorch LayerNorm epsilon: {pt_norm.eps}")
    print(f"MLX LayerNorm epsilon: {mlx_norm.eps}")

if __name__ == "__main__":
    main()
