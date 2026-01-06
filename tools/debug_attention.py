#!/usr/bin/env python3
"""
Debug the Attention layer to find why it diverges from PyTorch.
"""

import sys
import os
import torch
import numpy as np
import mlx.core as mx

sys.path.append(os.getcwd())
sys.path.append("rvc")

from rvc.lib.algorithm.synthesizers import Synthesizer as PTSynthesizer
from rvc_mlx.lib.mlx.synthesizers import Synthesizer as MLXSynthesizer

def compare(name, mlx_val, pt_val):
    """Quick comparison helper."""
    mlx_np = np.array(mlx_val) if isinstance(mlx_val, mx.array) else mlx_val
    pt_np = pt_val.cpu().numpy() if isinstance(pt_val, torch.Tensor) else pt_val

    if mlx_np.shape != pt_np.shape:
        print(f"{name}: ⚠️ SHAPE MISMATCH - MLX={mlx_np.shape}, PT={pt_np.shape}")
        return

    diff = np.abs(mlx_np - pt_np)
    max_diff = diff.max()
    rmse = np.sqrt(np.mean(diff**2))

    status = "✅" if max_diff < 0.01 else "⚠️" if max_diff < 0.1 else "❌"

    print(f"{name}: {status} max={max_diff:.6f}, RMSE={rmse:.6f}")

def main():
    # Load models
    pt_path = "/Users/mcruz/Library/Application Support/Replay/com.replay.Replay/models/Drake/model.pth"
    mlx_path = "rvc_mlx/models/checkpoints/Drake.npz"

    print("Loading models...\n")
    pt_ckpt = torch.load(pt_path, map_location="cpu")
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

    # Get first attention layer
    pt_attn = net_g_pt.enc_p.encoder.attn_layers[0]
    mlx_attn = getattr(net_g_mlx.enc_p.encoder, "attn_0")

    print("=== Checking Attention Weights ===\n")

    # Check Q, K, V weights
    for name in ["conv_q", "conv_k", "conv_v", "conv_o"]:
        pt_w = getattr(pt_attn, name).weight.data.numpy()
        mlx_w = np.array(getattr(mlx_attn, name).weight)

        # Conv1d weights: PT (Out, In, K), MLX (Out, K, In)
        # Transpose MLX back to PT format
        if mlx_w.ndim == 3:
            mlx_w = mlx_w.transpose(0, 2, 1)

        print(f"{name}.weight:")
        print(f"  PT shape: {pt_w.shape}, MLX shape: {mlx_w.shape}")

        diff = np.abs(pt_w - mlx_w)
        print(f"  Diff: max={diff.max():.10f}, mean={diff.mean():.10f}")
        if diff.max() < 1e-6:
            print(f"  ✅ Match")
        else:
            print(f"  ❌ Mismatch!")
        print()

    # Test attention forward pass
    print("\n=== Testing Attention Forward Pass ===\n")

    # Create test input
    test_input_pt = torch.randn(1, 192, 100)  # (B, C, T)
    test_input_mlx = mx.array(test_input_pt.numpy()).transpose(0, 2, 1)  # (B, T, C)

    # Create attention mask
    from rvc_mlx.lib.mlx.commons import sequence_mask as mlx_sequence_mask
    from rvc.lib.algorithm.commons import sequence_mask as pt_sequence_mask

    lengths_pt = torch.tensor([100]).long()
    x_mask_pt = pt_sequence_mask(lengths_pt, 100).unsqueeze(1).to(test_input_pt.dtype)
    attn_mask_pt = x_mask_pt.unsqueeze(2) * x_mask_pt.unsqueeze(-1)

    lengths_mlx = mx.array([100], dtype=mx.int32)
    x_mask_mlx = mlx_sequence_mask(lengths_mlx, 100)[:, :, None]
    x_mask_b = x_mask_mlx.astype(mx.float32)
    attn_mask_mlx = x_mask_b * x_mask_b.transpose(0, 2, 1)
    attn_mask_mlx = attn_mask_mlx[:, None, :, :]

    print(f"Input shapes: PT={test_input_pt.shape}, MLX={test_input_mlx.shape}")
    print(f"Mask shapes: PT={attn_mask_pt.shape}, MLX={attn_mask_mlx.shape}\n")

    # Forward pass
    with torch.no_grad():
        output_pt = pt_attn(test_input_pt, test_input_pt, attn_mask_pt)
        output_mlx = mlx_attn(test_input_mlx, test_input_mlx, attn_mask=attn_mask_mlx)

    print(f"Output shapes: PT={output_pt.shape}, MLX={output_mlx.shape}")

    # Transpose MLX to match PT
    output_mlx_np = np.array(output_mlx).transpose(0, 2, 1)
    output_pt_np = output_pt.numpy()

    print(f"\nPT output: [{output_pt_np.min():.4f}, {output_pt_np.max():.4f}], mean={output_pt_np.mean():.4f}")
    print(f"MLX output: [{output_mlx_np.min():.4f}, {output_mlx_np.max():.4f}], mean={output_mlx_np.mean():.4f}")

    diff = np.abs(output_mlx_np - output_pt_np)
    print(f"\nDiff: max={diff.max():.6f}, mean={diff.mean():.6f}, RMSE={np.sqrt(np.mean(diff**2)):.6f}")

    if diff.max() < 0.01:
        print("✅ Attention outputs match!")
    elif diff.max() < 0.1:
        print("⚠️ Small divergence in attention")
    else:
        print("❌ Large divergence in attention!")

    # Detailed step-by-step
    print("\n\n=== Step-by-Step Attention Breakdown ===\n")

    with torch.no_grad():
        # Step 1: Q, K, V projections
        q_pt = pt_attn.conv_q(test_input_pt)
        k_pt = pt_attn.conv_k(test_input_pt)
        v_pt = pt_attn.conv_v(test_input_pt)

        q_mlx = mlx_attn.conv_q(test_input_mlx)
        k_mlx = mlx_attn.conv_k(test_input_mlx)
        v_mlx = mlx_attn.conv_v(test_input_mlx)

        compare("Q projection", q_mlx.transpose(0, 2, 1), q_pt)
        compare("K projection", k_mlx.transpose(0, 2, 1), k_pt)
        compare("V projection", v_mlx.transpose(0, 2, 1), v_pt)

        # Step 2: Reshape for multi-head
        # PT: (B, C, T) -> (B, n_heads, T, head_dim)
        # MLX: (B, T, C) -> (B, n_heads, T, head_dim)

        n_heads = pt_kwargs["n_heads"]
        head_dim = pt_kwargs["hidden_channels"] // n_heads

        # PyTorch reshaping
        B, C, T = q_pt.shape
        q_pt_heads = q_pt.view(B, n_heads, head_dim, T).transpose(2, 3)  # (B, n_heads, T, head_dim)
        k_pt_heads = k_pt.view(B, n_heads, head_dim, T).transpose(2, 3)
        v_pt_heads = v_pt.view(B, n_heads, head_dim, T).transpose(2, 3)

        # MLX reshaping
        B, T, C = q_mlx.shape
        q_mlx_heads = q_mlx.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)  # (B, n_heads, T, head_dim)
        k_mlx_heads = k_mlx.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
        v_mlx_heads = v_mlx.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)

        compare("Q after reshape", q_mlx_heads, q_pt_heads)
        compare("K after reshape", k_mlx_heads, k_pt_heads)
        compare("V after reshape", v_mlx_heads, v_pt_heads)

        # Step 3: Attention scores
        scale = head_dim ** -0.5
        scores_pt = torch.matmul(q_pt_heads, k_pt_heads.transpose(-2, -1)) * scale
        scores_mlx = (q_mlx_heads @ k_mlx_heads.transpose(0, 1, 3, 2)) * scale

        compare("Attention scores (before mask)", scores_mlx, scores_pt)

        # Step 4: Apply mask
        # PT mask is (B, 1, T, T), MLX mask is (B, 1, T, T)
        scores_pt_masked = scores_pt + (1.0 - attn_mask_pt) * -1e4

        # MLX attention uses mx.where
        attn_mask_mlx_expanded = attn_mask_mlx  # Already (B, 1, T, T)
        scores_mlx_masked = mx.where(attn_mask_mlx_expanded == 0, -1e4, scores_mlx)

        compare("Scores after mask", scores_mlx_masked, scores_pt_masked)

        # Step 5: Softmax
        attn_weights_pt = torch.softmax(scores_pt_masked, dim=-1)
        attn_weights_mlx = mx.softmax(scores_mlx_masked, axis=-1)

        compare("Attention weights (after softmax)", attn_weights_mlx, attn_weights_pt)

        # Step 6: Apply to V
        out_pt = torch.matmul(attn_weights_pt, v_pt_heads)
        out_mlx = attn_weights_mlx @ v_mlx_heads

        compare("Output (before reshape)", out_mlx, out_pt)

        # Step 7: Reshape back
        out_pt = out_pt.transpose(2, 3).contiguous().view(B, C, T)
        out_mlx = out_mlx.transpose(0, 2, 1, 3).reshape(B, T, C)

        compare("Output (after reshape)", out_mlx.transpose(0, 2, 1), out_pt)

        # Step 8: Output projection
        final_pt = pt_attn.conv_o(out_pt)
        final_mlx = mlx_attn.conv_o(out_mlx)

        compare("Final output (after conv_o)", final_mlx.transpose(0, 2, 1), final_pt)

if __name__ == "__main__":
    main()
