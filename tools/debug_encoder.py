#!/usr/bin/env python3
"""
Debug the Encoder layer by comparing PyTorch vs MLX at each sublayer.
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

    print(f"{name}:")
    print(f"  {status} Max diff: {max_diff:.6f}, RMSE: {rmse:.6f}")
    print(f"  MLX: [{mlx_np.min():.4f}, {mlx_np.max():.4f}], mean={mlx_np.mean():.4f}")
    print(f"  PT:  [{pt_np.min():.4f}, {pt_np.max():.4f}], mean={pt_np.mean():.4f}")

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

    # Create identical inputs
    print("Creating test inputs...\n")
    seq_len = 100

    # Create input that matches after embeddings
    phone_pt = torch.randn(1, seq_len, 768)
    phone_mlx = mx.array(phone_pt.numpy())

    pitch_pt = torch.full((1, seq_len), 50).long()
    pitch_mlx = mx.full((1, seq_len), 50).astype(mx.int32)

    print("=== Encoder Detailed Comparison ===\n")

    with torch.no_grad():
        # Get embeddings (we know these match)
        x_pt = net_g_pt.enc_p.emb_phone(phone_pt) + net_g_pt.enc_p.emb_pitch(pitch_pt)
        x_mlx = net_g_mlx.enc_p.emb_phone(phone_mlx) + net_g_mlx.enc_p.emb_pitch(pitch_mlx)

        import math
        x_pt = x_pt * math.sqrt(pt_kwargs["hidden_channels"])
        x_mlx = x_mlx * math.sqrt(pt_kwargs["hidden_channels"])

        x_pt = torch.nn.functional.leaky_relu(x_pt, 0.1)
        x_mlx = mx.maximum(0.1 * x_mlx, x_mlx)

        print("Input to Encoder (after embeddings + scale + lrelu):")
        compare("  Input", x_mlx, x_pt)
        print()

        # PyTorch transposes to (B, C, T)
        x_pt = x_pt.transpose(1, -1)  # (B, T, C) -> (B, C, T)
        print(f"PyTorch after transpose: {x_pt.shape}")
        print(f"MLX (no transpose): {x_mlx.shape}\n")

        # Create masks
        from rvc_mlx.lib.mlx.commons import sequence_mask as mlx_sequence_mask
        from rvc.lib.algorithm.commons import sequence_mask as pt_sequence_mask

        # PyTorch mask
        lengths_pt = torch.tensor([seq_len]).long()
        x_mask_pt = pt_sequence_mask(lengths_pt, x_pt.size(2)).unsqueeze(1).to(x_pt.dtype)
        print(f"PyTorch x_mask: {x_mask_pt.shape}")

        # MLX mask
        lengths_mlx = mx.array([seq_len], dtype=mx.int32)
        x_mask_mlx = mlx_sequence_mask(lengths_mlx, x_mlx.shape[1])[:, :, None]
        print(f"MLX x_mask: {x_mask_mlx.shape}\n")

        # === Layer 0 ===
        print("=== LAYER 0 ===\n")

        # PyTorch Layer 0
        pt_enc = net_g_pt.enc_p.encoder
        pt_attn = pt_enc.attn_layers[0]
        pt_norm1 = pt_enc.norm_layers_1[0]
        pt_ffn = pt_enc.ffn_layers[0]
        pt_norm2 = pt_enc.norm_layers_2[0]

        # MLX Layer 0
        mlx_enc = net_g_mlx.enc_p.encoder
        mlx_attn = getattr(mlx_enc, "attn_0")
        mlx_norm1 = getattr(mlx_enc, "norm1_0")
        mlx_ffn = getattr(mlx_enc, "ffn_0")
        mlx_norm2 = getattr(mlx_enc, "norm2_0")

        # Attention mask
        attn_mask_pt = x_mask_pt.unsqueeze(2) * x_mask_pt.unsqueeze(-1)
        print(f"PyTorch attn_mask: {attn_mask_pt.shape}")

        # MLX attention mask (computed in Encoder.__call__)
        x_mask_b = x_mask_mlx.astype(mx.float32)
        # x_mask_mlx is (B, L, 1), use the ndim==3 logic from Encoder
        attn_mask_mlx = x_mask_b * x_mask_b.transpose(0, 2, 1)  # (B, L, 1) * (B, 1, L) -> (B, L, L)
        attn_mask_mlx = attn_mask_mlx[:, None, :, :]  # -> (B, 1, L, L)
        print(f"MLX attn_mask: {attn_mask_mlx.shape}\n")

        # Apply x_mask
        x_pt_masked = x_pt * x_mask_pt
        x_mlx_masked = x_mlx * x_mask_mlx

        # Step 1: Attention
        print("Step 1: Attention")
        y_pt = pt_attn(x_pt_masked, x_pt_masked, attn_mask_pt)
        y_mlx = mlx_attn(x_mlx_masked, x_mlx_masked, attn_mask=attn_mask_mlx)

        # PyTorch attention returns (B, C, T), MLX returns (B, T, C)
        # Need to transpose MLX to compare
        y_mlx_transposed = y_mlx.transpose(0, 2, 1)
        compare("  Attention output", y_mlx_transposed, y_pt)
        print()

        # Step 2: Dropout (disabled in eval mode, should be identity)
        y_pt = pt_enc.drop(y_pt)
        y_mlx = mlx_enc.drop(y_mlx)

        # Step 3: First LayerNorm
        print("Step 2: LayerNorm 1 (x + attn)")
        x_pt = pt_norm1(x_pt_masked + y_pt)
        x_mlx = mlx_norm1(x_mlx_masked + y_mlx)

        x_mlx_transposed = x_mlx.transpose(0, 2, 1)
        compare("  After norm1", x_mlx_transposed, x_pt)
        print()

        # Step 4: FFN
        print("Step 3: FFN")
        y_pt = pt_ffn(x_pt, x_mask_pt)
        y_mlx = mlx_ffn(x_mlx, x_mask_mlx)

        y_mlx_transposed = y_mlx.transpose(0, 2, 1)
        compare("  FFN output", y_mlx_transposed, y_pt)
        print()

        # Step 5: Second LayerNorm
        print("Step 4: LayerNorm 2 (x + ffn)")
        y_pt = pt_enc.drop(y_pt)
        y_mlx = mlx_enc.drop(y_mlx)

        x_pt = pt_norm2(x_pt + y_pt)
        x_mlx = mlx_norm2(x_mlx + y_mlx)

        x_mlx_transposed = x_mlx.transpose(0, 2, 1)
        compare("  After norm2 (Layer 0 output)", x_mlx_transposed, x_pt)
        print()

        # Final encoder output (all layers)
        print("=== Full Encoder (all layers) ===\n")
        x_pt_full = net_g_pt.enc_p.encoder(x_pt.transpose(1, -1).transpose(1, -1), x_mask_pt)
        x_mlx_full = net_g_mlx.enc_p.encoder(x_mlx_masked, x_mask_mlx)

        x_mlx_full_transposed = x_mlx_full.transpose(0, 2, 1)
        compare("Final encoder output", x_mlx_full_transposed, x_pt_full)

if __name__ == "__main__":
    main()
