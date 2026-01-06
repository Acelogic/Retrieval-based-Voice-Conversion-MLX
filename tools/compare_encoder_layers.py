#!/usr/bin/env python3
"""
Layer-by-layer comparison of Text Encoder to find divergence point.
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

def compare(name, mlx_val, pt_val, transpose_mlx=None):
    """Quick comparison helper."""
    mlx_np = np.array(mlx_val) if isinstance(mlx_val, mx.array) else mlx_val
    pt_np = pt_val.cpu().numpy() if isinstance(pt_val, torch.Tensor) else pt_val

    if transpose_mlx:
        mlx_np = mlx_np.transpose(transpose_mlx)

    if mlx_np.shape != pt_np.shape:
        print(f"{name}:")
        print(f"  ⚠️ SHAPE MISMATCH: MLX={mlx_np.shape}, PT={pt_np.shape}")
        return

    diff = np.abs(mlx_np - pt_np)
    print(f"{name}:")
    print(f"  Shapes: MLX={mlx_np.shape}, PT={pt_np.shape}")
    print(f"  MLX range: [{mlx_np.min():.6f}, {mlx_np.max():.6f}], mean={mlx_np.mean():.6f}")
    print(f"  PT range: [{pt_np.min():.6f}, {pt_np.max():.6f}], mean={pt_np.mean():.6f}")
    print(f"  Max diff: {diff.max():.6f}, RMSE: {np.sqrt(np.mean(diff**2)):.6f}")

    if diff.max() < 0.01:
        print(f"  ✅ Match")
    elif diff.max() < 0.1:
        print(f"  ⚠️  Small divergence")
    else:
        print(f"  ❌ DIVERGENCE!")
    print()

def main():
    # Load models
    pt_path = "/Users/mcruz/Library/Application Support/Replay/com.replay.Replay/models/Drake/model.pth"
    mlx_path = "rvc_mlx/models/checkpoints/Drake.npz"

    print("Loading PyTorch model...")
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

    print("Loading MLX model...")
    net_g_mlx = MLXSynthesizer(**pt_kwargs)
    net_g_mlx.load_weights(mlx_path, strict=False)

    # Create identical inputs
    print("\nCreating test inputs...\n")
    seq_len = 100

    phone_pt = torch.randn(1, seq_len, 768)
    phone_mlx = mx.array(phone_pt.numpy())

    pitch_pt = torch.full((1, seq_len), 50).long()
    pitch_mlx = mx.full((1, seq_len), 50).astype(mx.int32)

    lengths_pt = torch.tensor([seq_len]).long()
    lengths_mlx = mx.array([seq_len], dtype=mx.int32)

    print("=== Text Encoder Layer-by-Layer ===\n")

    with torch.no_grad():
        # Step 1: Phone embedding
        phone_emb_pt = net_g_pt.enc_p.emb_phone(phone_pt)
        phone_emb_mlx = net_g_mlx.enc_p.emb_phone(phone_mlx)

        # Check raw shapes first
        print(f"Raw shapes - PyTorch: {phone_emb_pt.shape}, MLX: {phone_emb_mlx.shape}\n")

        compare("1. Phone embedding", phone_emb_mlx, phone_emb_pt)

        # Step 2: Pitch embedding
        if pitch_pt is not None:
            pitch_emb_pt = net_g_pt.enc_p.emb_pitch(pitch_pt)
            pitch_emb_mlx = net_g_mlx.enc_p.emb_pitch(pitch_mlx)

            print(f"Raw shapes - PyTorch: {pitch_emb_pt.shape}, MLX: {pitch_emb_mlx.shape}\n")

            compare("2. Pitch embedding", pitch_emb_mlx, pitch_emb_pt)

            # Step 3: Sum phone + pitch
            x_pt = phone_emb_pt + pitch_emb_pt
            x_mlx = phone_emb_mlx + pitch_emb_mlx

            compare("3. Phone + Pitch", x_mlx, x_pt)
        else:
            x_pt = phone_emb_pt
            x_mlx = phone_emb_mlx

        # Step 4: Scale by sqrt(hidden_channels)
        import math
        hidden_channels = pt_kwargs["hidden_channels"]
        x_pt = x_pt * math.sqrt(hidden_channels)
        x_mlx = x_mlx * math.sqrt(hidden_channels)

        compare("4. After scaling", x_mlx, x_pt, transpose_mlx=(0, 2, 1))

        # Step 5: LeakyReLU
        x_pt = torch.nn.functional.leaky_relu(x_pt, 0.1)
        x_mlx = mx.maximum(0.1 * x_mlx, x_mlx)  # LeakyReLU

        compare("5. After LeakyReLU", x_mlx, x_pt, transpose_mlx=(0, 2, 1))

        # Step 6: Encoder (attention + FFN layers)
        # This is the complex part - let's just check the final output
        from rvc_mlx.lib.mlx.commons import sequence_mask

        x_mask_pt = net_g_pt.enc_p.encoder.attn_layers[0].make_pad_mask(lengths_pt, x_pt.size(2))
        x_mask_mlx = sequence_mask(lengths_mlx, x_mlx.shape[1])[:, :, None]

        compare("6a. Mask", x_mask_mlx, x_mask_pt, transpose_mlx=(0, 2, 1))

        # Run through encoder
        x_enc_pt = net_g_pt.enc_p.encoder(x_pt, x_mask_pt)
        x_enc_mlx = net_g_mlx.enc_p.encoder(x_mlx, x_mask_mlx)

        compare("6b. After Encoder", x_enc_mlx, x_enc_pt, transpose_mlx=(0, 2, 1))

        # Step 7: Projection (Conv1d)
        # PyTorch expects (B, C, T), MLX expects (B, T, C)
        # PyTorch encoder returns (B, C, T), MLX returns (B, T, C)

        stats_pt = net_g_pt.enc_p.proj(x_enc_pt)
        stats_mlx = net_g_mlx.enc_p.proj(x_enc_mlx)

        compare("7. After projection (stats)", stats_mlx, stats_pt, transpose_mlx=(0, 2, 1))

        # Step 8: Split into m and logs
        # PyTorch splits on dim 1 (channels), MLX on dim 2 (channels)
        m_pt, logs_pt = torch.split(stats_pt, stats_pt.size(1) // 2, dim=1)
        m_mlx, logs_mlx = mx.split(stats_mlx, 2, axis=-1)

        compare("8a. m (mean)", m_mlx, m_pt, transpose_mlx=(0, 2, 1))
        compare("8b. logs (log-variance)", logs_mlx, logs_pt, transpose_mlx=(0, 2, 1))

if __name__ == "__main__":
    main()
