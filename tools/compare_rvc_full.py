#!/usr/bin/env python3
"""
Compare PyTorch RVC vs MLX RVC inference to find where they diverge.
"""

import sys
import os
import torch
import numpy as np
import librosa
import mlx.core as mx

sys.path.append(os.getcwd())
sys.path.append("rvc")

from rvc.lib.algorithm.synthesizers import Synthesizer as PTSynthesizer
from rvc_mlx.lib.mlx.synthesizers import Synthesizer as MLXSynthesizer

def compare_arrays(name, mlx_arr, pt_arr, threshold=0.01, transpose_mlx=None):
    """Compare MLX and PyTorch arrays.

    Args:
        transpose_mlx: If provided, transpose MLX array axes to match PyTorch format.
                      E.g., (0, 2, 1) to convert (B, T, C) -> (B, C, T)
    """
    mlx_np = np.array(mlx_arr) if isinstance(mlx_arr, mx.array) else mlx_arr
    pt_np = pt_arr.cpu().numpy() if isinstance(pt_arr, torch.Tensor) else pt_arr

    # Transpose MLX if needed to match PyTorch format
    if transpose_mlx is not None:
        mlx_np = mlx_np.transpose(transpose_mlx)

    # Handle different shapes
    if mlx_np.shape != pt_np.shape:
        print(f"\n{name}:")
        print(f"  ⚠️  SHAPE MISMATCH!")
        print(f"  MLX: {mlx_np.shape} (after transpose {transpose_mlx})" if transpose_mlx else f"  MLX: {mlx_np.shape}")
        print(f"  PT:  {pt_np.shape}")
        return False

    diff = np.abs(mlx_np - pt_np)
    max_diff = diff.max()
    mean_diff = diff.mean()
    rmse = np.sqrt(np.mean(diff ** 2))

    print(f"\n{name}:")
    print(f"  Shapes: MLX={mlx_np.shape}, PT={pt_np.shape}")
    print(f"  MLX: range=[{mlx_np.min():.6f}, {mlx_np.max():.6f}], mean={mlx_np.mean():.6f}")
    print(f"  PT:  range=[{pt_np.min():.6f}, {pt_np.max():.6f}], mean={pt_np.mean():.6f}")
    print(f"  Diff: max={max_diff:.6f}, mean={mean_diff:.6f}, RMSE={rmse:.6f}")

    if max_diff < threshold:
        print(f"  ✅ Match (diff < {threshold})")
        return True
    else:
        print(f"  ❌ Divergence (diff >= {threshold})")
        return False

def main(pt_model_path, mlx_model_path):
    print("=== RVC PyTorch vs MLX Comparison ===\n")

    # Load PyTorch model
    print("Loading PyTorch model...")
    pt_ckpt = torch.load(pt_model_path, map_location="cpu")
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

    print(f"Model config: SR={pt_kwargs['sr']}, inter_channels={pt_kwargs['inter_channels']}")

    net_g_pt = PTSynthesizer(**pt_kwargs)
    net_g_pt.load_state_dict(pt_ckpt["weight"], strict=False)
    net_g_pt.eval()
    net_g_pt.remove_weight_norm()

    # Load MLX model
    print("\nLoading MLX model...")
    net_g_mlx = MLXSynthesizer(**pt_kwargs)

    # Load weights
    weights = mx.load(mlx_model_path)
    net_g_mlx.load_weights(mlx_model_path, strict=False)

    print("Models loaded.\n")

    # Create test inputs
    print("Creating test inputs...")
    seq_len = 100

    # Phone embeddings (from HuBERT/ContentVec)
    phone_pt = torch.randn(1, seq_len, 768)
    phone_mlx = mx.array(phone_pt.numpy())

    # F0 (pitch)
    f0_pt = torch.full((1, seq_len), 220.0)  # A3 note
    f0_mlx = mx.full((1, seq_len), 220.0)

    # Pitch (coarse, for embedding)
    pitch_pt = torch.full((1, seq_len), 50).long()
    pitch_mlx = mx.full((1, seq_len), 50).astype(mx.int32)

    # Speaker ID
    sid_pt = torch.tensor([0]).long()
    sid_mlx = mx.array([0], dtype=mx.int32)

    lengths_pt = torch.tensor([seq_len]).long()
    lengths_mlx = mx.array([seq_len], dtype=mx.int32)

    print(f"  phone: {phone_pt.shape}")
    print(f"  f0: {f0_pt.shape}")
    print(f"  pitch: {pitch_pt.shape}")

    # === Step-by-step comparison ===

    with torch.no_grad():
        # 1. Text Encoder
        print("\n--- Text Encoder ---")
        m_p_pt, logs_p_pt, x_mask_pt = net_g_pt.enc_p(phone_pt, pitch_pt, lengths_pt)
        m_p_mlx, logs_p_mlx, x_mask_mlx = net_g_mlx.enc_p(phone_mlx, pitch_mlx, lengths_mlx)

        # MLX now returns in PyTorch format (B, C, T), no transpose needed
        compare_arrays("Text Encoder m_p (mean)", m_p_mlx, m_p_pt, threshold=0.1)
        compare_arrays("Text Encoder logs_p (log-variance)", logs_p_mlx, logs_p_pt, threshold=0.1)

        # 2. Posterior Encoder (if using)
        # Skip for now, focus on inference path

        # 3. Generator/Decoder
        print("\n--- Generator/Decoder ---")

        # PT: decoder expects m_p (mean), f0, g
        # m_p shape: (B, T, C) for MLX, (B, C, T) for PyTorch usually
        # f0 shape: (B, T)

        # Use m_p directly as generator input
        x_gen_pt = m_p_pt
        x_gen_mlx = m_p_mlx

        print(f"Generator input shape: PT={x_gen_pt.shape}, MLX={x_gen_mlx.shape}")

        # Generator forward
        audio_pt = net_g_pt.dec(x_gen_pt, f0_pt, g=None)
        audio_mlx = net_g_mlx.dec(x_gen_mlx, f0_mlx, g=None)

        # MLX now returns in PyTorch format (B, C, T), no transpose needed
        compare_arrays("Generator Output (Audio)", audio_mlx, audio_pt, threshold=0.1)

        # Final stats
        print("\n=== Final Output Stats ===")
        audio_pt_np = audio_pt.squeeze().numpy()
        audio_mlx_np = np.array(audio_mlx).squeeze()

        print(f"PyTorch: shape={audio_pt_np.shape}, range=[{audio_pt_np.min():.6f}, {audio_pt_np.max():.6f}]")
        print(f"MLX:     shape={audio_mlx_np.shape}, range=[{audio_mlx_np.min():.6f}, {audio_mlx_np.max():.6f}]")

        # Check if silent
        if np.abs(audio_pt_np).max() < 1e-6:
            print("⚠️  PyTorch output is SILENT!")
        if np.abs(audio_mlx_np).max() < 1e-6:
            print("⚠️  MLX output is SILENT!")

        # Check correlation
        if len(audio_pt_np) == len(audio_mlx_np) and len(audio_pt_np) > 0:
            corr = np.corrcoef(audio_pt_np, audio_mlx_np)[0, 1]
            print(f"\nCorrelation: {corr:.6f}")
            if corr > 0.9:
                print("✅ Outputs are highly correlated")
            elif corr > 0.5:
                print("⚠️  Outputs are somewhat correlated")
            else:
                print("❌ Outputs are poorly correlated")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_model", required=True, help="PyTorch .pth model path")
    parser.add_argument("--mlx_model", required=True, help="MLX .npz model path")
    args = parser.parse_args()

    main(args.pt_model, args.mlx_model)
