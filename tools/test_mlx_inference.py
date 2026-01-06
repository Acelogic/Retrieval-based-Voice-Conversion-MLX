#!/usr/bin/env python3
"""
Quick test of MLX inference to see what's happening.
"""

import sys
import os
import numpy as np
import soundfile as sf
import mlx.core as mx

sys.path.append(os.getcwd())

from rvc_mlx.infer.infer_mlx import RVC_MLX

def test_inference(model_path, output_path="test-audio/mlx_test_output.wav"):
    print(f"=== Testing MLX Inference ===")
    print(f"Model: {model_path}\n")

    try:
        # Load model
        print("Loading model...")
        rvc = RVC_MLX(model_path)
        sr = getattr(rvc, 'target_sr', getattr(rvc, 'sr', getattr(rvc, 'tgt_sr', 48000)))
        print(f"Model loaded. SR={sr}")

        # Create test input
        print("\nCreating test input...")
        seq_len = 100

        # Phone features (simulating HuBERT output)
        phone = mx.random.normal((1, seq_len, 768))
        f0 = mx.full((1, seq_len), 220.0)  # A3 note
        pitch = mx.full((1, seq_len), 50, dtype=mx.int32)
        sid = mx.array([0], dtype=mx.int32)
        lengths = mx.array([seq_len], dtype=mx.int32)

        print(f"  phone: {phone.shape}")
        print(f"  f0: {f0.shape}")
        print(f"  pitch: {pitch.shape}")

        # Text Encoder
        print("\n--- Text Encoder ---")
        m_p, logs_p, x_mask = rvc.net_g.enc_p(phone, pitch, lengths)
        m_p_np = np.array(m_p)
        logs_p_np = np.array(logs_p)

        print(f"  m_p shape: {m_p.shape}")
        print(f"  logs_p shape: {logs_p.shape}")
        print(f"  x_mask shape: {x_mask.shape}")
        print(f"  m_p range: [{m_p_np.min():.6f}, {m_p_np.max():.6f}], mean={m_p_np.mean():.6f}")
        print(f"  logs_p range: [{logs_p_np.min():.6f}, {logs_p_np.max():.6f}], mean={logs_p_np.mean():.6f}")

        # Check if output is reasonable
        if np.abs(m_p_np).max() < 1e-6:
            print("  ⚠️  m_p is nearly ZERO!")
        elif np.isnan(m_p_np).any():
            print("  ❌ m_p contains NaN!")
        elif np.isinf(m_p_np).any():
            print("  ❌ m_p contains Inf!")
        else:
            print("  ✅ m_p looks reasonable")

        # Generator
        print("\n--- Generator ---")
        audio = rvc.net_g.dec(m_p, f0, g=None)
        audio_np = np.array(audio).squeeze()
        print(f"  Output shape: {audio.shape}")
        print(f"  Output range: [{audio_np.min():.6f}, {audio_np.max():.6f}]")
        print(f"  Output mean: {audio_np.mean():.6f}")

        # Check audio
        if np.abs(audio_np).max() < 1e-6:
            print("  ❌ Audio is SILENT!")
        elif np.isnan(audio_np).any():
            print("  ❌ Audio contains NaN!")
        elif np.isinf(audio_np).any():
            print("  ❌ Audio contains Inf!")
        else:
            print("  ✅ Audio has signal")

        # Save
        if len(audio_np) > 0:
            sf.write(output_path, audio_np, sr)
            print(f"\n✅ Saved to {output_path}")
        else:
            print("\n❌ No audio to save")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to MLX model (.npz)")
    parser.add_argument("--output", default="test-audio/mlx_test_output.wav")
    args = parser.parse_args()

    test_inference(args.model_path, args.output)
