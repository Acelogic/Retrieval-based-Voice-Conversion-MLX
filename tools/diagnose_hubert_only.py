#!/usr/bin/env python3
"""
Compare HuBERT feature extraction between PyTorch and MLX.
"""

import os
import sys
import numpy as np
import librosa

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.getcwd())


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default="test-audio/coder_audio_stock.wav")
    args = parser.parse_args()

    print("=" * 70)
    print("HUBERT FEATURE EXTRACTION PARITY DIAGNOSTIC")
    print("=" * 70)
    
    # Load audio (first 2 seconds for speed)
    audio, sr = librosa.load(args.audio, sr=16000)
    if len(audio) > 16000 * 2:
        audio = audio[:16000 * 2]
    print(f"\nAudio: {args.audio}")
    print(f"Duration: {len(audio)/sr:.2f}s, Samples: {len(audio)}")
    
    # PyTorch HuBERT
    print("\n--- PyTorch HuBERT (ContentVec) ---")
    try:
        import torch
        from rvc.lib.utils import load_embedding
        
        hubert_pt = load_embedding("contentvec", None).eval()
        
        audio_pt = torch.from_numpy(audio).float().unsqueeze(0)
        
        with torch.no_grad():
            feats = hubert_pt(audio_pt)["last_hidden_state"]
        
        feats_pt = feats.squeeze(0).numpy()
        print(f"  Shape: {feats_pt.shape}")
        print(f"  Range: [{feats_pt.min():.4f}, {feats_pt.max():.4f}]")
        print(f"  Mean: {feats_pt.mean():.4f}, Std: {feats_pt.std():.4f}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        feats_pt = None
    
    # MLX HuBERT  
    print("\n--- MLX HuBERT ---")
    try:
        import mlx.core as mx
        from rvc_mlx.lib.mlx.hubert import HubertModel, HubertConfig
        
        h_path = "rvc_mlx/models/embedders/contentvec/hubert_mlx.npz"
        if not os.path.exists(h_path):
            print(f"  ❌ Weights not found: {h_path}")
            feats_mlx = None
        else:
            config = HubertConfig(classifier_proj_size=768)
            hubert_mlx = HubertModel(config)
            hubert_mlx.load_weights(h_path, strict=False)
            mx.eval(hubert_mlx.parameters())
            
            audio_mx = mx.array(audio)[None, :]
            feats = hubert_mlx(audio_mx)
            mx.eval(feats)
            
            feats_mlx = np.array(feats).squeeze(0)
            print(f"  Shape: {feats_mlx.shape}")
            print(f"  Range: [{feats_mlx.min():.4f}, {feats_mlx.max():.4f}]")
            print(f"  Mean: {feats_mlx.mean():.4f}, Std: {feats_mlx.std():.4f}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        feats_mlx = None
    
    # Comparison
    if feats_pt is not None and feats_mlx is not None:
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)
        
        # Ensure same shape
        min_len = min(feats_pt.shape[0], feats_mlx.shape[0])
        if feats_pt.shape[0] != feats_mlx.shape[0]:
            print(f"\n⚠️  Length mismatch: PT={feats_pt.shape[0]}, MLX={feats_mlx.shape[0]}")
            feats_pt = feats_pt[:min_len]
            feats_mlx = feats_mlx[:min_len]
        
        # Correlation
        corr = np.corrcoef(feats_pt.flatten(), feats_mlx.flatten())[0, 1]
        print(f"\nOverall Correlation: {corr:.6f}")
        
        # Difference
        diff = np.abs(feats_pt - feats_mlx)
        print(f"Max Difference: {diff.max():.6f}")
        print(f"Mean Difference: {diff.mean():.6f}")
        print(f"RMSE: {np.sqrt(np.mean((feats_pt - feats_mlx)**2)):.6f}")
        
        # Per-frame correlation
        frame_corrs = [np.corrcoef(feats_pt[i], feats_mlx[i])[0, 1] for i in range(min(10, min_len))]
        print(f"\nPer-frame correlations (first 10): {[f'{c:.4f}' for c in frame_corrs]}")
        
        # Verdict
        print("\n" + "=" * 70)
        print("VERDICT")
        print("=" * 70)
        
        if corr > 0.99:
            print("✅ EXCELLENT - HuBERT features match well")
        elif corr > 0.9:
            print("⚠️  GOOD - Minor differences")
        elif corr > 0.5:
            print("⚠️  MODERATE - Noticeable differences")
        else:
            print("❌ POOR - HuBERT features diverge significantly!")
            print("   This is a major source of audio parity issues!")


if __name__ == "__main__":
    main()
