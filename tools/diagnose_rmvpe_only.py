#!/usr/bin/env python3
"""
Focused RMVPE F0 comparison between PyTorch and MLX.

This script compares RMVPE pitch detection outputs to find divergence.
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
    print("RMVPE F0 PARITY DIAGNOSTIC")
    print("=" * 70)
    
    # Load audio
    audio, sr = librosa.load(args.audio, sr=16000)
    print(f"\nAudio: {args.audio}")
    print(f"Duration: {len(audio)/sr:.2f}s, Samples: {len(audio)}")
    
    # Use shorter segment for speed
    if len(audio) > 16000 * 5:
        audio = audio[:16000 * 5]
        print(f"Using first 5 seconds: {len(audio)} samples")
    
    # PyTorch RMVPE
    print("\n--- PyTorch RMVPE ---")
    try:
        from rvc.lib.predictors.RMVPE import RMVPE0Predictor as RMVPE_PT
        pt_path = "rvc/models/predictors/rmvpe.pt"
        if not os.path.exists(pt_path):
            print(f"❌ PyTorch RMVPE not found: {pt_path}")
            f0_pt = None
        else:
            rmvpe_pt = RMVPE_PT(pt_path, device="cpu")
            f0_pt = rmvpe_pt.infer_from_audio(audio, thred=0.03)
            print(f"  Shape: {f0_pt.shape}")
            print(f"  Range: [{f0_pt.min():.1f}, {f0_pt.max():.1f}] Hz")
            voiced_pt = f0_pt > 0
            print(f"  Voiced: {voiced_pt.sum()}/{len(f0_pt)} ({100*voiced_pt.sum()/len(f0_pt):.1f}%)")
            if voiced_pt.sum() > 0:
                print(f"  Mean F0 (voiced): {f0_pt[voiced_pt].mean():.2f} Hz")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        f0_pt = None
    
    # MLX RMVPE
    print("\n--- MLX RMVPE ---")
    try:
        from rvc_mlx.lib.mlx.rmvpe import RMVPE0Predictor as RMVPE_MLX
        mlx_path = "rvc_mlx/models/predictors/rmvpe_mlx.npz"
        if not os.path.exists(mlx_path):
            print(f"❌ MLX RMVPE not found: {mlx_path}")
            f0_mlx = None
        else:
            rmvpe_mlx = RMVPE_MLX(mlx_path)
            f0_mlx = rmvpe_mlx.infer_from_audio(audio, thred=0.03)
            print(f"  Shape: {f0_mlx.shape}")
            print(f"  Range: [{f0_mlx.min():.1f}, {f0_mlx.max():.1f}] Hz")
            voiced_mlx = f0_mlx > 0
            print(f"  Voiced: {voiced_mlx.sum()}/{len(f0_mlx)} ({100*voiced_mlx.sum()/len(f0_mlx):.1f}%)")
            if voiced_mlx.sum() > 0:
                print(f"  Mean F0 (voiced): {f0_mlx[voiced_mlx].mean():.2f} Hz")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        f0_mlx = None
    
    # Comparison
    if f0_pt is not None and f0_mlx is not None:
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)
        
        # Ensure same length
        min_len = min(len(f0_pt), len(f0_mlx))
        f0_pt = f0_pt[:min_len]
        f0_mlx = f0_mlx[:min_len]
        
        # Voiced agreement
        voiced_pt = f0_pt > 0
        voiced_mlx = f0_mlx > 0
        voiced_agreement = np.mean(voiced_pt == voiced_mlx)
        print(f"\nVoiced Detection Agreement: {voiced_agreement*100:.2f}%")
        
        # Correlation
        corr = np.corrcoef(f0_pt, f0_mlx)[0, 1]
        print(f"Overall F0 Correlation: {corr:.4f}")
        
        # Voiced-only comparison
        voiced_both = voiced_pt & voiced_mlx
        if voiced_both.sum() > 0:
            f0_pt_v = f0_pt[voiced_both]
            f0_mlx_v = f0_mlx[voiced_both]
            
            corr_voiced = np.corrcoef(f0_pt_v, f0_mlx_v)[0, 1]
            print(f"Voiced F0 Correlation: {corr_voiced:.4f}")
            
            # Cents error
            cents = 1200 * np.abs(np.log2(f0_mlx_v / (f0_pt_v + 1e-8)))
            print(f"Mean Cents Error: {np.mean(cents):.1f} cents")
            print(f"Median Cents Error: {np.median(cents):.1f} cents")
            
            # Semitones
            semitones = cents / 100
            print(f"Mean Semitone Error: {np.mean(semitones):.2f} semitones")
            
            # Absolute Hz difference
            hz_diff = np.abs(f0_pt_v - f0_mlx_v)
            print(f"Mean Hz Difference: {np.mean(hz_diff):.2f} Hz")
            
            # Plot sample if matplotlib available
            try:
                import matplotlib.pyplot as plt
                
                fig, axes = plt.subplots(3, 1, figsize=(12, 8))
                
                # F0 contours
                axes[0].plot(f0_pt, label='PyTorch', alpha=0.7)
                axes[0].plot(f0_mlx, label='MLX', alpha=0.7)
                axes[0].set_ylabel('F0 (Hz)')
                axes[0].legend()
                axes[0].set_title(f'F0 Contours (Correlation: {corr:.4f})')
                
                # Difference
                axes[1].plot(f0_pt - f0_mlx, color='red', alpha=0.7)
                axes[1].axhline(0, color='black', linestyle='--', alpha=0.3)
                axes[1].set_ylabel('Difference (Hz)')
                axes[1].set_title(f'F0 Difference (Mean: {np.mean(f0_pt - f0_mlx):.2f} Hz)')
                
                # Voiced agreement
                axes[2].fill_between(range(len(f0_pt)), voiced_pt.astype(float), alpha=0.5, label='PT Voiced')
                axes[2].fill_between(range(len(f0_mlx)), voiced_mlx.astype(float) * 0.5, alpha=0.5, label='MLX Voiced')
                axes[2].set_ylabel('Voiced')
                axes[2].set_xlabel('Frame')
                axes[2].legend()
                axes[2].set_title(f'Voiced Detection Agreement: {voiced_agreement*100:.1f}%')
                
                plt.tight_layout()
                plt.savefig('logs/rmvpe_comparison.png', dpi=150)
                print(f"\n📊 Plot saved to: logs/rmvpe_comparison.png")
            except ImportError:
                print("\n  (matplotlib not available for plotting)")
        
        # Verdict
        print("\n" + "=" * 70)
        print("VERDICT")
        print("=" * 70)
        
        if corr > 0.95 and voiced_agreement > 0.98:
            print("✅ EXCELLENT - F0 outputs match well")
        elif corr > 0.8 and voiced_agreement > 0.95:
            print("⚠️  GOOD - Minor differences, acceptable for voice conversion")
        elif voiced_agreement > 0.9:
            print("⚠️  MODERATE - Voiced detection is OK, but F0 values differ")
            print("   This may cause audible pitch differences in output")
        else:
            print("❌ POOR - Significant divergence in RMVPE outputs")
            print("   This is likely causing the audio parity issue!")


if __name__ == "__main__":
    main()
