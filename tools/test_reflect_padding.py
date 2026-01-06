#!/usr/bin/env python3
"""
Test RMVPE F0 prediction with reflect padding fix.
"""

import sys
import os
import torch
import librosa
import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "rvc")

# Force reload
for mod in list(sys.modules.keys()):
    if 'rvc_mlx' in mod or 'rvc.lib' in mod:
        del sys.modules[mod]

from rvc.lib.predictors.RMVPE import RMVPE0Predictor as PyTorchRMVPE
from rvc_mlx.lib.mlx.rmvpe import RMVPE0Predictor as MLXRMVPE

def main():
    print("=== Testing RMVPE with Reflect Padding ===\n")

    # Load audio
    audio, sr = librosa.load("test-audio/coder_audio_stock.wav", sr=16000, mono=True)
    audio = audio[:sr // 2]  # 0.5 seconds
    print(f"Audio shape: {audio.shape} ({len(audio)/16000:.2f}s)\n")

    # MLX prediction
    print("--- MLX Prediction ---")
    mlx_predictor = MLXRMVPE()
    f0_mlx = mlx_predictor.infer_from_audio(audio, thred=0.03)

    voiced_mlx = f0_mlx > 0
    f0_mean_mlx = f0_mlx[voiced_mlx].mean() if voiced_mlx.any() else 0

    print(f"F0 shape: {f0_mlx.shape}")
    print(f"Voiced frames: {voiced_mlx.sum()} / {len(f0_mlx)} ({100*voiced_mlx.sum()/len(f0_mlx):.1f}%)")
    print(f"F0 range: [{f0_mlx[voiced_mlx].min():.2f}, {f0_mlx[voiced_mlx].max():.2f}] Hz" if voiced_mlx.any() else "No voiced frames")
    print(f"F0 mean: {f0_mean_mlx:.2f} Hz" if voiced_mlx.any() else "F0 mean: N/A")

    # PyTorch prediction
    print("\n--- PyTorch Prediction ---")
    pt_predictor = PyTorchRMVPE("rvc/models/predictors/rmvpe.pt", device="cpu")
    f0_pt = pt_predictor.infer_from_audio(audio, thred=0.03)
    if isinstance(f0_pt, torch.Tensor):
        f0_pt = f0_pt.squeeze().cpu().numpy()
    elif f0_pt.ndim > 1:
        f0_pt = f0_pt.squeeze()

    voiced_pt = f0_pt > 0
    f0_mean_pt = f0_pt[voiced_pt].mean() if voiced_pt.any() else 0

    print(f"F0 shape: {f0_pt.shape}")
    print(f"Voiced frames: {voiced_pt.sum()} / {len(f0_pt)} ({100*voiced_pt.sum()/len(f0_pt):.1f}%)")
    print(f"F0 range: [{f0_pt[voiced_pt].min():.2f}, {f0_pt[voiced_pt].max():.2f}] Hz" if voiced_pt.any() else "No voiced frames")
    print(f"F0 mean: {f0_mean_pt:.2f} Hz" if voiced_pt.any() else "F0 mean: N/A")

    # Compare
    print("\n--- Comparison ---")
    if voiced_mlx.any() and voiced_pt.any():
        f0_diff = abs(f0_mean_mlx - f0_mean_pt)
        f0_error_pct = 100 * f0_diff / f0_mean_pt
        print(f"F0 mean difference: {f0_diff:.2f} Hz ({f0_error_pct:.1f}%)")

        if f0_error_pct < 5:
            print("✅ F0 accuracy is excellent (<5% error)")
        elif f0_error_pct < 10:
            print("✅ F0 accuracy is good (<10% error)")
        else:
            print("❌ F0 accuracy needs improvement")

    voiced_diff = abs(voiced_mlx.sum() - voiced_pt.sum())
    voiced_error_pct = 100 * voiced_diff / voiced_pt.sum() if voiced_pt.any() else 0
    print(f"Voiced frames difference: {voiced_diff} ({voiced_error_pct:.1f}%)")

    if voiced_error_pct < 10:
        print("✅ Voiced detection is excellent (<10% error)")
    elif voiced_error_pct < 50:
        print("⚠️  Voiced detection is okay (<50% error)")
    else:
        print("❌ Voiced detection needs improvement (>50% error)")

if __name__ == "__main__":
    main()
