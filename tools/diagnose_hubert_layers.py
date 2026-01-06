#!/usr/bin/env python3
"""
Compare HuBERT layer-by-layer between PyTorch and MLX to find divergence.
"""

import os
import sys
import numpy as np
import librosa

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.getcwd())


def compare_arrays(name, pt_arr, mlx_arr, detailed=False):
    """Compare two arrays and return correlation."""
    pt_np = pt_arr.detach().cpu().numpy() if hasattr(pt_arr, 'detach') else np.asarray(pt_arr)
    mlx_np = np.array(mlx_arr) if hasattr(mlx_arr, '__iter__') else np.asarray(mlx_arr)
    
    if pt_np.shape != mlx_np.shape:
        print(f"  {name}: SHAPE MISMATCH PT={pt_np.shape} MLX={mlx_np.shape}")
        return None
    
    corr = np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1]
    diff = np.abs(pt_np - mlx_np)
    
    status = "✅" if corr > 0.99 else ("⚠️" if corr > 0.9 else "❌")
    
    print(f"  {name}: {status} corr={corr:.4f}, max_diff={diff.max():.4f}, mean_diff={diff.mean():.4f}")
    
    if detailed:
        print(f"    PT:  range=[{pt_np.min():.4f}, {pt_np.max():.4f}], mean={pt_np.mean():.4f}")
        print(f"    MLX: range=[{mlx_np.min():.4f}, {mlx_np.max():.4f}], mean={mlx_np.mean():.4f}")
    
    return corr


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default="test-audio/coder_audio_stock.wav")
    args = parser.parse_args()

    print("=" * 70)
    print("HUBERT LAYER-BY-LAYER COMPARISON")
    print("=" * 70)
    
    # Load audio (first 0.5 seconds for speed)
    audio, sr = librosa.load(args.audio, sr=16000)
    audio = audio[:16000 // 2]  # 0.5 seconds
    print(f"\nAudio: {len(audio)/16000:.2f}s, {len(audio)} samples")
    
    # ===== PYTORCH =====
    print("\n" + "=" * 50)
    print("Loading PyTorch HuBERT...")
    print("=" * 50)
    
    import torch
    from rvc.lib.utils import load_embedding
    
    hubert_pt = load_embedding("contentvec", None).eval()
    audio_pt = torch.from_numpy(audio).float().unsqueeze(0)
    
    # ===== MLX =====
    print("\nLoading MLX HuBERT...")
    
    import mlx.core as mx
    from rvc_mlx.lib.mlx.hubert import HubertModel, HubertConfig
    
    h_path = "rvc_mlx/models/embedders/contentvec/hubert_mlx.npz"
    config = HubertConfig(classifier_proj_size=768)
    hubert_mlx = HubertModel(config)
    hubert_mlx.load_weights(h_path, strict=False)
    mx.eval(hubert_mlx.parameters())
    
    audio_mx = mx.array(audio)[None, :]
    
    # ===== STAGE 1: Feature Extractor (CNN layers) =====
    print("\n" + "-" * 50)
    print("STAGE 1: Feature Extractor (7 CNN layers)")
    print("-" * 50)
    
    with torch.no_grad():
        # PyTorch feature extraction
        x_pt = audio_pt.unsqueeze(1)  # (1, 1, T)
        
        for i, conv in enumerate(hubert_pt.feature_extractor.conv_layers):
            x_pt = conv(x_pt)
            
            # MLX equivalent
            if i == 0:
                x_mlx = audio_mx[:, :, None]  # (1, T, 1)
            x_mlx = hubert_mlx.feature_extractor.conv_layers[i](x_mlx)
            mx.eval(x_mlx)
            
            # PyTorch: (B, C, L) -> MLX: (B, L, C)
            pt_compare = x_pt.transpose(1, 2)  # (B, L, C)
            compare_arrays(f"Conv Layer {i}", pt_compare, x_mlx, detailed=(i == 0))
    
    # Store feature extractor outputs
    feat_pt = x_pt.transpose(1, 2)  # (B, L, C)
    feat_mlx = x_mlx
    
    print(f"\n  Feature Extractor Output: PT={feat_pt.shape}, MLX={feat_mlx.shape}")
    
    # ===== STAGE 2: Feature Projection =====
    print("\n" + "-" * 50)
    print("STAGE 2: Feature Projection (LayerNorm + Linear)")
    print("-" * 50)
    
    with torch.no_grad():
        # PyTorch
        proj_pt = hubert_pt.feature_projection(feat_pt)
        
        # MLX
        proj_mlx = hubert_mlx.feature_projection(feat_mlx)
        mx.eval(proj_mlx)
    
    compare_arrays("Feature Projection", proj_pt, proj_mlx, detailed=True)
    
    # ===== STAGE 3: Positional Embeddings =====
    print("\n" + "-" * 50)
    print("STAGE 3: Positional Conv Embedding")
    print("-" * 50)
    
    with torch.no_grad():
        # PyTorch positional embedding
        pos_pt = hubert_pt.encoder.pos_conv_embed(proj_pt)
        
        # MLX
        pos_mlx = hubert_mlx.encoder.pos_conv_embed(proj_mlx)
        mx.eval(pos_mlx)
    
    compare_arrays("Pos Conv Embed", pos_pt, pos_mlx, detailed=True)
    
    # ===== STAGE 4: Encoder Pre-LayerNorm =====
    print("\n" + "-" * 50)
    print("STAGE 4: Encoder LayerNorm + Dropout")
    print("-" * 50)
    
    with torch.no_grad():
        # PyTorch
        norm_pt = hubert_pt.encoder.layer_norm(pos_pt)
        # No dropout in eval mode
        
        # MLX
        norm_mlx = hubert_mlx.encoder.layer_norm(pos_mlx)
        # No dropout in eval mode
        mx.eval(norm_mlx)
    
    compare_arrays("Encoder LayerNorm", norm_pt, norm_mlx, detailed=True)
    
    # ===== STAGE 5: Transformer Layers =====
    print("\n" + "-" * 50)
    print("STAGE 5: Transformer Layers (12 layers)")
    print("-" * 50)
    
    hidden_pt = norm_pt
    hidden_mlx = norm_mlx
    
    with torch.no_grad():
        for i in range(min(12, len(hubert_pt.encoder.layers))):
            hidden_pt = hubert_pt.encoder.layers[i](hidden_pt)[0]
            hidden_mlx = hubert_mlx.encoder.layers[i](hidden_mlx)
            mx.eval(hidden_mlx)
            
            corr = compare_arrays(f"Layer {i}", hidden_pt, hidden_mlx)
            
            if corr is not None and corr < 0.5:
                print(f"\n  ❌ DIVERGENCE DETECTED at Layer {i}! Stopping.")
                break
    
    # ===== STAGE 6: Final Projection =====
    print("\n" + "-" * 50)
    print("STAGE 6: Final Projection")
    print("-" * 50)
    
    with torch.no_grad():
        # PyTorch final projection
        final_pt = hubert_pt.proj(hidden_pt)
        
        # MLX
        final_mlx = hubert_mlx.final_proj(hidden_mlx)
        mx.eval(final_mlx)
    
    compare_arrays("Final Projection", final_pt, final_mlx, detailed=True)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nLook for the first layer with low correlation (❌) - that's where divergence starts.")


if __name__ == "__main__":
    main()
