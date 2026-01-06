#!/usr/bin/env python3
"""
Diagnose Parity: Compare intermediate outputs between PyTorch and MLX RVC.

This script systematically compares each stage of the RVC pipeline to find 
where divergence occurs:
1. Audio preprocessing
2. HuBERT feature extraction  
3. RMVPE F0 pitch detection
4. Feature upsampling
5. Synthesizer inference

Usage:
    python tools/diagnose_divergence.py \
        --input test-audio/sample.wav \
        --pt-model weights/model.pth \
        --mlx-model weights/model.npz
"""

import os
import sys
import argparse
import numpy as np
import json

sys.path.insert(0, os.getcwd())

def set_seeds(seed=42):
    """Set deterministic seeds for both frameworks."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    try:
        import mlx.core as mx
        mx.random.seed(seed)
    except ImportError:
        pass

def compare_outputs(name, pt_arr, mlx_arr, detailed=True):
    """Compare two arrays and report metrics."""
    import torch
    import mlx.core as mx
    
    # Convert to numpy
    if isinstance(pt_arr, torch.Tensor):
        pt_np = pt_arr.detach().cpu().numpy()
    else:
        pt_np = np.asarray(pt_arr)
        
    if isinstance(mlx_arr, mx.array):
        mlx_np = np.array(mlx_arr)
    else:
        mlx_np = np.asarray(mlx_arr)
    
    print(f"\n{'='*60}")
    print(f"Stage: {name}")
    print(f"{'='*60}")
    print(f"  PT Shape:  {pt_np.shape}")
    print(f"  MLX Shape: {mlx_np.shape}")
    
    if pt_np.shape != mlx_np.shape:
        print(f"  ❌ SHAPE MISMATCH!")
        # Try to compare what we can
        if pt_np.size == mlx_np.size:
            print(f"  Attempting reshape comparison...")
            mlx_np = mlx_np.reshape(pt_np.shape)
        else:
            return {"match": False, "reason": "shape_mismatch"}
    
    # Statistics
    pt_stats = {"min": pt_np.min(), "max": pt_np.max(), "mean": pt_np.mean(), "std": pt_np.std()}
    mlx_stats = {"min": mlx_np.min(), "max": mlx_np.max(), "mean": mlx_np.mean(), "std": mlx_np.std()}
    
    print(f"  PT:  min={pt_stats['min']:.6f}, max={pt_stats['max']:.6f}, mean={pt_stats['mean']:.6f}, std={pt_stats['std']:.6f}")
    print(f"  MLX: min={mlx_stats['min']:.6f}, max={mlx_stats['max']:.6f}, mean={mlx_stats['mean']:.6f}, std={mlx_stats['std']:.6f}")
    
    # Difference metrics
    diff = np.abs(pt_np - mlx_np)
    max_diff = diff.max()
    mean_diff = diff.mean()
    rmse = np.sqrt(np.mean((pt_np - mlx_np) ** 2))
    
    # Correlation
    if pt_np.size > 1:
        corr = np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1]
    else:
        corr = 1.0 if np.allclose(pt_np, mlx_np) else 0.0
        
    print(f"\n  Difference Metrics:")
    print(f"    Max Diff:  {max_diff:.6f}")
    print(f"    Mean Diff: {mean_diff:.6f}")
    print(f"    RMSE:      {rmse:.6f}")
    print(f"    Correlation: {corr:.6f}")
    
    # Verdict
    if corr > 0.99 and max_diff < 0.01:
        print(f"  ✅ EXCELLENT MATCH (corr>{0.99}, max_diff<{0.01})")
        status = "excellent"
    elif corr > 0.95:
        print(f"  ⚠️  GOOD MATCH (corr>{0.95})")
        status = "good"
    elif corr > 0.8:
        print(f"  ⚠️  MODERATE MATCH (corr>{0.8})")
        status = "moderate"
    else:
        print(f"  ❌ POOR MATCH - THIS IS WHERE DIVERGENCE OCCURS")
        status = "poor"
    
    return {
        "match": status in ["excellent", "good"],
        "status": status,
        "max_diff": float(max_diff),
        "mean_diff": float(mean_diff),
        "rmse": float(rmse),
        "correlation": float(corr) if not np.isnan(corr) else 0.0,
        "pt_shape": list(pt_np.shape),
        "mlx_shape": list(mlx_np.shape),
    }


def diagnose_hubert(audio_np, seed=42):
    """Compare HuBERT feature extraction."""
    print("\n" + "="*70)
    print("STAGE 1: HuBERT Feature Extraction")
    print("="*70)
    
    import torch
    import mlx.core as mx
    
    set_seeds(seed)
    
    # PyTorch HuBERT
    from fairseq import checkpoint_utils
    
    hubert_pt_path = "rvc/models/predictors/hubert/hubert_base.pt"
    if not os.path.exists(hubert_pt_path):
        print(f"  ⚠️  PyTorch HuBERT not found at {hubert_pt_path}")
        return None
    
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [hubert_pt_path], suffix=""
    )
    hubert_pt = models[0].eval()
    
    # MLX HuBERT
    from rvc_mlx.lib.mlx.hubert import HubertModel, HubertConfig
    
    mlx_hubert_path = "weights/hubert/hubert_base.npz"
    if not os.path.exists(mlx_hubert_path):
        print(f"  ⚠️  MLX HuBERT not found at {mlx_hubert_path}")
        return None
        
    config = HubertConfig()
    hubert_mlx = HubertModel(config)
    hubert_mlx.load_weights(mlx_hubert_path)
    
    # Prepare input
    audio_pt = torch.from_numpy(audio_np).float().unsqueeze(0)
    audio_mlx = mx.array(audio_np)[None, :]
    
    print(f"  Input audio shape: {audio_np.shape}")
    
    # Run inference
    with torch.no_grad():
        # PyTorch forward
        padding_mask = torch.zeros(audio_pt.shape, dtype=torch.bool)
        pt_result = hubert_pt.extract_features(
            source=audio_pt,
            padding_mask=padding_mask,
            output_layer=12
        )
        feats_pt = pt_result[0]  # (B, L, 768)
    
    # MLX forward
    feats_mlx = hubert_mlx(audio_mlx)
    mx.eval(feats_mlx)
    
    return compare_outputs("HuBERT Features", feats_pt, feats_mlx)


def diagnose_rmvpe(audio_np, seed=42):
    """Compare RMVPE F0 extraction."""
    print("\n" + "="*70)
    print("STAGE 2: RMVPE F0 Pitch Extraction")
    print("="*70)
    
    import torch
    import mlx.core as mx
    
    set_seeds(seed)
    
    # PyTorch RMVPE
    pt_rmvpe_path = "rvc/models/predictors/rmvpe.pt"
    if os.path.exists(pt_rmvpe_path):
        from rvc.lib.predictors.RMVPE import RMVPE0Predictor as RMVPE_PT
        rmvpe_pt = RMVPE_PT(pt_rmvpe_path, device="cpu")
    else:
        print(f"  ⚠️  PyTorch RMVPE not found at {pt_rmvpe_path}")
        return None
    
    # MLX RMVPE
    mlx_rmvpe_path = "weights/rmvpe/rmvpe.npz"
    if os.path.exists(mlx_rmvpe_path):
        from rvc_mlx.lib.mlx.rmvpe import RMVPE0Predictor as RMVPE_MLX
        rmvpe_mlx = RMVPE_MLX(mlx_rmvpe_path)
    else:
        print(f"  ⚠️  MLX RMVPE not found at {mlx_rmvpe_path}")
        return None
    
    print(f"  Input audio shape: {audio_np.shape}")
    
    # Run F0 extraction
    f0_pt = rmvpe_pt.infer_from_audio(audio_np, thred=0.03)
    f0_mlx = rmvpe_mlx.infer_from_audio(audio_np, thred=0.03)
    
    print(f"\n  F0 Output Shapes: PT={f0_pt.shape}, MLX={f0_mlx.shape}")
    
    # Compare F0
    result = compare_outputs("RMVPE F0", f0_pt, f0_mlx)
    
    # Additional F0-specific analysis
    voiced_pt = f0_pt > 0
    voiced_mlx = f0_mlx > 0
    
    voiced_agreement = np.mean(voiced_pt == voiced_mlx)
    print(f"\n  Voiced Detection Agreement: {voiced_agreement*100:.2f}%")
    
    # Compare voiced F0 values only
    voiced_both = voiced_pt & voiced_mlx
    if voiced_both.sum() > 0:
        f0_pt_voiced = f0_pt[voiced_both]
        f0_mlx_voiced = f0_mlx[voiced_both]
        cents_error = 1200 * np.abs(np.log2(f0_mlx_voiced / (f0_pt_voiced + 1e-8)))
        print(f"  Voiced F0 Mean Cents Error: {np.mean(cents_error):.2f} cents")
    
    result["voiced_agreement"] = float(voiced_agreement)
    return result


def diagnose_synthesizer(pt_model_path, mlx_model_path, audio_np, seed=42):
    """Compare Synthesizer with identical inputs."""
    print("\n" + "="*70)
    print("STAGE 3: Synthesizer Comparison (Identical Inputs)")
    print("="*70)
    
    import torch
    import mlx.core as mx
    
    set_seeds(seed)
    
    # Load PyTorch Synthesizer
    print("\n  Loading PyTorch model...")
    from rvc.lib.algorithm.synthesizers import Synthesizer as PT_Synthesizer
    
    pt_ckpt = torch.load(pt_model_path, map_location="cpu")
    config = pt_ckpt.get("config", pt_ckpt.get("params", {}))
    
    if isinstance(config, list):
        kwargs = {
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
            "sr": config[17] if len(config) > 17 else 40000,
            "use_f0": True,
            "text_enc_hidden_dim": 768,
            "vocoder": "NSF",
        }
    else:
        kwargs = dict(config)
    
    net_g_pt = PT_Synthesizer(**kwargs)
    net_g_pt.load_state_dict(pt_ckpt["weight"], strict=False)
    net_g_pt.eval()
    net_g_pt.remove_weight_norm()
    
    # Load MLX Synthesizer  
    print("  Loading MLX model...")
    from rvc_mlx.lib.mlx.synthesizers import Synthesizer as MLX_Synthesizer
    
    net_g_mlx = MLX_Synthesizer(**kwargs)
    net_g_mlx.load_weights(mlx_model_path, strict=False)
    
    # Create IDENTICAL test inputs (random but same for both)
    np.random.seed(seed)
    seq_len = 100
    
    # Phone features (from HuBERT) - random but identical
    phone_np = np.random.randn(1, seq_len, 768).astype(np.float32)
    phone_pt = torch.from_numpy(phone_np)
    phone_mlx = mx.array(phone_np)
    
    # F0 - simple sine wave pitch contour
    t = np.linspace(0, 1, seq_len)
    f0_raw = 220 + 50 * np.sin(2 * np.pi * 2 * t)  # A3 with vibrato
    f0_pt = torch.from_numpy(f0_raw.astype(np.float32)).unsqueeze(0)
    f0_mlx = mx.array(f0_raw.astype(np.float32))[None, :]
    
    # Coarse pitch (quantized)
    f0_mel = 1127 * np.log(1 + f0_raw / 700)
    f0_mel_min = 1127 * np.log(1 + 50 / 700)
    f0_mel_max = 1127 * np.log(1 + 1100 / 700)
    f0_coarse = np.clip((f0_mel - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1, 1, 255).astype(np.int64)
    pitch_pt = torch.from_numpy(f0_coarse).unsqueeze(0).long()
    pitch_mlx = mx.array(f0_coarse.astype(np.int32))[None, :]
    
    # Speaker ID
    sid_pt = torch.tensor([0]).long()
    sid_mlx = mx.array([0], dtype=mx.int32)
    
    # Lengths
    lengths_pt = torch.tensor([seq_len]).long()
    lengths_mlx = mx.array([seq_len], dtype=mx.int32)
    
    print(f"\n  Test inputs:")
    print(f"    phone: {phone_np.shape}")
    print(f"    f0: {f0_raw.shape} (range: {f0_raw.min():.1f} - {f0_raw.max():.1f} Hz)")
    print(f"    pitch (coarse): {f0_coarse.shape}")
    
    results = {}
    
    # Compare Text Encoder
    print("\n  --- Sub-stage: Text Encoder ---")
    with torch.no_grad():
        m_p_pt, logs_p_pt, x_mask_pt = net_g_pt.enc_p(phone_pt, pitch_pt, lengths_pt)
    m_p_mlx, logs_p_mlx, x_mask_mlx = net_g_mlx.enc_p(phone_mlx, pitch_mlx, lengths_mlx)
    mx.eval(m_p_mlx, logs_p_mlx, x_mask_mlx)
    
    results["text_encoder_m_p"] = compare_outputs("TextEncoder m_p (mean)", m_p_pt, m_p_mlx)
    results["text_encoder_logs_p"] = compare_outputs("TextEncoder logs_p (log-var)", logs_p_pt, logs_p_mlx)
    
    # Compare Decoder/Generator
    print("\n  --- Sub-stage: Generator/Decoder ---")
    with torch.no_grad():
        # Use the encoder output as generator input
        audio_pt_out = net_g_pt.dec(m_p_pt, f0_pt, g=None)
    audio_mlx_out = net_g_mlx.dec(m_p_mlx, f0_mlx, g=None)
    mx.eval(audio_mlx_out)
    
    results["decoder_audio"] = compare_outputs("Decoder Output (Audio)", audio_pt_out, audio_mlx_out)
    
    # Full inference
    print("\n  --- Sub-stage: Full Synthesizer.infer ---")
    with torch.no_grad():
        audio_pt_full, _, _ = net_g_pt.infer(phone_pt, lengths_pt, pitch_pt, f0_pt, sid_pt)
    audio_mlx_full, _, _ = net_g_mlx.infer(phone_mlx, lengths_mlx, pitch_mlx, f0_mlx, sid_mlx)
    mx.eval(audio_mlx_full)
    
    results["full_infer"] = compare_outputs("Full Synthesizer.infer", audio_pt_full, audio_mlx_full)
    
    return results


def diagnose_full_pipeline(pt_model_path, mlx_model_path, audio_path, seed=42):
    """Run full pipeline comparison."""
    import librosa
    import soundfile as sf
    
    print("\n" + "#"*70)
    print("# FULL PIPELINE PARITY DIAGNOSTIC")
    print("#"*70)
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    print(f"\nLoaded audio: {audio_path}")
    print(f"  Duration: {len(audio)/sr:.2f}s, Sample rate: {sr}Hz, Samples: {len(audio)}")
    
    all_results = {}
    
    # Stage 1: HuBERT
    try:
        all_results["hubert"] = diagnose_hubert(audio, seed)
    except Exception as e:
        print(f"\n  ❌ HuBERT comparison failed: {e}")
        all_results["hubert"] = {"error": str(e)}
    
    # Stage 2: RMVPE
    try:
        all_results["rmvpe"] = diagnose_rmvpe(audio, seed)
    except Exception as e:
        print(f"\n  ❌ RMVPE comparison failed: {e}")
        all_results["rmvpe"] = {"error": str(e)}
    
    # Stage 3: Synthesizer
    try:
        all_results["synthesizer"] = diagnose_synthesizer(pt_model_path, mlx_model_path, audio, seed)
    except Exception as e:
        print(f"\n  ❌ Synthesizer comparison failed: {e}")
        import traceback
        traceback.print_exc()
        all_results["synthesizer"] = {"error": str(e)}
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    
    for stage, result in all_results.items():
        if result is None:
            print(f"  {stage}: SKIPPED (model not found)")
        elif "error" in result:
            print(f"  {stage}: ❌ ERROR - {result['error']}")
        elif isinstance(result, dict) and "status" in result:
            status_icon = {"excellent": "✅", "good": "⚠️", "moderate": "⚠️", "poor": "❌"}.get(result["status"], "?")
            print(f"  {stage}: {status_icon} {result['status'].upper()} (corr={result.get('correlation', 'N/A'):.4f})")
        elif isinstance(result, dict):
            # Nested results (synthesizer sub-stages)
            for sub_stage, sub_result in result.items():
                if isinstance(sub_result, dict) and "status" in sub_result:
                    status_icon = {"excellent": "✅", "good": "⚠️", "moderate": "⚠️", "poor": "❌"}.get(sub_result["status"], "?")
                    print(f"  {stage}/{sub_stage}: {status_icon} {sub_result['status'].upper()} (corr={sub_result.get('correlation', 'N/A'):.4f})")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Diagnose PyTorch vs MLX RVC divergence")
    parser.add_argument("--input", type=str, required=True, help="Input audio file")
    parser.add_argument("--pt-model", type=str, required=True, help="PyTorch model (.pth)")
    parser.add_argument("--mlx-model", type=str, required=True, help="MLX model (.npz)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-json", type=str, help="Save results to JSON file")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    if not os.path.exists(args.pt_model):
        print(f"Error: PyTorch model not found: {args.pt_model}")
        sys.exit(1)
        
    if not os.path.exists(args.mlx_model):
        print(f"Error: MLX model not found: {args.mlx_model}")
        sys.exit(1)
    
    results = diagnose_full_pipeline(args.pt_model, args.mlx_model, args.input, args.seed)
    
    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.save_json}")


if __name__ == "__main__":
    main()
