#!/usr/bin/env python3
"""
Debug script to run Python RVC-MLX inference with detailed logging
Matches iOS debug output format for direct comparison
"""

import numpy as np
import mlx.core as mx
import sys
import os

sys.path.append(os.getcwd())

from rvc_mlx.lib.mlx.synthesizers import Synthesizer
from rvc_mlx.lib.mlx.hubert import HubertModel, HubertConfig
from rvc_mlx.lib.mlx.rmvpe import RMVPE0Predictor
from rvc_mlx.infer.pipeline_mlx import AudioProcessor
import soundfile as sf
import librosa

def load_model_mlx(model_path):
    """Load RVC model weights"""
    print(f"Loading model from {model_path}")
    weights = mx.load(model_path)
    
    # Remap keys
    from rvc_mlx.infer.infer_mlx import remap_keys
    weights = remap_keys(weights)
    
    # Initialize synthesizer with v2 40k defaults
    net_g = Synthesizer(
        spec_channels=1025,
        segment_size=32,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[10, 10, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16, 4, 4],
        spk_embed_dim=109,
        gin_channels=256,
        sr=40000,
        use_f0=True
    )
    
    net_g.load_weights(list(weights.items()), strict=False)
    net_g.eval()
    mx.eval(net_g.parameters())
    
    return net_g

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--protect", type=float, default=0.33)
    args = parser.parse_args()
    
    # Load models
    print("="*60)
    print("PYTHON MLX RVC INFERENCE DEBUG")
    print("="*60)
    
    print("\n1. Loading models...")
    net_g = load_model_mlx(args.model)
    
    hubert_model = HubertModel(HubertConfig(classifier_proj_size=768))
    hubert_path = "rvc_mlx/models/embedders/contentvec/hubert_mlx.npz"
    if os.path.exists(hubert_path):
        hubert_model.load_weights(hubert_path, strict=False)
        hubert_model.eval()
        mx.eval(hubert_model.parameters())
    
    rmvpe = RMVPE0Predictor()
    if hasattr(rmvpe, 'model'):
        rmvpe.model.eval()
    
    # Load audio
    print(f"\n2. Loading audio: {args.input}")
    audio, sr = sf.read(args.input)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
    print(f"   Audio: {len(audio)} samples ({len(audio)/16000:.3f}s)")
    
    # High-pass filter
    print("\n3. Applying high-pass filter (48Hz)...")
    from scipy import signal
    bh, ah = signal.butter(5, 48, btype='high', fs=16000)
    audio_filtered = signal.filtfilt(bh, ah, audio)
    print(f"   Filtered RMS: {np.sqrt(np.mean(audio_filtered**2)):.6f}")
    
    # Pad
    pad_samples = 1600
    audio_padded = np.pad(audio_filtered, (pad_samples, pad_samples), mode='edge')
    
    # HuBERT features
    print("\n4. Extracting HuBERT features...")
    audio_mx = mx.array(audio_padded)[None, :]
    feats = hubert_model(audio_mx)
    print(f"   HuBERT output shape: {feats.shape}")
    print(f"   HuBERT stats: min={feats.min().item():.6f}, max={feats.max().item():.6f}")
    
    # Store raw features
    feats0 = feats
    
    # F0 extraction
    print("\n5. Extracting F0 (RMVPE)...")
    f0 = rmvpe.infer_from_audio(audio_padded, thred=0.03)
    print(f"   F0 shape: {f0.shape}")
    print(f"   F0 stats: min={f0.min():.6f}, max={f0.max():.6f}, mean={f0.mean():.6f}")
    print(f"   F0 voiced: {(f0 > 0).sum()}/{len(f0)} ({(f0 > 0).sum()/len(f0)*100:.1f}%)")
    print(f"   F0 (First 20): {f0[:20].tolist()}")
    
    # Upsample features
    print("\n6. Upsampling features (50fps -> 100fps)...")
    B, L, C = feats.shape
    feats = mx.broadcast_to(feats[:, :, None, :], (B, L, 2, C)).reshape(B, L*2, C)
    feats0_up = mx.broadcast_to(feats0[:, :, None, :], (B, L, 2, C)).reshape(B, L*2, C)
    print(f"   Upsampled shape: {feats.shape}")
    
    # Pitch quantization
    print("\n7. Calculating pitch buckets...")
    f0_min, f0_max = 50.0, 1100.0
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0 > 0] = (f0_mel[f0 > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    pitch = np.rint(f0_mel).astype(int)
    print(f"   Pitch (First 20): {pitch[:20].tolist()}")
    
    # Sync lengths
    p_len = min(feats.shape[1], len(f0))
    feats = feats[:, :p_len, :]
    feats0_synced = feats0_up[:, :p_len, :]
    f0_final = f0[:p_len]
    pitch_final = pitch[:p_len]
    
    # Feature protection
    print(f"\n8. Applying feature protection (protect={args.protect})...")
    if args.protect < 0.5:
        voiced_mask = f0_final > 0
        protect_weight = np.where(voiced_mask, 1.0, args.protect)
        protect_weight_mx = mx.array(protect_weight)[None, :, None]
        feats = feats * protect_weight_mx + feats0_synced * (1.0 - protect_weight_mx)
        print(f"   Applied protection blending")
        print(f"   Feats stats after protection: min={feats.min().item():.6f}, max={feats.max().item():.6f}")
    
    # Phone features (first channel of first 20 frames)
    phone_data = np.array(feats[0, :min(20, p_len), 0])
    print(f"   Phone[0] (First 20): {['%.4f' % x for x in phone_data]}")
    
    # Synthesizer inference
    print(f"\n9. Running Synthesizer...")
    print(f"   Input shapes: phone={feats.shape}, pitch={pitch_final.shape}, f0={f0_final.shape}")
    
    phone_mx = feats
    pitch_mx = mx.array(pitch_final)[None, :]
    nsff0_mx = mx.array(f0_final)[None, :, None]
    p_len_mx = mx.array([p_len])
    sid_mx = mx.array([0])
    
    audio_out, _, _ = net_g.infer(phone_mx, p_len_mx, pitch_mx, nsff0_mx, sid_mx)
    mx.eval(audio_out)
    
    print(f"   Output shape: {audio_out.shape}")
    print(f"   Output stats: min={audio_out.min().item():.6f}, max={audio_out.max().item():.6f}")
    
    # Extract core (remove padding)
    output_ratio = 40000 / 16000
    crop_samples = int(pad_samples * output_ratio)
    audio_out_np = np.array(audio_out[0, crop_samples:-crop_samples, 0])
    
    print(f"   After crop: {len(audio_out_np)} samples")
    
    # Normalization
    print("\n10. Normalizing output...")
    audio_max = np.abs(audio_out_np).max() / 0.99
    if audio_max > 1:
        audio_out_np = audio_out_np / audio_max
        print(f"    Normalized by {audio_max:.6f}")
    
    # Save
    print(f"\n11. Saving to {args.output}")
    sf.write(args.output, audio_out_np, 40000)
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)

if __name__ == "__main__":
    main()
