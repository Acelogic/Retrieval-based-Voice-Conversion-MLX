#!/usr/bin/env python3
"""Quick comparison test - run Python MLX inference and export intermediate values"""
import os
import sys
import numpy as np
import mlx.core as mx
import soundfile as sf

# Add path
sys.path.insert(0, '/Users/mcruz/Developer/Retrieval-based-Voice-Conversion-MLX')

from rvc_mlx.lib.mlx.synthesizers import Synthesizer
from rvc_mlx.lib.mlx.hubert import HubertModel, HubertConfig
from rvc_mlx.lib.mlx.rmvpe import RMVPE0Predictor

def main():
    # Load test audio (same as iOS uses)
    test_audio_path = "/Users/mcruz/Developer/Retrieval-based-Voice-Conversion-MLX/test_audio.wav"
    if not os.path.exists(test_audio_path):
        print(f"Please create test audio at {test_audio_path}")
        print("Use a short 2-3 second speech clip at 16kHz mono")
        return
    
    audio, sr = sf.read(test_audio_path)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
    print(f"Loaded audio: {len(audio)} samples ({len(audio)/16000:.2f}s)")
    
    # Load models
    print("Loading HuBERT...")
    hubert = HubertModel(HubertConfig())
    hubert_weights = dict(mx.load("/Users/mcruz/Developer/Retrieval-based-Voice-Conversion-MLX/Demos/iOS/RVCNative/RVCNativePackage/Sources/RVCNativeFeature/Assets/hubert_base.safetensors"))
    hubert.load_weights(list(hubert_weights.items()))
    
    print("Loading RMVPE...")
    rmvpe = RMVPE0Predictor()
    
    print("Loading Synthesizer...")
    synth = Synthesizer(
        spec_channels=1025,
        segment_size=32,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.0,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[10, 10, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16, 4, 4],
        spk_embed_dim=256,
        gin_channels=256,
        sr=40000,
        use_f0=True
    )
    synth_weights = dict(mx.load("/Users/mcruz/Developer/Retrieval-based-Voice-Conversion-MLX/Demos/iOS/RVCNative/RVCNativePackage/Sources/RVCNativeFeature/Assets/coder.safetensors"))
    synth.load_weights(list(synth_weights.items()))
    
    # Run inference
    audio_mx = mx.array(audio)[None, :]  # [1, T]
    
    # HuBERT
    print("\n=== HuBERT ===")
    feats = hubert(audio_mx)
    print(f"Features shape: {feats.shape}")
    print(f"Features stats: min={float(feats.min()):.4f}, max={float(feats.max()):.4f}")
    
    # RMVPE
    print("\n=== RMVPE ===")
    f0 = rmvpe.infer_from_audio(audio, thred=0.03)
    print(f"F0 shape: {f0.shape}")
    print(f"F0 stats: min={f0.min():.2f}, max={f0.max():.2f}, mean={f0.mean():.2f}")
    print(f"F0 zeros (unvoiced): {np.sum(f0 == 0)} / {len(f0)}")
    
    # Upsample features (2x)
    B, L, C = feats.shape
    feats_up = mx.broadcast_to(feats[:, :, None, :], (B, L, 2, C)).reshape(B, L*2, C)
    print(f"Upsampled features: {feats_up.shape}")
    
    # Pitch processing
    p_len = min(feats_up.shape[1], len(f0))
    feats_up = feats_up[:, :p_len, :]
    f0 = f0[:p_len]
    
    # Quantize pitch
    f0_mel_min = 1127 * np.log(1 + 50 / 700)
    f0_mel_max = 1127 * np.log(1 + 1100 / 700)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    pitch = np.rint(f0_mel).astype(np.int32)
    
    pitch_mx = mx.array(pitch)[None, :]  # [1, L]
    pitchf_mx = mx.array(f0)[None, :]    # [1, L] - continuous F0 in Hz
    
    print(f"\n=== Pre-Synth ===")
    print(f"phone: {feats_up.shape}")
    print(f"pitch (quantized): {pitch_mx.shape}, range [{int(pitch_mx.min())}, {int(pitch_mx.max())}]")
    print(f"pitchf (Hz): {pitchf_mx.shape}, range [{float(pitchf_mx.min()):.1f}, {float(pitchf_mx.max()):.1f}]")
    
    # Synthesize
    print("\n=== Synthesizer ===")
    sid = mx.array([0])
    phone_lengths = mx.array([p_len])
    
    output, _, _ = synth.infer(
        feats_up,
        phone_lengths,
        pitch_mx,
        pitchf_mx,  # This is nsff0 - continuous F0
        sid
    )
    
    print(f"Output shape: {output.shape}")
    print(f"Output stats: min={float(output.min()):.4f}, max={float(output.max()):.4f}")
    
    # Save
    out_audio = np.array(output[0, :, 0])
    sf.write("/Users/mcruz/Developer/Retrieval-based-Voice-Conversion-MLX/test_output_python.wav", out_audio, 40000)
    print(f"\nSaved to test_output_python.wav ({len(out_audio)} samples)")

if __name__ == "__main__":
    main()
