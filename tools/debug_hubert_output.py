
import mlx.core as mx
import numpy as np
import soundfile as sf
import librosa
import os
import sys

sys.path.append(os.getcwd())

from rvc_mlx.lib.mlx.hubert import HubertModel, HubertConfig

def test_hubert_output(audio_path, model_path):
    print("Loading audio...")
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1: audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
    print("\nLoading Hubert...")
    conf = HubertConfig(classifier_proj_size=768)
    model = HubertModel(conf)
    model.load_weights(model_path, strict=False)
    
    x = mx.array(audio)[None, :]
    print(f"Input shape: {x.shape}")
    
    try:
        y = model(x)
        print(f"Output shape: {y.shape}")
        
        y_np = np.array(y)
        print(f"Output stats: min={y_np.min()}, max={y_np.max()}, mean={y_np.mean()}, std={y_np.std()}")
        
        # Check for NaN/Inf
        if np.isnan(y_np).any():
             print("❌ Output contains NaNs!")
        if np.isinf(y_np).any():
             print("❌ Output contains Infs!")
             
        # Check first frame vs last frame to see if it's dynamic
        print(f"Frame 0 (first 5 dims): {y_np[0,0,:5]}")
        print(f"Frame 50 (first 5 dims): {y_np[0,50,:5]}")
        
        if np.allclose(y_np[0,0,:], y_np[0,50,:], atol=1e-3):
             print("⚠️ Warning: Output seems constant/collapsed!")
        else:
             print("✅ Output varies over time.")

    except Exception as e:
        print(f"Error running Hubert: {e}")

if __name__ == "__main__":
    test_hubert_output("test-audio/coder_audio_stock.wav", "rvc_mlx/models/embedders/contentvec/hubert_mlx.npz")
