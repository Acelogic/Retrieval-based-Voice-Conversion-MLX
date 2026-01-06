
import mlx.core as mx
import numpy as np
import soundfile as sf
import librosa
import os
import sys

sys.path.append(os.getcwd())

from rvc_mlx.lib.mlx.rmvpe import RMVPE0Predictor

def check_f0(audio_path):
    print(f"Loading {audio_path}...")
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1: audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
    print("Loading RMVPE...")
    rmvpe = RMVPE0Predictor() 
    # Usually minimal load args if weights are in default location
    
    print("Running F0 extraction...")
    f0 = rmvpe.infer_from_audio(audio, thred=0.03)
    
    print(f"F0 shape: {f0.shape}")
    print(f"F0 stats: min={f0.min()}, max={f0.max()}, mean={f0.mean()}, voiced_frames={np.sum(f0 > 0)}")
    
    # Check if we have voiced frames
    if np.sum(f0 > 0) < 10:
         print("❌ Very few voiced frames detected!")
    else:
         print("✅ Voiced frames detected.")

if __name__ == "__main__":
    check_f0("test-audio/coder_audio_stock.wav")
