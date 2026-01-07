import sys
import numpy as np
import librosa
import soundfile as sf
import argparse

def load_audio(file_path, sr=16000):
    audio, samplerate = sf.read(file_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if samplerate != sr:
        audio = librosa.resample(audio, orig_sr=samplerate, target_sr=sr)
    return audio.astype(np.float32)

def compare(path1, path2):
    print(f"Comparing:")
    print(f"  A: {path1}")
    print(f"  B: {path2}")
    
    sr = 16000
    y1 = load_audio(path1, sr=sr)
    y2 = load_audio(path2, sr=sr)
    
    min_len = min(len(y1), len(y2))
    y1 = y1[:min_len]
    y2 = y2[:min_len]
    
    # Waveform correlation
    corr = np.corrcoef(y1, y2)[0, 1]
    
    # Spectral correlation
    n_fft = 1024
    hop_length = 256
    S1 = np.abs(librosa.stft(y1, n_fft=n_fft, hop_length=hop_length))
    S2 = np.abs(librosa.stft(y2, n_fft=n_fft, hop_length=hop_length))
    
    mel1 = librosa.feature.melspectrogram(S=S1**2, sr=sr, n_mels=80)
    mel2 = librosa.feature.melspectrogram(S=S2**2, sr=sr, n_mels=80)
    
    log_mel1 = librosa.power_to_db(mel1, ref=np.max)
    log_mel2 = librosa.power_to_db(mel2, ref=np.max)
    
    spec_corr = np.corrcoef(log_mel1.flatten(), log_mel2.flatten())[0, 1]
    
    print(f"\nResults:")
    print(f"  Waveform Corr:   {corr:.6f}")
    print(f"  Spectrogram Corr: {spec_corr:.6f}")
    
    if spec_corr > 0.95:
        print("  ✅ PASS (Perceptually Identical)")
    else:
        print("  ❌ FAIL (Divergent Audio)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_a")
    parser.add_argument("file_b")
    args = parser.parse_args()
    compare(args.file_a, args.file_b)
