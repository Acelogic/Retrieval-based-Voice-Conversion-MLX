import numpy as np
import librosa
import torch
import soundfile as sf

def test_mel():
    # Config matching RMVPE.swift
    sr = 16000
    n_fft = 1024
    hop_length = 160
    win_length = 1024
    n_mels = 128
    fmin = 30
    fmax = 8000
    
    # 1. Mel Filterbank
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=True, norm="slaney")
    print(f"DEBUG: Mel Filterbank Stats - shape: {mel_basis.shape}, min: {mel_basis.min():.6f}, max: {mel_basis.max():.6f}, mean: {mel_basis.mean():.6f}")
    print(f"DEBUG: Mel Filterbank [0, :10]: {mel_basis[0, :10]}")
    
    # 2. Audio processing (Replicating PipelineMLX)
    audio, _ = librosa.load("test-audio/coder_audio_stock.wav", sr=sr)
    
    # A. HighPass Filter (PipelineMLX)
    from scipy import signal
    bh, ah = signal.butter(5, 48, btype="high", fs=16000)
    audio = signal.filtfilt(bh, ah, audio)
    
    # B. Pipeline Padding (PipelineMLX: t_pad=1600)
    # Swift RVCInference line 422: padSamples = 1600
    p_pad = 1600
    audio_pipeline = np.pad(audio, (p_pad, p_pad), mode='reflect')
    
    # C. RMVPE Internal Padding (RMVPE.swift / rmvpe.py: pad_len=512)
    # The input to RMVPE is audio_pipeline
    r_pad = 512
    # Swift RMVPE.swift logs "Audio Padded [0...20]" refers to THIS final padded array
    audio_final = np.pad(audio_pipeline, (r_pad, r_pad), mode='reflect')
    
    print(f"DEBUG: Audio Pipeline+RMVPE Padded [0...20]: {audio_final[:20]}")
    print(f"DEBUG: Audio Pipeline+RMVPE Center [512-10...512+10]: {audio_final[502:522]}")

    # STFT using Librosa
    stft = librosa.stft(audio_final, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann', center=False)
    magnitude = np.abs(stft)
    
    mel = mel_basis @ magnitude
    log_mel = np.log(np.maximum(mel, 1e-5))
    
    print(f"DEBUG: Mel Spectrogram Stats: min {log_mel.min():.6f}, max {log_mel.max():.6f}, mean {log_mel.mean():.6f}")
    print(f"DEBUG: Mel[0, :10] (Frame 0 First 10 bins): {log_mel[:10, 0]}")

if __name__ == "__main__":
    test_mel()
