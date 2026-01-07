
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
import sys

def calculate_log_mel_spectrogram(audio, sr=48000, n_mels=128, n_fft=2048, hop_length=512):
    # Use consistent params
    window = np.hanning(n_fft)
    stft = scipy.signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft-hop_length, window='hann')[2]
    magnitude = np.abs(stft)
    
    # Manual Mel filterbank construction or just simple log mag spec for correlation
    # For RVC matching, Log Magnitude Spectrogram correlation is the standard metric.
    # We'll use Log Magnitude.
    
    log_spec = np.log1p(magnitude)
    return log_spec

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 compare_correlation.py <file1.wav> <file2.wav>")
        sys.exit(1)
        
    f1 = sys.argv[1]
    f2 = sys.argv[2]
    
    sr1, data1 = wav.read(f1)
    sr2, data2 = wav.read(f2)
    
    if sr1 != sr2:
        print(f"Sample rate mismatch: {sr1} vs {sr2}")
        sys.exit(1)
        
    # Crop to min length
    min_len = min(len(data1), len(data2))
    data1 = data1[:min_len]
    data2 = data2[:min_len]
    
    # Normalize
    data1 = data1.astype(np.float32) / 32768.0 if data1.dtype == np.int16 else data1
    data2 = data2.astype(np.float32) / 32768.0 if data2.dtype == np.int16 else data2
    
    # Calculate Spectrograms
    spec1 = calculate_log_mel_spectrogram(data1, sr=sr1)
    spec2 = calculate_log_mel_spectrogram(data2, sr=sr2)
    
    # Correlation
    # Pearson correlation per frame? Or flattened?
    # Usually we want global correlation or mean frame correlation.
    
    flat1 = spec1.flatten()
    flat2 = spec2.flatten()
    
    corr = np.corrcoef(flat1, flat2)[0, 1]
    
    print(f"Spectral Correlation: {corr:.4f}")
    
    if corr > 0.73:
        print("PASS (> 0.73)")
    else:
        print("FAIL (< 0.73)")

if __name__ == "__main__":
    main()
