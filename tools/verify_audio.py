#!/usr/bin/env python3
"""
Verify MLX RVC output quality through spectral analysis.

This script analyzes the MLX output independently to verify:
1. Proper frequency content matching vocal range
2. No clipping or distortion
3. Proper sample rate utilization
4. Signal energy distribution
"""

import sys
import numpy as np
import soundfile as sf
import argparse

sys.path.insert(0, '.')

def analyze_audio(audio_path):
    """Analyze audio file for quality metrics."""
    print(f"\n=== Analyzing: {audio_path} ===")
    
    audio, sr = sf.read(audio_path)
    duration = len(audio) / sr
    
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Samples: {len(audio)}")
    
    # Amplitude stats
    print(f"\nAmplitude Statistics:")
    print(f"  Max: {np.max(audio):.6f}")
    print(f"  Min: {np.min(audio):.6f}")
    print(f"  Mean: {np.mean(audio):.6f}")
    print(f"  Std: {np.std(audio):.6f}")
    print(f"  RMS: {np.sqrt(np.mean(audio**2)):.6f}")
    
    # Check for clipping
    clip_threshold = 0.99
    clipped_samples = np.sum(np.abs(audio) > clip_threshold)
    clip_pct = clipped_samples / len(audio) * 100
    print(f"\nClipping Analysis:")
    print(f"  Samples > {clip_threshold}: {clipped_samples} ({clip_pct:.4f}%)")
    if clip_pct > 0.1:
        print("  ⚠️  Warning: Significant clipping detected!")
    else:
        print("  ✅ No significant clipping")
    
    # DC offset check
    dc_offset = np.mean(audio)
    if abs(dc_offset) > 0.01:
        print(f"\n⚠️  DC offset detected: {dc_offset:.6f}")
    else:
        print(f"\n✅ Minimal DC offset: {dc_offset:.6f}")
    
    # Spectral analysis (simple FFT)
    try:
        from scipy.fft import rfft, rfftfreq
        
        # Use first 2 seconds for analysis
        chunk = audio[:min(len(audio), sr * 2)]
        
        # Apply window
        window = np.hanning(len(chunk))
        spectrum = np.abs(rfft(chunk * window))
        freqs = rfftfreq(len(chunk), 1/sr)
        
        # Find peak frequency
        peak_idx = np.argmax(spectrum)
        peak_freq = freqs[peak_idx]
        
        # Energy distribution
        low_energy = np.sum(spectrum[freqs < 300]**2)  # Under 300 Hz
        mid_energy = np.sum(spectrum[(freqs >= 300) & (freqs < 3000)]**2)  # 300-3000 Hz (voice)
        high_energy = np.sum(spectrum[freqs >= 3000]**2)  # Above 3000 Hz
        total_energy = low_energy + mid_energy + high_energy
        
        print(f"\nSpectral Analysis (first 2s):")
        print(f"  Peak frequency: {peak_freq:.1f} Hz")
        print(f"  Low freq (<300 Hz): {low_energy/total_energy*100:.1f}%")
        print(f"  Mid freq (300-3000 Hz): {mid_energy/total_energy*100:.1f}%")
        print(f"  High freq (>3000 Hz): {high_energy/total_energy*100:.1f}%")
        
        if mid_energy/total_energy > 0.3:
            print("  ✅ Good vocal frequency content")
        else:
            print("  ⚠️  Low vocal frequency content - may sound muffled or tinny")
            
    except ImportError:
        print("\n(scipy not available for spectral analysis)")
    
    # Zero-crossing rate (indicates voicing quality)
    zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio))))
    zcr = zero_crossings / len(audio) * sr  # per second
    print(f"\nZero-crossing rate: {zcr:.0f} Hz")
    if 500 < zcr < 5000:
        print("  ✅ ZCR in expected range for voice")
    else:
        print("  ⚠️  ZCR outside typical voice range")
    
    return {
        'sr': sr,
        'duration': duration,
        'max_amp': np.max(np.abs(audio)),
        'rms': np.sqrt(np.mean(audio**2)),
        'dc_offset': dc_offset,
        'clip_pct': clip_pct,
    }


def compare_waveforms(file1, file2):
    """Compare two audio files at waveform level."""
    print(f"\n=== Comparing: {file1} vs {file2} ===")
    
    audio1, sr1 = sf.read(file1)
    audio2, sr2 = sf.read(file2)
    
    if sr1 != sr2:
        print(f"⚠️  Sample rates differ: {sr1} vs {sr2}")
        return
    
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]
    
    # Compute metrics
    mse = np.mean((audio1 - audio2)**2)
    rmse = np.sqrt(mse)
    
    # Correlation (but avoid hanging on large arrays)
    if min_len > 500000:
        # Subsample for correlation
        step = min_len // 100000
        a1_sub = audio1[::step]
        a2_sub = audio2[::step]
        corr = np.corrcoef(a1_sub, a2_sub)[0, 1]
        print("(Using subsampled correlation due to large file size)")
    else:
        corr = np.corrcoef(audio1, audio2)[0, 1]
    
    print(f"\nWaveform Comparison:")
    print(f"  MSE: {mse:.10f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  Correlation: {corr:.6f}")
    
    # Normalized comparison
    audio1_norm = audio1 / (np.max(np.abs(audio1)) + 1e-10)
    audio2_norm = audio2 / (np.max(np.abs(audio2)) + 1e-10)
    
    mse_norm = np.mean((audio1_norm - audio2_norm)**2)
    
    if min_len > 500000:
        step = min_len // 100000
        corr_norm = np.corrcoef(audio1_norm[::step], audio2_norm[::step])[0, 1]
    else:
        corr_norm = np.corrcoef(audio1_norm, audio2_norm)[0, 1]
    
    print(f"\nNormalized Comparison:")
    print(f"  Normalized MSE: {mse_norm:.10f}")
    print(f"  Normalized Correlation: {corr_norm:.6f}")
    
    # Quality assessment
    print("\n" + "="*50)
    if corr_norm > 0.99:
        print("✅ EXCELLENT: Nearly identical outputs")
    elif corr_norm > 0.95:
        print("✅ GOOD: Very similar outputs (minor numerical differences)")
    elif corr_norm > 0.8:
        print("⚠️  MODERATE: Noticeable differences")
    elif corr_norm > 0.5:
        print("⚠️  FAIR: Significant differences (different voice characteristics)")
    else:
        print("❌ POOR: Outputs appear to be fundamentally different")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and compare RVC audio outputs")
    parser.add_argument("files", nargs="+", help="Audio files to analyze")
    parser.add_argument("--compare", action="store_true", help="Compare first two files")
    args = parser.parse_args()
    
    for f in args.files:
        analyze_audio(f)
    
    if args.compare and len(args.files) >= 2:
        compare_waveforms(args.files[0], args.files[1])
