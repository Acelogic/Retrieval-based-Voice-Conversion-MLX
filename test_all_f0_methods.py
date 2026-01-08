#!/usr/bin/env python3
"""Test all F0 extraction methods with real audio inference."""

import os
import sys
import time
import numpy as np
import librosa
import soundfile as sf

# Add project to path
sys.path.insert(0, '/Users/mcruz/Developer/Retrieval-based-Voice-Conversion-MLX')

from rvc_mlx.lib.mlx.pitch_extractors import PitchExtractor

# Test parameters
AUDIO_FILE = "test-audio/input_16k.wav"
OUTPUT_DIR = "test_results/f0_comparison"
METHODS = ["rmvpe", "dio", "pm", "harvest", "fcpe"]  # Skip crepe (needs weights)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load audio
    print(f"Loading audio: {AUDIO_FILE}")
    audio, sr = librosa.load(AUDIO_FILE, sr=16000)
    duration = len(audio) / sr
    print(f"  Duration: {duration:.2f}s, Sample rate: {sr} Hz")
    
    results = {}
    
    print("\n" + "="*60)
    print("Testing F0 extraction methods")
    print("="*60)
    
    for method in METHODS:
        print(f"\n[{method.upper()}]")
        try:
            # Create extractor
            start = time.time()
            extractor = PitchExtractor(method=method, sample_rate=16000, hop_size=160)
            init_time = time.time() - start
            
            # Extract F0
            start = time.time()
            f0 = extractor.extract(audio, f0_min=50, f0_max=1100)
            extract_time = time.time() - start
            
            # Analyze results
            voiced_mask = f0 > 0
            voiced_ratio = voiced_mask.mean()
            
            if voiced_mask.sum() > 0:
                voiced_f0 = f0[voiced_mask]
                mean_f0 = voiced_f0.mean()
                std_f0 = voiced_f0.std()
                min_f0 = voiced_f0.min()
                max_f0 = voiced_f0.max()
            else:
                mean_f0 = std_f0 = min_f0 = max_f0 = 0
            
            results[method] = {
                'f0': f0,
                'voiced_ratio': voiced_ratio,
                'mean_f0': mean_f0,
                'std_f0': std_f0,
                'min_f0': min_f0,
                'max_f0': max_f0,
                'extract_time': extract_time,
            }
            
            print(f"  Frames: {len(f0)}")
            print(f"  Voiced: {voiced_ratio*100:.1f}%")
            print(f"  F0 range: {min_f0:.1f} - {max_f0:.1f} Hz")
            print(f"  Mean F0: {mean_f0:.1f} Hz (±{std_f0:.1f})")
            print(f"  Time: {extract_time:.3f}s ({duration/extract_time:.1f}x realtime)")
            
            # Save F0 contour
            np.save(f"{OUTPUT_DIR}/{method}_f0.npy", f0)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results[method] = None
    
    # Compare methods
    print("\n" + "="*60)
    print("Method Comparison")
    print("="*60)
    
    valid_methods = [m for m in METHODS if results.get(m) is not None]
    
    if len(valid_methods) >= 2:
        # Correlation matrix
        print("\nF0 Correlation Matrix:")
        print(f"{'':>10}", end='')
        for m in valid_methods:
            print(f"{m:>10}", end='')
        print()
        
        for m1 in valid_methods:
            print(f"{m1:>10}", end='')
            f0_1 = results[m1]['f0']
            for m2 in valid_methods:
                f0_2 = results[m2]['f0']
                # Resample if different lengths
                if len(f0_1) != len(f0_2):
                    from scipy.ndimage import zoom
                    f0_2_resampled = zoom(f0_2, len(f0_1) / len(f0_2))
                else:
                    f0_2_resampled = f0_2
                
                # Calculate correlation on voiced regions
                both_voiced = (f0_1 > 0) & (f0_2_resampled > 0)
                if both_voiced.sum() > 10:
                    corr = np.corrcoef(f0_1[both_voiced], f0_2_resampled[both_voiced])[0, 1]
                    print(f"{corr:>10.3f}", end='')
                else:
                    print(f"{'N/A':>10}", end='')
            print()
    
    # Speed comparison
    print("\nSpeed Comparison:")
    for method in valid_methods:
        r = results[method]
        rtf = duration / r['extract_time']
        print(f"  {method:>10}: {r['extract_time']:.3f}s ({rtf:.1f}x realtime)")
    
    print(f"\nResults saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
