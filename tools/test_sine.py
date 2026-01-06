
import mlx.core as mx
import numpy as np
import soundfile as sf
import os
import sys

sys.path.append(os.getcwd())

from rvc_mlx.lib.mlx.generators import SineGenerator

def test_sine():
    sr = 48000
    upsample = 480
    gen = SineGenerator(sr, harmonic_num=0)
    
    # 1 second of 440Hz
    # Frames = 100 Hz = 100 frames.
    f0 = mx.ones((1, 100, 1)) * 440
    
    sine_wavs, uv, noise = gen(f0, upsample)
    
    # sine_wavs: (1, 100*480, 1) = (1, 48000, 1)
    # Flatten
    audio = np.array(sine_wavs).flatten()
    
    print(f"Generated {len(audio)} samples.")
    sf.write("test-audio/sine_test.wav", audio, sr)
    print("Saved test-audio/sine_test.wav")
    
    # Check continuity manually (or visually in waveform, but here we check numbers)
    # Check around frame boundaries (every 480 samples)
    # sample 479 and 480
    print(f"Sample 479: {audio[479]}")
    print(f"Sample 480: {audio[480]}")
    print(f"Sample 481: {audio[481]}")
    
    # Difference should be small (continuous)
    diff = abs(audio[480] - audio[479])
    print(f"Diff 479-480: {diff}")
    
    # Compare with expected sine
    # sin(2*pi*440*t)
    t = np.arange(48000) / 48000
    expected = np.sin(2 * np.pi * 440 * t)
    
    # Phase might start random or aligned?
    # Our generator has random phase logic?
    # random_phase = ...
    # So we can't compare exact values, but frequency should match.
    
    pass

if __name__ == "__main__":
    test_sine()
