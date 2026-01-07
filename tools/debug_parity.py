#!/usr/bin/env python3
"""
Debug script to compare Python MLX intermediate outputs with Swift debug values.
Helps identify where divergence occurs in the RVC pipeline.
"""

import numpy as np
import mlx.core as mx
import soundfile as sf
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rvc_mlx.lib.mlx.hubert import HubertModel, HubertConfig
from rvc_mlx.lib.mlx.rmvpe import RMVPE0Predictor


def load_audio(path, target_sr=16000):
    """Load and resample audio to target sample rate"""
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def main():
    audio_path = "test-audio/input_16k.wav"

    print("=" * 60)
    print("Python MLX Debug Output - For Comparison with Swift")
    print("=" * 60)

    # Load audio
    audio = load_audio(audio_path)
    audio_mx = mx.array(audio)

    print(f"\n[Audio]")
    print(f"  Shape: [{len(audio)}]")
    print(f"  Min: {audio.min():.7f}, Max: {audio.max():.7f}")
    print(f"  First 20: {audio[:20].tolist()}")

    # HuBERT - Use classifier_proj_size=768 to match RVC inference (no projection)
    print(f"\n[HuBERT]")
    hubert = HubertModel(HubertConfig(classifier_proj_size=768))
    weights_path = "rvc_mlx/models/hubert_base.safetensors"
    if Path(weights_path).exists():
        weights = mx.load(weights_path)
        hubert.load_weights(list(weights.items()))
        print(f"  Loaded weights from {weights_path}")

    audio_batch = audio_mx[None, :]
    hubert_out = hubert(audio_batch)
    hubert_np = np.array(hubert_out)

    print(f"  Output shape: {list(hubert_out.shape)}")
    print(f"  [0,0,:5]: {hubert_np[0,0,:5].tolist()}")
    print(f"  Min: {hubert_np.min():.7f}, Max: {hubert_np.max():.7f}")

    # RMVPE
    print(f"\n[RMVPE]")
    rmvpe = RMVPE0Predictor()

    # Get mel spectrogram
    mel = rmvpe.mel_spectrogram(audio_mx)
    mel_np = np.array(mel)
    print(f"  Mel shape: {list(mel.shape)}")
    print(f"  Mel min: {mel_np.min():.7f}, max: {mel_np.max():.7f}")
    print(f"  Mel[0, :10]: {mel_np[0, :10].tolist()}")

    # Get F0
    f0 = rmvpe.infer_from_audio(audio_mx, thred=0.03)
    f0_np = np.array(f0)

    print(f"  F0 shape: {list(f0.shape)}")
    print(f"  F0 (First 20): {f0_np[:20].flatten().tolist()}")
    print(f"  F0 min: {f0_np.min():.2f}, max: {f0_np.max():.2f}")
    voiced = f0_np[f0_np > 0]
    if len(voiced) > 0:
        print(f"  Voiced frames: {len(voiced)}/{len(f0_np.flatten())} ({100*len(voiced)/len(f0_np.flatten()):.1f}%)")
        print(f"  Voiced mean: {voiced.mean():.2f} Hz")

    # Compare with Swift values
    print("\n" + "=" * 60)
    print("Comparison with Swift Debug Output")
    print("=" * 60)

    # Swift values from benchmark output
    swift_hubert_first5 = [-0.17370184, 0.24801248, 0.0789628, 0.003588026, -0.036665328]
    swift_f0_first20 = [107.1552, 109.2477, 109.6007, 109.9130, 109.9564, 110.2392, 111.2937,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        111.7255, 110.5157, 110.4216, 109.9594, 109.7525]

    print("\n[HuBERT Comparison]")
    py_hubert = hubert_np[0, 0, :5].tolist()
    print(f"  Python: {py_hubert}")
    print(f"  Swift:  {swift_hubert_first5}")
    hubert_diff = np.abs(np.array(py_hubert) - np.array(swift_hubert_first5))
    print(f"  Diff:   {hubert_diff.tolist()}")
    print(f"  Max diff: {hubert_diff.max():.6f}")

    print("\n[F0 Comparison]")
    py_f0 = f0_np[:20].flatten().tolist()
    print(f"  Python: {[f'{x:.4f}' for x in py_f0]}")
    print(f"  Swift:  {[f'{x:.4f}' for x in swift_f0_first20]}")
    f0_diff = np.abs(np.array(py_f0) - np.array(swift_f0_first20))
    print(f"  Max diff: {f0_diff.max():.4f} Hz")

    # Check if F0 pattern matches (ignore absolute values)
    py_voiced_mask = np.array(py_f0) > 0
    swift_voiced_mask = np.array(swift_f0_first20) > 0
    voicing_match = np.sum(py_voiced_mask == swift_voiced_mask) / len(py_f0)
    print(f"  Voicing pattern match: {voicing_match*100:.1f}%")


if __name__ == "__main__":
    main()
