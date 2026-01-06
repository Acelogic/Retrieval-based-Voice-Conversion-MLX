#!/usr/bin/env python3
"""
Quick test to verify HuBERT output dimension in Python MLX implementation.
"""

import os
import sys
import numpy as np

# Set environment variable before imports
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_hubert_dimension():
    """Test what dimension HuBERT actually outputs in Python MLX."""
    print("=" * 80)
    print("Testing HuBERT Output Dimension in Python MLX")
    print("=" * 80)

    try:
        from rvc.infer.infer import VoiceConverter
        import soundfile as sf

        model_dir = "/Users/mcruz/Library/Application Support/Replay/com.replay.Replay/models/Slim Shady"
        model_file = os.path.join(model_dir, "model.pth")
        index_file = os.path.join(model_dir, "model.index")

        if not os.path.exists(model_file):
            print(f"❌ Model not found at {model_file}")
            return False

        print(f"Model: {model_file}")
        print(f"Index: {index_file}")

        # Create test audio (5 seconds at 16kHz)
        print("\nCreating 5s test audio...")
        audio = np.random.randn(16000 * 5).astype(np.float32) * 0.5

        # Save as temp file
        temp_input = "/tmp/test_input.wav"
        temp_output = "/tmp/test_output_mlx.wav"
        sf.write(temp_input, audio, 16000)
        print(f"✅ Saved test audio to {temp_input}")

        print("\nInitializing VoiceConverter with MLX backend...")
        vc = VoiceConverter()

        print("\nRunning inference with MLX backend...")
        print("📊 Watch for DEBUG prints showing HuBERT dimension:")
        print("")

        result = vc.convert_audio(
            audio_input_path=temp_input,
            audio_output_path=temp_output,
            model_path=model_file,
            index_path=index_file,
            pitch=0,
            f0_method="rmvpe",
            index_rate=0.5,
            volume_envelope=1.0,
            protect=0.33,
            hop_length=128,
            f0_autotune=False,
            split_audio=False,
            embedder_model="contentvec",
            backend="mlx"  # Use MLX backend!
        )

        if os.path.exists(temp_output):
            print(f"\n✅ SUCCESS! Output saved to {temp_output}")

            # Check output
            output_audio, sr = sf.read(temp_output)
            print(f"   Output shape: {output_audio.shape}")
            print(f"   Sample rate: {sr}")
            return True
        else:
            print(f"\n❌ Output file not created")
            return False

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hubert_dimension()
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)
    sys.exit(0 if success else 1)
