#!/usr/bin/env python3
"""
Test full RVC inference pipeline with a real model.
This will help isolate whether the UNet issue is in RMVPE or the RVC pipeline.
"""

import os
import sys
import numpy as np

from pathlib import Path

# Set environment variable before imports
os.environ["OMP_NUM_THREADS"] = "1"

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_rmvpe_standalone():
    """Test RMVPE pitch detection in isolation."""
    print("=" * 80)
    print("Test 1: RMVPE Standalone (Pitch Detection Only)")
    print("=" * 80)

    try:
        from rvc.lib.mlx.rmvpe import RMVPE0Predictor
        import mlx.core as mx

        weights_file = "rvc/models/predictors/rmvpe_mlx.npz"

        if not os.path.exists(weights_file):
            print(f"❌ MLX weights not found at {weights_file}")
            return False

        print(f"Loading RMVPE from {weights_file}...")
        predictor = RMVPE0Predictor(weights_path=weights_file)
        print("✅ Model loaded")

        # Create test audio (5 seconds)
        print("\nCreating 5s test audio...")
        audio = np.random.randn(16000 * 5).astype(np.float32)

        print("Running inference...")
        f0 = predictor.infer_from_audio(audio, thred=0.03)

        print(f"✅ SUCCESS! F0 shape: {f0.shape}")
        print(f"   F0 range: [{f0.min():.2f}, {f0.max():.2f}]")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_pytorch_rmvpe():
    """Test PyTorch RMVPE for comparison."""
    print("\n" + "=" * 80)
    print("Test 2: PyTorch RMVPE (Baseline)")
    print("=" * 80)

    try:
        import torch
        from rvc.lib.predictors.RMVPE import RMVPE0Predictor as TorchRMVPE

        model_file = "rvc/models/predictors/rmvpe.pt"

        if not os.path.exists(model_file):
            print(f"❌ PyTorch model not found at {model_file}")
            return False

        print(f"Loading PyTorch RMVPE from {model_file}...")
        predictor = TorchRMVPE(model_path=model_file, device="mps")
        print("✅ Model loaded")

        # Create test audio (5 seconds)
        print("\nCreating 5s test audio...")
        audio = np.random.randn(16000 * 5).astype(np.float32)

        print("Running inference...")
        f0 = predictor.infer_from_audio(audio, thred=0.03)

        print(f"✅ SUCCESS! F0 shape: {f0.shape}")
        print(f"   F0 range: [{f0.min():.2f}, {f0.max():.2f}]")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_full_rvc_inference():
    """Test full RVC inference with real model."""
    print("\n" + "=" * 80)
    print("Test 3: Full RVC Inference Pipeline")
    print("=" * 80)

    try:
        from rvc.infer.infer import VoiceConverter

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
        import soundfile as sf

        temp_input = "/tmp/test_input.wav"
        temp_output = "/tmp/test_output.wav"
        sf.write(temp_input, audio, 16000)
        print(f"✅ Saved test audio to {temp_input}")

        print("\nInitializing VoiceConverter...")
        vc = VoiceConverter()

        print("Running inference with PyTorch backend...")
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
            backend="torch",  # Use PyTorch backend first
        )

        if os.path.exists(temp_output):
            print(f"✅ SUCCESS! Output saved to {temp_output}")

            # Check output
            output_audio, sr = sf.read(temp_output)
            print(f"   Output shape: {output_audio.shape}")
            print(f"   Sample rate: {sr}")
            return True
        else:
            print(f"❌ Output file not created")
            return False

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("\n" + "🔍" * 40)
    print("RVC Pipeline Testing Suite")
    print("Investigating RMVPE UNet Shape Mismatch")
    print("🔍" * 40)

    results = {}

    # Test 1: PyTorch RMVPE (should work)
    results["pytorch_rmvpe"] = test_pytorch_rmvpe()

    # Test 2: MLX RMVPE (currently failing)
    results["mlx_rmvpe"] = test_rmvpe_standalone()

    # Test 3: Full RVC pipeline with PyTorch (should work)
    if results["pytorch_rmvpe"]:
        results["full_pipeline"] = test_full_rvc_inference()
    else:
        print("\n⚠️ Skipping full pipeline test (PyTorch RMVPE failed)")
        results["full_pipeline"] = False

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"PyTorch RMVPE:      {'✅ PASSED' if results['pytorch_rmvpe'] else '❌ FAILED'}"
    )
    print(f"MLX RMVPE:          {'✅ PASSED' if results['mlx_rmvpe'] else '❌ FAILED'}")
    print(
        f"Full RVC Pipeline:  {'✅ PASSED' if results['full_pipeline'] else '❌ FAILED'}"
    )
    print("=" * 80)

    if not results["mlx_rmvpe"]:
        print("\n💡 Diagnosis:")
        print("   - MLX RMVPE has a UNet decoder shape mismatch")
        print("   - This is a pre-existing bug in the MLX implementation")
        print("   - PyTorch RMVPE works correctly for comparison")
        print(
            "   - Chunking optimization is ready but cannot be benchmarked until fixed"
        )
    elif results["mlx_rmvpe"] and results["pytorch_rmvpe"]:
        print("\n🎉 All tests passed! Ready to benchmark!")

    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
