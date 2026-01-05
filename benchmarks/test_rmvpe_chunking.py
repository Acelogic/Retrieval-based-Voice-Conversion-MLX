#!/usr/bin/env python3
"""
Test script to validate RMVPE chunking implementation.
Tests equivalence between chunked and single-pass processing.
"""

import numpy as np
import mlx.core as mx
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from rvc.lib.mlx.rmvpe import RMVPE0Predictor

def test_chunking_equivalence():
    """Test that chunked processing produces identical results to single-pass."""
    print("=" * 60)
    print("RMVPE Chunking Equivalence Test")
    print("=" * 60)

    # Model path
    model_path = "/Users/mcruz/Library/Application Support/Replay/com.replay.Replay/models"
    weights_path = os.path.join(model_path, "rmvpe.mlx.safetensors")

    if not os.path.exists(weights_path):
        print(f"ERROR: Model not found at {weights_path}")
        print("Please ensure the model is converted and available.")
        return False

    print(f"\nLoading RMVPE model from: {weights_path}")
    predictor = RMVPE0Predictor(weights_path=weights_path)
    print("Model loaded successfully!")

    # Test cases with different audio lengths
    test_cases = [
        ("Short audio (16k frames, ~160s)", 16000),
        ("Exactly chunk size (32k frames, ~320s)", 32000),
        ("Medium audio (48k frames, ~480s)", 48000),
        ("Long audio (64k frames, ~640s)", 64000),
        ("Very long audio (100k frames, ~1000s)", 100000),
    ]

    all_passed = True

    for name, n_frames in test_cases:
        print(f"\n{'─' * 60}")
        print(f"Test Case: {name}")
        print(f"{'─' * 60}")

        # Create synthetic mel spectrogram
        # Shape: (n_mels, T) where n_mels=128
        print(f"Creating synthetic mel spectrogram: (128, {n_frames})")
        mel = np.random.randn(128, n_frames).astype(np.float32)

        # Test 1: Single-pass (chunk_size=None)
        print("Running single-pass mode (chunk_size=None)...")
        hidden_single = predictor.mel2hidden(mel, chunk_size=None)
        print(f"  Output shape: {hidden_single.shape}")

        # Test 2: Chunked processing (chunk_size=32000)
        print("Running chunked mode (chunk_size=32000)...")
        hidden_chunked = predictor.mel2hidden(mel, chunk_size=32000)
        print(f"  Output shape: {hidden_chunked.shape}")

        # Verify shapes match
        if hidden_single.shape != hidden_chunked.shape:
            print(f"  ❌ FAILED: Shape mismatch!")
            print(f"     Single-pass: {hidden_single.shape}")
            print(f"     Chunked: {hidden_chunked.shape}")
            all_passed = False
            continue

        # Convert to numpy for comparison
        hidden_single_np = np.array(hidden_single)
        hidden_chunked_np = np.array(hidden_chunked)

        # Check equivalence
        are_equal = np.allclose(hidden_single_np, hidden_chunked_np, rtol=1e-5, atol=1e-6)

        if are_equal:
            # Calculate statistics
            max_diff = np.max(np.abs(hidden_single_np - hidden_chunked_np))
            mean_diff = np.mean(np.abs(hidden_single_np - hidden_chunked_np))

            print(f"  ✅ PASSED: Outputs are equivalent!")
            print(f"     Max absolute difference: {max_diff:.2e}")
            print(f"     Mean absolute difference: {mean_diff:.2e}")
        else:
            print(f"  ❌ FAILED: Outputs differ significantly!")
            max_diff = np.max(np.abs(hidden_single_np - hidden_chunked_np))
            mean_diff = np.mean(np.abs(hidden_single_np - hidden_chunked_np))
            print(f"     Max absolute difference: {max_diff:.2e}")
            print(f"     Mean absolute difference: {mean_diff:.2e}")
            all_passed = False

    # Summary
    print(f"\n{'=' * 60}")
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("Chunking implementation is correct and produces identical results.")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please review the implementation.")
    print("=" * 60)

    return all_passed

def test_edge_cases():
    """Test edge cases and special scenarios."""
    print("\n" + "=" * 60)
    print("RMVPE Edge Cases Test")
    print("=" * 60)

    model_path = "/Users/mcruz/Library/Application Support/Replay/com.replay.Replay/models"
    weights_path = os.path.join(model_path, "rmvpe.mlx.safetensors")

    if not os.path.exists(weights_path):
        print(f"ERROR: Model not found at {weights_path}")
        return False

    print(f"\nLoading RMVPE model...")
    predictor = RMVPE0Predictor(weights_path=weights_path)
    print("Model loaded!")

    test_cases = [
        ("Very short (1k frames)", 1000),
        ("Edge: 32001 frames (just over chunk)", 32001),
        ("Edge: 31999 frames (just under chunk)", 31999),
    ]

    all_passed = True

    for name, n_frames in test_cases:
        print(f"\n{name}:")
        mel = np.random.randn(128, n_frames).astype(np.float32)

        try:
            hidden = predictor.mel2hidden(mel, chunk_size=32000)
            print(f"  ✅ Processed successfully: {hidden.shape}")
        except Exception as e:
            print(f"  ❌ FAILED with error: {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL EDGE CASES PASSED!")
    else:
        print("❌ SOME EDGE CASES FAILED!")
    print("=" * 60)

    return all_passed

if __name__ == "__main__":
    # Set environment variable for faiss
    os.environ["OMP_NUM_THREADS"] = "1"

    print("\nRMVPE Chunking Implementation Test Suite")
    print("Testing MLX RMVPE mel2hidden chunking implementation\n")

    # Run tests
    eq_passed = test_chunking_equivalence()
    edge_passed = test_edge_cases()

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Equivalence Tests: {'✅ PASSED' if eq_passed else '❌ FAILED'}")
    print(f"Edge Case Tests: {'✅ PASSED' if edge_passed else '❌ FAILED'}")
    print("=" * 60)

    sys.exit(0 if (eq_passed and edge_passed) else 1)
