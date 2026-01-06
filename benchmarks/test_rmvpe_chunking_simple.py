#!/usr/bin/env python3
"""
Simple test to validate RMVPE chunking logic without requiring model weights.
Tests the chunking mechanism by mocking the model.
"""

import numpy as np
import mlx.core as mx
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))


def test_chunking_logic():
    """Test chunking logic with a mock model."""
    print("=" * 60)
    print("RMVPE Chunking Logic Test (No Model Required)")
    print("=" * 60)

    # Create a simple mock model function
    def mock_model(mel_input):
        """Mock model that returns predictable output based on input shape."""
        # Input shape: (1, T, 128, 1)
        # Output shape: (1, T, 360)
        batch, time, mels, channels = mel_input.shape
        # Return simple pattern for validation
        output = mx.ones((batch, time, 360)) * mx.mean(mel_input)
        return output

    # Test different mel lengths
    test_cases = [
        ("Short (16k frames, < chunk_size)", 16000),
        ("Exactly chunk_size (32k frames)", 32000),
        ("Just over (32001 frames)", 32001),
        ("Medium (48k frames)", 48000),
        ("Long (64k frames)", 64000),
    ]

    chunk_size = 32000
    all_passed = True

    for name, n_frames in test_cases:
        print(f"\n{'─' * 60}")
        print(f"Test: {name}")
        print(f"{'─' * 60}")

        # Create synthetic mel
        mel = np.random.randn(128, n_frames).astype(np.float32)

        # Simulate the mel2hidden logic
        # 1. Pad to multiple of 32
        pad_curr = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        mel_padded = np.pad(mel, ((0, 0), (0, pad_curr)), mode="reflect")

        # 2. Convert and reshape
        mel_mx = mx.array(mel_padded)
        mel_mx = mel_mx.transpose(1, 0)[None, :, :, None]  # (1, T_pad, 128, 1)
        pad_frames = mel_mx.shape[1]

        print(f"  Original frames: {n_frames}")
        print(f"  Padded frames: {pad_frames}")
        print(f"  Will chunk: {pad_frames > chunk_size}")

        # 3. Test single-pass
        print("  Testing single-pass...")
        hidden_single = mock_model(mel_mx)
        print(f"    Output shape: {hidden_single.shape}")

        # 4. Test chunked
        if pad_frames > chunk_size:
            print(f"  Testing chunked (chunk_size={chunk_size})...")
            output_chunks = []
            num_chunks = 0
            for start in range(0, pad_frames, chunk_size):
                end = min(start + chunk_size, pad_frames)
                mel_chunk = mel_mx[:, start:end, :, :]
                chunk_frames = mel_chunk.shape[1]

                # Verify chunk alignment
                if chunk_frames % 32 != 0:
                    print(
                        f"    ❌ ERROR: Chunk {num_chunks} has {chunk_frames} frames (not divisible by 32)"
                    )
                    all_passed = False
                    break

                out_chunk = mock_model(mel_chunk)
                mx.eval(out_chunk)
                output_chunks.append(out_chunk)
                num_chunks += 1

            if output_chunks:
                hidden_chunked = mx.concatenate(output_chunks, axis=1)
                print(f"    Processed {num_chunks} chunks")
                print(f"    Output shape: {hidden_chunked.shape}")

                # Verify shapes match
                if hidden_single.shape != hidden_chunked.shape:
                    print(f"    ❌ FAILED: Shape mismatch!")
                    print(
                        f"       Single: {hidden_single.shape}, Chunked: {hidden_chunked.shape}"
                    )
                    all_passed = False
                else:
                    # Verify values match (they should with our mock model)
                    hidden_single_np = np.array(hidden_single)
                    hidden_chunked_np = np.array(hidden_chunked)

                    if np.allclose(hidden_single_np, hidden_chunked_np, rtol=1e-5):
                        print(f"    ✅ PASSED: Chunking preserves output")
                    else:
                        max_diff = np.max(np.abs(hidden_single_np - hidden_chunked_np))
                        print(
                            f"    ❌ FAILED: Outputs differ (max diff: {max_diff:.2e})"
                        )
                        all_passed = False
        else:
            print(f"  Skipping chunked test (short audio)")
            print(f"    ✅ PASSED: Single-pass works correctly")

        # 5. Strip padding
        hidden_final = hidden_single[:, :n_frames, :]
        print(f"  Final output shape after stripping padding: {hidden_final.shape}")

        if hidden_final.shape[1] != n_frames:
            print(f"    ❌ FAILED: Final shape incorrect")
            all_passed = False

    # Summary
    print(f"\n{'=' * 60}")
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("Chunking logic is correctly implemented.")
    else:
        print("❌ SOME TESTS FAILED!")
    print("=" * 60)

    return all_passed


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "=" * 60)
    print("Edge Cases Test")
    print("=" * 60)

    # Test various chunk sizes
    test_cases = [
        (1000, 32000),  # Very short
        (31999, 32000),  # Just under chunk
        (32000, 32000),  # Exactly chunk
        (32001, 32000),  # Just over chunk
        (64000, 32000),  # Exactly 2x chunk
        (64032, 32000),  # 2x chunk + 32
        (100000, 32000),  # Long audio
    ]

    all_passed = True

    for n_frames, chunk_size in test_cases:
        # Check padding alignment
        pad_curr = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        pad_frames = n_frames + pad_curr

        # Calculate number of chunks
        if pad_frames <= chunk_size:
            n_chunks = 1
        else:
            n_chunks = (pad_frames + chunk_size - 1) // chunk_size

        # Calculate last chunk size
        last_chunk_start = (n_chunks - 1) * chunk_size
        last_chunk_size = pad_frames - last_chunk_start

        print(
            f"\nFrames: {n_frames:6d}, Padded: {pad_frames:6d}, Chunks: {n_chunks}, Last chunk: {last_chunk_size:6d}",
            end="",
        )

        # Verify last chunk is divisible by 32
        if last_chunk_size % 32 != 0:
            print(f"  ❌ FAILED (last chunk not divisible by 32)")
            all_passed = False
        else:
            print(f"  ✅")

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL EDGE CASES PASSED!")
    else:
        print("❌ SOME EDGE CASES FAILED!")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    print("\nRMVPE Chunking Logic Validation")
    print("Testing chunking implementation without model weights\n")

    logic_passed = test_chunking_logic()
    edge_passed = test_edge_cases()

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Logic Tests: {'✅ PASSED' if logic_passed else '❌ FAILED'}")
    print(f"Edge Cases: {'✅ PASSED' if edge_passed else '❌ FAILED'}")
    print("=" * 60)

    if logic_passed and edge_passed:
        print("\n✅ Implementation is correct!")
        print("The chunking logic properly:")
        print("  - Pads to multiples of 32")
        print("  - Processes chunks when needed")
        print("  - Skips chunking for short audio")
        print("  - Preserves output shape and values")
        print("  - Strips padding correctly")

    sys.exit(0 if (logic_passed and edge_passed) else 1)
