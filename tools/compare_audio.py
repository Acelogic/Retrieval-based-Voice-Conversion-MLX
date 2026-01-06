import numpy as np
import soundfile as sf
import sys


def compare_audio(file1, file2):
    try:
        data1, sr1 = sf.read(file1)
        data2, sr2 = sf.read(file2)

        print(f"File 1: {file1}")
        print(f"  Shape: {data1.shape}, Sample Rate: {sr1}")
        print(
            f"  Mean: {np.mean(data1):.6f}, Std: {np.std(data1):.6f}, Max: {np.max(data1):.6f}"
        )

        print(f"File 2: {file2}")
        print(f"  Shape: {data2.shape}, Sample Rate: {sr2}")
        print(
            f"  Mean: {np.mean(data2):.6f}, Std: {np.std(data2):.6f}, Max: {np.max(data2):.6f}"
        )

        if sr1 != sr2:
            print("Warning: Sample rates differ!")

        if data1.shape != data2.shape:
            print(f"Warning: Shapes differ! {data1.shape} vs {data2.shape}")
            min_len = min(len(data1), len(data2))
            data1 = data1[:min_len]
            data2 = data2[:min_len]

        mse = np.mean((data1 - data2) ** 2)
        print(f"Mean Squared Error: {mse:.10f}")

        # Cross correlation
        correlation = np.corrcoef(data1.flatten(), data2.flatten())[0, 1]
        print(f"Correlation: {correlation:.6f}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_audio.py <file1> <file2>")
    else:
        compare_audio(sys.argv[1], sys.argv[2])
