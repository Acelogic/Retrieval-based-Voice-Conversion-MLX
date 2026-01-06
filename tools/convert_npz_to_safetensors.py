import numpy as np
from safetensors.numpy import save_file
import sys
import os


def convert(npz_path):
    if not os.path.exists(npz_path):
        print(f"Error: File {npz_path} not found")
        return

    print(f"Loading {npz_path}...")
    data = np.load(npz_path)
    tensors = {k: data[k] for k in data.files}

    output_path = npz_path.replace(".npz", ".safetensors")
    print(f"Saving to {output_path}...")
    save_file(tensors, output_path)
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_npz_to_safetensors.py <file.npz>")
    else:
        convert(sys.argv[1])
