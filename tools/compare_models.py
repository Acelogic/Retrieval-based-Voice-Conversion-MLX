import sys
import mlx.core as mx
import numpy as np

def compare(file1, file2):
    print(f"Loading {file1}...")
    w1 = mx.load(file1)
    print(f"Loading {file2}...")
    w2 = mx.load(file2)
    
    keys1 = set(w1.keys())
    keys2 = set(w2.keys())
    
    if keys1 != keys2:
        print(f"Key mismatch!")
        print(f"Unique to 1: {keys1 - keys2}")
        print(f"Unique to 2: {keys2 - keys1}")
        # return False # Continue to check common keys
        
    common_keys = keys1.intersection(keys2)
    print(f"Comparing {len(common_keys)} common keys...")
    
    all_match = True
    for k in common_keys:
        a = w1[k]
        b = w2[k]
        if not mx.array_equal(a, b):
            diff = mx.abs(a - b).max()
            if diff > 1e-5:
                print(f"MISMATCH at {k}: max diff {diff}")
                all_match = False
                break
            
    if all_match:
        print("SUCCESS: Models match!")
    else:
        print("FAILURE: Models differ.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_models.py file1 file2")
        sys.exit(1)
    compare(sys.argv[1], sys.argv[2])
