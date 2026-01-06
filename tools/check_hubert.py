
import mlx.core as mx
import numpy as np

def check_hubert(path):
    print(f"Loading {path}...")
    try:
        weights = mx.load(path)
    except Exception as e:
        print(f"Error loading: {e}")
        return

    # Check a key
    print(f"Keys: {len(weights)} keys found.")
    
    # Try to find projection layer or final layer to guess output dim
    # final_proj usually "encoder.layers.11.final_layer_norm" or "final_proj"
    
    # Check for 'projection' or 'final'
    if "final_proj.weight" in weights:
         w = weights["final_proj.weight"]
         print(f"final_proj.weight: {w.shape}")
    elif "encoder.layers.11.linear2.weight" in weights:
         w = weights["encoder.layers.11.linear2.weight"]
         print(f"Last encoder layer linear2: {w.shape}")
    elif "encoder.layers.11.fc2.weight" in weights:
         w = weights["encoder.layers.11.fc2.weight"]
         print(f"Last encoder layer fc2: {w.shape}")
         
    # Check first layer
    if "encoder.layers.0.linear1.weight" in weights:
         w = weights["encoder.layers.0.linear1.weight"]
         print(f"First layer linear1: {w.shape}")

if __name__ == "__main__":
    check_hubert("rvc_mlx/models/embedders/contentvec/hubert_mlx.npz")
