
import torch
import sys

def inspect(path):
    print(f"Loading {path}...")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False) # set weights_only=False to support older serializations if needed, but risky. For inspection ok.
    except Exception as e:
        print(f"Error loading: {e}")
        return

    print(f"Type: {type(ckpt)}")
    if isinstance(ckpt, dict):
        if "weight" in ckpt and isinstance(ckpt["weight"], dict):
             print("\nInspecting 'weight' sub-dict:")
             for k, v in ckpt["weight"].items():
                 if "dec.ups.0.weight" in k:
                     print(f"  {k}: {v.shape}")

if __name__ == "__main__":
    inspect(sys.argv[1])
