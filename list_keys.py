import torch
import sys

def list_keys(pth_path):
    try:
        cpt = torch.load(pth_path, map_location="cpu")
        weights = cpt.get("weight", cpt)
        keys = sorted(weights.keys())
        for k in keys:
            if "flow" in k:
                print(f"{k}: {weights[k].shape}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python list_keys.py <pth_path>")
    else:
        list_keys(sys.argv[1])
