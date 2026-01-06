import sys
import os
import mlx.core as mx

# Add project root to sys.path
sys.path.append(os.getcwd())

from rvc.lib.mlx.convert import convert_weights

MODEL_PATH = "/Users/mcruz/Library/Application Support/Replay/com.replay.Replay/models/Slim Shady/model.pth"
OUTPUT_DIR = "Demos/iOS/RVCDemo/Resources"


def main():
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = MODEL_PATH

    print(f"Converting model: {model_path}")

    try:
        mlx_weights, config = convert_weights(model_path)
    except Exception as e:
        print(f"Error converting weights: {e}")
        return

    # Create output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "model.npz")

    print(f"Saving to {out_path}...")
    mx.savez(out_path, **mlx_weights)
    print("Done!")


if __name__ == "__main__":
    main()
