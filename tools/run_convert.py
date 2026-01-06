import sys
import os
import mlx.core as mx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rvc.lib.mlx.convert import convert_weights


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pth_path = os.path.join(project_root, "rvc/models/pretraineds/hifi-gan/f0G40k.pth")
    save_path = os.path.join(
        project_root, "rvc/models/checkpoints/hifi-gan-f0G40k.safetensors"
    )

    # Ensure dir exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Converting {pth_path}...")
    weights, config = convert_weights(pth_path)

    print(f"Saving to {save_path}...")
    mx.save_safetensors(save_path, weights)
    print("Done!")


if __name__ == "__main__":
    main()
