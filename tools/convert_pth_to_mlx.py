import numpy as np
import mlx.core as mx
import torch
import argparse
import os


def remove_weight_norm(state_dict):
    new_dict = {}
    keys = list(state_dict.keys())
    processed_prefixes = set()

    for k in keys:
        if k.endswith(".weight_g"):
            prefix = k[:-9]
            if prefix in processed_prefixes:
                continue

            w_g = state_dict[k]
            w_v = state_dict[prefix + ".weight_v"]

            norm_v = np.linalg.norm(w_v, axis=(1, 2) if w_v.ndim == 3 else 1)
            if w_v.ndim == 3:
                norm_v = norm_v[:, None, None]
            else:
                norm_v = norm_v[:, None]

            w = w_g * (w_v / norm_v)
            new_dict[prefix + ".weight"] = w
            processed_prefixes.add(prefix)

        elif k.endswith(".weight_v"):
            pass
        else:
            new_dict[k] = state_dict[k]

    return new_dict


def convert_weights(pth_path, output_path):
    print(f"Loading {pth_path}...")
    if pth_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file

            state_dict = load_file(pth_path)
        except ImportError:
            print(
                "Error: safetensors not installed. Please install it with 'pip install safetensors'."
            )
            return
        cpt = {
            "weight": state_dict,
            "config": {},
        }  # Mock cpt for safetensors which usually just has weights
    else:
        cpt = torch.load(pth_path, map_location="cpu")
        if "weight" in cpt:
            state_dict = cpt["weight"]
        elif "model" in cpt:
            state_dict = cpt["model"]
        else:
            state_dict = cpt

    np_dict = {k: v.numpy() for k, v in state_dict.items()}
    np_dict = remove_weight_norm(np_dict)

    mlx_dict = {}
    for k, v in np_dict.items():
        if "emb" in k and "weight" in k:
            pass
        elif "weight" in k and v.ndim == 3:
            if "ups" in k:
                v = v.transpose(1, 2, 0)
            else:
                v = v.transpose(0, 2, 1)
        elif "weight" in k and v.ndim == 2 and "linear" in k.lower():
            v = v.transpose()  # (Out, In) -> (In, Out)

        mlx_dict[k] = mx.array(v)

    print(f"Saving to {output_path}...")
    mx.savez(output_path, **mlx_dict)

    # Also save config if available
    if "config" in cpt:
        import json

        config_path = os.path.splitext(output_path)[0] + ".json"
        with open(config_path, "w") as f:
            json.dump(cpt["config"], f)
        print(f"Saved config to {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input .pth file")
    parser.add_argument("output", help="Output .npz file")
    args = parser.parse_args()

    convert_weights(args.input, args.output)
