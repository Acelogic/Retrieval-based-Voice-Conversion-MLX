import numpy as np
import mlx.core as mx
import torch


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

            # dim=0, 1, 2 depending on layer.
            # Conv1d/Linear usually dim=0 (out_channels) for PyTorch weight norm if not transposed yet
            # However, torch.nn.utils.weight_norm default dim=0.

            # Check dimensions
            # Conv1d: (Out, In, K)
            # Linear: (Out, In)
            # ConvTranspose1d: (In, Out, K) -> weight_norm dim usually 1? No, usually 1 for In?
            # Actually RVC uses weight_norm on Conv1d and Linear. Default dim=0.

            if w_v.ndim == 3:
                norm_v = np.linalg.norm(w_v, axis=(1, 2))
                norm_v = norm_v[:, None, None]
            else:
                norm_v = np.linalg.norm(w_v, axis=1)
                norm_v = norm_v[:, None]

            w = w_g * (w_v / norm_v)

            new_dict[prefix + ".weight"] = w
            processed_prefixes.add(prefix)

        elif k.endswith(".weight_v"):
            prefix = k[:-9]
            if prefix not in processed_prefixes:
                pass
        else:
            new_dict[k] = state_dict[k]

    return new_dict


def convert_and_save(pth_path, output_path):
    print(f"Loading {pth_path}...")
    cpt = torch.load(pth_path, map_location="cpu")
    state_dict = cpt["weight"]

    np_dict = {k: v.numpy() for k, v in state_dict.items()}
    np_dict = remove_weight_norm(np_dict)

    mlx_dict = {}
    for k, v in np_dict.items():
        if "conv" in k and "weight" in k and v.ndim == 3:
            if "ups" in k:
                # ConvTranspose1d: PyTorch (In, Out, K) -> MLX Conv1d (Out, K, In)
                v = v.transpose(1, 2, 0)
            else:
                # Regular Conv1d: PyTorch (Out, In, K) -> MLX (Out, K, In)
                v = v.transpose(0, 2, 1)
        elif "linear" in k and "weight" in k:
            # PyTorch (Out, In) -> MLX (In, Out)
            v = v.transpose()

        mlx_dict[k] = mx.array(v)

    print(f"Saving to {output_path}...")
    mx.savez(output_path, **mlx_dict)
    print("Done!")


if __name__ == "__main__":
    pth = "/Users/mcruz/Library/Application Support/Replay/com.replay.Replay/models/Coder999V2/CoderV2_250e_1000s.pth"
    out = "coder.npz"
    convert_and_save(pth, out)
