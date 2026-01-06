import torch
import torch.nn as nn
import numpy as np
import mlx.core as mx
import os
import sys

# Ensure rvc is in path
sys.path.append(os.getcwd())

from rvc.lib.utils import load_embedding


def convert():
    print("Loading PyTorch Hubert...")
    pt_model = load_embedding("contentvec")
    pt_model.eval()
    state_dict = pt_model.state_dict()
    mlx_weights = {}

    print(f"Total PyTorch keys: {len(state_dict)}")

    h_model = pt_model.hubert if hasattr(pt_model, "hubert") else pt_model

    # Manually extract pos_conv weight
    try:
        w = h_model.encoder.pos_conv_embed.conv.weight
        pos_conv_weight = w.detach().cpu().numpy()
        # PyTorch Conv1d: (Out, In/G, K) -> MLX mx.conv1d: (Out, K, In/G)
        pos_conv_weight = pos_conv_weight.transpose(0, 2, 1)
        mlx_weights["encoder.pos_conv_embed.weight"] = mx.array(pos_conv_weight)
        print("Manually extracted encoder.pos_conv_embed.weight")
    except Exception as e:
        print(f"Warning: Manual pos_conv extraction failed: {e}")

    for key, val in state_dict.items():
        new_key = key

        # 1. Strip 'hubert.' prefix
        if new_key.startswith("hubert."):
            new_key = new_key[len("hubert.") :]

        # 2. Skip masked_spec_embed
        if "masked_spec_embed" in new_key:
            continue

        # 3. Skip weight_norm parametrizations
        if "pos_conv_embed" in new_key and (
            ".weight" in new_key
            or "parametrizations" in new_key
            or "weight_" in new_key
        ):
            if ".weight" in new_key or "weight_" in new_key:
                continue

        # 4. Rename pos_conv_embed.conv -> pos_conv_embed
        if "pos_conv_embed.conv." in new_key:
            new_key = new_key.replace("pos_conv_embed.conv.", "pos_conv_embed.")

        # Transpositions
        val_np = val.cpu().detach().numpy()

        # Conv1d logic (Feature Extractor)
        if "feature_extractor.conv_layers" in new_key and "weight" in new_key:
            # (Out, In, K) -> (Out, K, In)
            if val_np.ndim == 3:
                val_np = val_np.transpose(0, 2, 1)

        # NO TRANSPOSE for Linear weights!
        # MLX nn.Linear (Out, In) matches PyTorch (Out, In).

        mlx_weights[new_key] = mx.array(val_np)

    print(f"Final MLX keys: {len(mlx_weights)}")
    save_path = os.path.join(
        os.getcwd(), "rvc", "models", "embedders", "contentvec", "hubert_mlx.npz"
    )
    mx.savez(save_path, **mlx_weights)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    convert()
