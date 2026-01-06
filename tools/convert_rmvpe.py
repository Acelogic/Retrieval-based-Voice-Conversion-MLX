import torch
import numpy as np
import mlx.core as mx
import os
import sys
import re

sys.path.append(os.getcwd())


def convert():
    print("Loading PyTorch RMVPE...")
    model_path = "rvc/models/predictors/rmvpe.pt"

    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        print(f"RMVPE model not found at {model_path}. Please verify path.")
        return

    print(f"Converting {len(ckpt)} weights...")
    mlx_weights = {}

    for key, val in ckpt.items():
        val = val.cpu().detach().numpy()
        new_key = key

        # Skip num_batches_tracked
        if "num_batches_tracked" in key:
            continue

        # ==========================================
        # KEY RENAMING RULES
        # Match model parameter paths from tree_flatten output:
        # - unet.encoder.layers.0.blocks.0.conv1.weight
        # - fc.layers.0.0.forward_grus.0.Wx
        # ==========================================

        # 1. Encoder/Intermediate/Decoder layers: conv.Y -> blocks.Y
        if "encoder.layers." in new_key or "intermediate.layers." in new_key:
            new_key = re.sub(r"\.conv\.(\d+)\.", r".blocks.\1.", new_key)

        if "decoder.layers." in new_key:
            new_key = re.sub(r"\.conv2\.(\d+)\.", r".blocks.\1.", new_key)

        # 2. ConvBlockRes internal: conv.0 -> conv1, conv.1 -> bn1, conv.3 -> conv2, conv.4 -> bn2
        if ".blocks." in new_key and ".conv." in new_key:
            new_key = re.sub(r"\.conv\.0\.", r".conv1.", new_key)
            new_key = re.sub(r"\.conv\.1\.", r".bn1.", new_key)
            new_key = re.sub(r"\.conv\.3\.", r".conv2.", new_key)
            new_key = re.sub(r"\.conv\.4\.", r".bn2.", new_key)

        # 3. Decoder conv1_trans
        if "decoder.layers." in new_key:
            new_key = re.sub(r"\.conv1\.0\.", r".conv1_trans.", new_key)
            new_key = re.sub(r"\.conv1\.1\.", r".bn1.", new_key)

        # 4. GRU weights and biases
        # PT: weight_ih (3H, I) -> MLX: Wx (I, 3H) [transpose]
        # PT: weight_hh (3H, H) -> MLX: Wh (H, 3H) [transpose]
        # PT: bias_ih (3H,) -> MLX: b (3H,) [direct]
        # PT: bias_hh (3H,) -> MLX: bhn (H,) [extract last H elements]
        #     In GRU, bias_hh covers [reset, update, new] gates
        #     MLX bhn only needs "new" gate portion = last H elements
        if "fc.0.gru." in key:
            # Hidden size is 256, so bias_hh has 768 elements, we take last 256
            H = 256

            # Backward first (reverse keys)
            new_key = new_key.replace(
                "fc.0.gru.weight_ih_l0_reverse", "bigru.backward_grus.0.Wx"
            )
            new_key = new_key.replace(
                "fc.0.gru.weight_hh_l0_reverse", "bigru.backward_grus.0.Wh"
            )
            new_key = new_key.replace(
                "fc.0.gru.bias_ih_l0_reverse", "bigru.backward_grus.0.b"
            )
            if "bias_hh_l0_reverse" in key:
                new_key = "bigru.backward_grus.0.bhn"
                val = val[-H:]  # Extract last H elements

            # Forward
            new_key = new_key.replace(
                "fc.0.gru.weight_ih_l0", "bigru.forward_grus.0.Wx"
            )
            new_key = new_key.replace(
                "fc.0.gru.weight_hh_l0", "bigru.forward_grus.0.Wh"
            )
            new_key = new_key.replace("fc.0.gru.bias_ih_l0", "bigru.forward_grus.0.b")
            if "bias_hh_l0" in key and "reverse" not in key:
                new_key = "bigru.forward_grus.0.bhn"
                val = val[-H:]  # Extract last H elements

        # 5. FC Linear: fc.1 -> linear
        if "fc.1." in new_key:
            new_key = new_key.replace("fc.1.", "linear.")

        # ==========================================
        # VALUE TRANSPOSITIONS
        # ==========================================

        # ConvTranspose2d FIRST (before regular Conv2d check)
        # PT decoder.layers.X.conv1.0.weight: (In, Out, H, W) -> MLX (Out, H, W, In)
        if "conv1.0." in key and "weight" in key and val.ndim == 4:
            val = val.transpose(1, 2, 3, 0)  # (In,Out,H,W) -> (Out,H,W,In)
        # Regular Conv2d (exclude conv1.0 which is ConvTranspose2d)
        elif (
            ("conv" in key or key.startswith("cnn"))
            and "weight" in key
            and val.ndim == 4
        ):
            val = val.transpose(0, 2, 3, 1)  # (Out,In,H,W) -> (Out,H,W,In)

        # GRU weights transpose (PT: 3H x In -> MLX: In x 3H)
        if ".Wx" in new_key or ".Wh" in new_key:
            if val.ndim == 2:
                val = val.transpose(1, 0)

        mlx_weights[new_key] = mx.array(val)

    print(f"Converted {len(mlx_weights)} tensors.")

    # Sample fc keys
    fc_keys = [k for k in mlx_weights.keys() if "fc." in k]
    print("FC-related keys:", fc_keys)

    save_path = "rvc/models/predictors/rmvpe.safetensors"
    mx.save_safetensors(save_path, mlx_weights)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    convert()
