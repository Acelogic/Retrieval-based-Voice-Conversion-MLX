
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
    
    # Pre-process biases to combine them where needed
    # We'll map them to the E2E structure in rmvpe.py
    for key, val in ckpt.items():
        val = val.cpu().detach().numpy()
        new_key = key
        
        if "num_batches_tracked" in key:
            continue

        # Remap to unet structure
        if new_key.startswith("unet."):
            # 1. Encoder/Intermediate/Decoder layers: conv.Y -> blocks.Y
            if "encoder.layers." in new_key or "intermediate.layers." in new_key:
                new_key = re.sub(r'\.conv\.(\d+)\.', r'.blocks.\1.', new_key)
            if "decoder.layers." in new_key:
                new_key = re.sub(r'\.conv2\.(\d+)\.', r'.blocks.\1.', new_key)
            
            # 2. ConvBlockRes internal
            if ".blocks." in new_key and ".conv." in new_key:
                new_key = re.sub(r'\.conv\.0\.', r'.conv1.', new_key)
                new_key = re.sub(r'\.conv\.1\.', r'.bn1.', new_key)
                new_key = re.sub(r'\.conv\.3\.', r'.conv2.', new_key)
                new_key = re.sub(r'\.conv\.4\.', r'.bn2.', new_key)
            
            # 3. Decoder conv1_trans
            if "decoder.layers." in new_key:
                new_key = re.sub(r'\.conv1\.0\.', r'.conv1_trans.', new_key)
                new_key = re.sub(r'\.conv1\.1\.', r'.bn1.', new_key)

        # 4. FC layer structure mapping
        # PT: fc.0.gru... -> fc.bigru...
        # PT: fc.1... -> fc.linear...
        if new_key.startswith("fc.0.gru."):
            layer_path = "fc.bigru"
            if "l0_reverse" in new_key:
                direction = "backward_grus.0"
            else:
                direction = "forward_grus.0"

            if "weight_ih" in new_key:
                new_key = f"{layer_path}.{direction}.weight_ih"
            elif "weight_hh" in new_key:
                new_key = f"{layer_path}.{direction}.weight_hh"
            elif "bias_ih" in new_key:
                new_key = f"{layer_path}.{direction}.bias_ih"
            elif "bias_hh" in new_key:
                new_key = f"{layer_path}.{direction}.bias_hh"

        if new_key.startswith("fc.1."):
            new_key = new_key.replace("fc.1.", "fc.linear.")
            # No transpose for Linear (PT is O, I; MLX is O, I)

        # Transpositions for Conv layers
        if "weight" in key and val.ndim == 4:
            if "conv1_trans" in new_key:
                val = val.transpose(1, 2, 3, 0) # (In, Out, H, W) -> (Out, H, W, In)
            else:
                val = val.transpose(0, 2, 3, 1) # (Out, In, H, W) -> (Out, H, W, In)

        mlx_weights[new_key] = mx.array(val)
        
    print(f"Converted {len(mlx_weights)} tensors.")
    
    save_path = "rvc_mlx/models/predictors/rmvpe_mlx.npz"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    mx.savez(save_path, **mlx_weights)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    convert()


if __name__ == "__main__":
    convert()
