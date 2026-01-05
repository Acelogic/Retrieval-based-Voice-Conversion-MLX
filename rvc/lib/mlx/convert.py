import numpy as np
import mlx.core as mx
import torch

def remove_weight_norm(state_dict):
    """
    Simulates weight norm removal by fusing weight_g and weight_v.
    state_dict is a dict of {key: numpy array}.
    """
    new_dict = {}
    keys = list(state_dict.keys())
    processed_prefixes = set()

    for k in keys:
        if k.endswith(".weight_g"):
            prefix = k[:-9]
            if prefix in processed_prefixes: continue
            
            w_g = state_dict[k]
            w_v = state_dict[prefix + ".weight_v"]
            
            # Fuse: w = g * (v / ||v||)
            # PyTorch weight norm dim is usually 0 for Conv1d/Linear
            # dim=0
            norm_v = np.linalg.norm(w_v, axis=(1, 2) if w_v.ndim == 3 else 1) # simple dim check
            # reshape norm to match w_v for broadcast
            if w_v.ndim == 3:
                norm_v = norm_v[:, None, None]
            else:
                norm_v = norm_v[:, None]
                
            w = w_g * (w_v / norm_v)
            
            new_dict[prefix + ".weight"] = w
            processed_prefixes.add(prefix)
            
        elif k.endswith(".weight_v"):
            prefix = k[:-9]
            if prefix not in processed_prefixes:
                # wait for g
                pass
        else:
            new_dict[k] = state_dict[k]
            
    return new_dict

def convert_weights(pth_path):
    """
    Loads a .pth file, converts to numpy, removes weight norm, and prepares for MLX.
    """
    cpt = torch.load(pth_path, map_location="cpu")
    state_dict = cpt["weight"]
    
    # Convert to numpy
    np_dict = {k: v.numpy() for k, v in state_dict.items()}
    
    # Remove weight norm
    np_dict = remove_weight_norm(np_dict)
    
    # Transpose Conv weights for MLX logic (N, L, C) input -> (Out, Time, In) weight?
    # PyTorch Conv1d weight: (Out, In, Kernel)
    # MLX Conv1d weight: (Out, Kernel, In)
    
    mlx_dict = {}
    for k, v in np_dict.items():
        if "conv" in k and "weight" in k and v.ndim == 3:
             if "ups" in k:
                 # ConvTranspose1d: PyTorch (In, Out, K) -> MLX Conv1d (Out, K, In)
                 # Permute (1, 2, 0)
                 v = v.transpose(1, 2, 0)
             else:
                 # Regular Conv1d: PyTorch (Out, In, K) -> MLX (Out, K, In)
                 # Permute (0, 2, 1)
                 v = v.transpose(0, 2, 1)
        elif "emb" in k and "weight" in k:
             # Embedding weight (Num, Dim). PyTorch/MLX same.
             pass
        elif "linear" in k and "weight" in k:
             # Linear weight (Out, In). PyTorch uses x @ W.T + b usually. 
             # MLX Linear weight is (In, Out). 
             # Wait, PyTorch nn.Linear stores (Out, In). 
             # MLX nn.Linear stores (Out, In)? No, MLX internal is (In, Out).
             # Let's check MLX docs mental cache: 
             # MLX nn.Linear(input_dims, output_dims). weight is (input_dims, output_dims).
             # So we must transpose PyTorch (Out, In) -> (In, Out).
             v = v.transpose()
             
        mlx_dict[k] = mx.array(v)
        
    return mlx_dict, cpt["config"]
