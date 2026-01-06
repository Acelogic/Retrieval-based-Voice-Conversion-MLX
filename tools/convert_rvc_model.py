
import torch
import numpy as np
import mlx.core as mx
import os
import sys
import re
import argparse
import json
import shutil

def remap_key_name(key):
    """
    Remap PyTorch RVC key names to MLX RVC key names.
    This consolidates the logic found in infer_mlx.py and applies it permanently.
    """
    new_key = key
    
    # 1. Decoder (dec) remapping
    # Structure: dec.resblocks.0.convs1.0.weight -> dec.resblock_0.c1_0.weight
    
    # ResBlocks
    # dec.resblocks.X.convsY.Z.weight
    match = re.search(r"dec\.resblocks\.(\d+)\.(convs[12])\.(\d+)\.(.+)", key)
    if match:
        idx, group, layer_idx, rest = match.groups()
        # group is convs1 or convs2 -> c1 or c2
        short_group = "c1" if group == "convs1" else "c2"
        return f"dec.resblock_{idx}.{short_group}_{layer_idx}.{rest}"
    
    # Upsamples
    # dec.ups.X.weight -> dec.up_X.weight
    match = re.search(r"dec\.ups\.(\d+)\.(.+)", key)
    if match:
        idx, rest = match.groups()
        return f"dec.up_{idx}.{rest}"
        
    # Noise Convs
    # dec.noise_convs.X.weight -> dec.noise_conv_X.weight
    match = re.search(r"dec\.noise_convs\.(\d+)\.(.+)", key)
    if match:
        idx, rest = match.groups()
        return f"dec.noise_conv_{idx}.{rest}"
        
    # 2. Text Encoder (enc_p) remapping
    # enc_p.encoder.attn_layers.0.conv_q.weight -> enc_p.encoder.attn_0.conv_q.weight
    match = re.search(r"enc_p\.encoder\.attn_layers\.(\d+)\.(.+)", key)
    if match:
        idx, rest = match.groups()
        return f"enc_p.encoder.attn_{idx}.{rest}"
        
    # enc_p.encoder.norm_layers_1.0.weight -> enc_p.encoder.norm1_0.weight
    match = re.search(r"enc_p\.encoder\.norm_layers_1\.(\d+)\.(.+)", key)
    if match:
        idx, rest = match.groups()
        return f"enc_p.encoder.norm1_{idx}.{rest}"
        
    # enc_p.encoder.norm_layers_2.0.weight -> enc_p.encoder.norm2_0.weight
    match = re.search(r"enc_p\.encoder\.norm_layers_2\.(\d+)\.(.+)", key)
    if match:
        idx, rest = match.groups()
        return f"enc_p.encoder.norm2_{idx}.{rest}"
        
    # enc_p.encoder.ffn_layers.0.conv_1.weight -> enc_p.encoder.ffn_0.conv_1.weight
    match = re.search(r"enc_p\.encoder\.ffn_layers\.(\d+)\.(.+)", key)
    if match:
        idx, rest = match.groups()
        return f"enc_p.encoder.ffn_{idx}.{rest}"
        
    # 3. Flow remapping (ResidualCouplingBlock)
    # flow.flows.0.enc.in_layers.0.weight -> flow.flow_0.enc.in_layer_0.weight
    
    # flow.flows.X...
    if key.startswith("flow.flows."):
        parts = key.split(".")
        flow_idx = int(parts[2])
        
        # PyTorch flow list contains [Layer, Flip, Layer, Flip...]
        # We map Layer indices (0, 2, 4...) to MLX indices (0, 1, 2...)
        # Flip modules (1, 3, 5...) have no weights usually, but if they did, we'd skip or handle them.
        
        real_idx = flow_idx // 2
        prefix = f"flow.flow_{real_idx}"
        
        rest_parts = parts[3:]
        # map enc.in_layers.Y -> enc.in_layer_Y
        # map enc.res_skip_layers.Y -> enc.res_skip_layer_Y
        
        current_rest = ".".join(rest_parts)
        
        match_in = re.search(r"enc\.in_layers\.(\d+)\.(.+)", current_rest)
        if match_in:
            l_idx, r = match_in.groups()
            return f"{prefix}.enc.in_layer_{l_idx}.{r}"
            
        match_res = re.search(r"enc\.res_skip_layers\.(\d+)\.(.+)", current_rest)
        if match_res:
            l_idx, r = match_res.groups()
            return f"{prefix}.enc.res_skip_layer_{l_idx}.{r}"
            
        # fallback for pre/post
        return f"{prefix}.{current_rest}"
        
    # 4. Posterior Encoder (enc_q) - same as TextEncoder usually?
    # enc_q.pre, enc_q.proj are simple.
    # enc_q.enc.in_layers... like WaveNet in Flow
    if key.startswith("enc_q.enc."):
        # enc_q.enc.in_layers.0.weight
        match_in = re.search(r"enc_q\.enc\.in_layers\.(\d+)\.(.+)", key)
        if match_in:
             l_idx, r = match_in.groups()
             return f"enc_q.enc.in_layer_{l_idx}.{r}"

        match_res = re.search(r"enc_q\.enc\.res_skip_layers\.(\d+)\.(.+)", key)
        if match_res:
             l_idx, r = match_res.groups()
             return f"enc_q.enc.res_skip_layer_{l_idx}.{r}"
             
    return key

def convert_weights(model_path, output_path):
    print(f"Loading PyTorch model from {model_path}...")
    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        print(f"Model not found at {model_path}")
        return

    # Handle if 'weight' key exists (standard RVC) or if it's a raw state dict
    if "weight" in ckpt:
        state_dict = ckpt["weight"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        # Check if it *is* a state dict
        if "dec.conv_pre.weight" in ckpt or "generator.conv_pre.weight" in ckpt:
             state_dict = ckpt
        else:
             print("Could not find 'weight' or 'model' in checkpoint and it doesn't look like a valid RVC model.")
             return

    # Check for config in ckpt
    config = None
    if "config" in ckpt:
        print("Found config in checkpoint.")
        config = ckpt["config"]
        
    print(f"Converting {len(state_dict)} raw tensors...")
    
    # Group parameters by layer to handle Weight Norm
    # Map: layer_prefix -> { 'weight': val, 'weight_g': val, 'weight_v': val, 'bias': val }
    layer_groups = {}
    
    for key, val in state_dict.items():
        if "num_batches_tracked" in key:
            continue
            
        val = val.cpu().detach().numpy()
        
        # Determine suffix
        # Handle LayerNorm gamma/beta (older PyTorch convention)
        if key.endswith(".gamma"):
            base = key[:-6]
            type_ = "gamma"
        elif key.endswith(".beta"):
            base = key[:-5]
            type_ = "beta"
        elif key.endswith(".weight"):
            base = key[:-7]
            type_ = "weight"
        elif key.endswith(".weight_g"):
            base = key[:-9]
            type_ = "weight_g"
        elif key.endswith(".weight_v"):
            base = key[:-9]
            type_ = "weight_v"
        elif key.endswith(".bias"):
            base = key[:-5]
            type_ = "bias"
        else:
            base = key
            type_ = "other"
            
        if base not in layer_groups:
            layer_groups[base] = {}
        layer_groups[base][type_] = val

    mlx_weights = {}
    
    print(f"Processing {len(layer_groups)} layers...")
    
    for base_key, params in layer_groups.items():
        # fusion logic
        final_weight = None
        # Handle beta (LayerNorm bias) before general bias
        if "beta" in params:
            final_bias = params["beta"]
        else:
            final_bias = params.get("bias", None)

        # Handle LayerNorm: gamma->weight, beta->bias
        if "gamma" in params:
            final_weight = params["gamma"]
        elif "weight" in params:
            final_weight = params["weight"]
        elif "weight_g" in params and "weight_v" in params:
            # Fuse weight norm
            # w = g * (v / ||v||)
            # PyTorch weight_norm:
            # v is parameter
            # g is parameter (magnitude)
            # dim is usually 0 (out_channels) for Conv1d/ConvTranspose1d/Conv2d
            
            v = params["weight_v"]
            g = params["weight_g"]
            
            # Helper to compute norm along correct dimensions
            # Generic logic: norm over all dims EXCEPT dim 0? 
            # PyTorch default dim=0.
            # For Conv1d (Out, In, K), dim=0 is Out. Norm over (In, K).
            # v shape: (Out, In, K). g shape: (Out, 1, 1).
            
            # Check v shape
            # We want to normalize v such that ||v|| across [1, 2, ...] is 1?
            # Actually PyTorch weight_norm default dim=0. 
            # This means for every index in dim 0, we compute norm of the slice, and verify matches g.
            # fused = v * (g / norm(v))
            
            if v.ndim == 3:
                # Conv1d/ConvTranspose1d
                # Norm over (1, 2)
                norm_v = np.linalg.norm(v, axis=(1, 2), keepdims=True)
            elif v.ndim == 4:
                # Conv2d
                norm_v = np.linalg.norm(v, axis=(1, 2, 3), keepdims=True)
            elif v.ndim == 2:
                # Linear (Out, In)
                norm_v = np.linalg.norm(v, axis=1, keepdims=True)
            else:
                # 1D or other?
                # If 1D, just norm
                 norm_v = np.linalg.norm(v, keepdims=True)
            
            # g usually matches shape (C_out, 1, ...) but in checkpoint it might be (C, 1, 1) or just (C).
            # Ensure g broadcasts
            # If g is 1D (C,): reshape to match norm_v
            if g.ndim == 1:
                target_shape = [1] * v.ndim
                target_shape[0] = g.shape[0]
                g = g.reshape(target_shape)
            
            final_weight = v * (g / (norm_v + 1e-8))
            
        elif "other" in params:
             # Just pass through
             final_weight = params["other"]
             # If it's a scalar or something weird, we handle it as weight
        
        # 1. Remap Keys and Transpose
        if final_weight is not None:
             # Construct key
             # Special case: emb_rel embeddings should not have .weight suffix
             if "emb_rel" in base_key:
                 full_key = base_key
             else:
                 full_key = f"{base_key}.weight"
             new_key = remap_key_name(full_key)

             val = final_weight
             # Transpose logic matches previous script
             if val.ndim == 4:
                # Conv2d: (Out, In, H, W) -> (Out, H, W, In)
                val = val.transpose(0, 2, 3, 1)
             elif val.ndim == 3:
                # Don't transpose relative position embeddings (emb_rel_k, emb_rel_v)
                if "emb_rel" in base_key:
                    # These are (n_heads, 2*window+1, head_dim), keep as-is
                    pass
                # Conv1d / ConvTranspose1d
                elif "dec.ups" in base_key or "dec.up_" in new_key:
                    # ConvTranspose1d: PyTorch (In, Out, K) -> MLX (Out, K, In)
                    # Use (1, 2, 0)
                    val = val.transpose(1, 2, 0)
                else:
                    # Conv1d: PyTorch (Out, In, K) -> MLX (Out, K, In)
                    # Use (0, 2, 1)
                    val = val.transpose(0, 2, 1)

             mlx_weights[new_key] = mx.array(val)
             
        if final_bias is not None:
             full_key = f"{base_key}.bias"
             new_key = remap_key_name(full_key)
             mlx_weights[new_key] = mx.array(final_bias)

    print(f"Converted {len(mlx_weights)} tensors.")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mx.savez(output_path, **mlx_weights)
    print(f"Saved weights to {output_path}")
    
    # Handle Config
    # If config found in checkpoint, save it as json
    # Or look for config.json next to model_path
    config_params = None
    if config is not None:
        config_params = config
    else:
         # Try finding config.json
         base_path = os.path.splitext(model_path)[0]
         json_path = base_path + ".json"
         if os.path.exists(json_path):
             print(f"Found external config at {json_path}")
             with open(json_path, 'r') as f:
                 config_params = json.load(f)
    
    if config_params:
         out_json_path = os.path.splitext(output_path)[0] + ".json"
         with open(out_json_path, 'w') as f:
             json.dump(config_params, f, indent=2)
         print(f"Saved config to {out_json_path}")
    else:
         print("Warning: No configuration found. Inference might use defaults which can lead to mismatches (e.g. 40k vs 48k).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RVC PyTorch model to MLX")
    parser.add_argument("model_path", type=str, help="Path to input .pth model")
    parser.add_argument("output_path", type=str, help="Path to output .npz model")
    
    args = parser.parse_args()
    convert_weights(args.model_path, args.output_path)
