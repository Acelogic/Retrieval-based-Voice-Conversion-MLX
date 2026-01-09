
import torch
import numpy as np
import mlx.core as mx
import os
import sys
import re
import argparse
import json
import shutil

def remap_discriminator_key(key):
    """
    Remap PyTorch discriminator key names to MLX discriminator key names.

    PyTorch structure:
    - discriminators.0 = DiscriminatorS
    - discriminators.1-N = DiscriminatorP(period)

    MLX structure:
    - discriminator_s = DiscriminatorS (with grouped convolutions for conv_1-4)
    - discriminator_p_0, discriminator_p_1, ... = DiscriminatorP
    """
    # DiscriminatorS: discriminators.0.convs.N -> discriminator_s.conv_N
    # Note: conv_1 through conv_4 use grouped convolutions - handled in split function
    match = re.search(r"discriminators\.0\.convs\.(\d+)\.(.+)", key)
    if match:
        idx, rest = match.groups()
        return f"discriminator_s.conv_{idx}.{rest}"

    # DiscriminatorS: discriminators.0.conv_post -> discriminator_s.conv_post
    match = re.search(r"discriminators\.0\.conv_post\.(.+)", key)
    if match:
        rest = match.groups()[0]
        return f"discriminator_s.conv_post.{rest}"

    # DiscriminatorP: discriminators.N.convs.M -> discriminator_p_{N-1}.conv_M
    match = re.search(r"discriminators\.(\d+)\.convs\.(\d+)\.(.+)", key)
    if match:
        disc_idx, conv_idx, rest = match.groups()
        disc_idx = int(disc_idx)
        if disc_idx > 0:  # Skip discriminator 0 (DiscriminatorS)
            p_idx = disc_idx - 1
            return f"discriminator_p_{p_idx}.conv_{conv_idx}.{rest}"

    # DiscriminatorP: discriminators.N.conv_post -> discriminator_p_{N-1}.conv_post
    match = re.search(r"discriminators\.(\d+)\.conv_post\.(.+)", key)
    if match:
        disc_idx, rest = match.groups()
        disc_idx = int(disc_idx)
        if disc_idx > 0:
            p_idx = disc_idx - 1
            return f"discriminator_p_{p_idx}.conv_post.{rest}"

    return key


# DiscriminatorS grouped convolution configuration
# conv_idx: (in_channels, out_channels, groups)
DISCRIMINATOR_S_GROUPED_CONVS = {
    1: (16, 64, 4),
    2: (64, 256, 16),
    3: (256, 1024, 64),
    4: (1024, 1024, 256),
}


def split_grouped_conv_weights(base_key, weight, bias):
    """
    Split grouped convolution weights into separate per-group weights for MLX.

    PyTorch grouped conv: weight shape (out_channels, in_channels/groups, kernel_size)
    MLX GroupedConv1d: separate conv_i for each group with shape (out/groups, kernel, in/groups)

    Returns list of (key, weight) pairs for the split weights.
    """
    # Check if this is a grouped conv in DiscriminatorS
    match = re.search(r"discriminator_s\.conv_(\d+)\.", base_key)
    if not match:
        return None

    conv_idx = int(match.group(1))
    if conv_idx not in DISCRIMINATOR_S_GROUPED_CONVS:
        return None

    in_channels, out_channels, groups = DISCRIMINATOR_S_GROUPED_CONVS[conv_idx]
    out_per_group = out_channels // groups

    # Split weights by group
    # PyTorch weight: (out_channels, in_channels/groups, kernel_size)
    # Each group uses output channels [g*out_per_group : (g+1)*out_per_group]
    results = []

    if weight is not None:
        # Weight has shape (out_channels, kernel_size, in_per_group) in MLX format
        # (already transposed from PyTorch format by caller)
        for g in range(groups):
            start = g * out_per_group
            end = (g + 1) * out_per_group
            group_weight = weight[start:end]  # (out_per_group, kernel, in_per_group)
            new_key = base_key.replace(f"conv_{conv_idx}.", f"conv_{conv_idx}.conv_{g}.")
            results.append((new_key, group_weight))

    if bias is not None:
        # Bias has shape (out_channels,) - split similarly
        for g in range(groups):
            start = g * out_per_group
            end = (g + 1) * out_per_group
            group_bias = bias[start:end]  # (out_per_group,)
            new_key = base_key.replace(f"conv_{conv_idx}.", f"conv_{conv_idx}.conv_{g}.")
            new_key = new_key.replace(".weight", ".bias")
            results.append((new_key, group_bias))

    return results


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

def convert_discriminator_weights(model_path, output_path):
    """
    Convert discriminator weights from PyTorch to MLX format.

    Args:
        model_path: Path to input .pth discriminator model
        output_path: Path to output .npz model
    """
    print(f"Loading PyTorch discriminator from {model_path}...")
    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        print(f"Model not found at {model_path}")
        return

    # Handle different checkpoint formats
    if "weight" in ckpt:
        state_dict = ckpt["weight"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        # Assume it's a raw state dict
        state_dict = ckpt

    print(f"Converting {len(state_dict)} discriminator tensors...")

    # Group parameters by layer for weight norm fusion
    layer_groups = {}

    for key, val in state_dict.items():
        if "num_batches_tracked" in key:
            continue

        val = val.cpu().detach().numpy()

        # Determine suffix
        if key.endswith(".weight"):
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

    print(f"Processing {len(layer_groups)} discriminator layers...")

    for base_key, params in layer_groups.items():
        final_weight = None
        final_bias = params.get("bias", None)

        if "weight" in params:
            final_weight = params["weight"]
        elif "weight_g" in params and "weight_v" in params:
            # Fuse weight norm
            v = params["weight_v"]
            g = params["weight_g"]

            try:
                if v.ndim == 3:
                    # Conv1d: norm over (1, 2)
                    norm_v = np.linalg.norm(v, axis=(1, 2), keepdims=True)
                elif v.ndim == 4:
                    # Conv2d: norm over (1, 2, 3)
                    norm_v = np.linalg.norm(v, axis=(1, 2, 3), keepdims=True)
                elif v.ndim == 2:
                    # Linear
                    norm_v = np.linalg.norm(v, axis=1, keepdims=True)
                elif v.ndim == 1:
                    # 1D: norm as scalar
                    norm_v = np.linalg.norm(v, keepdims=True)
                else:
                    norm_v = np.linalg.norm(v, keepdims=True)
            except ValueError as e:
                print(f"  Warning: Cannot compute norm for {base_key} with shape {v.shape}, using simple norm")
                norm_v = np.linalg.norm(v.flatten(), keepdims=True).reshape([1] * v.ndim)

            if g.ndim == 1:
                target_shape = [1] * v.ndim
                target_shape[0] = g.shape[0]
                g = g.reshape(target_shape)

            final_weight = v * (g / (norm_v + 1e-8))
        elif "other" in params:
            final_weight = params["other"]

        if final_weight is not None:
            full_key = f"{base_key}.weight"
            new_key = remap_discriminator_key(full_key)

            val = final_weight
            # Transpose for MLX
            if val.ndim == 4:
                # Conv2d: (Out, In, H, W) -> (Out, H, W, In)
                val = val.transpose(0, 2, 3, 1)
            elif val.ndim == 3:
                # Conv1d: (Out, In, K) -> (Out, K, In)
                val = val.transpose(0, 2, 1)

            # Check if this is a grouped convolution that needs splitting
            split_results = split_grouped_conv_weights(new_key, val, final_bias)
            if split_results:
                for split_key, split_val in split_results:
                    mlx_weights[split_key] = mx.array(split_val)
                continue  # Skip the normal bias handling below since it's handled in split

            mlx_weights[new_key] = mx.array(val)

        if final_bias is not None:
            full_key = f"{base_key}.bias"
            new_key = remap_discriminator_key(full_key)
            mlx_weights[new_key] = mx.array(final_bias)

    print(f"Converted {len(mlx_weights)} discriminator tensors.")

    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    mx.savez(output_path, **mlx_weights)
    print(f"Saved discriminator weights to {output_path}")


def convert_both(generator_path, discriminator_path, output_dir):
    """
    Convert both generator and discriminator models.

    Args:
        generator_path: Path to generator .pth
        discriminator_path: Path to discriminator .pth
        output_dir: Output directory for converted models
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert generator
    gen_output = os.path.join(output_dir, "generator.npz")
    convert_weights(generator_path, gen_output)

    # Convert discriminator
    disc_output = os.path.join(output_dir, "discriminator.npz")
    convert_discriminator_weights(discriminator_path, disc_output)

    print(f"\nConverted models saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RVC PyTorch model to MLX")
    subparsers = parser.add_subparsers(dest="command", help="Conversion commands")

    # Generator conversion (default behavior)
    gen_parser = subparsers.add_parser("generator", help="Convert generator model")
    gen_parser.add_argument("model_path", type=str, help="Path to input .pth model")
    gen_parser.add_argument("output_path", type=str, help="Path to output .npz model")

    # Discriminator conversion
    disc_parser = subparsers.add_parser("discriminator", help="Convert discriminator model")
    disc_parser.add_argument("model_path", type=str, help="Path to input .pth model")
    disc_parser.add_argument("output_path", type=str, help="Path to output .npz model")

    # Both models
    both_parser = subparsers.add_parser("both", help="Convert both generator and discriminator")
    both_parser.add_argument("--generator", "-g", type=str, required=True, help="Path to generator .pth")
    both_parser.add_argument("--discriminator", "-d", type=str, required=True, help="Path to discriminator .pth")
    both_parser.add_argument("--output-dir", "-o", type=str, required=True, help="Output directory")

    args = parser.parse_args()

    if args.command == "generator":
        convert_weights(args.model_path, args.output_path)
    elif args.command == "discriminator":
        convert_discriminator_weights(args.model_path, args.output_path)
    elif args.command == "both":
        convert_both(args.generator, args.discriminator, args.output_dir)
    else:
        # Default: treat positional args as generator conversion (backward compat)
        if len(sys.argv) >= 3:
            convert_weights(sys.argv[1], sys.argv[2])
        else:
            parser.print_help()
