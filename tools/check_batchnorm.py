#!/usr/bin/env python3
"""
Check if BatchNorm layers are loading and using running statistics.
"""

import sys
import os
import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.getcwd())

# Force reload
for mod in list(sys.modules.keys()):
    if 'rvc_mlx' in mod:
        del sys.modules[mod]

from rvc_mlx.lib.mlx.rmvpe import RMVPE0Predictor

def check_batchnorm_params(model, prefix=""):
    """Recursively check all BatchNorm layers."""
    batchnorm_info = []

    # Check if this is a BatchNorm layer
    if isinstance(model, nn.BatchNorm):
        info = {
            "name": prefix,
            "weight_shape": model.weight.shape if hasattr(model, 'weight') else None,
            "bias_shape": model.bias.shape if hasattr(model, 'bias') else None,
            "running_mean_shape": model.running_mean.shape if hasattr(model, 'running_mean') else None,
            "running_var_shape": model.running_var.shape if hasattr(model, 'running_var') else None,
            "weight_range": (float(model.weight.min()), float(model.weight.max())) if hasattr(model, 'weight') else None,
            "bias_range": (float(model.bias.min()), float(model.bias.max())) if hasattr(model, 'bias') else None,
            "running_mean_range": (float(model.running_mean.min()), float(model.running_mean.max())) if hasattr(model, 'running_mean') else None,
            "running_var_range": (float(model.running_var.min()), float(model.running_var.max())) if hasattr(model, 'running_var') else None,
        }
        batchnorm_info.append(info)

    # Recursively check submodules
    if hasattr(model, '__dict__'):
        for name, submodule in model.__dict__.items():
            if isinstance(submodule, nn.Module):
                child_prefix = f"{prefix}.{name}" if prefix else name
                batchnorm_info.extend(check_batchnorm_params(submodule, child_prefix))

    return batchnorm_info

def main():
    print("=== Checking BatchNorm Parameters ===\n")

    # Load model
    predictor = RMVPE0Predictor()

    # Check what weights were loaded
    print("--- Loaded Weight Keys ---")
    weights_dict = dict(np.load("rvc_mlx/models/predictors/rmvpe_mlx.npz"))
    bn_keys = [k for k in weights_dict.keys() if 'bn' in k.lower() or 'batch' in k.lower()]
    print(f"Found {len(bn_keys)} BatchNorm-related keys in weight file:")
    for key in bn_keys[:10]:  # Show first 10
        print(f"  {key}: shape={weights_dict[key].shape}")
    if len(bn_keys) > 10:
        print(f"  ... and {len(bn_keys)-10} more")

    # Check model BatchNorm layers
    print("\n--- BatchNorm Layers in Model ---")
    bn_layers = check_batchnorm_params(predictor.model)
    print(f"Found {len(bn_layers)} BatchNorm layers in model:")
    for i, info in enumerate(bn_layers[:5]):  # Show first 5
        print(f"\n{i+1}. {info['name']}:")
        print(f"   weight: {info['weight_shape']}, range={info['weight_range']}")
        print(f"   bias: {info['bias_shape']}, range={info['bias_range']}")
        print(f"   running_mean: {info['running_mean_shape']}, range={info['running_mean_range']}")
        print(f"   running_var: {info['running_var_shape']}, range={info['running_var_range']}")
    if len(bn_layers) > 5:
        print(f"\n... and {len(bn_layers)-5} more BatchNorm layers")

    # Check first encoder BatchNorm specifically
    print("\n--- First Encoder BatchNorm ---")
    bn = predictor.model.unet.encoder.bn
    print(f"Type: {type(bn)}")
    print(f"Weight: {bn.weight.shape}, range=[{float(mx.min(bn.weight)):.4f}, {float(mx.max(bn.weight)):.4f}]")
    print(f"Bias: {bn.bias.shape}, range=[{float(mx.min(bn.bias)):.4f}, {float(mx.max(bn.bias)):.4f}]")
    if hasattr(bn, 'running_mean'):
        print(f"Running mean: {bn.running_mean.shape}, range=[{float(mx.min(bn.running_mean)):.4f}, {float(mx.max(bn.running_mean)):.4f}]")
        print(f"Running var: {bn.running_var.shape}, range=[{float(mx.min(bn.running_var)):.4f}, {float(mx.max(bn.running_var)):.4f}]")
    else:
        print("❌ No running_mean/running_var attributes!")

    # Test BatchNorm forward pass
    print("\n--- Testing BatchNorm Forward Pass ---")
    test_input = mx.random.normal(shape=(1, 64, 128, 1))
    output = bn(test_input)
    print(f"Input: shape={test_input.shape}, range=[{float(mx.min(test_input)):.4f}, {float(mx.max(test_input)):.4f}]")
    print(f"Output: shape={output.shape}, range=[{float(mx.min(output)):.4f}, {float(mx.max(output)):.4f}]")
    print(f"Output mean: {float(mx.mean(output)):.4f}, std: {float(mx.std(output)):.4f}")
    print("Expected: mean≈0, std≈1 if using running stats")

if __name__ == "__main__":
    main()
