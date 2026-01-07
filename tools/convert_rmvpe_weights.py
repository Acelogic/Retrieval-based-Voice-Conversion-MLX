#!/usr/bin/env python3
"""Convert Python RMVPE NPZ weights to Swift-compatible safetensors.

The Swift PyTorchGRU uses exactly the same weight names as Python:
- weight_ih, weight_hh, bias_ih, bias_hh
"""
import numpy as np
import mlx.core as mx

def convert_rmvpe_weights():
    # Load Python NPZ
    npz_path = 'rvc_mlx/models/predictors/rmvpe_mlx.npz'
    npz = dict(np.load(npz_path))
    print(f'Loaded {len(npz)} weights from NPZ')
    
    converted = {}
    
    for key, value in npz.items():
        new_key = key
        new_value = mx.array(value)
        
        # Remove 'fc.' prefix from GRU and linear keys
        if key.startswith('fc.'):
            new_key = key[3:]  # Remove 'fc.'
        
        # Convert Python snake_case property names to Swift camelCase
        # bigru.forward_grus.0 -> bigru.forwardGRUs.0
        # bigru.backward_grus.0 -> bigru.backwardGRUs.0
        new_key = new_key.replace('forward_grus', 'forwardGRUs')
        new_key = new_key.replace('backward_grus', 'backwardGRUs')
        
        # PyTorchGRU uses the same weight names as Python:
        # weight_ih, weight_hh, bias_ih, bias_hh
        # No conversion needed for GRU weights - they stay as-is!
        
        # For BatchNorm, rename gamma/beta to weight/bias if present
        if '.gamma' in new_key:
            new_key = new_key.replace('.gamma', '.weight')
        elif '.beta' in new_key:
            new_key = new_key.replace('.beta', '.bias')
        
        converted[new_key] = new_value
        
        if new_key != key:
            print(f'  {key} {value.shape} -> {new_key} {new_value.shape}')
    
    # Save as safetensors
    output_path = 'Demos/iOS/RVCNative/RVCNativePackage/Sources/RVCNativeFeature/Assets/rmvpe.safetensors'
    mx.save_safetensors(output_path, converted)
    print(f'\nSaved {len(converted)} weights to {output_path}')
    
    # Verify
    reloaded = dict(mx.load(output_path))
    print(f'Verification: {len(reloaded)} keys loaded')
    
    # Print GRU weights to confirm names
    print('\nGRU weight keys:')
    for k in sorted(reloaded.keys()):
        if 'GRU' in k:
            print(f'  {k}: {reloaded[k].shape}')

if __name__ == "__main__":
    convert_rmvpe_weights()
