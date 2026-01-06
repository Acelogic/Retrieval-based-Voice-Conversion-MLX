
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import json
import os
import sys

# Add path
sys.path.append(os.getcwd())

from rvc_mlx.lib.mlx.synthesizers import Synthesizer

def debug_weights(model_path):
    print(f"Loading weights from {model_path}...")
    weights = mx.load(model_path)
    
    # Check stats of some weights
    print("Checking specific weight stats in file:")
    keys_to_check = [
        "dec.conv_pre.weight",
        "dec.up_0.weight", 
        "dec.resblock_0.c1_0.weight",
        "enc_p.encoder.attn_0.conv_q.weight"
    ]
    
    for k in keys_to_check:
        if k in weights:
            w = weights[k]
            print(f"  {k}: shape={w.shape}, mean={mx.mean(w).item():.6f}, std={mx.std(w).item():.6f}")
        else:
            print(f"  {k}: NOT FOUND")
            
    # Load Config
    config_path = os.path.splitext(model_path)[0] + ".json"
    with open(config_path, 'r') as f:
        conf = json.load(f)
    
    # Initialize Synthesizer
    print("\nInitializing Synthesizer...")
    # Extract params from config (simplified logic from infer_mlx)
    # Assume config is dictionary as saved by convert_rvc_model
    if isinstance(conf, dict) and "model" in conf:
        m_conf = conf["model"]
    else:
        # Fallback for list or different structure, but we saved it as dict with "model" usually?
        # Actually convert_rvc_model saved whatever was in the checkpoint.
        # If the checkpoint had 'config' key which was a list (common in RVC), it's a list.
        m_conf = {}
        pass
        
    # We'll use default args and try to rely on what infer_mlx does, 
    # but to be safe let's just use the classes and load weights manually and check if they change.
    
    # Instantiate with minimal args just to get the structure
    # We need to match the actual model config though or shapes will mismatch.
    # Let's import RVC_MLX from infer_mlx to reuse its logic
    from rvc_mlx.infer.infer_mlx import RVC_MLX
    
    rvc = RVC_MLX(model_path)
    
    # Check if weights loaded into model match weights in file
    print("\nVerifying loaded model weights...")
    
    # Check dec.conv_pre.weight
    model_w = rvc.net_g.dec.conv_pre.weight
    file_w = weights["dec.conv_pre.weight"]
    
    # Check difference
    diff = mx.abs(model_w - file_w).sum()
    print(f"  dec.conv_pre.weight diff: {diff.item()}")
    
    # Check up_0
    model_w = rvc.net_g.dec.up_0.weight
    # In file it was remapped to dec.up_0.weight by convert_rvc_model
    # The Model expects dec.up_0.weight?
    # RVC_MLX.remap_keys remaps "dec.ups.0.weight" -> "dec.up_0.weight"
    # But convert_rvc_model ALREADY remapped it to "dec.up_0.weight".
    # So RVC_MLX.remap_keys might be re-remapping or missing it?
    
    if "dec.up_0.weight" in weights:
         # infer_mlx remap_keys looks for "dec.ups.0.weight".
         # It does NOT handle "dec.up_0.weight" if already remapped?
         pass
         
    # Let's check what infer_mlx does.
    # infer_mlx.remap_keys:
    # if new_key.startswith("dec.ups."): -> remaps
    # It assumes keys consist of "dec.ups...".
    # If keys are ALREADY "dec.up_0...", it doesn't match "dec.ups.".
    # So it keeps "dec.up_0...".
    # net_g.load_weights expects... what?
    # net_g.dec has attribute up_0. 
    # load_weights matching logic uses parameters().
    # parameters of net_g.dec.up_0 will be named "dec.up_0.weight" (recursively name fixed?)
    # MLX load_weights matches based on key matching layer tree.
    # "dec" -> "up_0" -> "weight".
    # So "dec.up_0.weight" is the correct key for MLX to load!
    
    # So if file has "dec.up_0.weight", it should load fine.
    
    model_w = rvc.net_g.dec.up_0.weight
    if "dec.up_0.weight" in weights:
        file_w = weights["dec.up_0.weight"]
        diff = mx.abs(model_w - file_w).sum()
        print(f"  dec.up_0.weight diff: {diff.item()}")
    else:
        print("  dec.up_0.weight NOT IN FILE")

    # Run a dummy input forward pass on Generator
    print("\nTesting Generator Forward Pass...")
    B, L, D = 1, 100, 256 # internal dims
    # Generator input x is (B, L, inter_channels) = 192 typically
    inter_channels = rvc.net_g.dec.conv_pre.weight.shape[2] # (Out, K, In) -> In is shape[2]
    print(f"  Detected inter_channels: {inter_channels}")
    
    x = mx.random.normal((1, 100, inter_channels))
    f0 = mx.random.normal((1, 100)) * 440
    # Generator call: x, f0, g=None
    
    try:
        audio = rvc.net_g.dec(x, f0)
        print(f"  Generator Output shape: {audio.shape}")
        print(f"  Generator Output Stats: min={mx.min(audio).item()}, max={mx.max(audio).item()}, mean={mx.mean(audio).item()}")
        
        if mx.abs(article).max().item() < 1e-6:
             print("  ⚠️ OUTPUT IS SILENT/ZERO!")
        else:
             print("  ✅ Output has signal.")
    except Exception as e:
        print(f"  ❌ Generator Error: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    args = parser.parse_args()
    debug_weights(args.model_path)
