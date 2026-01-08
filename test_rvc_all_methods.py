#!/usr/bin/env python3
"""Test full RVC inference with all F0 extraction methods."""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/mcruz/Developer/Retrieval-based-Voice-Conversion-MLX')

from rvc_mlx.infer.infer_mlx import RVC_MLX

# Parameters
MODEL_PATH = "weights/Drake.npz"
AUDIO_INPUT = "test-audio/input_16k.wav"
OUTPUT_DIR = "test_results/f0_methods_rvc"
METHODS = ["rmvpe", "dio", "pm", "harvest", "fcpe"]  # Skip crepe (needs weights)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading RVC model...")
    start = time.time()
    rvc = RVC_MLX(MODEL_PATH)
    load_time = time.time() - start
    print(f"  Model loaded in {load_time:.2f}s")
    
    print("\n" + "="*60)
    print("Running voice conversion with each F0 method")
    print("="*60)
    
    results = {}
    
    for method in METHODS:
        output_file = f"{OUTPUT_DIR}/output_{method}.wav"
        print(f"\n[{method.upper()}]")
        
        try:
            start = time.time()
            rvc.infer(
                audio_input=AUDIO_INPUT,
                audio_output=output_file,
                pitch=0,
                f0_method=method,
                index_path=""
            )
            elapsed = time.time() - start
            
            # Get file size
            file_size = os.path.getsize(output_file)
            
            results[method] = {
                'time': elapsed,
                'output': output_file,
                'size': file_size
            }
            
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Output: {output_file}")
            print(f"  Size: {file_size/1024:.1f} KB")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[method] = None
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    print(f"\n{'Method':<12} {'Time (s)':<12} {'Output File'}")
    print("-"*60)
    for method in METHODS:
        if results.get(method):
            r = results[method]
            print(f"{method:<12} {r['time']:<12.2f} {r['output']}")
        else:
            print(f"{method:<12} {'FAILED':<12}")
    
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("\nListen to the files to compare quality:")
    for method in METHODS:
        if results.get(method):
            print(f"  open {results[method]['output']}")

if __name__ == "__main__":
    main()
