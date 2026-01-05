#!/usr/bin/env python3
import subprocess
import re
import numpy as np
import os
import sys
from pathlib import Path

# Configuration
INPUT_AUDIO = "TestAudio/coder_audio_stock.wav"
MODEL_PATH = "/Users/mcruz/Library/Application Support/Replay/com.replay.Replay/models/Slim Shady/model.pth"
INDEX_PATH = "/Users/mcruz/Library/Application Support/Replay/com.replay.Replay/models/Slim Shady/model.index"
OUTPUT_DIR = "TestAudio"
NUM_RUNS = 3

def run_inference(backend):
    output_path = os.path.join(OUTPUT_DIR, f"output_{backend}_bench.wav")
    cmd = [
        "conda", "run", "-n", "rvc", "python", "rvc_cli.py", "infer",
        "--backend", backend,
        "--input_path", INPUT_AUDIO,
        "--output_path", output_path,
        "--pth_path", MODEL_PATH,
        "--index_path", INDEX_PATH
    ]
    
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    
    print(f"Running {backend} inference...")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if result.returncode != 0:
        print(f"Error running {backend} inference: {result.stderr}")
        return None
    
    # Parse time from output: "Conversion completed at '...' in 2.83 seconds."
    match = re.search(r"Conversion completed at .* in ([\d\.]+) seconds", result.stdout)
    if match:
        return float(match.group(1))
    else:
        print(f"Could not find timing in output for {backend}")
        print(result.stdout)
        return None

def main():
    if not os.path.exists(INPUT_AUDIO):
        print(f"Input audio not found: {INPUT_AUDIO}")
        return

    results = {"torch": [], "mlx": []}
    
    for i in range(NUM_RUNS):
        print(f"\n--- Run {i+1}/{NUM_RUNS} ---")
        
        # Torch
        t_torch = run_inference("torch")
        if t_torch:
            results["torch"].append(t_torch)
            print(f"Torch: {t_torch:.3f}s")
            
        # MLX
        t_mlx = run_inference("mlx")
        if t_mlx:
            results["mlx"].append(t_mlx)
            print(f"MLX: {t_mlx:.3f}s")

    print("\n" + "="*50)
    print(f"{'Backend':<10} | {'Median':<10} | {'Mean':<10} | {'Std Dev':<10}")
    print("-"*50)
    
    for backend in ["torch", "mlx"]:
        times = results[backend]
        if times:
            median = np.median(times)
            mean = np.mean(times)
            std = np.std(times)
            print(f"{backend:<10} | {median:<10.3f} | {mean:<10.3f} | {std:<10.3f}")
        else:
            print(f"{backend:<10} | ERROR      | ERROR      | ERROR")
            
    if results["torch"] and results["mlx"]:
        m_torch = np.median(results["torch"])
        m_mlx = np.median(results["mlx"])
        speedup = m_torch / m_mlx
        print("\n" + "="*50)
        print(f"MLX is {speedup:.2f}x faster (median) than PyTorch")
        print("="*50)

if __name__ == "__main__":
    main()
