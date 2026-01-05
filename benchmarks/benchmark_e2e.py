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
        "/usr/bin/time", "-l",
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
        return None, None
    
    # Parse time from output: "Conversion completed at '...' in 2.83 seconds."
    time_match = re.search(r"Conversion completed at .* in ([\d\.]+) seconds", result.stdout)
    
    # Parse memory from stderr: "1540096  maximum resident set size"
    mem_match = re.search(r"(\d+)\s+maximum resident set size", result.stderr)
    
    t = float(time_match.group(1)) if time_match else None
    m = int(mem_match.group(1)) / (1024 * 1024) if mem_match else None # Convert to MB
    
    if t is None:
        print(f"Could not find timing in output for {backend}")
        print(result.stdout)
        
    return t, m

def main():
    if not os.path.exists(INPUT_AUDIO):
        print(f"Input audio not found: {INPUT_AUDIO}")
        return

    results = {"torch": {"time": [], "mem": []}, "mlx": {"time": [], "mem": []}}
    
    for i in range(NUM_RUNS):
        print(f"\n--- Run {i+1}/{NUM_RUNS} ---")
        
        # Torch
        t_torch, m_torch = run_inference("torch")
        if t_torch:
            results["torch"]["time"].append(t_torch)
            results["torch"]["mem"].append(m_torch)
            print(f"Torch: {t_torch:.3f}s, {m_torch:.1f} MB")
            
        # MLX
        t_mlx, m_mlx = run_inference("mlx")
        if t_mlx:
            results["mlx"]["time"].append(t_mlx)
            results["mlx"]["mem"].append(m_mlx)
            print(f"MLX: {t_mlx:.3f}s, {m_mlx:.1f} MB")

    print("\n" + "="*70)
    print(f"{'Backend':<10} | {'Median Time':<12} | {'Median Mem':<12} | {'Max Mem':<10}")
    print("-" * 70)
    
    for backend in ["torch", "mlx"]:
        times = results[backend]["time"]
        mems = results[backend]["mem"]
        if times:
            t_med = np.median(times)
            m_med = np.median(mems)
            m_max = np.max(mems)
            print(f"{backend:<10} | {t_med:<12.3f} | {m_med:<12.1f} MB | {m_max:<10.1f} MB")
        else:
            print(f"{backend:<10} | ERROR        | ERROR")
            
    if results["torch"] and results["mlx"]:
        m_torch = np.median(results["torch"])
        m_mlx = np.median(results["mlx"])
        speedup = m_torch / m_mlx
        print("\n" + "="*50)
        print(f"MLX is {speedup:.2f}x faster (median) than PyTorch")
        print("="*50)

if __name__ == "__main__":
    main()
