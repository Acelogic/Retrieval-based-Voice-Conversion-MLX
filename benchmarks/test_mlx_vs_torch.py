import os
import sys
import numpy as np
import soundfile as sf
import time
from pathlib import Path

# Set environment variable before imports
os.environ["OMP_NUM_THREADS"] = "1"

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rvc.infer.infer import VoiceConverter

def test_backends_comparison():
    print("=" * 80)
    print("Comparing Torch and MLX Backends")
    print("=" * 80)
    
    model_dir = "/Users/mcruz/Library/Application Support/Replay/com.replay.Replay/models/Slim Shady"
    model_file = os.path.join(model_dir, "model.pth")
    index_file = os.path.join(model_dir, "model.index")
    input_audio = "TestAudio/coder_audio_stock.wav"
    
    if not os.path.exists(input_audio):
        print(f"❌ Input audio not found at {input_audio}")
        return
        
    vc = VoiceConverter()
    
    results = {}
    
    for backend in ["torch", "mlx"]:
        print(f"\nRunning {backend.upper()} backend...")
        output_file = f"TestAudio/output_{backend}_test.wav"
        
        start_time = time.time()
        vc.convert_audio(
            audio_input_path=input_audio,
            audio_output_path=output_file,
            model_path=model_file,
            index_path=index_file,
            pitch=0,
            f0_method="rmvpe",
            index_rate=0.5,
            volume_envelope=1.0,
            protect=0.33,
            hop_length=128,
            f0_autotune=False,
            split_audio=False,
            embedder_model="contentvec",
            backend=backend
        )
        elapsed = time.time() - start_time
        print(f"✅ {backend.upper()} finished in {elapsed:.2f}s")
        
        data, sr = sf.read(output_file)
        results[backend] = {
            "data": data,
            "sr": sr,
            "elapsed": elapsed
        }
        
    # Comparison
    data_torch = results["torch"]["data"]
    data_mlx = results["mlx"]["data"]
    
    min_len = min(len(data_torch), len(data_mlx))
    data_torch = data_torch[:min_len]
    data_mlx = data_mlx[:min_len]
    
    mse = np.mean((data_torch - data_mlx)**2)
    correlation = np.corrcoef(data_torch.flatten(), data_mlx.flatten())[0, 1]
    
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(f"MSE: {mse:.10f}")
    print(f"Correlation: {correlation:.6f}")
    print(f"Speedup: {results['torch']['elapsed'] / results['mlx']['elapsed']:.2x}")
    print("=" * 80)
    
    if correlation > 0.7:
        print("✅ Python MLX implementation produces similar results to Torch.")
    else:
        print("❌ Python MLX results are significantly different from Torch!")

if __name__ == "__main__":
    test_backends_comparison()
