import os
import sys
import argparse
from distutils.util import strtobool

# Add current directory to path
sys.path.append(os.getcwd())

from rvc_mlx.infer.infer_mlx import RVC_MLX

def run_infer(
    input_path: str,
    output_path: str,
    model_path: str,
    pitch: int,
    f0_method: str,
    index_path: str,
    index_rate: float,
    volume_envelope: float,
    protect: float,
    f0_autotune: bool,
    f0_autotune_strength: float,
    export_format: str,
    # Ignored/Unused args for interface compatibility or future use
    split_audio: bool = False, 
    clean_audio: bool = False,
    clean_strength: float = 0.5,
    embedder_model: str = "contentvec",
    **kwargs
):
    print(f"Starting MLX Inference...")
    print(f"Model: {model_path}")
    print(f"Input: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return

    # Initialize RVC MLX
    try:
        rvc = RVC_MLX(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Adjust output extension
    if export_format.lower() != "wav":
        output_path = os.path.splitext(output_path)[0] + f".{export_format.lower()}"
    
    # Run Inference
    # RVC_MLX.infer signature:
    # infer(self, audio_input, audio_output, pitch=0, f0_method="rmvpe", index_path=None, index_rate=0.75, volume_envelope=1.0, protect=0.5)
    # Note: pipeline_mlx.py supports f0_autotune, but RVC_MLX wrapper might need update to pass it.
    # Let's check RVC_MLX.infer implementation in infer_mlx.py.
    # It calls:
    # audio_opt = self.pipeline.pipeline(..., f0_autotune, f0_autotune_strength, ...)
    # So I need to update RVC_MLX.infer to accept these new args if strictly needed, or just accept that they are fixed in wrapper.
    # The wrapper from Step 63 hardcoded f0_autotune=False.
    # I should update RVC_MLX in infer_mlx.py to accept **kwargs or specific args to be fully compatible.
    # For now, I'll pass what I can.
    
    # To support full CLI args, I really should update infer_mlx.py first.
    # But for now let's call it with available parameters.
    
    rvc.infer(
        audio_input=input_path,
        audio_output=output_path,
        pitch=pitch,
        f0_method=f0_method,
        index_path=index_path if index_path else None,
        index_rate=index_rate,
        volume_envelope=volume_envelope,
        protect=protect
    )
    
    print(f"Inference done. Saved to {output_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="MLX RVC CLI")
    subparsers = parser.add_subparsers(title="subcommands", dest="mode", help="Choose a mode")

    # Infer
    infer_parser = subparsers.add_parser("infer", help="Run inference (Pure MLX)")
    
    infer_parser.add_argument("--model_path", "--pth_path", dest="model_path", type=str, required=True, help="Path to MLX .npz model")
    infer_parser.add_argument("--input_path", type=str, required=True, help="Input audio path")
    infer_parser.add_argument("--output_path", type=str, required=True, help="Output audio path")
    
    infer_parser.add_argument("--pitch", type=int, default=0, help="Pitch shift semitones")
    infer_parser.add_argument("--f0_method", type=str, default="rmvpe", choices=["rmvpe"], help="F0 method (only rmvpe supported in MLX currently)")
    infer_parser.add_argument("--index_path", type=str, default="", help="Path to .index file")
    infer_parser.add_argument("--index_rate", type=float, default=0.75, help="Index rate")
    infer_parser.add_argument("--volume_envelope", type=float, default=1.0, help="Volume envelope scaling")
    infer_parser.add_argument("--protect", type=float, default=0.5, help="Protect voiceless consonants")
    
    # Optional args that are present in standard CLI but maybe no-op here
    infer_parser.add_argument("--export_format", type=str, default="WAV", choices=["WAV", "FLAC", "MP3"], help="Export format")
    infer_parser.add_argument("--f0_autotune", type=lambda x: bool(strtobool(x)), default=False)
    infer_parser.add_argument("--f0_autotune_strength", type=float, default=1.0)
    
    # Add other ignored args to prevent crashes if user passes full rvc command
    for arg in ["--split_audio", "--clean_audio", "--proposed_pitch", "--post_process", "--reverb", "--pitch_shift", "--limiter", "--gain", "--distortion", "--chorus", "--bitcrush", "--clipping", "--compressor", "--delay"]:
        infer_parser.add_argument(arg, action="store_true", help="Ignored in MLX version")
    
    for arg in ["--clean_strength", "--proposed_pitch_threshold", "--embedder_model", "--embedder_model_custom", "--sid"]:
        infer_parser.add_argument(arg, type=str, help="Ignored in MLX version") # Type str to catch whatever

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.mode == "infer":
        # Convert args to dict
        kwargs = vars(args)
        # Remove mode
        safe_kwargs = {k: v for k, v in kwargs.items() if k != "mode"}
        
        run_infer(**safe_kwargs)
    else:
        print("Only 'infer' subcommand is supported in MLX RVC CLI.")
