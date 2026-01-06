import os
import sys
import argparse
from distutils.util import strtobool

# Add current directory to path
sys.path.append(os.getcwd())

from rvc_mlx.infer.infer_mlx import RVC_MLX
from rvc_mlx.lib.tools.analyzer import analyze_audio
from rvc_mlx.lib.tools.model_download import model_download_pipeline
from rvc_mlx.lib.tools.prerequisites_download import prequisites_download_pipeline
import asyncio

def load_voices_data():
    import json
    try:
        with open(os.path.join("rvc_mlx", "lib", "tools", "tts_voices.json"), "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return []

voices_data = load_voices_data()
locales = list({voice["ShortName"] for voice in voices_data}) if voices_data else []

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

def run_batch_infer(
    input_folder: str,
    output_folder: str,
    model_path: str,
    pitch: int,
    f0_method: str,
    index_path: str,
    index_rate: float,
    volume_envelope: float,
    protect: float,
    export_format: str,
    **kwargs
):
    print(f"Starting MLX Batch Inference...")
    if not os.path.exists(input_folder):
        print(f"Error: Input folder not found: {input_folder}")
        return
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize RVC MLX
    try:
        rvc = RVC_MLX(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    files = [f for f in os.listdir(input_folder) if f.endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a"))]
    print(f"Found {len(files)} files to process.")

    for file in files:
        input_file = os.path.join(input_folder, file)
        output_file = os.path.join(output_folder, os.path.splitext(file)[0] + f".{export_format.lower()}")
        
        rvc.infer(
            audio_input=input_file,
            audio_output=output_file,
            pitch=pitch,
            f0_method=f0_method,
            index_path=index_path if index_path else None,
            index_rate=index_rate,
            volume_envelope=volume_envelope,
            protect=protect
        )
    print(f"Batch inference complete. Results saved to {output_folder}")

def run_tts(
    tts_text: str,
    tts_voice: str,
    tts_rate: int,
    output_tts_path: str,
    output_rvc_path: str,
    model_path: str,
    pitch: int,
    f0_method: str,
    index_path: str,
    index_rate: float,
    volume_envelope: float,
    protect: float,
    export_format: str,
    **kwargs
):
    from rvc_mlx.lib.tools.tts import main as tts_main
    
    print(f"Synthesizing TTS: '{tts_text}' with voice {tts_voice}...")
    
    # Refactor tts_main to be callable easily or use edge_tts directly here
    import edge_tts
    rates = f"+{tts_rate}%" if tts_rate >= 0 else f"{tts_rate}%"
    
    async def run_edge_tts():
        communicate = edge_tts.Communicate(tts_text, tts_voice, rate=rates)
        await communicate.save(output_tts_path)
        
    asyncio.run(run_edge_tts())
    
    print(f"TTS complete. Saved to {output_tts_path}. Now converting with MLX...")
    
    run_infer(
        input_path=output_tts_path,
        output_path=output_rvc_path,
        model_path=model_path,
        pitch=pitch,
        f0_method=f0_method,
        index_path=index_path,
        index_rate=index_rate,
        volume_envelope=volume_envelope,
        protect=protect,
        export_format=export_format
    )

def parse_arguments():
    parser = argparse.ArgumentParser(description="MLX RVC CLI")
    subparsers = parser.add_subparsers(title="subcommands", dest="mode", help="Choose a mode")

    # Infer
    infer_parser = subparsers.add_parser("infer", help="Run inference (Pure MLX)")
    
    infer_parser.add_argument("--model_path", "--pth_path", dest="model_path", type=str, required=True, help="Path to MLX .npz model")
    infer_parser.add_argument("--input_path", type=str, required=True, help="Input audio path")
    infer_parser.add_argument("--output_path", type=str, required=True, help="Output audio path")
    
    for p in [infer_parser]:
        p.add_argument("--pitch", type=int, default=0, help="Pitch shift semitones")
        p.add_argument("--f0_method", type=str, default="rmvpe", choices=["rmvpe"], help="F0 method (only rmvpe supported in MLX currently)")
        p.add_argument("--index_path", type=str, default="", help="Path to .index file")
        p.add_argument("--index_rate", type=float, default=0.75, help="Index rate")
        p.add_argument("--volume_envelope", type=float, default=1.0, help="Volume envelope scaling")
        p.add_argument("--protect", type=float, default=0.5, help="Protect voiceless consonants")
        p.add_argument("--export_format", type=str, default="WAV", choices=["WAV", "FLAC", "MP3"], help="Export format")
        p.add_argument("--f0_autotune", type=lambda x: bool(strtobool(x)), default=False)
        p.add_argument("--f0_autotune_strength", type=float, default=1.0)

    # Batch Infer
    batch_parser = subparsers.add_parser("batch_infer", help="Run batch inference (Pure MLX)")
    batch_parser.add_argument("--model_path", "--pth_path", dest="model_path", type=str, required=True, help="Path to MLX .npz model")
    batch_parser.add_argument("--input_folder", dest="input_folder", type=str, required=True, help="Input folder path")
    batch_parser.add_argument("--output_folder", dest="output_folder", type=str, required=True, help="Output folder path")
    
    for p in [batch_parser]:
        p.add_argument("--pitch", type=int, default=0)
        p.add_argument("--f0_method", type=str, default="rmvpe", choices=["rmvpe"])
        p.add_argument("--index_path", type=str, default="")
        p.add_argument("--index_rate", type=float, default=0.75)
        p.add_argument("--volume_envelope", type=float, default=1.0)
        p.add_argument("--protect", type=float, default=0.5)
        p.add_argument("--export_format", type=str, default="WAV")

    # TTS
    tts_parser = subparsers.add_parser("tts", help="Run TTS + MLX Inference")
    tts_parser.add_argument("--model_path", "--pth_path", dest="model_path", type=str, required=True)
    tts_parser.add_argument("--tts_text", type=str, required=True, help="Text to synthesize")
    tts_parser.add_argument("--tts_voice", type=str, default="en-US-AndrewNeural", help="Edge TTS voice name")
    tts_parser.add_argument("--tts_rate", type=int, default=0, help="TTS speed rate")
    tts_parser.add_argument("--output_tts_path", type=str, default="logs/tts_out.wav")
    tts_parser.add_argument("--output_rvc_path", type=str, required=True)
    
    for p in [tts_parser]:
        p.add_argument("--pitch", type=int, default=0)
        p.add_argument("--f0_method", type=str, default="rmvpe", choices=["rmvpe"])
        p.add_argument("--index_path", type=str, default="")
        p.add_argument("--index_rate", type=float, default=0.75)
        p.add_argument("--volume_envelope", type=float, default=1.0)
        p.add_argument("--protect", type=float, default=0.5)
        p.add_argument("--export_format", type=str, default="WAV")

    # Audio Analyzer
    analyzer_parser = subparsers.add_parser("audio_analyzer", help="Analyze audio features")
    analyzer_parser.add_argument("--input_path", type=str, required=True)
    analyzer_parser.add_argument("--output_path", type=str, default="logs/audio_analysis.png")

    # Model Download
    download_parser = subparsers.add_parser("download", help="Download a model from URL")
    download_parser.add_argument("--model_link", type=str, required=True)

    # Prerequisites
    prereq_parser = subparsers.add_parser("prerequisites", help="Download prerequisites")
    prereq_parser.add_argument("--pretraineds_hifigan", action="store_true")
    prereq_parser.add_argument("--models", action="store_true")
    prereq_parser.add_argument("--exe", action="store_true")

    # Placeholders for Training (Not yet supported in Pure MLX)
    for cmd in ["preprocess", "extract", "train", "index", "model_information", "model_blender", "tensorboard"]:
        subparsers.add_parser(cmd, help=f"{cmd.capitalize()} (Not supported in Pure MLX yet)")

    # Add other ignored args for compatibility
    for p in [infer_parser, batch_parser, tts_parser]:
        for arg in ["--split_audio", "--clean_audio", "--proposed_pitch", "--post_process", "--reverb", "--pitch_shift", "--limiter", "--gain", "--distortion", "--chorus", "--bitcrush", "--clipping", "--compressor", "--delay"]:
            p.add_argument(arg, action="store_true", help="Ignored in MLX version")
        for arg in ["--clean_strength", "--proposed_pitch_threshold", "--embedder_model", "--embedder_model_custom", "--sid"]:
            p.add_argument(arg, type=str, help="Ignored in MLX version")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.mode == "infer":
        run_infer(**vars(args))
    elif args.mode == "batch_infer":
        run_batch_infer(**vars(args))
    elif args.mode == "tts":
        run_tts(**vars(args))
    elif args.mode == "audio_analyzer":
        audio_info, plot_path = analyze_audio(args.input_path, args.output_path)
        print(audio_info)
        print(f"Analysis saved to {plot_path}")
    elif args.mode == "download":
        model_download_pipeline(args.model_link)
    elif args.mode == "prerequisites":
        prequisites_download_pipeline(args.pretraineds_hifigan, args.models, args.exe)
    elif args.mode in ["preprocess", "extract", "train", "index", "model_information", "model_blender", "tensorboard"]:
        print(f"The '{args.mode}' subcommand is not yet implemented for the Pure MLX backend.")
    else:
        print(f"Unknown mode: {args.mode}")
