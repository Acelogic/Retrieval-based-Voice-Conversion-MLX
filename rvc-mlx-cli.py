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
        protect=protect,
        f0_autotune=f0_autotune,
        f0_autotune_strength=f0_autotune_strength
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
            protect=protect,
            f0_autotune=kwargs.get("f0_autotune", False),
            f0_autotune_strength=kwargs.get("f0_autotune_strength", 1.0)
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
        f0_autotune=kwargs.get("f0_autotune", False),
        f0_autotune_strength=kwargs.get("f0_autotune_strength", 1.0),
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
    prereq_parser.add_argument("--pretraineds_hifigan", action="store_true", help="Download base HiFi-GAN pretrains")
    prereq_parser.add_argument("--models", action="store_true", help="Download RMVPE, FCPE, ContentVec models")
    prereq_parser.add_argument("--exe", action="store_true", help="Download executables (Windows only)")
    prereq_parser.add_argument("--titan", action="store_true", help="Download TITAN community pretrain (recommended for training)")

    # Convert
    convert_parser = subparsers.add_parser("convert", help="Convert PyTorch model to MLX")
    convert_parser.add_argument("--model_path", "-i", dest="model_path", type=str, required=True, help="Path to input PyTorch model (.pth)")
    convert_parser.add_argument("--output_path", "-o", dest="output_path", type=str, required=True, help="Path to output MLX model (.npz)")

    # Preprocess - Audio slicing and normalization
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess audio for training (slice, normalize)")
    preprocess_parser.add_argument("--model_name", type=str, required=True, help="Name for the training experiment")
    preprocess_parser.add_argument("--input_folder", type=str, required=True, help="Folder containing audio files")
    preprocess_parser.add_argument("--sample_rate", type=int, default=40000, choices=[32000, 40000, 48000], help="Target sample rate")
    preprocess_parser.add_argument("--cut_mode", type=str, default="Automatic", choices=["Automatic", "Simple", "Skip"], help="Audio cutting mode")
    preprocess_parser.add_argument("--cpu_cores", type=int, default=4, help="Number of CPU cores for parallel processing")

    # Extract - Feature extraction (HuBERT + F0)
    extract_parser = subparsers.add_parser("extract", help="Extract features (HuBERT embeddings + F0 pitch)")
    extract_parser.add_argument("--model_name", type=str, required=True, help="Name of the training experiment")
    extract_parser.add_argument("--f0_method", type=str, default="rmvpe", choices=["rmvpe"], help="F0 extraction method")
    extract_parser.add_argument("--embedder_model", type=str, default="contentvec", help="Embedder model to use")
    extract_parser.add_argument("--cpu_cores", type=int, default=4, help="Number of CPU cores")

    # Train - Main training command
    train_parser = subparsers.add_parser("train", help="Train/fine-tune RVC model")
    train_parser.add_argument("--model_name", type=str, required=True, help="Name of the training experiment")
    train_parser.add_argument("--sample_rate", type=int, default=40000, choices=[32000, 40000, 48000], help="Sample rate")
    train_parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    train_parser.add_argument("--total_epoch", type=int, default=200, help="Total training epochs")
    train_parser.add_argument("--save_every_epoch", type=int, default=10, help="Save checkpoint every N epochs")
    train_parser.add_argument("--pretrain_g", type=str, default="", help="Path to pretrained generator (.pth or .npz)")
    train_parser.add_argument("--pretrain_d", type=str, default="", help="Path to pretrained discriminator (.pth or .npz)")
    train_parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
    train_parser.add_argument("--use_aim", type=lambda x: bool(strtobool(x)), default=True, help="Use Aim for experiment tracking")
    train_parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID (ignored on MLX, uses Metal)")
    train_parser.add_argument("--overtraining_detector", type=lambda x: bool(strtobool(x)), default=True, help="Enable overtraining detection (recommended)")
    train_parser.add_argument("--overtraining_patience", type=int, default=10, help="Epochs without improvement before stopping")
    train_parser.add_argument("--auto_batch_size", type=lambda x: bool(strtobool(x)), default=False, help="Auto-select batch size based on dataset duration")
    train_parser.add_argument("--pretrain", type=str, default="base", choices=["base", "titan"], help="Pretrained model to use")
    train_parser.add_argument("--vocoder", type=str, default="hifigan", choices=["hifigan", "refinegan"], help="Vocoder type (refinegan has higher fidelity)")

    # Placeholders for not-yet-implemented commands
    for cmd in ["index", "model_information", "model_blender", "tensorboard"]:
        subparsers.add_parser(cmd, help=f"{cmd.capitalize()} (Not yet supported in Pure MLX)")

    # Add other ignored args for compatibility
    for p in [infer_parser, batch_parser, tts_parser]:
        for arg in ["--split_audio", "--clean_audio", "--proposed_pitch", "--post_process", "--reverb", "--pitch_shift", "--limiter", "--gain", "--distortion", "--chorus", "--bitcrush", "--clipping", "--compressor", "--delay"]:
            p.add_argument(arg, action="store_true", help="Ignored in MLX version")
        for arg in ["--clean_strength", "--proposed_pitch_threshold", "--embedder_model", "--embedder_model_custom", "--sid"]:
            p.add_argument(arg, type=str, help="Ignored in MLX version")

    return parser.parse_args()

def run_convert(model_path, output_path, **kwargs):
    print(f"Converting model from {model_path} to {output_path}...")
    try:
        from tools import convert_rvc_model
        convert_rvc_model.convert_weights(model_path, output_path)
    except ImportError:
        print("Error: Could not import tools.convert_rvc_model. Make sure you are in the project root.")
    except Exception as e:
        print(f"Error converting model: {e}")


def run_preprocess(model_name, input_folder, sample_rate=40000, cut_mode="Automatic", cpu_cores=4, **kwargs):
    """Preprocess audio files for training."""
    print(f"Starting MLX preprocessing for '{model_name}'...")
    print(f"Input folder: {input_folder}")
    print(f"Sample rate: {sample_rate}")
    print(f"Cut mode: {cut_mode}")

    try:
        from rvc_mlx.preprocess.audio_slicer import preprocess_audio

        exp_dir = os.path.join("logs", model_name)
        os.makedirs(exp_dir, exist_ok=True)

        preprocess_audio(
            input_dir=input_folder,
            exp_dir=exp_dir,
            sr=sample_rate,
            cut_mode=cut_mode,
            num_workers=cpu_cores,
        )
        print(f"Preprocessing complete. Output saved to {exp_dir}")
    except ImportError as e:
        print(f"Error: Could not import preprocessing module: {e}")
    except Exception as e:
        print(f"Error during preprocessing: {e}")


def run_extract(model_name, f0_method="rmvpe", embedder_model="contentvec", cpu_cores=4, **kwargs):
    """Extract features (F0 and embeddings) from preprocessed audio."""
    print(f"Starting MLX feature extraction for '{model_name}'...")
    print(f"F0 method: {f0_method}")
    print(f"Embedder: {embedder_model}")

    try:
        from rvc_mlx.preprocess.feature_extractor import FeatureExtractor

        exp_dir = os.path.join("logs", model_name)
        if not os.path.exists(exp_dir):
            print(f"Error: Experiment directory not found: {exp_dir}")
            print("Run 'preprocess' first to create the experiment directory.")
            return

        extractor = FeatureExtractor(
            exp_dir=exp_dir,
            f0_method=f0_method,
        )

        count = extractor.extract_all()
        print(f"Feature extraction complete. Processed {count} files.")

        # Build training filelist
        print("Building training filelist...")
        from rvc_mlx.preprocess.dataset_builder import build_dataset
        filelist_path = build_dataset(exp_dir, val_ratio=0.1)
        print(f"Created filelist at {filelist_path}")

    except ImportError as e:
        print(f"Error: Could not import feature extractor: {e}")
    except Exception as e:
        print(f"Error during feature extraction: {e}")


def run_train(
    model_name,
    sample_rate=40000,
    batch_size=8,
    total_epoch=200,
    save_every_epoch=10,
    pretrain_g="",
    pretrain_d="",
    learning_rate=1e-4,
    use_aim=True,
    overtraining_detector=True,
    overtraining_patience=10,
    auto_batch_size=False,
    pretrain="base",
    **kwargs
):
    """Train/fine-tune RVC model using MLX."""
    print(f"Starting MLX training for '{model_name}'...")
    print(f"Sample rate: {sample_rate}")

    exp_dir = os.path.join("logs", model_name)

    # Auto batch size based on dataset duration (AI Hub recommendation)
    if auto_batch_size:
        try:
            from rvc_mlx.train.overtraining_detector import get_smart_batch_size
            sliced_dir = os.path.join(exp_dir, "sliced")
            if os.path.exists(sliced_dir):
                batch_size = get_smart_batch_size(sliced_dir, sample_rate, verbose=True)
            else:
                print("Warning: Sliced audio directory not found. Using default batch size.")
        except ImportError:
            print("Warning: Could not import batch size detector. Using default.")

    print(f"Batch size: {batch_size}")
    print(f"Total epochs: {total_epoch}")
    print(f"Learning rate: {learning_rate}")
    print(f"Overtraining detection: {'enabled' if overtraining_detector else 'disabled'}")

    # Resolve pretrain paths based on pretrain option
    if pretrain == "titan" and not pretrain_g:
        pretrain_g = "rvc/models/pretraineds/titan/f0G40k.pth"
        pretrain_d = "rvc/models/pretraineds/titan/f0D40k.pth"
        print(f"Using TITAN pretrain")
    elif pretrain == "base" and not pretrain_g:
        pretrain_g = f"rvc/models/pretraineds/hifi-gan/f0G{sample_rate // 1000}k.pth"
        pretrain_d = f"rvc/models/pretraineds/hifi-gan/f0D{sample_rate // 1000}k.pth"
        print(f"Using base RVC v2 pretrain")

    try:
        from rvc_mlx.train.trainer import RVCTrainer, TrainingConfig
        from rvc_mlx.train.data_loader import DataLoader, RVCDataset, RVCCollator
        from rvc_mlx.monitoring.aim_tracker import create_tracker

        exp_dir = os.path.join("logs", model_name)
        if not os.path.exists(exp_dir):
            print(f"Error: Experiment directory not found: {exp_dir}")
            print("Run 'preprocess' and 'extract' first.")
            return

        # Check for filelist
        filelist_path = os.path.join(exp_dir, "filelist.txt")
        if not os.path.exists(filelist_path):
            print(f"Error: Filelist not found: {filelist_path}")
            print("Run 'extract' first to generate the filelist.")
            return

        # Determine if fine-tuning (pretrained weights provided)
        is_finetuning = bool(pretrain_g and os.path.exists(pretrain_g))

        # Create training config
        config = TrainingConfig(
            sample_rate=sample_rate,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=total_epoch,
            save_every_epoch=save_every_epoch,
            checkpoint_dir=os.path.join(exp_dir, "checkpoints"),
            enable_overtraining_detection=overtraining_detector,
            overtraining_patience=overtraining_patience,
            is_finetuning=is_finetuning,
        )

        # Create dataset and dataloader
        dataset = RVCDataset(filelist_path, sample_rate=sample_rate)
        collator = RVCCollator()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)

        # Create tracker
        tracker = None
        if use_aim:
            tracker = create_tracker(experiment_name=model_name)

        # Create models
        from rvc_mlx.lib.mlx.synthesizers import Synthesizer
        from rvc_mlx.train.discriminators import MultiPeriodDiscriminator

        print("Creating generator model...")
        net_g = Synthesizer(
            spec_channels=1025,
            segment_size=config.segment_size,
            inter_channels=192,
            hidden_channels=192,
            filter_channels=768,
            n_heads=2,
            n_layers=6,
            kernel_size=3,
            p_dropout=0.0,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[10, 10, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[16, 16, 4, 4],
            spk_embed_dim=256,
            gin_channels=256,
            sr=sample_rate,
            use_f0=True,
        )

        print("Creating discriminator model...")
        net_d = MultiPeriodDiscriminator()

        # Load pretrained weights if available
        import mlx.core as mx
        from rvc_mlx.infer.infer_mlx import remap_keys  # Use inference module's key remapping
        if pretrain_g and os.path.exists(pretrain_g):
            print(f"Loading generator pretrain from {pretrain_g}...")
            if pretrain_g.endswith('.npz'):
                # MLX format - load and remap keys
                g_weights = dict(mx.load(pretrain_g))
                g_weights = remap_keys(g_weights)  # Remap to model's naming convention
                net_g.load_weights(list(g_weights.items()), strict=False)
                print(f"  Loaded {len(g_weights)} generator weights")
            elif pretrain_g.endswith('.pth'):
                # PyTorch format - need to convert
                import torch
                ckpt = torch.load(pretrain_g, map_location="cpu", weights_only=True)
                state_dict = ckpt["model"] if "model" in ckpt else ckpt
                g_weights = {}
                for k, v in state_dict.items():
                    arr = v.numpy()
                    # Transpose Conv weights for MLX
                    if "weight" in k and len(arr.shape) == 3:
                        arr = arr.transpose(0, 2, 1)
                    g_weights[k] = mx.array(arr)
                g_weights = remap_keys(g_weights)
                net_g.load_weights(list(g_weights.items()), strict=False)
                print(f"  Converted and loaded {len(g_weights)} generator weights")
        if pretrain_d and os.path.exists(pretrain_d):
            print(f"Loading discriminator pretrain from {pretrain_d}...")
            if pretrain_d.endswith('.npz'):
                # MLX format - load directly (discriminator key names match)
                d_weights = dict(mx.load(pretrain_d))
                net_d.load_weights(list(d_weights.items()), strict=False)
                print(f"  Loaded {len(d_weights)} discriminator weights")
            elif pretrain_d.endswith('.pth'):
                # PyTorch format - simple conversion (transpose conv weights)
                import torch
                ckpt = torch.load(pretrain_d, map_location="cpu", weights_only=True)
                state_dict = ckpt["model"] if "model" in ckpt else ckpt
                d_weights = {}
                for k, v in state_dict.items():
                    arr = v.numpy()
                    # Transpose Conv weights
                    if "weight" in k and len(arr.shape) == 3:
                        arr = arr.transpose(0, 2, 1)  # Conv1d: (out, in, k) -> (out, k, in)
                    elif "weight" in k and len(arr.shape) == 4:
                        arr = arr.transpose(0, 2, 3, 1)  # Conv2d: (out, in, h, w) -> (out, h, w, in)
                    d_weights[k] = mx.array(arr)
                net_d.load_weights(list(d_weights.items()), strict=False)
                print(f"  Converted and loaded {len(d_weights)} discriminator weights")

        # Create trainer
        trainer = RVCTrainer(
            net_g=net_g,
            net_d=net_d,
            config=config,
            train_loader=dataloader,
            val_loader=None,
        )

        # Start training
        print("Starting training loop...")
        trainer.train(
            epochs=total_epoch,
            save_every=save_every_epoch,
            checkpoint_dir=os.path.join(exp_dir, "checkpoints"),
        )

        print(f"Training complete. Checkpoints saved to {os.path.join(exp_dir, 'checkpoints')}")

    except ImportError as e:
        print(f"Error: Could not import training modules: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

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
        prequisites_download_pipeline(args.pretraineds_hifigan, args.models, args.exe, getattr(args, 'titan', False))
    elif args.mode == "convert":
        run_convert(**vars(args))
    elif args.mode == "preprocess":
        run_preprocess(**vars(args))
    elif args.mode == "extract":
        run_extract(**vars(args))
    elif args.mode == "train":
        run_train(**vars(args))
    elif args.mode in ["index", "model_information", "model_blender", "tensorboard"]:
        print(f"The '{args.mode}' subcommand is not yet implemented for the Pure MLX backend.")
    else:
        print(f"Unknown mode: {args.mode}")
