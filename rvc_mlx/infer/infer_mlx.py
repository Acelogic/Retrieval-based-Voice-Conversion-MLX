import os
import sys
import numpy as np
import mlx.core as mx
import soundfile as sf
import librosa
import argparse

# Add current directory to path for imports
sys.path.append(os.getcwd())

from rvc_mlx.lib.mlx.synthesizers import Synthesizer
from rvc_mlx.lib.mlx.hubert import HubertModel, HubertConfig
from rvc_mlx.lib.mlx.rmvpe import RMVPE0Predictor
from rvc_mlx.infer.pipeline_mlx import PipelineMLX


def load_audio(file_path, sr=16000):
    try:
        audio, samplerate = sf.read(file_path)
    except Exception as e:
        print(f"Error loading audio {file_path}: {e}")
        return None

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    if samplerate != sr:
        audio = librosa.resample(audio, orig_sr=samplerate, target_sr=sr)

    return audio


class RVC_MLX:
    def __init__(self, model_path, config=None):
        self.config = config  # Should contain simple config
        self.model_path = model_path

        # Load Model
        print(f"Loading MLX model from {model_path}...")
        # Assume .npz format for pure MLX
        weights = mx.load(model_path)

        # Infer parameters from weights or config
        # We need a Config object.
        # For now, use a simple class or dict
        class ConfigObj:
            def __init__(self):
                self.x_pad = 1
                self.x_query = 6
                self.x_center = 38
                self.x_max = 41
                self.device = "gpu"  # MLX handles this
                self.is_half = False  # MLX handles fp16/fp32 auto

        self.pipeline_config = ConfigObj()

        # Determine synthesizer args from weights if possible, or use defaults for RVC v2
        # Default RVC v2 params
        spec_channels = 1025
        segment_size = 32
        inter_channels = 192
        hidden_channels = 192
        filter_channels = 768
        n_heads = 2
        n_layers = 6
        kernel_size = 3
        p_dropout = 0
        resblock = "1"
        resblock_kernel_sizes = [3, 7, 11]
        resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        upsample_rates = [10, 8, 2, 2]  # 40k -> [10, 8, 2, 2], 48k -> [10, 8, 2, 2, ?]
        # Wait, standard RVC v2 40k:
        # [10, 8, 2, 2] -> 10*8*2*2 = 320. 320*hop? hop=sample_rate/something?
        # RVC uses hop_length=320? No?
        # Standard: sr=40000. hop?

        # To strictly match, we should load config from .json next to weights or embedded.
        # But for "strip pytorch", user might provide weights + config.
        # Let's try to load config.json if exists
        config_path = os.path.splitext(model_path)[0] + ".json"
        if os.path.exists(config_path):
            import json

            with open(config_path, "r") as f:
                conf = json.load(f)
            # Parse conf
            # Synthesizer args...
            # ...

        # Minimal defaults for 40k v2 logic
        upsample_initial_channel = 512
        upsample_kernel_sizes = [16, 16, 4, 4]
        spk_embed_dim = 109  # Default
        gin_channels = 256
        sr = 40000  # Default

        # Check speakers from weights
        if "emb_g.weight" in weights:
            spk_embed_dim = weights["emb_g.weight"].shape[0]

        # Initialize Synthesizer
        self.net_g = Synthesizer(
            spec_channels,
            segment_size,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            spk_embed_dim,
            gin_channels,
            sr,
            use_f0=True,
        )

        # Load weights
        self.net_g.update(weights)
        mx.eval(self.net_g.parameters())

        self.tgt_sr = sr

        if not hasattr(self, "hubert_model") or self.hubert_model is None:
            print("Loading MLX Hubert...")
            conf = HubertConfig(classifier_proj_size=768)
            self.hubert_model = HubertModel(conf)
            # Load weights
            # Try rvc_mlx/models path
            h_path = os.path.join(
                "rvc_mlx", "models", "embedders", "contentvec", "hubert_mlx.npz"
            )

            if not os.path.exists(h_path):
                # Try legacy/dev path
                h_path = "rvc/models/embedders/contentvec/hubert_mlx.npz"

            if os.path.exists(h_path):
                self.hubert_model.load_weights(h_path, strict=False)
                mx.eval(self.hubert_model.parameters())
            else:
                print(
                    f"Error: Hubert weights not found at {h_path}. Please place hubert_mlx.npz there."
                )

        self.load_rmvpe()

        self.pipeline = PipelineMLX(
            self.tgt_sr, self.pipeline_config, self.hubert_model, self.rmvpe_model
        )

    def load_rmvpe(self):
        print("Loading RMVPE...")
        self.rmvpe_model = RMVPE0Predictor()
        # RMVPE0Predictor loads weights internally from default path or we can pass path

    def infer(
        self,
        audio_input,
        audio_output,
        pitch=0,
        f0_method="rmvpe",
        index_path=None,
        index_rate=0.75,
        volume_envelope=1.0,
        protect=0.5,
    ):

        print(f"Processing {audio_input}...")
        audio = load_audio(audio_input)
        if audio is None:
            return

        # Call pipeline
        # pipeline(model, net_g, sid, audio, pitch, f0_method, file_index, index_rate, pitch_guidance, volume_envelope, version, protect, ...)

        audio_opt = self.pipeline.pipeline(
            self.hubert_model,
            self.net_g,
            0,  # sid default 0
            audio,
            pitch,
            f0_method,
            index_path,
            index_rate,
            True,  # pitch_guidance
            volume_envelope,
            "v2",  # version
            protect,
            False,  # f0_autotune
            1.0,
            False,
            155.0,
        )

        # Save
        sf.write(audio_output, audio_opt, self.tgt_sr)
        print(f"Saved to {audio_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="Path to MLX model weights (.npz)"
    )
    parser.add_argument("--input", type=str, required=True, help="Input audio path")
    parser.add_argument("--output", type=str, required=True, help="Output audio path")
    parser.add_argument("--pitch", type=int, default=0, help="Pitch shift")
    parser.add_argument(
        "--f0-method", type=str, default="rmvpe", help="F0 method (rmvpe)"
    )
    parser.add_argument("--index", type=str, default="", help="Path to .index file")

    args = parser.parse_args()

    rvc = RVC_MLX(args.model)
    rvc.infer(args.input, args.output, args.pitch, args.f0_method, args.index)
