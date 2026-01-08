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

def remap_keys(weights):
    new_weights = {}
    for k, v in weights.items():
        new_key = k
        # Decoder remapping
        if new_key.startswith("dec.resblocks."):
            # dec.resblocks.0.convs1.0.weight -> dec.resblock_0.c1_0.weight
            parts = new_key.split(".")
            idx = parts[2]
            c_type = "c1" if "convs1" in new_key else "c2"
            c_idx = parts[4]
            rest = parts[5:] # weight, bias
            new_key = f"dec.resblock_{idx}.{c_type}_{c_idx}.{'.'.join(rest)}"
        elif new_key.startswith("dec.ups."):
            # dec.ups.0.weight -> dec.up_0.weight
            parts = new_key.split(".")
            idx = parts[2]
            rest = parts[3:]
            new_key = f"dec.up_{idx}.{'.'.join(rest)}"
        elif new_key.startswith("dec.noise_convs."):
            parts = new_key.split(".")
            idx = parts[2]
            rest = parts[3:]
            new_key = f"dec.noise_conv_{idx}.{'.'.join(rest)}"
            
        # Encoder remapping
        elif new_key.startswith("enc_p.encoder.attn_layers."):
            # enc_p.encoder.attn_layers.0.conv_q.weight -> enc_p.encoder.attn_0.conv_q.weight
            parts = new_key.split(".")
            idx = parts[3]
            rest = parts[4:]
            new_key = f"enc_p.encoder.attn_{idx}.{'.'.join(rest)}"
        elif new_key.startswith("enc_p.encoder.norm_layers_1."):
            parts = new_key.split(".")
            idx = parts[3]
            rest = parts[4:]
            new_key = f"enc_p.encoder.norm1_{idx}.{'.'.join(rest)}"
        elif new_key.startswith("enc_p.encoder.norm_layers_2."):
            parts = new_key.split(".")
            idx = parts[3]
            rest = parts[4:]
            new_key = f"enc_p.encoder.norm2_{idx}.{'.'.join(rest)}"
        elif new_key.startswith("enc_p.encoder.ffn_layers."):
            parts = new_key.split(".")
            idx = parts[3]
            rest = parts[4:]
            new_key = f"enc_p.encoder.ffn_{idx}.{'.'.join(rest)}"
            
        # Flow remapping
        elif new_key.startswith("flow.flows."):
            # flow.flows.0.enc.in_layers.0.weight -> flow.flow_0.enc.in_layer_0.weight
            parts = new_key.split(".")
            f_idx = parts[2]
            if "in_layers" in new_key:
                l_idx = parts[5]
                rest = parts[6:]
                new_key = f"flow.flow_{f_idx}.enc.in_layer_{l_idx}.{'.'.join(rest)}"
            elif "res_skip_layers" in new_key:
                l_idx = parts[5]
                rest = parts[6:]
                new_key = f"flow.flow_{f_idx}.enc.res_skip_layer_{l_idx}.{'.'.join(rest)}"
            else:
                rest = parts[3:]
                new_key = f"flow.flow_{f_idx}.{'.'.join(rest)}"

        # Common layer params: gamma -> weight, beta -> bias for LayerNorm
        if new_key.endswith(".gamma"):
            new_key = new_key.replace(".gamma", ".weight")
        elif new_key.endswith(".beta"):
            new_key = new_key.replace(".beta", ".bias")

        new_weights[new_key] = v
    return new_weights

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
        self.config = config # Should contain simple config
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
                self.device = "gpu" # MLX handles this
                self.is_half = False # MLX handles fp16/fp32 auto
                
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
        upsample_rates = [10, 8, 2, 2] # 40k -> [10, 8, 2, 2], 48k -> [10, 8, 2, 2, ?]
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
             with open(config_path, 'r') as f:
                 conf = json.load(f)
             # Parse conf
             # Synthesizer args...
             # ...
             pass
        
        # Minimal defaults for 40k v2 logic
        upsample_initial_channel = 512
        upsample_kernel_sizes = [16, 16, 4, 4]
        spk_embed_dim = 109 # Default
        gin_channels = 256
        sr = 40000 # Default
        upsample_rates = [10, 10, 2, 2] # Correct for 40k

        # Load config if exists
        config_path = os.path.splitext(model_path)[0] + ".json"
        if os.path.exists(config_path):
             import json
             with open(config_path, 'r') as f:
                 conf = json.load(f)
                 if isinstance(conf, list):
                      # RVC JSON export format (list)
                      # [spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, 
                      #  n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, 
                      #  resblock_dilation_sizes, upsample_rates, upsample_initial_channel, 
                      #  upsample_kernel_sizes, spk_embed_dim, gin_channels, sr]
                      if len(conf) >= 18:
                           spec_channels = conf[0]
                           segment_size = conf[1]
                           inter_channels = conf[2]
                           hidden_channels = conf[3]
                           filter_channels = conf[4]
                           n_heads = conf[5]
                           n_layers = conf[6]
                           kernel_size = conf[7]
                           p_dropout = conf[8]
                           resblock = conf[9]
                           resblock_kernel_sizes = conf[10]
                           resblock_dilation_sizes = conf[11]
                           upsample_rates = conf[12]
                           upsample_initial_channel = conf[13]
                           upsample_kernel_sizes = conf[14]
                           spk_embed_dim = conf[15]
                           gin_channels = conf[16]
                           sr = conf[17]
                 elif isinstance(conf, dict):
                      if "data" in conf:
                           sr = conf["data"].get("sampling_rate", sr)
                      if "model" in conf:
                           m_conf = conf["model"]
                           upsample_rates = m_conf.get("upsample_rates", upsample_rates)
                           upsample_initial_channel = m_conf.get("upsample_initial_channel", upsample_initial_channel)
                           upsample_kernel_sizes = m_conf.get("upsample_kernel_sizes", upsample_kernel_sizes)
                           gin_channels = m_conf.get("gin_channels", gin_channels)
                           hidden_channels = m_conf.get("hidden_channels", hidden_channels)
                           inter_channels = m_conf.get("inter_channels", inter_channels)
                           filter_channels = m_conf.get("filter_channels", filter_channels)
                           n_heads = m_conf.get("n_heads", n_heads)
                           n_layers = m_conf.get("n_layers", n_layers)
        
        print(f"DEBUG: Model configuration - Sample Rate: {sr}, Upsample Rates: {upsample_rates}, Total Upsample: {np.prod(upsample_rates)}")

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
            use_f0=True
        )
        
        # Load weights
        weights = remap_keys(weights)
        self.net_g.load_weights(list(weights.items()), strict=False)
        self.net_g.eval() # Disable dropout
        mx.eval(self.net_g.parameters())
        
        self.tgt_sr = sr
        
        if not hasattr(self, 'hubert_model') or self.hubert_model is None:
            print("Loading MLX Hubert...")
            conf = HubertConfig(classifier_proj_size=768)
            self.hubert_model = HubertModel(conf)
            # Load weights
            # Try rvc_mlx/models path
            h_path = os.path.join("rvc_mlx", "models", "embedders", "contentvec", "hubert_mlx.npz")
            
            if not os.path.exists(h_path):
                 # Try legacy/dev path
                 h_path = "rvc/models/embedders/contentvec/hubert_mlx.npz"

            
            if os.path.exists(h_path):
                 self.hubert_model.load_weights(h_path, strict=False)
                 self.hubert_model.eval() # Disable dropout
                 mx.eval(self.hubert_model.parameters())
            else:
                 print(f"Error: Hubert weights not found at {h_path}. Please place hubert_mlx.npz there.")

        self.load_rmvpe()
        
        self.pipeline = PipelineMLX(self.tgt_sr, self.pipeline_config, self.hubert_model, self.rmvpe_model)



    def load_rmvpe(self):
        print("Loading RMVPE...")
        self.rmvpe_model = RMVPE0Predictor() 
        if hasattr(self.rmvpe_model, 'model'):
             self.rmvpe_model.model.eval()
        # RMVPE0Predictor loads weights internally from default path or we can pass path

    def infer(self, 
              audio_input, 
              audio_output, 
              pitch=0, 
              f0_method="rmvpe", 
              index_path=None, 
              index_rate=0.75, 
              volume_envelope=1.0, 
              protect=0.5,
              f0_autotune=False,
              f0_autotune_strength=1.0):
              
        print(f"Processing {audio_input}...")
        audio = load_audio(audio_input)
        if audio is None: return

        # Call pipeline
        # pipeline(model, net_g, sid, audio, pitch, f0_method, file_index, index_rate, pitch_guidance, volume_envelope, version, protect, ...)
        
        audio_opt = self.pipeline.pipeline(
            self.hubert_model,
            self.net_g,
            0, # sid default 0
            audio,
            pitch,
            f0_method,
            index_path,
            index_rate,
            True, # pitch_guidance
            volume_envelope,
            "v2", # version
            protect,
            f0_autotune, 
            f0_autotune_strength, 
            False,
            155.0
        )
        
        # Save
        sf.write(audio_output, audio_opt, self.tgt_sr)
        print(f"Saved to {audio_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to MLX model weights (.npz)")
    parser.add_argument("--input", type=str, required=True, help="Input audio path")
    parser.add_argument("--output", type=str, required=True, help="Output audio path")
    parser.add_argument("--pitch", type=int, default=0, help="Pitch shift")
    parser.add_argument("--f0-method", type=str, default="rmvpe",
                        choices=["rmvpe", "dio", "pm", "harvest", "crepe", "crepe-tiny", "fcpe"],
                        help="F0 extraction method: rmvpe (default), dio, pm, harvest, crepe, crepe-tiny, fcpe")
    parser.add_argument("--index", type=str, default="", help="Path to .index file")
    
    args = parser.parse_args()
    
    rvc = RVC_MLX(args.model)
    rvc.infer(args.input, args.output, args.pitch, args.f0_method, args.index)
