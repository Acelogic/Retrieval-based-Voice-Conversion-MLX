
import os
import sys
import numpy as np
import mlx.core as mx
import librosa
import scipy.signal
import json

try:
    import faiss
except ImportError:
    faiss = None
    print("Warning: faiss not installed. Index usage will be disabled.")

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc_mlx.configs.config import Config
from rvc_mlx.infer.pipeline_mlx import Autotune, AudioProcessor
from rvc_mlx.lib.mlx.synthesizers import Synthesizer
from rvc_mlx.lib.mlx.rmvpe import RMVPE0Predictor as RMVPE 
from rvc_mlx.lib.mlx.utils import load_embedding, HubertModelWithFinalProj

def circular_write(new_data, target):
    """
    Circular write for numpy arrays (mutable).
    new_data: (L,)
    target: (Target_L,)
    """
    offset = new_data.shape[0]
    target[:-offset] = target[offset:]
    target[-offset:] = new_data
    return target 

class RealtimeVoiceConverter:
    """
    A class for performing realtime voice conversion using the Retrieval-Based Voice Conversion (RVC) method.
    """

    def __init__(self, weight_root):
        """
        Initializes the RealtimeVoiceConverter with default configuration, and sets up models and parameters.
        """
        self.config = Config()  # Load configuration
        self.tgt_sr = None  # Target sampling rate for the output audio
        self.net_g = None  # Generator network for voice conversion
        self.cpt = None  # Checkpoint for loading model weights
        self.version = None  # Model version
        self.use_f0 = None  # Whether the model uses F0
        # load weights and setup model network.
        self.load_model(weight_root)
        self.setup_network()

    def load_model(self, weight_root):
        """
        Loads the model weights from the specified path.
        """
        if not weight_root or not os.path.isfile(weight_root):
             self.cpt = None
             return

        try:
            # Check extension
            if weight_root.endswith(".pt"):
                 # Try substitution
                 alt_path = weight_root.replace(".pt", ".safetensors")
                 if not os.path.exists(alt_path):
                     alt_path = weight_root.replace(".pt", ".npz")
                 
                 if os.path.exists(alt_path):
                     print(f"Substituted {weight_root} with {alt_path}")
                     weight_root = alt_path
                 else:
                     raise RuntimeError(f"Cannot load {weight_root} (PyTorch format) in pure MLX mode. Please convert to .safetensors or .npz first using the migration tools.")

            # Load weights using MLX
            if weight_root.endswith(".npz") or weight_root.endswith(".safetensors"):
                weights = mx.load(weight_root)
                self.cpt = {"weight": weights}
                
                # Deduce config from adjacent file or defaults
                config_path = os.path.join(os.path.dirname(weight_root), "config.json")
                if os.path.exists(config_path):
                     with open(config_path, "r") as f:
                         self.cpt.update(json.load(f))
                else:
                     # Default Config fallback (Assuming v2 40k)
                     self.cpt["config"] = [
                        1025, 32, 192, 192, 768, 2, 6, 3, 0.1, "1", 
                        [3,7,11], [[1,3,5], [1,3,5], [1,3,5]], 
                        [10,8,2,2], 512, [3,7,11], 109, 256, 40000
                     ]
                     
                     # Version check heuristic
                     # If "enc_p.encoder.layers.0.attn.q_proj.weight" exists, check dim
                     if "enc_p.encoder.layers.0.attn.q_proj.weight" in weights:
                          w = weights["enc_p.encoder.layers.0.attn.q_proj.weight"]
                          if w.shape[-1] == 256: # v1
                              self.cpt["version"] = "v1"
                              self.cpt["config"][-3] = 256 # spk_embed_dim
                              self.cpt["config"][4] = 256 # filter_channels? No v1 filter channels usually 768, but hidden size 256?
                              # Need precise v1 config if used. But v2 is standard now.
                          else:
                              self.cpt["version"] = "v2"
                     else:
                          self.cpt["version"] = "v2"

                     self.cpt["f0"] = 1 # Assume f0=1 (True)
                     self.cpt["vocoder"] = "HiFi-GAN"
            else:
                 raise ValueError("Unsupported model file format. Use .safetensors or .npz")
        
        except Exception as e:
            print(f"Error loading model: {e}")
            self.cpt = None

    def setup_network(self):
        """
        Sets up the network configuration based on the loaded checkpoint.
        """
        if self.cpt is not None:
            self.tgt_sr = self.cpt["config"][-1]
            
            # Update embed dim from weights if available to be safe
            if "emb_g.weight" in self.cpt["weight"]:
                self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]
            
            self.use_f0 = self.cpt.get("f0", 1)

            self.version = self.cpt.get("version", "v2") # Default to v2
            self.text_enc_hidden_dim = 768 if self.version == "v2" else 256
            self.vocoder = self.cpt.get("vocoder", "HiFi-GAN")
            
            self.net_g = Synthesizer(
                *self.cpt["config"],
                use_f0=self.use_f0,
                text_enc_hidden_dim=self.text_enc_hidden_dim,
                vocoder=self.vocoder,
            )

            self.net_g.load_weights(list(self.cpt["weight"].items()), strict=False)
            mx.eval(self.net_g.parameters()) # ensure loaded

    def inference(
        self,
        feats: mx.array,
        p_len: mx.array,
        sid: mx.array,
        pitch: mx.array,
        pitchf: mx.array,
    ):
        # Synthesizer.infer returns: o, x_mask, (z, z_p, m_p, logs_p)
        output, _, _ = self.net_g.infer(feats, p_len, pitch, pitchf, sid)
        # output: (B, 1, L_audio) -> (B, L_audio) (since C=1 usually)
        
        # Audio output (B, L)
        # Clip
        out = output[0, 0]
        out = mx.maximum(out, -1.0)
        out = mx.minimum(out, 1.0)
        return out


class Realtime_Pipeline:
    def __init__(
        self,
        vc: RealtimeVoiceConverter,
        hubert_model: HubertModelWithFinalProj = None,
        index=None,
        big_npy=None,
        f0_method: str = "rmvpe",
        sid: int = 0,
    ):
        self.vc = vc
        self.hubert_model = hubert_model
        self.index = index
        self.big_npy = big_npy
        self.use_f0 = vc.use_f0
        self.version = vc.version
        self.f0_method = f0_method
        self.sample_rate = 16000
        self.tgt_sr = vc.tgt_sr
        self.window = 160
        self.model_window = self.tgt_sr // 100
        self.f0_min = 50.0
        self.f0_max = 1100.0
        
        self.sid = mx.array([sid], dtype=mx.int64)
        
        self.autotune = Autotune()
        self.f0_model = None

    def get_f0(
        self,
        x: np.ndarray,
        pitch: np.ndarray = None,
        pitchf: np.ndarray = None,
        f0_up_key: int = 0,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1.0,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
    ):
        """
        Estimates the fundamental frequency (F0) of a given audio signal.
        Input x is numpy array (16k mono).
        """
        
        # RMVPE MLX
        if self.f0_method == "rmvpe":
            if self.f0_model is None:
                self.f0_model = RMVPE()
            f0 = self.f0_model.infer_from_audio(x, thred=0.03)
        else:
             # Fallback to RMVPE
             if self.f0_model is None:
                self.f0_model = RMVPE()
             f0 = self.f0_model.infer_from_audio(x)
             
        # f0 adjustments (Numpy ops)
        if f0_autotune:
            f0 = self.autotune.autotune_f0(f0, f0_autotune_strength)
        
        if proposed_pitch:
             valid_f0 = np.where(f0 > 0)[0]
             if len(valid_f0) > 1:
                 median_f0 = np.median(f0[valid_f0])
                 if median_f0 > 0:
                      up_key = 12 * np.log2(proposed_pitch_threshold / median_f0)
                      up_key = max(-12, min(12, up_key))
                      f0 *= pow(2, (f0_up_key + up_key) / 12)
        else:
            f0 *= pow(2, f0_up_key / 12)
            
        # Coarse pitch
        f0_mel = 1127.0 * np.log(1.0 + f0 / 700.0)
        f0_mel = (f0_mel - self.f0_min) * 254 / (self.f0_max - self.f0_min) + 1
        f0_mel = np.clip(f0_mel, 1, 255)
        f0_coarse = np.rint(f0_mel).astype(np.int64)
        
        # Circular write buffers
        if pitch is not None and pitchf is not None:
             circular_write(f0_coarse, pitch)
             circular_write(f0, pitchf)
             
             return mx.array(pitch)[None, :], mx.array(pitchf)[None, :]
        else:
             return mx.array(f0_coarse)[None, :], mx.array(f0)[None, :]
             
    def _interpolate(self, x, scale_factor):
        # x: (B, C, L)
        # Nearest neighbor interpolation
        B, C, L = x.shape
        x = mx.broadcast_to(x[..., None], (B, C, L, scale_factor))
        x = x.reshape(B, C, L * scale_factor)
        return x

    def voice_conversion(
        self,
        audio: np.ndarray, # Buffer input (Numpy)
        pitch: np.ndarray = None,
        pitchf: np.ndarray = None,
        f0_up_key: int = 0,
        index_rate: float = 0.5,
        p_len: int = 0,
        silence_front: int = 0,
        skip_head: int = None,
        return_length: int = None,
        protect: float = 0.5,
        volume_envelope: float = 1,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
        reduced_noise=None,
        board=None,
    ):
        """
        Performs realtime voice conversion.
        """
        # Audio for f0
        audio_f0 = audio[silence_front:]
        
        mx_pitch, mx_pitchf = (None, None)
        if self.use_f0:
             mx_pitch, mx_pitchf = self.get_f0(
                 audio_f0, pitch, pitchf, f0_up_key, f0_autotune, f0_autotune_strength, proposed_pitch, proposed_pitch_threshold
             )
        
        # Hubert
        # Input: (B, T)
        feats_in = mx.array(audio)[None, :] # (1, L)
        
        # Forward pass (HubertModelWithFinalProj)
        # returns (B, T, C)
        feats = self.hubert_model(feats_in) 
        
        # Feats padding/copying last frame
        feats = mx.concatenate([feats, feats[:, -1:, :]], axis=1)
        
        feats0 = feats if self.use_f0 else None
        
        # Index Search (Faiss uses Numpy)
        if self.index is not None and index_rate > 0:
             npy = np.array(feats[0]) # (T, C)
             
             skip_offset = skip_head // 2
             search_query = npy[skip_offset:]
             if search_query.shape[0] > 0:
                 score, ix = self.index.search(search_query, k=8)
                 weight = np.square(1 / score)
                 weight /= weight.sum(axis=1, keepdims=True)
                 
                 neighbors = self.big_npy[ix]
                 new_feats = np.sum(neighbors * np.expand_dims(weight, axis=2), axis=1)
                 
                 # Blend
                 npy[skip_offset:] = new_feats * index_rate + (1 - index_rate) * npy[skip_offset:]
                 
                 feats = mx.array(npy)[None, :]

        # Upsample features
        # feats: (B, T, C). Need (B, C, T) for interpolate?
        feats = feats.transpose(0, 2, 1) # (B, C, T)
        feats = self._interpolate(feats, 2) # (B, C, 2T)
        feats = feats.transpose(0, 2, 1) # (B, 2T, C)
        
        # Clip to p_len
        if self.use_f0:
            mx_pitch = mx_pitch[:, :p_len]
            mx_pitchf = mx_pitchf[:, :p_len]
            feats = feats[:, :p_len, :]

            feats0 = feats0.transpose(0, 2, 1)
            feats0 = self._interpolate(feats0, 2)
            feats0 = feats0.transpose(0, 2, 1)
            feats0 = feats0[:, :p_len, :]
            
            # mask pitch
            mx_pitch = mx_pitch[:, -p_len:]
            mx_pitchf = mx_pitchf[:, -p_len:]
            
            # Pitch protection
            if protect < 0.5:
                pitchff = mx_pitchf
                pitchff_mask = pitchff > 0
                pitchff_val = mx.where(pitchff_mask, 1.0, protect)
                
                pitchff_val = pitchff_val[..., None]
                feats = feats * pitchff_val + feats0 * (1 - pitchff_val)
                
        else:
             mx_pitch, mx_pitchf = None, None
             feats = feats[:, :p_len, :]
        
        p_len_arr = mx.array([p_len], dtype=mx.int64)
        
        # Inference
        out_audio = self.vc.inference(feats, p_len_arr, self.sid, mx_pitch, mx_pitchf)
        
        # Convert out_audio to numpy for post-process (pedalboard, RMS)
        out_audio_np = np.array(out_audio)
        
        # Volume Envelope
        if volume_envelope != 1:
             out_audio_np = AudioProcessor.change_rms(
                 audio, # Input raw audio (Numpy)
                 self.sample_rate,
                 out_audio_np,
                 self.tgt_sr,
                 volume_envelope
             )
             
        # Resample logic
        scaled_window = int(np.floor(1.0 * self.model_window))
        if scaled_window != self.model_window:
             out_audio_np = librosa.resample(out_audio_np, orig_sr=scaled_window, target_sr=self.model_window)

        if reduced_noise is not None:
             try:
                 out_audio_np = reduced_noise(out_audio_np, self.tgt_sr)
             except Exception:
                 pass
             
        if board is not None:
             try:
                 out_audio_np = board(out_audio_np, self.tgt_sr)
             except Exception:
                 pass
         
        return out_audio_np


def load_faiss_index(file_index):
    if faiss is None:
        return None, None
    if file_index != "" and os.path.exists(file_index):
        try:
            index = faiss.read_index(file_index)
            big_npy = index.reconstruct_n(0, index.ntotal)
        except Exception as error:
            print(f"An error occurred reading the FAISS index: {error}")
            index = big_npy = None
    else:
        index = big_npy = None

    return index, big_npy


def create_pipeline(
    model_path: str = None,
    index_path: str = None,
    f0_method: str = "rmvpe",
    embedder_model: str = None,
    embedder_model_custom: str = None,
    sid: int = 0,
):
    """
    Initialize real-time voice conversion pipeline.
    """
    vc = RealtimeVoiceConverter(model_path)
    
    idx_path = index_path.strip().strip('"') if index_path else "" 
    index, big_npy = load_faiss_index(idx_path)

    hubert_model = load_embedding(embedder_model, embedder_model_custom)
    
    pipeline = Realtime_Pipeline(
        vc,
        hubert_model,
        index,
        big_npy,
        f0_method,
        sid,
    )

    return pipeline
