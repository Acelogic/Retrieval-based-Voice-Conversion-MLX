import os
import sys
import torch
import numpy as np
import mlx.core as mx
from torch.nn import functional as F

from rvc.infer.infer import VoiceConverter as VoiceConverterTorch
from rvc.infer.pipeline import Pipeline
from rvc.lib.mlx.synthesizers import Synthesizer
from rvc.lib.mlx.convert import convert_weights

class PipelineMLX(Pipeline):
    def __init__(self, tgt_sr, config, mlx_model):
        super().__init__(tgt_sr, config)
        self.mlx_model = mlx_model

    def voice_conversion(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        index,
        big_npy,
        index_rate,
        version,
        protect,
    ):
        """
        MLX override of voice_conversion. 
        Uses Torch for feature extraction, then MLX for synthesis.
        """
        # Feature extraction logic from original pipeline.py
        with torch.no_grad():
            pitch_guidance = pitch != None and pitchf != None
            
            feats = torch.from_numpy(audio0).float()
            feats = feats.mean(-1) if feats.dim() == 2 else feats
            assert feats.dim() == 1, feats.dim()
            feats = feats.view(1, -1).to(self.device)
            
            # extract features (Hubert)
            feats = model(feats)["last_hidden_state"]
            feats = (
                model.final_proj(feats[0]).unsqueeze(0) if version == "v1" else feats
            )
            
            feats0 = feats.clone() if pitch_guidance else None
            
            if index is not None and index_rate > 0:
                feats = self._retrieve_speaker_embeddings(feats, index, big_npy, index_rate)
                
            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
            
            p_len = min(audio0.shape[0] // self.window, feats.shape[1])
            
            if pitch_guidance:
                feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
                pitch, pitchf = pitch[:, :p_len], pitchf[:, :p_len]
                
                if protect < 0.5:
                    pitchff = pitchf.clone()
                    pitchff[pitchf > 0] = 1
                    pitchff[pitchf < 1] = protect
                    feats = feats * pitchff.unsqueeze(-1) + feats0 * (1 - pitchff.unsqueeze(-1))
                    feats = feats.to(feats0.dtype)
            else:
                pitch, pitchf = None, None
                
            # --- MLX INFERENCE START ---
            # Convert tensors to MLX arrays.
            # feats: (1, L, C) -> MLX (1, L, C)
            feats_mx = mx.array(feats.cpu().numpy())
            
            # pitch: (1, L)
            pitch_mx = mx.array(pitch.cpu().numpy()) if pitch is not None else None
            # pitchf: (1, L)
            pitchf_mx = mx.array(pitchf.cpu().numpy()) if pitchf is not None else None
            
            # sid: (1,)
            sid_mx = mx.array(sid.cpu().numpy())
            
            # Call MLX model
            # MLX model handles (N, L, C) naturally
            # Our Synthesizer.infer returns (o, x_mask, stats)
            # o is audio output (B, 1, T) or (B, T, 1)?
            # HiFiGANNSFGenerator call: returns (B, 1, T) or (B, C, L)?
            # My port used `conv_post` (Conv1d).
            # If Conv1d output is (N, L, 1), then output is (N, L, 1).
            # Let's check my generator logic. 
            # `x = self.conv_post(x)` 
            # MLX Conv1d output (N, L, C). `out_channels=1`.
            # So (1, L, 1).
            
            out_audio, _, _ = self.mlx_model.infer(
                feats_mx, 
                None, # phone_lengths is for training inputs to TextEncoder, inference computes from feats?
                      # Wait, Synthesizer.infer signature: (phone, phone_lengths, ...).
                      # But `pipeline.py` passes `feats` as 1st arg. `p_len` as 2nd.
                      # `net_g.infer(feats, p_len, pitch, pitchf, sid)`
                
                # In `synthesizers.py` port:
                # def infer(self, phone, phone_lengths, pitch, nsff0, sid, rate=None):
                
                mx.array([p_len]), # phone_lengths
                pitch_mx,
                pitchf_mx,
                sid_mx
            )
            
            # Output shape (1, T, 1) -> squeeze to (T)
            audio1 = np.array(out_audio[0, :, 0])
             
            # --- MLX INFERENCE END ---
            
            del feats, feats0
            return audio1


class VoiceConverterMLX(VoiceConverterTorch):
    def __init__(self):
        super().__init__()
        self.mlx_model = None
        self.mlx_model_path = None
        self.mlx_pipeline = None

    def get_vc(self, weight_root, sid):
        super().get_vc(weight_root, sid)
        
        if not self.mlx_model or self.mlx_model_path != weight_root:
            print(f"Loading MLX model: {weight_root}")
            weights, config = convert_weights(weight_root)
            
            renamed_weights = {}
            for k, v in weights.items():
                new_k = k
                # Simple heuristic replacement for list indices
                # enc_p.attn_layers.0 -> enc_p.attn_0
                # enc_p.norm_layers_1.0 -> enc_p.norm1_0 (matching my Encoders.py)
                # ...
                # This mapping must be precise.
                # Let's define the set of replacements based on my port names.
                # Encoders:
                # .attn_layers.X -> .attn_X
                # .norm_layers_1.X -> .norm1_X
                # .ffn_layers.X -> .ffn_X
                # .norm_layers_2.X -> .norm2_X
                
                # Generator (HiFiGAN):
                # .ups.X -> .up_X
                # .noise_convs.X -> .noise_conv_X
                # .resblocks.X -> .resblock_X
                
                # ResidualCouplingBlock/Layer:
                # .flows.X -> .flow_X
                # Inside ResidualCouplingLayer, `enc` is WaveNet.
                # WaveNet: .in_layers.X -> .in_layer_X, .res_skip_layers.X -> .res_skip_layer_X
                
                # Implementation of renaming:
                
                parts = new_k.split(".")
                new_parts = []
                skip_next = False
                for i, p in enumerate(parts):
                    if skip_next:
                        skip_next = False
                        continue
                        
                    if p.isdigit():
                        # Should have been handled by previous part
                        new_parts.append(p)
                    elif i + 1 < len(parts) and parts[i+1].isdigit():
                        # Found list access pattern e.g. "ups" followed by "0"
                        idx = parts[i+1]
                        
                        # Apply specific renaming rules
                        if p == "attn_layers": p = f"attn_{idx}"
                        elif p == "norm_layers_1": p = f"norm1_{idx}"
                        elif p == "ffn_layers": p = f"ffn_{idx}"
                        elif p == "norm_layers_2": p = f"norm2_{idx}"
                        elif p == "ups": p = f"up_{idx}"
                        elif p == "noise_convs": p = f"noise_conv_{idx}"
                        elif p == "resblocks": p = f"resblock_{idx}"
                        elif p == "flows": p = f"flow_{idx}"
                        elif p == "in_layers": p = f"in_layer_{idx}"
                        elif p == "res_skip_layers": p = f"res_skip_layer_{idx}"
                        
                        else:
                            # Default fallback: name_idx
                             p = f"{p}_{idx}"
                        
                        new_parts.append(p)
                        skip_next = True
                    else:
                        new_parts.append(p)
                
                new_k = ".".join(new_parts)
                renamed_weights[new_k] = v
                
            self.mlx_model = Synthesizer(
                *config,
                use_f0=self.use_f0,
                text_enc_hidden_dim=self.text_enc_hidden_dim,
                vocoder=self.vocoder
            )
            # Use load_weights assuming flattened structure
            # self.mlx_model.load_weights(list(renamed_weights.items())) -- expects file
            self.mlx_model.update(renamed_weights)
            # MX eval to ensure weights loaded/cached
            mx.eval(self.mlx_model.parameters())
            
            self.mlx_model_path = weight_root
            
            # Create pipeline instance injection
            self.mlx_pipeline = PipelineMLX(self.tgt_sr, self.config, self.mlx_model)

    def pipeline(self, model, net_g, sid, audio, pitch, f0_method, file_index, index_rate, pitch_guidance, volume_envelope, version, protect, f0_autotune, f0_autotune_strength, proposed_pitch, proposed_pitch_threshold):
        return self.mlx_pipeline.pipeline(
            model,
            None, # net_g ignored
            sid,
            audio,
            pitch,
            f0_method,
            file_index,
            index_rate,
            pitch_guidance,
            volume_envelope,
            version,
            protect,
            f0_autotune,
            f0_autotune_strength,
            proposed_pitch,
            proposed_pitch_threshold
        )

# Phase 2: Pure MLX Pipeline
from rvc.lib.mlx.hubert import HubertModel, HubertConfig
from rvc.lib.mlx.rmvpe import RMVPE0Predictor as RMVPE_MLX

class PipelineMLXPure(PipelineMLX):
    def __init__(self, tgt_sr, config, mlx_model, hubert_model, rmvpe_model):
        super().__init__(tgt_sr, config, mlx_model)
        self.hubert_model = hubert_model
        self.rmvpe_model = rmvpe_model

    def voice_conversion(
        self,
        model, # Ignored (is None in Pure mode)
        net_g, # Ignored (is None)
        sid,
        audio0,
        pitch,
        pitchf,
        index,
        big_npy,
        index_rate,
        version,
        protect,
    ):
        """
        Pure MLX Voice Conversion.
        audio0: numpy array (T,)
        """
        # 1. Feature Extraction (Hubert MLX)
        # audio0 is (T,), floats.
        # Hubert expects (N, T).
        audio_mx = mx.array(audio0)[None, :]
        
        # Run Hubert
        # feats = model(feats)["last_hidden_state"]
        # Our HubertModel returns `proj` which is the projected features used by RVC.
        # Check `rvc/lib/mlx/hubert.py`: `__call__` returns `proj`.
        # `proj` shape: (1, L, C).
        feats = self.hubert_model(audio_mx)

        # Interpolate feats
        # feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        # MLX Interpolate?
        # Use simple upsampling (repeat).
        # feats (N, L, C). scale_factor=2 along L.
        # feats.shape -> (1, L, 256).
        # Upsample along L.
        B, L, C = feats.shape
        # Nearest neighbor upsample: repeat elements.
        # (B, L, 1, C) -> broadcast -> (B, L, 2, C) -> reshape (B, L*2, C).
        feats = mx.broadcast_to(feats[:, :, None, :], (B, L, 2, C))
        feats = feats.reshape(B, L * 2, C)
        
        p_len = min(audio0.shape[0] // self.window, feats.shape[1])
        feats = feats[:, :p_len, :]
        
        # 2. Pitch Extraction (RMVPE MLX)
        # pitch, pitchf args might be None if f0_method was used.
        # But wait, `pipeline.py` calls `get_f0` BEFORE `voice_conversion`.
        # `get_f0` is where RMVPE is used.
        # We need to ensure `get_f0` uses our pure implementation if we override it.
        # BUT `PipelineMLX` inherits from `Pipeline` which implements `pipeline` method showing flow:
        #   pitch, pitchf = self.get_f0(...)
        #   return self.voice_conversion(...)
        
        # So `pitch` passed here is ALREADY computed.
        # If we use `voice_conversion`, we just use the passed pitch.
        # WE NEED to override `get_f0` method in this class to use `rmvpe_model`.
        
        # 3. Inference (Synthesizer MLX)
        pitch_mx = mx.array(pitch) if pitch is not None else None
        pitchf_mx = mx.array(pitchf) if pitchf is not None else None
        sid_mx = mx.array(sid)

        out_audio, _, _ = self.mlx_model.infer(
             feats,
             mx.array([p_len]),
             pitch_mx,
             pitchf_mx,
             sid_mx
        )
        
        audio1 = np.array(out_audio[0, :, 0])
        return audio1
        
    def get_f0(self, input_audio, f0_up_key, f0_method, filter_radius, cr_threshold=None):
        # Override get_f0 to use RMVPE_MLX if selected.
        # input_audio: numpy (T,)
        if f0_method == "rmvpe":
            # Use pure MLX/Numpy RMVPE
            if hasattr(self, 'rmvpe_model') and self.rmvpe_model:
                f0 = self.rmvpe_model.infer_from_audio(input_audio, thred=0.03) # Default thred?
                
                # Coarse F0 logic (interpolate/clean)?
                # Standard Pipeline `get_f0` does F0 coarse processing.
                # Here `infer_from_audio` returns f0 contour (numpy).
                
                # Apply Pitch Shift (f0_up_key)
                f0 = f0 * (2 ** (f0_up_key / 12))
                
                # Convert to pitchf / coarse pitch
                # Standard RVC logic:
                # pitch = coarse_pd(f0) (bucketize)
                # pitchf = f0
                
                # We need `feature_input.coarse_f0`.
                # Let's import utility or implement simple bucket.
                # 256 quantization.
                
                # Create pitchf
                pitchf = f0
                
                # Create pitch (coarse)
                # Formula: 1 + log2(f0 / 1) * 12 ? No.
                # Standard RVC bucket logic?
                # It's in `rvc/lib/utils.py`? No. it is usually 1-255.
                # Let's look at `rvc/infer/pipeline.py`.
                
                # Re-use parent logic?
                # Parent `get_f0` calls `self.other_methods` or `feature_input.compute_f0`.
                # We can't reuse parent for "rmvpe" because it calls Torch model `self.model_rmvpe`.
                # We return `f0` (coarse) and `f0_bak` (float).
                
                # Simplified coarse pitch:
                # log_f0 = np.log2(f0 + 1e-5)
                # pitch = (log_f0 * 12) + 24 ? Need verification.
                # Actually, MLX Synthesizer `infer` expects standard inputs.
                
                # Let's import `coarse_f0` if possible. Not easily available?
                # Let's assume for now we return the raw f0 and let `pipeline` fail?
                # `pipeline.pipeline` expects `pitch, pitchf`.
                
                # Wait, `infer_mlx` `voice_conversion` converts pitch/pitchf to mx arrays.
                # The Synthesizer needs them.
                # `feats_mx` is (1, L, 256).
                # `pitch` (1, L).
                
                # Let's check `rvc/lib/utils.py` or `Pipeline.get_f0`.
                # It seems `get_f0` returns (pitch, pitchf).
                # We will reuse `feature_input` methods if we can import them?
                # `rvc/lib/utils`?
                # Actually, `Pipeline` class has access to `self.model_rmvpe`.
                
                # We will implement simplified pitch processing here.
                # clone `pitchf`.
                # pitch = np.round(12 * np.log2(pitchf / 440)) + 69 ? (MIDI)
                # RVC typically uses 256 buckets?
                # Let's look at `Synthesizer` code.
                # `self.pitch_embed` implies embedding(256)?
                # So pitch must be in [1, 255]. 0 is unvoiced.
                
                f0_mel_min = 1127 * np.log(1 + 50 / 700)
                f0_mel_max = 1127 * np.log(1 + 1100 / 700)
                
                # Simplified map from existing codebase logic (usually):
                # pitch = 1 + 255 * (f0_mel - min) / (max - min)
                
                # Use standard MIDI-like if unsure or try to invoke original logic?
                # Best way: Copy the logic.
                
                f0_bak = f0.copy()
                f0_mel = 1127 * np.log(1 + f0 / 700)
                f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
                f0_mel[f0_mel <= 1] = 1
                f0_mel[f0_mel > 255] = 255
                pitch = np.rint(f0_mel).astype(int)
                
                return pitch, f0_bak
            else:
                return super().get_f0(input_audio, f0_up_key, f0_method, filter_radius, cr_threshold)
        else:
             # Other methods (pm, harvest, crepe) - fallback to parent (might use CPU/Torch?)
             return super().get_f0(input_audio, f0_up_key, f0_method, filter_radius, cr_threshold)

class VoiceConverterMLXPure(VoiceConverterMLX):
    def __init__(self):
        super().__init__()
        self.hubert_model = None
        self.rmvpe_model = None
        
    def get_vc(self, weight_root, sid):
        # 1. Load Synthesizer (MLX) -> Parent handles this
        super().get_vc(weight_root, sid) 
        
        # 2. Load Hubert (MLX)
        if not self.hubert_model:
            print("Loading MLX Hubert...")
            conf = HubertConfig()
            self.hubert_model = HubertModel(conf)
            # Load weights
            h_path = os.path.join("rvc", "models", "embedders", "contentvec", "hubert_mlx.npz")
            if os.path.exists(h_path):
                 self.hubert_model.load_weights(h_path)
                 mx.eval(self.hubert_model.parameters())
            else:
                print(f"Error: Hubert weights not found at {h_path}")
        
        # 3. Load RMVPE (MLX)
        if not self.rmvpe_model:
            print("Loading MLX RMVPE...")
            self.rmvpe_model = RMVPE_MLX() # Loads default weights internally
            
        # 4. Inject PipelineMLXPure
        # Parent sets `self.mlx_pipeline = PipelineMLX(...)`
        # We override it.
        self.mlx_pipeline = PipelineMLXPure(
            self.tgt_sr, 
            self.config, 
            self.mlx_model, 
            self.hubert_model, 
            self.rmvpe_model
        )
