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
        # Delegate to self.mlx_pipeline instead of self.vc (Torch pipeline)
        # Note: self.vc is still initialized in super().get_vc with Torch model, which is fine (used for Hubert etc?)
        # `model` passed here is Hubert. `net_g` is Torch Generator (ignored by us).
        
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
