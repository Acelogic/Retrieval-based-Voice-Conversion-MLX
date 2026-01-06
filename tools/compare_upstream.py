import torch
import mlx.core as mx
import numpy as np
import sys
import os
import soundfile as sf
import librosa

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# PyTorch Imports
from rvc.lib.utils import load_embedding
from rvc.lib.predictors.RMVPE import RMVPE0Predictor as RMVPE_PT
from rvc.configs.config import Config

# MLX Imports
from rvc_mlx.lib.mlx.hubert import HubertModel as HubertMLX
from rvc_mlx.lib.mlx.rmvpe import RMVPE0Predictor as RMVPE_MLX

def compare_upstream():
    print("Initializing Upstream Comparison...")
    
    # Load Audio
    audio_path = "test-audio/coder_audio_stock.wav"
    print(f"Loading {audio_path}...")
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1: audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio = audio.astype(np.float32)
    
    # ---------------------------------------------------------
    # HuBERT Comparison
    # ---------------------------------------------------------
    print("\n--- Comparing HuBERT ---")
    
    # PyTorch
    config = Config()
    hubert_pt = load_embedding("contentvec", None).to(config.device)
    hubert_pt.eval()
    
    audio_pt = torch.from_numpy(audio).float().to(config.device).unsqueeze(0)
    
    print("Running PyTorch HuBERT...")
    # Check config
    if hasattr(hubert_pt, "config"):
        print(f"PT Config hidden_act: {getattr(hubert_pt.config, 'hidden_act', 'Unknown')}")
    elif hasattr(hubert_pt, "cfg"):
        # Fairseq model style
        print(f"PT Cfg activation: {getattr(hubert_pt.cfg.model, 'activation_fn', 'Unknown')}")
        
    with torch.no_grad():
        # RVC Pipeline behavior:
        feats_pt = audio_pt
        # Pipeline logic expects (1, T). audio_pt is (1, T).
        # if feats_pt.dim() == 2: feats_pt = feats_pt.mean(-1) # This caused the bug!
        # logic in pipeline.py assumes input is (T, C). Ours is (1, T).
        pass
        
        # PyTorch RVC uses the contentvec model which returns a dict
        # feats = model(feats)["last_hidden_state"]
        padding_mask = torch.BoolTensor(feats_pt.shape).fill_(False).to(config.device)
        inputs = {
            "source": feats_pt,
            "padding_mask": padding_mask,
            "output_layer": 12, # 12 for contentvec
        }
        # Note: calling forward directly on Fairseq model usually takes source
        # But load_embedding return might be wrapper.
        # Checking implementation: typical usage in RVC:
        # feats = model.extract_features(**inputs)[0] 
        # But let's try direct call as per standard Fairseq model if possible.
        # However, MLX port logic suggests:
        # feats = model(audio)
        
        # Let's inspect typical RVC pipeline.py
        # feats = model(feats)["last_hidden_state"]
        # which implies model forward returns dict with last_hidden_state.
        
        out_pt = hubert_pt(feats_pt)
        feats_pt_out = out_pt["last_hidden_state"]
        
        
        print("Iterating PyTorch HuBERT keys...")
        # hubert_pt is likely a FairseqHubertModel
        # Let's print some keys
        for name, param in hubert_pt.named_parameters():
             if "final_proj" in name or "out" in name or "proj" in name:
                  print(f"PT Param: {name} {param.shape}")
        
        # MLX
        from rvc_mlx.lib.mlx.hubert import HubertModel, HubertConfig
        
        print("Loading MLX HuBERT...")
        conf = HubertConfig(classifier_proj_size=768)
        hubert_mlx = HubertModel(conf)
        
        # Check path
        h_path = "rvc_mlx/models/embedders/contentvec/hubert_mlx.npz"
        if os.path.exists(h_path):
             # hubert_mlx.load_weights(h_path, strict=False)
             # Let's inspect the NPZ file directly first
             npz_weights = mx.load(h_path)
             print("Keys in hubert_mlx.npz:")
             for k in npz_weights.keys():
                  if "pos_conv_embed" in k:
                      print(f"NPZ Key: {k} {npz_weights[k].shape}")
                      
             
             hubert_mlx.load_weights(list(npz_weights.items()), strict=False)
             hubert_mlx.eval() # CRITICAL: Disable Dropout!
             
             print("Iterating MLX HuBERT keys...")
             # Traverse recursive tree or flat dict if parameters() returns flat
             # MLX nn.Module.parameters() returns nested dict/tree.
             # Use `utils.tree_flatten` or just a recursive printer helper.
             
             def print_tree(tree, prefix=""):
                 if isinstance(tree, mx.array):
                     print(f"MLX Param: {prefix} {tree.shape}")
                     # Check statistics to see if it looks random or learned?
                     return
                 if isinstance(tree, dict):
                     for k, v in tree.items():
                         print_tree(v, f"{prefix}.{k}" if prefix else k)
             
             print_tree(hubert_mlx.parameters())
             
             # Specific check for pos_conv_embed
             # Access via model hierarchy
             if hasattr(hubert_mlx.encoder.pos_conv_embed, 'weight'):
                 w = hubert_mlx.encoder.pos_conv_embed.weight
                 print(f"Pos Conv Weight Shape: {w.shape}")
                 print(f"Pos Conv Weight Mean: {mx.mean(w).item()}")
                 print(f"Pos Conv Weight Std: {np.std(np.array(w))}")
             
             if hasattr(hubert_mlx.encoder.pos_conv_embed, 'bias'):
                 b = hubert_mlx.encoder.pos_conv_embed.bias
                 if b is not None:
                     print(f"Pos Conv Bias Shape: {b.shape}")
                 else:
                     print("Pos Conv Bias is None")
             
             print("Running MLX HuBERT Feature Extractor...")
             audio_mx = mx.array(audio)[None, :]
             
             # MLX Feats
             feats_mlx_cnn = hubert_mlx.feature_extractor(audio_mx)
             # MLX Conv1d returns (N, L, C). PyTorch usually (N, C, L).
             
             # PyTorch Feats
             print("Running PyTorch HuBERT Feature Extractor...")
             # hubert_pt.feature_extractor(audio_pt) -> returns (N, C, L) or (N, L, C)?
             # Transformers implementation returns (N, C, L) usually for convs.
             # Actually `hubert_pt.feature_extractor` is a module.
             
             feats_pt_cnn = hubert_pt.feature_extractor(audio_pt)
             
             # Dimensions check
             print(f"PT CNN Out: {feats_pt_cnn.shape}")
             print(f"MLX CNN Out: {feats_mlx_cnn.shape}")
             
             if feats_pt_cnn.shape[1] == 512: # (N, C, L)
                 feats_pt_cnn = feats_pt_cnn.transpose(1, 2) # -> (N, L, C)
             
             feats_pt_cnn_np = feats_pt_cnn.cpu().numpy()
             feats_mlx_cnn_np = np.array(feats_mlx_cnn)
             
             diff_cnn = np.abs(feats_pt_cnn_np - feats_mlx_cnn_np)
             print(f"CNN Max Diff: {diff_cnn.max():.6f}")
             print(f"CNN Mean Diff: {diff_cnn.mean():.6f}")
             print(f"CNN RMSE: {np.sqrt(np.mean(diff_cnn**2)):.6f}")
             print(f"CNN Corr: {np.corrcoef(feats_pt_cnn_np.flatten(), feats_mlx_cnn_np.flatten())[0, 1]:.6f}")
             
             
             print("Comparing Feature Projection Weights...")
             # PT
             pt_ln_w = hubert_pt.feature_projection.layer_norm.weight.detach().cpu().numpy()
             pt_proj_w = hubert_pt.feature_projection.projection.weight.detach().cpu().numpy()
             
             # MLX
             mlx_ln_w = np.array(hubert_mlx.feature_projection.layer_norm.weight)
             mlx_proj_w = np.array(hubert_mlx.feature_projection.projection.weight)
             
             print(f"LN Weight Diff: {np.abs(pt_ln_w - mlx_ln_w).max()}")
             print(f"Proj Weight Diff: {np.abs(pt_proj_w - mlx_proj_w).max()}")
             
             print("Comparing Feature Projection...")
             # PT
             # feats_pt_cnn is already (N, L, 512) thanks to check above
             feats_pt_proj = hubert_pt.feature_projection(feats_pt_cnn)
             
             # MLX
             # feats_mlx_cnn is (N, L, 512)
             feats_mlx_proj = hubert_mlx.feature_projection(feats_mlx_cnn)
             
             diff_proj = np.abs(feats_pt_proj.cpu().numpy() - np.array(feats_mlx_proj))
             print(f"Proj Max Diff: {diff_proj.max():.6f}")
             
             print("Comparing Positional Embedding...")
             # PT
             # hubert_pt.encoder.pos_conv_embed(hidden_states)
             # Input is features (N, L, 768).
             feats_pt_pos = hubert_pt.encoder.pos_conv_embed(feats_pt_proj)
             
             # MLX
             feats_mlx_pos = hubert_mlx.encoder.pos_conv_embed(feats_mlx_proj)
             
             diff_pos = np.abs(feats_pt_pos.cpu().numpy() - np.array(feats_mlx_pos))
             print(f"PosEmbed Max Diff: {diff_pos.max():.6f}")
             print(f"PosEmbed Mean Diff: {diff_pos.mean():.6f}")

             # If PosEmbed differs, check weights again or padding logic.
             
             print("Running MLX HuBERT Full...")
             feats_mlx_out = hubert_mlx(audio_mx)
             
             # Compare Final
             feats_pt_np = feats_pt_out.cpu().numpy()
             feats_mlx_np = np.array(feats_mlx_out)
             
             # ... continue comparison
             
             print(f"PT feats shape: {feats_pt_np.shape}")
             print(f"MLX feats shape: {feats_mlx_np.shape}")
             
             # Verify consistency
             diff = np.abs(feats_pt_np - feats_mlx_np)
             print(f"Max Diff: {diff.max():.6f}")
             print(f"Mean Diff: {diff.mean():.6f}")
             print(f"RMSE: {np.sqrt(np.mean(diff**2)):.6f}")
             
             # Correlation per feature
             # Flatten
             corr = np.corrcoef(feats_pt_np.flatten(), feats_mlx_np.flatten())[0, 1]
             print(f"Corr: {corr:.6f}")
        else:
             print(f"MLX HuBERT weights not found at {h_path}")
    
    
    # ---------------------------------------------------------
    # RMVPE Comparison
    # ---------------------------------------------------------
    print("\n--- Comparing RMVPE ---")
    
    # PyTorch
    rmvpe_pt_path = "rvc/models/predictors/rmvpe.pt"
    if not os.path.exists(rmvpe_pt_path):
        print(f"RMVPE PT model not found at {rmvpe_pt_path}")
    else:
        print("Loading PyTorch RMVPE...")
        rmvpe_pt = RMVPE_PT(rmvpe_pt_path, device=config.device)
        
        print("Running PyTorch RMVPE...")
        f0_pt = rmvpe_pt.infer_from_audio(audio, thred=0.03)
        # f0_pt is numpy array
        
        # MLX
        rmvpe_mlx_path = "rvc_mlx/models/predictors/rmvpe.mx" # Or similar?
        # The infer_mlx uses:
        # self.rmvpe = RMVPE(model_path)
        # Where is the generic MLX model path?
        # Default might be rvc_mlx/models/predictors/rmvpe.onnx transformed?
        # Actually in `infer_mlx.py` it loads from somewhere.
        # Let's assume standard path or check.
        # `rvc_mlx/lib/mlx/rmvpe.py`
        
        # We need to find valid MLX weights.
        # Benchmark loaded "Loading RMVPE..."
        
        print("Loading MLX RMVPE...")
        try:
             # Try standard location
             rmvpe_mlx = RMVPE_MLX("rvc_mlx/models/predictors/rmvpe_mlx.npz")
             # Or we need to convert it first?
             # infer_mlx.py: using RMVPE(model_path) which can be .pt
             
             print("Running MLX RMVPE...")
             f0_mlx = rmvpe_mlx.infer_from_audio(audio, thred=0.03)
             
             # Compare
             print(f"PT F0 shape: {f0_pt.shape}")
             print(f"MLX F0 shape: {f0_mlx.shape}")
             
             # F0 is 1D array (time steps / hop)
             min_len = min(len(f0_pt), len(f0_mlx))
             f0_pt = f0_pt[:min_len]
             f0_mlx = f0_mlx[:min_len]
             
             diff = np.abs(f0_pt - f0_mlx)
             print(f"Max Diff: {diff.max():.6f}")
             print(f"Mean Diff: {diff.mean():.6f}")
             print(f"RMSE: {np.sqrt(np.mean(diff**2)):.6f}")
             print(f"Corr: {np.corrcoef(f0_pt, f0_mlx)[0,1]:.6f}")
             
        except Exception as e:
            print(f"MLX RMVPE Error: {e}")

if __name__ == "__main__":
    compare_upstream()
