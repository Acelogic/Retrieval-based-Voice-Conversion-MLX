
import os
import sys
import torch
import numpy as np
import soundfile as sf
import librosa

# Add path
sys.path.append(os.getcwd())

# Import PyTorch implementation (Using local rvc codebase)
from rvc.lib.algorithm.synthesizers import Synthesizer

def run_pytorch(model_path, input_audio, output_path):
    print(f"[PyTorch] Loading {model_path}...")
    ckpt = torch.load(model_path, map_location="cpu")
    
    # Config
    if "config" in ckpt:
        config_list = ckpt["config"]
    else:
        print("[PyTorch] No config (list) in ckpt, likely fails.")
        return

    # Map list to kwargs
    kwargs = {
        "spec_channels": config_list[0],
        "segment_size": config_list[1],
        "inter_channels": config_list[2],
        "hidden_channels": config_list[3],
        "filter_channels": config_list[4],
        "n_heads": config_list[5],
        "n_layers": config_list[6],
        "kernel_size": config_list[7],
        "p_dropout": config_list[8],
        "resblock": config_list[9],
        "resblock_kernel_sizes": config_list[10],
        "resblock_dilation_sizes": config_list[11],
        "upsample_rates": config_list[12],
        "upsample_initial_channel": config_list[13],
        "upsample_kernel_sizes": config_list[14],
        "spk_embed_dim": config_list[15],
        "gin_channels": config_list[16],
        "sr": config_list[17],
        "use_f0": True, # Always True for RVC?
        "text_enc_hidden_dim": 768, # Default for V2
        "vocoder": "HiFi-GAN"
    }

    print(f"[PyTorch] Initializing Synthesizer with SR={kwargs['sr']}...")
    net_g = Synthesizer(**kwargs)
         
    # Load weights
    net_g.load_state_dict(ckpt["weight"], strict=False)
    net_g.eval()
    net_g.remove_weight_norm()
    
    print("[PyTorch] Model loaded.")
    
    print("[PyTorch] Generating random test tone...")
    # x (Phone): (B, 768, L) for V2
    # f0: (B, L)
    # sid: (B,)
    
    L = 200
    x = torch.randn(1, 200, 768) # PyTorch TextEncoder (linear) expects (B, L, Dim) -> No, linear(768, 192).
    # wait. TextEncoder(channel, ..., n_layers, ...)
    # input `phone`. `emb_phone` is Linear.
    # So input must be (B, L, 768).
    
    f0 = torch.full((1, L), 440.0)
    sid = torch.tensor([0]).long()
    lengths = torch.tensor([L]).long()
    
    with torch.no_grad():
         # pitch (coarse), nsff0 (fine)
         pitchf = f0
         # coarse mock
         pitch = torch.full((1, L), 100).long()
         
         # infer(phone, phone_lengths, pitch, nsff0, sid, rate)
         audio, _, _ = net_g.infer(x, lengths, pitch, pitchf, sid)
         # output shape (1, T, 1) or (1, T)? 
         # infer returns `o, x_mask, ...`
         # `o` is `self.dec(...)`. Dec output (B, 1, T).
         
    audio_np = audio.squeeze().numpy()
    print(f"[PyTorch] Output stats: min={audio_np.min()}, max={audio_np.max()}, mean={audio_np.mean()}")
    sf.write(output_path, audio_np, kwargs['sr'])
    print(f"[PyTorch] Saved to {output_path}")

if __name__ == "__main__":
    run_pytorch(sys.argv[1], None, "test-audio/pytorch_output.wav")
