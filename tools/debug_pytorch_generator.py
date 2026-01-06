
import numpy as np
import torch
import sys
import os
import json

sys.path.append(os.getcwd())

# Direct import to avoid torchaudio
from rvc.lib.algorithm.generators.hifigan_nsf import HiFiGANNSFGenerator

def run_pytorch_generator(model_path, config_path):
    print(f"Loading {model_path}...")
    ckpt = torch.load(model_path, map_location="cpu")
    
    with open(config_path, 'r') as f:
        # Drake.json is usually a list
        config = json.load(f)
        
    # [spec, seg, inter, hidden, filter, heads, layers, kern, drop, res, res_k, res_d, ups, ups_init, ups_k, spk, gin, sr]
    # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
    
    inter_channels = config[2]
    resblock_kernel_sizes = config[10]
    resblock_dilation_sizes = config[11]
    upsample_rates = config[12]
    upsample_initial_channel = config[13]
    upsample_kernel_sizes = config[14]
    gin_channels = config[16]
    sr = config[17]
    
    print(f"Config: SR={sr}, Upsample={upsample_rates}")
    
    net_g = HiFiGANNSFGenerator(
        initial_channel=inter_channels,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
        upsample_rates=upsample_rates,
        upsample_initial_channel=upsample_initial_channel,
        upsample_kernel_sizes=upsample_kernel_sizes,
        gin_channels=gin_channels,
        sr=sr
    )
    
    # Load weights
    # ckpt is full checkpoint. Generator weights are in "weight" key for "dec" prefix?
    # No, usually "weight" key contains the generator state dict directly IF it's a generator checkpoint.
    # But RVC checkpoint contains {weight: {model_state}, config: ...}
    
    # Extract generator weights
    gen_state = {}
    if "weight" in ckpt:
        for k, v in ckpt["weight"].items():
            if k.startswith("dec."):
                gen_state[k[4:]] = v # Remove "dec." prefix
    
    # Load
    net_g.load_state_dict(gen_state, strict=True)
    net_g.eval()
    net_g.remove_weight_norm()
    
    # Dump first layer weight
    print(f"PyTorch dec.conv_pre.weight: {net_g.conv_pre.weight[0,0,0]}") # first element
    
    # Run Inference
    # input: x (inter_channels, L), f0 (1, L'), g (gin_channels, 1)
    
    L = 100
    x = torch.randn(1, inter_channels, L)
    # f0 length should be L * prod(upsample_rates)? 
    # forward(x, f0, g)
    # inside: m_source(f0, upp)
    # SineGen expects f0 length?
    # SineGen(x, u). x is f0.
    # It upsamples f0 by u.
    # So f0 should be length L?
    # Or L * ups?
    # RVC pipeline passes `nsff0` which is already upsampled?
    # In `Synthesizer.infer`:
    # z = flow(...)
    # o = dec(z, nsff0, g)
    # nsff0 is `pitchf`.
    # `input pitch` to pipeline is coarse.
    # `pitchf` is fine pitch per frame?
    # RVC `get_f0` returns f0 per frame (hop_size).
    # So f0 length = L.
    
    # Wait, `HiFiGANNSFGenerator.forward`:
    # har_source, _, _ = self.m_source(f0, self.upp)
    # `SourceModuleHnNSF.forward(x, upsample_factor)`:
    # `sine_wavs, ... = l_sin_gen(x, upsample_factor)`
    # `SineGenerator.forward(f0, upp)`:
    # `f0 = F.interpolate(f0, scale_factor=upp)`
    # So f0 input should be length L.
    
    f0 = torch.full((1, L), 440.0).float()
    g = torch.randn(1, gin_channels, 1)
    
    with torch.no_grad():
        out = net_g(x, f0, g)
        
    print(f"Output shape: {out.shape}")
    print(f"Output stats: {out.mean()}, {out.std()}")
    
    # Save inputs for MLX comparison
    np.savez("test_inputs.npz", x=x.numpy(), f0=f0.numpy(), g=g.numpy())
    np.save("pytorch_output.npy", out.numpy())
    print("Saved test_inputs.npz and pytorch_output.npy")

if __name__ == "__main__":
    run_pytorch_generator(
        "/Users/mcruz/Library/Application Support/Replay/com.replay.Replay/models/Drake/model.pth", 
        "rvc_mlx/models/checkpoints/Drake.json"
    )
