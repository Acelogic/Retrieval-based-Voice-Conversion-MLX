import torch
import mlx.core as mx
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rvc.lib.algorithm.synthesizers import Synthesizer as SynthesizerPT
from rvc_mlx.lib.mlx.synthesizers import Synthesizer as SynthesizerMLX

def compare_synthesizer():
    print("Initializing Models...")
    
    # Paths
    pt_path = "/Users/mcruz/Library/Application Support/Replay/com.replay.Replay/models/Drake/model.pth"
    mlx_path = "rvc_mlx/models/checkpoints/Drake.npz"
    
    # Load PyTorch
    ckpt_pt = torch.load(pt_path, map_location="cpu")
    config_pt = ckpt_pt["config"]
    # Adjust config for PyTorch instantiation if needed
    # SynthesizerPT(dev, spk_emb_dim, 256, 32, 512, 192, 16000, 32000, 0, 1024, 32, 128, 512, 0.99, 7)
    # Actually it takes: (device, 256, 256, 512, 128, 16000, ...)
    # Let's verify init signature or use dynamic
    # PT RVC typically: Synthesizer(dev, 256, 256, 512, 128, sr, ...)
    
    # Parse config
    # Default RVC v2 params (fallback)
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
    upsample_rates = [10, 10, 2, 2] 
    upsample_initial_channel = 512
    upsample_kernel_sizes = [16, 16, 4, 4]
    spk_embed_dim = 109 
    gin_channels = 256
    sr = 40000 
    use_sdp = True # Usually True

    if isinstance(config_pt, list) and len(config_pt) >= 18:
        spec_channels = config_pt[0]
        segment_size = config_pt[1]
        inter_channels = config_pt[2]
        hidden_channels = config_pt[3]
        filter_channels = config_pt[4]
        n_heads = config_pt[5]
        n_layers = config_pt[6]
        kernel_size = config_pt[7]
        p_dropout = config_pt[8]
        resblock = config_pt[9]
        resblock_kernel_sizes = config_pt[10]
        resblock_dilation_sizes = config_pt[11]
        upsample_rates = config_pt[12]
        upsample_initial_channel = config_pt[13]
        upsample_kernel_sizes = config_pt[14]
        spk_embed_dim = config_pt[15]
        gin_channels = config_pt[16]
        sr = config_pt[17]
    elif isinstance(config_pt, dict):
        if "sampling_rate" in config_pt: sr = config_pt.get("sampling_rate", sr)
        # Add other dict mappings if needed, but we saw list config
        
    print(f"Config loaded: sr={sr}, hidden={hidden_channels}, flow_layers={n_layers}")

    net_pt = SynthesizerPT(
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
        # n_speakers, gin_channels, sr, use_f0, ...
        # Positionals based on definition in rvc/lib/algorithm/synthesizers.py
        spk_embed_dim, # n_speakers (109)
        gin_channels,
        sr,
        True, # use_f0
        # Optional args: text_enc_hidden_dim=768, vocoder="HiFi-GAN"
        # 256 if is_half else 768? No, text_enc_feature is 256 or 768.
        # If config is half, maybe 256? RVC usually 256 for v2?
        # Let's verify hidden_channels (192).
        # filter_channels (768).
        # We should check if text_enc_hidden_dim is needed. 
        # Default is 768.
    )
    # Reload state dict
    msg = net_pt.load_state_dict(ckpt_pt["weight"], strict=False)
    # print(msg)
    net_pt.eval()
    net_pt.remove_weight_norm()
    
    # Load MLX
    net_mlx = SynthesizerMLX(
        spec_channels=spec_channels,
        segment_size=segment_size,
        inter_channels=inter_channels,
        hidden_channels=hidden_channels,
        filter_channels=filter_channels, # Note: filter_channels vs inter_channels?
        n_heads=n_heads,
        n_layers=n_layers,
        kernel_size=kernel_size,
        p_dropout=p_dropout,
        resblock=resblock,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
        upsample_rates=upsample_rates,
        upsample_initial_channel=upsample_initial_channel,
        upsample_kernel_sizes=upsample_kernel_sizes,
        spk_embed_dim=spk_embed_dim, # Match MLX arg name
        gin_channels=gin_channels,
        sr=sr,
        use_f0=True,
        # use_sdp not in MLX init signature based on outline?
        # Let's check init outline again... 
        # Outline did not show use_sdp. But checked for **kwargs? 
    )
    # Load weights
    weights = mx.load(mlx_path)
    net_mlx.load_weights(list(weights.items()), strict=False)
    
    print("Models Loaded.")
    
    # Dummy Inputs
    B, T_phone = 1, 200
    phone = torch.randint(0, 100, (B, T_phone)).long() # Phone indices? Or Embeddings?
    # RVC TextEncoder expects embeddings (B, T, D) for MLX (via HuBERT)
    # PyTorch enc_p expects (B, T, D) too?
    # RVC uses HuBERT contentvec (B, T, 256 or 768).
    dim = 768 # Hubert dim
    phone_pt = torch.randn(B, T_phone, dim)
    phone_mlx = mx.array(phone_pt.numpy())
    
    phone_lengths_pt = torch.tensor([T_phone]).long()
    phone_lengths_mlx = mx.array([T_phone])
    
    pitch_pt = torch.randint(0, 255, (B, T_phone)).long()
    pitch_mlx = mx.array(pitch_pt.numpy())
    
    nsff0_pt = torch.randn(B, T_phone)
    nsff0_mlx = mx.array(nsff0_pt.numpy())
    
    sid_pt = torch.tensor([0]).long()
    sid_mlx = mx.array([0])
    
    print("Running Inference...")
    with torch.no_grad():
        # (phone, phone_lengths, pitch, nsff0, sid, rate)
        o_pt, _, _ = net_pt.infer(phone_pt, phone_lengths_pt, pitch_pt, nsff0_pt, sid_pt)
        # Output o_pt: (B, 1, T_audio).
        
    o_mlx, _, _ = net_mlx.infer(phone_mlx, phone_lengths_mlx, pitch_mlx, nsff0_mlx, sid_mlx)
    # Output o_mlx: (B, T_audio, 1) or (B, T, 1)?
    
    # Compare
    o_pt_np = o_pt.squeeze().numpy()
    o_mlx_np = np.array(o_mlx).squeeze()
    
    print("\nResults:")
    print(f"PT shape: {o_pt_np.shape}")
    print(f"MLX shape: {o_mlx_np.shape}")
    
    # Numerical
    diff = np.abs(o_pt_np - o_mlx_np)
    print(f"Max diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")
    print(f"RMSE: {np.sqrt((diff**2).mean()):.6f}")
    print(f"Corr: {np.corrcoef(o_pt_np, o_mlx_np)[0,1]:.6f}")
    
    pt_rms = np.sqrt((o_pt_np**2).mean())
    mlx_rms = np.sqrt((o_mlx_np**2).mean())
    print(f"PT RMS: {pt_rms:.6f}")
    print(f"MLX RMS: {mlx_rms:.6f}")
    print(f"Ratio: {mlx_rms/pt_rms:.6f}")

    # Spectrogram Correlation
    try:
        import librosa
        n_fft = 1024
        hop_length = 256
        S_pt = np.abs(librosa.stft(o_pt_np, n_fft=n_fft, hop_length=hop_length))
        S_mlx = np.abs(librosa.stft(o_mlx_np, n_fft=n_fft, hop_length=hop_length))
        
        mel_pt = librosa.feature.melspectrogram(S=S_pt**2, sr=sr, n_mels=80)
        mel_mlx = librosa.feature.melspectrogram(S=S_mlx**2, sr=sr, n_mels=80)
        
        log_mel_pt = librosa.power_to_db(mel_pt, ref=np.max)
        # Use ref=np.max of PT to keep relative scale? Or per-spectrogram?
        # Usually per-spectrogram to verify shape. 
        log_mel_mlx = librosa.power_to_db(mel_mlx, ref=np.max)
        
        spec_corr = np.corrcoef(log_mel_pt.flatten(), log_mel_mlx.flatten())[0, 1]
        print(f"Spectrogram Corr: {spec_corr:.6f}")
        
    except ImportError:
        print("Skipped Spectrogram Corr (librosa missing)")

if __name__ == "__main__":
    compare_synthesizer()
