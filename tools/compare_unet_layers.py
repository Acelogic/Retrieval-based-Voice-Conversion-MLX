#!/usr/bin/env python3
"""
Compare UNet/CNN layer outputs between PyTorch and MLX.
"""

import sys
import os
import torch
import librosa
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "rvc")

# Force reload
for mod in list(sys.modules.keys()):
    if 'rvc_mlx' in mod or 'rvc.lib' in mod:
        del sys.modules[mod]

from rvc.lib.predictors.RMVPE import RMVPE0Predictor as PyTorchRMVPE
from rvc_mlx.lib.mlx.rmvpe import RMVPE0Predictor as MLXRMVPE

def compare_tensors(name, mlx_tensor, pt_tensor):
    """Compare MLX and PyTorch tensors."""
    mlx_np = np.array(mlx_tensor) if isinstance(mlx_tensor, mx.array) else mlx_tensor
    pt_np = pt_tensor.cpu().numpy() if isinstance(pt_tensor, torch.Tensor) else pt_tensor

    # Handle batch dimension
    if mlx_np.ndim > pt_np.ndim:
        mlx_np = mlx_np[0]
    elif pt_np.ndim > mlx_np.ndim:
        pt_np = pt_np[0]

    # Match shapes
    min_shape = tuple(min(m, p) for m, p in zip(mlx_np.shape, pt_np.shape))
    slices = tuple(slice(0, s) for s in min_shape)
    mlx_crop = mlx_np[slices]
    pt_crop = pt_np[slices]

    diff = np.abs(mlx_crop - pt_crop)

    print(f"\n{name}:")
    print(f"  MLX shape: {mlx_np.shape}, PT shape: {pt_np.shape}")
    print(f"  MLX range: [{mlx_np.min():.6f}, {mlx_np.max():.6f}]")
    print(f"  PT range:  [{pt_np.min():.6f}, {pt_np.max():.6f}]")
    print(f"  Max diff: {diff.max():.6f}, Mean diff: {diff.mean():.6f}")
    print(f"  RMSE: {np.sqrt(np.mean((mlx_crop - pt_crop) ** 2)):.6f}")

    if diff.max() > 0.01:
        print(f"  ⚠️  SIGNIFICANT DIFFERENCE!")
        return False
    else:
        print(f"  ✅ Match")
        return True

def main():
    print("=== Comparing UNet/CNN Layer Outputs ===\n")

    # Load audio
    audio, sr = librosa.load("test-audio/coder_audio_stock.wav", sr=16000, mono=True)
    audio = audio[:sr // 2]  # 0.5 seconds for faster testing

    # Initialize models
    mlx_predictor = MLXRMVPE()
    pt_predictor = PyTorchRMVPE("rvc/models/predictors/rmvpe.pt", device="cpu")

    # Get mel spectrograms
    mel_mlx = mlx_predictor.mel_spectrogram(audio)
    audio_torch = torch.from_numpy(audio).float().unsqueeze(0)
    mel_pt = pt_predictor.mel_extractor(audio_torch, center=True)

    compare_tensors("Mel Spectrogram", mel_mlx, mel_pt)

    # Prepare MLX input
    n_frames = mel_mlx.shape[-1]
    pad_curr = 32 * ((n_frames - 1) // 32 + 1) - n_frames
    mel_padded_mlx = mx.pad(mel_mlx, ((0, 0), (0, pad_curr)), mode='constant')
    mel_input_mlx = mel_padded_mlx.transpose(1, 0)[None, :, :, None]

    # Prepare PT input
    mel_input_pt = mel_pt.unsqueeze(1).transpose(2, 3)

    compare_tensors("Mel Input (after reshape)", mel_input_mlx, mel_input_pt)

    # === Encoder ===
    print("\n--- ENCODER ---")
    model_mlx = mlx_predictor.model
    model_pt = pt_predictor.model

    with torch.no_grad():
        # Encoder BatchNorm input
        x_mlx = model_mlx.unet.encoder.bn(mel_input_mlx)
        x_pt = model_pt.unet.encoder.bn(mel_input_pt)
        compare_tensors("Encoder BN output", x_mlx, x_pt)

        # Encoder layers
        concat_mlx = []
        concat_pt = []
        for i, (layer_mlx, layer_pt) in enumerate(zip(model_mlx.unet.encoder.layers, model_pt.unet.encoder.layers)):
            t_mlx, x_mlx = layer_mlx(x_mlx)
            t_pt, x_pt = layer_pt(x_pt)
            concat_mlx.append(t_mlx)
            concat_pt.append(t_pt)
            compare_tensors(f"Encoder Layer {i} skip", t_mlx, t_pt)
            compare_tensors(f"Encoder Layer {i} output", x_mlx, x_pt)

    # === Intermediate ===
    print("\n--- INTERMEDIATE ---")
    with torch.no_grad():
        x_mlx_inter = model_mlx.unet.intermediate(x_mlx)
        x_pt_inter = model_pt.unet.intermediate(x_pt)
        compare_tensors("Intermediate output", x_mlx_inter, x_pt_inter)

    # === Decoder ===
    print("\n--- DECODER ---")
    with torch.no_grad():
        x_mlx_dec = model_mlx.unet.decoder(x_mlx_inter, concat_mlx)
        x_pt_dec = model_pt.unet.decoder(x_pt_inter, concat_pt)
        compare_tensors("Decoder output", x_mlx_dec, x_pt_dec)

    # === CNN ===
    print("\n--- CNN ---")
    with torch.no_grad():
        x_mlx_cnn = model_mlx.cnn(x_mlx_dec)
        x_pt_cnn = model_pt.cnn(x_pt_dec)
        compare_tensors("CNN output", x_mlx_cnn, x_pt_cnn)

    # === Reshape for BiGRU ===
    print("\n--- RESHAPE ---")
    x_mlx_reshape = x_mlx_cnn.transpose(0, 1, 3, 2)
    B, T, C, M = x_mlx_reshape.shape
    cnn_out_mlx = x_mlx_reshape.reshape(B, T, C * M)

    x_pt_reshape = x_pt_cnn.transpose(1, 2).flatten(-2)
    compare_tensors("CNN reshaped", cnn_out_mlx, x_pt_reshape)

    # === BiGRU ===
    print("\n--- BIGRU ---")
    with torch.no_grad():
        bigru_out_mlx = model_mlx.fc.bigru(cnn_out_mlx)
        bigru_out_pt = model_pt.fc[0](x_pt_reshape)
        compare_tensors("BiGRU output", bigru_out_mlx, bigru_out_pt)

    # === Linear ===
    print("\n--- LINEAR ---")
    with torch.no_grad():
        linear_out_mlx = model_mlx.fc.linear(bigru_out_mlx)
        linear_out_pt = model_pt.fc[1](bigru_out_pt)
        compare_tensors("Linear output", linear_out_mlx, linear_out_pt)

    # === Sigmoid ===
    print("\n--- SIGMOID ---")
    with torch.no_grad():
        sigmoid_out_mlx = model_mlx.sigmoid(linear_out_mlx)
        sigmoid_out_pt = model_pt.fc[3](linear_out_pt)
        compare_tensors("Sigmoid output", sigmoid_out_mlx, sigmoid_out_pt)

if __name__ == "__main__":
    main()
