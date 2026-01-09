#!/usr/bin/env python3
"""Debug NaN that appears on training step 2 - with fixes."""

import sys
sys.path.insert(0, '/Users/mcruz/Developer/Retrieval-based-Voice-Conversion-MLX')

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from rvc_mlx.lib.mlx.synthesizers import Synthesizer
from rvc_mlx.train.discriminators import MultiPeriodDiscriminator
from rvc_mlx.train.losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from rvc_mlx.train.mel_processing import spectrogram
from rvc_mlx.train.data_loader import create_dataloader
from rvc_mlx.lib.mlx.commons import slice_segments

# Config
SEGMENT_SIZE = 32
HOP_LENGTH = 400
N_FFT = 2048
WIN_LENGTH = 2048
SPEC_CHANNELS = N_FFT // 2 + 1
KL_SCALE = 0.01  # Scale down KL loss to prevent gradient explosion
MAX_GRAD_NORM = 1.0

def clip_gradients(grads, max_norm, max_grad_value=1e3):
    """
    Clip gradient norm to max_norm.

    First replaces NaN/Inf and clamps extreme values, then scales by norm if needed.
    """
    # First pass: sanitize and clamp extreme values
    all_grads = []
    inf_grad_paths = []

    def sanitize_and_collect(g, path=""):
        if isinstance(g, dict):
            result = {}
            for k, v in g.items():
                result[k] = sanitize_and_collect(v, f"{path}.{k}")
            return result
        elif hasattr(g, 'shape'):
            mx.eval(g)
            # Check for inf/nan before clamping
            has_inf = mx.isinf(g).any().item()
            has_nan = mx.isnan(g).any().item()
            if has_inf or has_nan:
                inf_grad_paths.append(path)
            # Replace inf with max_value, nan with 0
            g_safe = mx.where(mx.isnan(g), mx.zeros_like(g), g)
            g_safe = mx.where(mx.isinf(g) & (g > 0), mx.full(g.shape, max_grad_value), g_safe)
            g_safe = mx.where(mx.isinf(g) & (g < 0), mx.full(g.shape, -max_grad_value), g_safe)
            # Then clamp
            g_clamped = mx.clip(g_safe, -max_grad_value, max_grad_value)
            mx.eval(g_clamped)
            all_grads.append(g_clamped)
            return g_clamped
        return g

    grads_sanitized = sanitize_and_collect(grads)

    if inf_grad_paths:
        print(f"  WARNING: Inf/NaN gradients in: {inf_grad_paths[:5]}{'...' if len(inf_grad_paths) > 5 else ''}")

    # Compute total norm from sanitized gradients
    total_norm_sq = 0.0
    for g in all_grads:
        total_norm_sq += mx.sum(g ** 2).item()
    total_norm = total_norm_sq ** 0.5

    # Scale if needed
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        def scale_grad(g):
            if isinstance(g, dict):
                return {k: scale_grad(v) for k, v in g.items()}
            elif hasattr(g, 'shape'):
                return g * clip_coef
            return g
        return scale_grad(grads_sanitized), total_norm
    return grads_sanitized, total_norm

def run_debug():
    print("=== Training Debug with Fixes ===\n")

    # Load models
    print("Loading models...")
    generator = Synthesizer(
        spec_channels=SPEC_CHANNELS,
        segment_size=SEGMENT_SIZE,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.0,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[10, 10, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16, 4, 4],
        spk_embed_dim=109,
        gin_channels=256,
        sr=40000,
        use_f0=True,
        text_enc_hidden_dim=768,
        vocoder_type="hifigan-nsf",
    )

    discriminator = MultiPeriodDiscriminator(version="v2")

    # Load pretrained weights
    print("Loading pretrained weights...")
    g_weights = mx.load("/Users/mcruz/Developer/Retrieval-based-Voice-Conversion-MLX/weights/f0G40k.npz")
    d_weights = mx.load("/Users/mcruz/Developer/Retrieval-based-Voice-Conversion-MLX/weights/f0D40k.npz")

    generator.load_weights(list(g_weights.items()), strict=False)
    discriminator.load_weights(list(d_weights.items()), strict=False)
    mx.eval(generator.parameters(), discriminator.parameters())
    print("  Weights loaded")

    # Freeze encoder
    generator.enc_p.freeze()
    print("  Encoder frozen")

    # Setup optimizers with lower learning rate
    lr_g = 1e-4
    lr_d = 1e-4
    optimizer_g = optim.AdamW(learning_rate=lr_g, betas=[0.8, 0.99], eps=1e-9, weight_decay=0.01)
    optimizer_d = optim.AdamW(learning_rate=lr_d, betas=[0.8, 0.99], eps=1e-9, weight_decay=0.01)
    print(f"  LR: G={lr_g}, D={lr_d}")
    print(f"  KL Scale: {KL_SCALE}")
    print(f"  Max Grad Norm: {MAX_GRAD_NORM}")

    # Load data
    print("\nLoading data...")
    dataloader = create_dataloader(
        "/Users/mcruz/Developer/Retrieval-based-Voice-Conversion-MLX/logs/billy_joel_test/filelist.txt",
        batch_size=2,
        shuffle=True,
        max_frames=200,
        hop_length=HOP_LENGTH,
        use_precomputed_spec=True,
    )

    # Define loss functions
    def compute_g_loss(generator, discriminator, batch):
        """Compute generator loss."""
        phone = batch["phone"]
        phone_lengths = batch["phone_lengths"]
        pitch = batch["pitch"]
        pitchf = batch["pitchf"]
        spec = batch["spec"]
        spec_lengths = batch["spec_lengths"]
        wave = batch["wave"]
        sid = batch["sid"]

        # Forward pass through generator
        (o, ids_slice, x_mask, y_mask,
         (z, z_p, m_p, logs_p, m_q, logs_q)) = generator.forward(
            phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid
        )

        # Compute mel spectrogram of generated audio
        mel_y_hat = spectrogram(o, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, center=True)
        mel_y_hat = mel_y_hat.transpose(0, 2, 1)  # (B, T, C) -> (B, C, T)

        # Get ground truth mel - stop gradient on indices
        ids_slice_sg = mx.stop_gradient(ids_slice)
        audio_segment_size = SEGMENT_SIZE * HOP_LENGTH
        y_slice = slice_segments(wave, ids_slice_sg * HOP_LENGTH, audio_segment_size, time_first=True)
        mel_y = spectrogram(y_slice, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, center=True)
        mel_y = mel_y.transpose(0, 2, 1)  # (B, T, C) -> (B, C, T)

        # Mel loss
        loss_mel = mx.abs(mel_y - mel_y_hat).mean() * 45

        # KL loss (scaled down to prevent gradient explosion)
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, x_mask) * KL_SCALE

        # Format audio for discriminator - stop gradient on ground truth
        y_hat_d = o  # (B, T, 1)
        y_d = mx.stop_gradient(y_slice[:, :, None])

        # Discriminator outputs
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = discriminator(y_d, y_hat_d)

        # Adversarial loss
        loss_gen = generator_loss(y_d_gs)

        # Feature matching loss
        loss_fm = feature_loss(fmap_rs, fmap_gs)

        # Total generator loss
        loss_g = loss_gen + loss_fm + loss_mel + loss_kl

        return loss_g, (loss_gen, loss_fm, loss_mel, loss_kl)

    def compute_d_loss(discriminator, y_real, y_fake):
        """Compute discriminator loss."""
        y_d = y_real[:, :, None]  # (B, T, 1)
        y_hat_d = y_fake[:, :, None]

        y_d_rs, y_d_gs, _, _ = discriminator(y_d, y_hat_d)

        loss_d = discriminator_loss(y_d_rs, y_d_gs)
        return loss_d

    def check_params_for_nan(model, name="model"):
        """Check for NaN/Inf in model parameters."""
        for key, val in model.parameters().items():
            if isinstance(val, dict):
                for k2, v2 in val.items():
                    if hasattr(v2, 'shape'):
                        mx.eval(v2)
                        if mx.isnan(v2).any().item():
                            return f"{name}.{key}.{k2}"
                        if mx.isinf(v2).any().item():
                            return f"{name}.{key}.{k2} (inf)"
            elif hasattr(val, 'shape'):
                mx.eval(val)
                if mx.isnan(val).any().item():
                    return f"{name}.{key}"
                if mx.isinf(val).any().item():
                    return f"{name}.{key} (inf)"
        return None

    # Training loop
    print("\n=== Starting Training ===")
    for step_idx, batch in enumerate(dataloader):
        if step_idx >= 50:  # Run 50 steps
            break

        print(f"\nStep {step_idx + 1}/50")

        # Check params before step
        nan_param = check_params_for_nan(generator, "generator")
        if nan_param:
            print(f"  BEFORE: NaN in {nan_param}")
            break

        # === Generator step ===
        loss_fn_g = nn.value_and_grad(generator, lambda gen: compute_g_loss(gen, discriminator, batch))
        (loss_g, aux), grads_g = loss_fn_g(generator)
        loss_gen, loss_fm, loss_mel, loss_kl = aux
        mx.eval(loss_g)

        print(f"  Loss: total={loss_g.item():.3f} (gen={loss_gen.item():.3f}, fm={loss_fm.item():.3f}, mel={loss_mel.item():.3f}, kl={loss_kl.item():.3f})")

        # Clip gradients
        grads_g, grad_norm_g = clip_gradients(grads_g, MAX_GRAD_NORM)

        # Update generator
        optimizer_g.update(generator, grads_g)
        mx.eval(generator.parameters(), optimizer_g.state)

        # Check params after update
        nan_param = check_params_for_nan(generator, "generator")
        if nan_param:
            print(f"  AFTER UPDATE: NaN in {nan_param}")
            break

        # === Discriminator step ===
        phone = batch["phone"]
        phone_lengths = batch["phone_lengths"]
        pitch = batch["pitch"]
        pitchf = batch["pitchf"]
        spec = batch["spec"]
        spec_lengths = batch["spec_lengths"]
        wave = batch["wave"]
        sid = batch["sid"]

        # Generate audio with updated generator
        (o_new, ids_slice_new, _, _, _) = generator.forward(
            phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid
        )
        o_new = mx.stop_gradient(o_new)
        mx.eval(o_new)

        # Slice ground truth
        audio_segment_size = SEGMENT_SIZE * HOP_LENGTH
        y_slice_new = slice_segments(wave, ids_slice_new * HOP_LENGTH, audio_segment_size, time_first=True)
        mx.eval(y_slice_new)

        # Check for NaN
        if mx.isnan(o_new).any().item():
            print("  ERROR: NaN in generated audio!")
            break

        loss_fn_d = nn.value_and_grad(discriminator, lambda disc: compute_d_loss(disc, y_slice_new, o_new.squeeze(-1)))
        loss_d, grads_d = loss_fn_d(discriminator)
        mx.eval(loss_d)

        # Clip gradients
        grads_d, grad_norm_d = clip_gradients(grads_d, MAX_GRAD_NORM)

        # Update discriminator
        optimizer_d.update(discriminator, grads_d)
        mx.eval(discriminator.parameters(), optimizer_d.state)

        print(f"  G: loss={loss_g.item():.3f} (gen={loss_gen.item():.3f}, fm={loss_fm.item():.3f}, mel={loss_mel.item():.3f}, kl={loss_kl.item():.3f}), grad={grad_norm_g:.2f}")
        print(f"  D: loss={loss_d.item():.3f}, grad={grad_norm_d:.2f}")

    print("\n=== Training Complete ===")

if __name__ == "__main__":
    run_debug()
