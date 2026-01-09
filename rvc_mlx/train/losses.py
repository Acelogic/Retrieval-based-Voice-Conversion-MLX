"""
Loss Functions for RVC MLX Training

MLX implementations of RVC loss functions.
"""

import mlx.core as mx
from typing import List, Tuple, Optional


def feature_loss(fmap_r: List[List[mx.array]], fmap_g: List[List[mx.array]]) -> mx.array:
    """
    Compute feature matching loss between real and generated feature maps.

    Args:
        fmap_r: List of feature maps from discriminator on real audio
        fmap_g: List of feature maps from discriminator on generated audio

    Returns:
        Feature matching loss value
    """
    loss = mx.array(0.0)
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss = loss + mx.mean(mx.abs(rl - gl))
    return loss * 2


def discriminator_loss(
    disc_real_outputs: List[mx.array],
    disc_generated_outputs: List[mx.array],
) -> mx.array:
    """
    Compute discriminator loss for real and generated outputs.

    Uses least-squares GAN loss.

    Args:
        disc_real_outputs: Discriminator outputs for real samples
        disc_generated_outputs: Discriminator outputs for generated samples

    Returns:
        Total discriminator loss
    """
    loss = mx.array(0.0)
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = mx.mean((1 - dr) ** 2)
        g_loss = mx.mean(dg ** 2)
        loss = loss + r_loss + g_loss
    return loss


def generator_loss(disc_outputs: List[mx.array]) -> mx.array:
    """
    Compute generator adversarial loss.

    Uses least-squares GAN loss.

    Args:
        disc_outputs: Discriminator outputs for generated samples

    Returns:
        Total generator loss
    """
    loss = mx.array(0.0)
    for dg in disc_outputs:
        loss = loss + mx.mean((1 - dg) ** 2)
    return loss


def kl_loss(
    z_p: mx.array,
    logs_q: mx.array,
    m_p: mx.array,
    logs_p: mx.array,
    z_mask: mx.array,
) -> mx.array:
    """
    Compute KL divergence loss between posterior and prior distributions.

    Args:
        z_p: Latent variable from flow (B, T, C) or (B, C, T)
        logs_q: Log variance of posterior (B, T, C) or (B, C, T)
        m_p: Mean of prior (B, T, C) or (B, C, T)
        logs_p: Log variance of prior (B, T, C) or (B, C, T)
        z_mask: Mask for valid positions (B, T, 1) or (B, 1, T)

    Returns:
        KL divergence loss
    """
    # Handle length mismatches by truncating to minimum length
    # This can happen due to different time resolutions between encoders
    min_len = min(z_p.shape[-1], logs_q.shape[-1], m_p.shape[-1], logs_p.shape[-1])
    z_p = z_p[..., :min_len]
    logs_q = logs_q[..., :min_len]
    m_p = m_p[..., :min_len]
    logs_p = logs_p[..., :min_len]
    if z_mask is not None:
        z_mask = z_mask[..., :min_len]

    # KL(q||p) = log(p) - log(q) - 0.5 + 0.5 * (z_p - m_p)^2 * exp(-2*log_p)
    # Clamp logs_p to prevent exp(-2*logs_p) from exploding when logs_p is very negative
    logs_p_clamped = mx.clip(logs_p, -10.0, 10.0)
    logs_q_clamped = mx.clip(logs_q, -10.0, 10.0)

    kl = logs_p_clamped - logs_q_clamped - 0.5 + 0.5 * ((z_p - m_p) ** 2) * mx.exp(-2.0 * logs_p_clamped)
    if z_mask is not None:
        kl = mx.sum(kl * z_mask)
        loss = kl / (mx.sum(z_mask) + 1e-8)  # Add epsilon to prevent division by zero
    else:
        loss = mx.mean(kl)

    # Clamp final loss to prevent extreme values
    loss = mx.clip(loss, -1000.0, 1000.0)
    return loss


def mel_loss(mel_real: mx.array, mel_gen: mx.array) -> mx.array:
    """
    Compute L1 loss on mel spectrograms.

    Args:
        mel_real: Real mel spectrogram
        mel_gen: Generated mel spectrogram

    Returns:
        L1 loss
    """
    return mx.mean(mx.abs(mel_real - mel_gen))


def discriminator_loss_scaled(
    disc_real: List[mx.array],
    disc_fake: List[mx.array],
    scale: float = 1.0,
) -> mx.array:
    """
    Compute scaled discriminator loss.

    Applies scaling to losses beyond the midpoint for multi-resolution discriminators.

    Args:
        disc_real: Discriminator outputs for real samples
        disc_fake: Discriminator outputs for generated samples
        scale: Scaling factor for second half of discriminators

    Returns:
        Total scaled discriminator loss
    """
    midpoint = len(disc_real) // 2
    loss = mx.array(0.0)

    for i, (d_real, d_fake) in enumerate(zip(disc_real, disc_fake)):
        real_loss = mx.mean((1 - d_real) ** 2)
        fake_loss = mx.mean(d_fake ** 2)
        total_loss = real_loss + fake_loss

        if i >= midpoint:
            total_loss = total_loss * scale

        loss = loss + total_loss

    return loss


def generator_loss_scaled(
    disc_outputs: List[mx.array],
    scale: float = 1.0,
) -> mx.array:
    """
    Compute scaled generator loss.

    Args:
        disc_outputs: Discriminator outputs for generated samples
        scale: Scaling factor for second half of discriminators

    Returns:
        Total scaled generator loss
    """
    midpoint = len(disc_outputs) // 2
    loss = mx.array(0.0)

    for i, d_fake in enumerate(disc_outputs):
        loss_value = mx.mean((1 - d_fake) ** 2)

        if i >= midpoint:
            loss_value = loss_value * scale

        loss = loss + loss_value

    return loss
