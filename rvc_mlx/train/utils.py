"""
Training Utilities for RVC MLX

Checkpoint management, learning rate scheduling, and other utilities.
"""

import os
import json
import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Any, Optional, Tuple
import numpy as np


class ExponentialLRScheduler:
    """
    Exponential learning rate scheduler.

    lr = initial_lr * decay^epoch
    """

    def __init__(self, initial_lr: float, decay: float = 0.999875):
        self.initial_lr = initial_lr
        self.decay = decay
        self.current_lr = initial_lr

    def step(self, epoch: int) -> float:
        """Update learning rate for epoch."""
        self.current_lr = self.initial_lr * (self.decay ** epoch)
        return self.current_lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr


def compute_grad_norm(grads: Dict[str, mx.array]) -> float:
    """Compute total gradient norm."""
    total_norm = 0.0
    for g in grads.values():
        if g is not None:
            total_norm += float(mx.sum(g ** 2).item())
    return np.sqrt(total_norm)


def clip_grad_norm(grads: Dict[str, mx.array], max_norm: float) -> Dict[str, mx.array]:
    """
    Clip gradient norm.

    Args:
        grads: Dictionary of gradients
        max_norm: Maximum norm

    Returns:
        Clipped gradients
    """
    total_norm = compute_grad_norm(grads)

    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        grads = {k: g * clip_coef if g is not None else g for k, g in grads.items()}

    return grads


def save_model_for_inference(
    net_g: nn.Module,
    output_path: str,
    config: Dict[str, Any],
    epoch: int = 0,
    step: int = 0,
    sample_rate: int = 40000,
):
    """
    Save model weights for inference (without discriminator and training state).

    Args:
        net_g: Generator model
        output_path: Output path (.npz)
        config: Model configuration
        epoch: Current epoch
        step: Current step
        sample_rate: Sample rate
    """
    # Get all parameters
    params = dict(net_g.parameters())

    # Filter out posterior encoder (enc_q) - only needed for training
    inference_params = {}
    for k, v in params.items():
        if not k.startswith("enc_q."):
            inference_params[k] = v

    # Save weights
    mx.savez(output_path, **inference_params)

    # Save metadata
    metadata = {
        "epoch": epoch,
        "step": step,
        "sr": sample_rate,
        "config": config,
        "f0": True,  # RVC uses F0
        "version": "mlx_v1",
    }

    meta_path = os.path.splitext(output_path)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved inference model to {output_path}")


def load_pretrained_weights(
    model: nn.Module,
    weights_path: str,
    strict: bool = False,
) -> Tuple[int, int]:
    """
    Load pretrained weights into model.

    Args:
        model: Model to load weights into
        weights_path: Path to weights file (.npz)
        strict: Raise error if keys don't match

    Returns:
        (loaded_count, skipped_count)
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # Load weights
    loaded_weights = dict(mx.load(weights_path))

    # Get model parameter keys
    model_params = dict(model.parameters())
    model_keys = set(model_params.keys())
    loaded_keys = set(loaded_weights.keys())

    # Find matching keys
    matched = model_keys & loaded_keys
    missing = model_keys - loaded_keys
    unexpected = loaded_keys - model_keys

    if strict and (missing or unexpected):
        raise ValueError(
            f"Weight mismatch. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )

    # Load matched weights
    weights_to_load = [(k, loaded_weights[k]) for k in matched]
    model.load_weights(weights_to_load)

    if missing:
        print(f"Warning: {len(missing)} parameters not found in checkpoint")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected keys in checkpoint")

    return len(matched), len(missing) + len(unexpected)


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False


class MovingAverage:
    """Exponential moving average for smoothing metrics."""

    def __init__(self, smoothing: float = 0.99):
        self.smoothing = smoothing
        self.value = None

    def update(self, new_value: float) -> float:
        """Update and return smoothed value."""
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.smoothing * self.value + (1 - self.smoothing) * new_value
        return self.value

    def get(self) -> Optional[float]:
        """Get current smoothed value."""
        return self.value
