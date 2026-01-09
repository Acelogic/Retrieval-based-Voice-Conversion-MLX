"""
RVC Trainer for MLX

Main training orchestrator using MLX's value_and_grad for efficient training.
"""

import os
import time
import json
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.optimizers import clip_grad_norm
import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from tqdm import tqdm
import math

from .losses import feature_loss, discriminator_loss, generator_loss, kl_loss, mel_loss
from .discriminators import MultiPeriodDiscriminator
from .mel_processing import mel_spectrogram, get_mel_config
from .data_loader import DataLoader, create_dataloader
from .overtraining_detector import OvertrainingDetector, OvertTrainingStatus


def sanitize_gradients(grads: Dict, max_grad_value: float = 1e3) -> Tuple[Dict, bool]:
    """
    Sanitize gradients by replacing NaN/Inf with finite values.

    MLX's clip_grad_norm doesn't handle inf values properly - when any gradient
    contains inf, the norm computation overflows and all gradients get zeroed.
    This function replaces inf/nan with clamped finite values first.

    Args:
        grads: Nested dict of gradient arrays
        max_grad_value: Maximum allowed gradient value magnitude

    Returns:
        Tuple of (sanitized gradients, had_inf_or_nan flag)
    """
    had_issues = False

    def sanitize(g):
        nonlocal had_issues
        if isinstance(g, dict):
            return {k: sanitize(v) for k, v in g.items()}
        elif hasattr(g, 'shape'):
            mx.eval(g)
            # Check for issues
            has_inf = mx.isinf(g).any().item()
            has_nan = mx.isnan(g).any().item()
            if has_inf or has_nan:
                had_issues = True
                # Replace nan with 0, inf with max_value
                g_safe = mx.where(mx.isnan(g), mx.zeros_like(g), g)
                g_safe = mx.where(mx.isinf(g) & (g > 0), mx.full(g.shape, max_grad_value), g_safe)
                g_safe = mx.where(mx.isinf(g) & (g < 0), mx.full(g.shape, -max_grad_value), g_safe)
                mx.eval(g_safe)
                return g_safe
            # Clamp extreme values even if not inf
            g_clamped = mx.clip(g, -max_grad_value, max_grad_value)
            mx.eval(g_clamped)
            return g_clamped
        return g

    return sanitize(grads), had_issues


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    sample_rate: int = 40000
    hop_length: int = 320
    segment_size: int = 12800  # Samples per training segment

    # Training
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.8, 0.99)
    eps: float = 1e-9
    lr_decay: float = 0.999875
    batch_size: int = 4
    epochs: int = 200

    # Fine-tuning (lower lr for pretrained models)
    finetune_lr_scale: float = 1.0  # Scale learning rate (1.0 = no scaling, as in successful earlier run)
    is_finetuning: bool = False  # Set to True when loading pretrained weights

    # Freeze encoder during fine-tuning (CRITICAL for voice conversion)
    # The text encoder (enc_p) should remain frozen to preserve pretrained representations
    # Only train decoder, flow, and posterior encoder
    freeze_encoder: bool = True  # Freeze enc_p during fine-tuning

    # Discriminator learning rate scale
    # Lower values slow discriminator learning to prevent it from dominating (common: 0.5-1.0)
    d_lr_scale: float = 0.2  # Discriminator LR = learning_rate * d_lr_scale

    # Discriminator update threshold
    # Skip D updates when D loss is below this threshold (D is winning too much)
    # Higher threshold = more conservative D updates, preventing collapse
    d_loss_threshold: float = 1.0  # Skip D update if loss < threshold (was 0.5, increased)

    # Gradient clipping (prevents NaN from exploding gradients)
    max_grad_norm: float = 1.0  # Clip gradients to this norm

    # Loss weights
    c_mel: float = 45.0
    c_kl: float = 1.0

    # Checkpointing
    save_every_epoch: int = 10
    checkpoint_dir: str = "checkpoints"

    # Overtraining detection (based on AI Hub recommendations)
    enable_overtraining_detection: bool = True
    overtraining_patience: int = 10  # Epochs without improvement to detect plateau
    overtraining_rising_patience: int = 5  # Consecutive rising epochs to stop
    auto_stop_on_overtraining: bool = True  # Automatically stop when detected

    # Warmup mode: disable adversarial training for first N epochs
    # This stabilizes initial training by only using mel+kl loss (no discriminator)
    warmup_epochs: int = 0  # 0 = no warmup (full GAN training from start)
    disable_adversarial: bool = False  # Completely disable adversarial losses (debug mode)


def slice_segments(x: mx.array, ids_str: mx.array, segment_size: int) -> mx.array:
    """
    Slice segments from batch.

    Args:
        x: Input tensor (B, T, C) or (B, T)
        ids_str: Start indices (B,)
        segment_size: Segment length

    Returns:
        Sliced segments (B, segment_size, C) or (B, segment_size)
    """
    batch_size = x.shape[0]
    segments = []

    for b in range(batch_size):
        start = int(ids_str[b].item())
        end = start + segment_size
        if x.ndim == 3:
            segment = x[b, start:end, :]
            # Pad if segment is shorter than expected
            if segment.shape[0] < segment_size:
                pad_len = segment_size - segment.shape[0]
                padding = mx.zeros((pad_len, segment.shape[1]))
                segment = mx.concatenate([segment, padding], axis=0)
        else:
            segment = x[b, start:end]
            # Pad if segment is shorter than expected
            if segment.shape[0] < segment_size:
                pad_len = segment_size - segment.shape[0]
                padding = mx.zeros((pad_len,))
                segment = mx.concatenate([segment, padding], axis=0)
        segments.append(segment)

    return mx.stack(segments, axis=0)


def rand_slice_segments(x: mx.array, x_lengths: mx.array, segment_size: int) -> Tuple[mx.array, mx.array]:
    """
    Randomly slice segments from batch.

    Args:
        x: Input tensor (B, T, C)
        x_lengths: Actual lengths (B,)
        segment_size: Segment length

    Returns:
        Sliced segments, start indices
    """
    batch_size = x.shape[0]
    ids_str = []

    for b in range(batch_size):
        max_start = max(0, int(x_lengths[b].item()) - segment_size)
        if max_start > 0:
            start = int(np.random.randint(0, max_start))
        else:
            start = 0
        ids_str.append(start)

    ids_str = mx.array(np.array(ids_str, dtype=np.int32))
    segments = slice_segments(x, ids_str, segment_size)

    return segments, ids_str


class RVCTrainer:
    """
    RVC Trainer using MLX.

    Handles training loop with generator and discriminator updates.
    """

    def __init__(
        self,
        net_g: nn.Module,
        net_d: MultiPeriodDiscriminator,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        self.net_g = net_g
        self.net_d = net_d
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Freeze encoder if configured (CRITICAL for fine-tuning)
        # The text encoder (enc_p) should remain frozen to preserve pretrained representations
        if config.freeze_encoder and config.is_finetuning:
            if hasattr(net_g, 'enc_p'):
                net_g.enc_p.freeze()
                print("Frozen: enc_p (TextEncoder) - preserving pretrained representations")
            else:
                print("Warning: enc_p not found in generator, cannot freeze encoder")

        # Mel config
        self.mel_config = get_mel_config(config.sample_rate)

        # Determine actual learning rates
        self.base_lr = config.learning_rate
        if config.is_finetuning:
            self.current_lr_g = config.learning_rate * config.finetune_lr_scale
            print(f"Fine-tuning mode: G LR scaled from {config.learning_rate} to {self.current_lr_g}")
        else:
            self.current_lr_g = config.learning_rate

        # Discriminator uses scaled LR to prevent it from dominating
        self.current_lr_d = self.current_lr_g * config.d_lr_scale
        print(f"Discriminator LR: {self.current_lr_d:.2e} (scale: {config.d_lr_scale})")

        # Optimizers with separate learning rates
        self.optim_g = optim.AdamW(
            learning_rate=self.current_lr_g,
            betas=config.betas,
            eps=config.eps,
        )
        self.optim_d = optim.AdamW(
            learning_rate=self.current_lr_d,
            betas=config.betas,
            eps=config.eps,
        )

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")

        # History
        self.loss_history = {
            "g_total": [],
            "d_total": [],
            "mel": [],
            "kl": [],
            "fm": [],
            "gen": [],
        }

        # Overtraining detection
        self.overtraining_detector = None
        if config.enable_overtraining_detection:
            self.overtraining_detector = OvertrainingDetector(
                patience=config.overtraining_patience,
                rising_patience=config.overtraining_rising_patience,
            )

    def _compute_generator_loss(
        self,
        batch: Dict[str, mx.array],
    ) -> Tuple[mx.array, Dict[str, mx.array], mx.array, mx.array]:
        """
        Compute generator loss.

        Returns:
            loss_total: Total generator loss
            loss_dict: Individual loss components
            y_hat: Generated audio
            wave_slice: Sliced ground truth audio (matches y_hat shape)
        """
        phone = batch["phone"]
        phone_lengths = batch["phone_lengths"]
        pitch = batch["pitch"]
        pitchf = batch["pitchf"]
        spec = batch["spec"]  # Spectrogram (B, spec_channels, T_spec)
        spec_lengths = batch["spec_lengths"]
        wave = batch["wave"]  # Raw waveform for discriminator
        wave_lengths = batch["wave_lengths"]
        sid = batch["sid"]

        # Forward through generator
        # Note: This requires synthesizer to have forward() method
        # y (spec) is used for posterior encoder, not waveform
        y_hat, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = self.net_g.forward(
            phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid
        )

        # Slice ground truth audio to match generated
        segment_size = self.config.segment_size
        wave_slice = slice_segments(
            wave[:, :, None] if wave.ndim == 2 else wave,
            ids_slice * self.config.hop_length,
            segment_size,
        )

        # Ensure shapes match
        if wave_slice.ndim == 2:
            wave_slice = wave_slice[:, :, None]
        if y_hat.ndim == 2:
            y_hat = y_hat[:, :, None]

        # Match lengths (generator output may differ slightly from segment_size)
        min_wave_len = min(wave_slice.shape[1], y_hat.shape[1])
        wave_slice = wave_slice[:, :min_wave_len, :]
        y_hat = y_hat[:, :min_wave_len, :]

        # Check if we should use adversarial losses (not in warmup mode)
        in_warmup = self.config.warmup_epochs > 0 and self.epoch < self.config.warmup_epochs
        use_adversarial = not self.config.disable_adversarial and not in_warmup

        # Mel spectrograms
        wave_mel = self.mel_config.compute(wave_slice[:, :, 0])
        y_hat_mel = self.mel_config.compute(y_hat[:, :, 0])

        # Match lengths (generator output may differ slightly from segment_size)
        min_mel_len = min(wave_mel.shape[1], y_hat_mel.shape[1])
        wave_mel = wave_mel[:, :min_mel_len, :]
        y_hat_mel = y_hat_mel[:, :min_mel_len, :]

        # Losses
        loss_mel = mel_loss(wave_mel, y_hat_mel) * self.config.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, y_mask) * self.config.c_kl

        # Clamp reconstruction losses to prevent NaN
        loss_mel = mx.clip(loss_mel, 0.0, 10000.0)
        loss_kl = mx.clip(loss_kl, 0.0, 100.0)

        if use_adversarial:
            # Discriminator outputs (no grad for generator loss)
            _, y_d_hat_g, fmap_r, fmap_g = self.net_d(wave_slice, y_hat)
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen = generator_loss(y_d_hat_g)

            # Clamp adversarial losses to prevent NaN propagation
            # Large values indicate unstable training - clamp to prevent explosion
            loss_fm = mx.clip(loss_fm, 0.0, 1000.0)
            loss_gen = mx.clip(loss_gen, 0.0, 100.0)

            loss_total = loss_gen + loss_fm + loss_mel + loss_kl
        else:
            # Warmup mode: only mel + kl loss (no discriminator)
            loss_fm = mx.array(0.0)
            loss_gen = mx.array(0.0)
            loss_total = loss_mel + loss_kl

        loss_dict = {
            "mel": loss_mel,
            "kl": loss_kl,
            "fm": loss_fm,
            "gen": loss_gen,
            "total": loss_total,
        }

        return loss_total, loss_dict, y_hat, wave_slice

    def _compute_discriminator_loss(
        self,
        wave_real: mx.array,
        wave_fake: mx.array,
    ) -> mx.array:
        """Compute discriminator loss."""
        y_d_rs, y_d_gs, _, _ = self.net_d(wave_real, mx.stop_gradient(wave_fake))
        loss_d = discriminator_loss(y_d_rs, y_d_gs)
        return loss_d

    def train_step(self, batch: Dict[str, mx.array]) -> Dict[str, float]:
        """
        Single training step.

        Returns dict of loss values.
        """
        # Generator forward and loss
        def g_loss_fn(net_g):
            loss_total, loss_dict, y_hat, wave_slice = self._compute_generator_loss(batch)
            return loss_total, (loss_dict, y_hat, wave_slice)

        # Compute generator gradients
        loss_and_grad_fn = nn.value_and_grad(self.net_g, g_loss_fn)
        (loss_g, (loss_dict_g, y_hat, wave_slice)), grads_g = loss_and_grad_fn(self.net_g)

        # Sanitize gradients (replace inf/nan with finite values)
        # This prevents clip_grad_norm from zeroing all gradients when any contains inf
        grads_g, had_grad_issues = sanitize_gradients(grads_g)

        # Clip generator gradients to prevent NaN
        grads_g, g_norm = clip_grad_norm(grads_g, max_norm=self.config.max_grad_norm)

        # Check for NaN in loss and skip update if detected
        loss_g_val = float(loss_g.item())
        if math.isnan(loss_g_val) or math.isinf(loss_g_val):
            print(f"WARNING: NaN/Inf in generator loss, skipping update")
            del grads_g
            return {
                "g_total": float('nan'),
                "d_total": float('nan'),
                "mel": float('nan'),
                "kl": float('nan'),
                "fm": float('nan'),
                "gen": float('nan'),
            }

        # Update generator
        self.optim_g.update(self.net_g, grads_g)

        # CRITICAL: Evaluate generator update immediately to prevent NaN accumulation
        # The delayed eval pattern was causing numerical instability
        mx.eval(self.net_g.parameters(), self.optim_g.state)

        # Free generator gradients
        del grads_g

        # Check if we should update discriminator (not in warmup mode)
        in_warmup = self.config.warmup_epochs > 0 and self.epoch < self.config.warmup_epochs
        use_adversarial = not self.config.disable_adversarial and not in_warmup

        d_loss_val = 0.0
        d_updated = False

        if use_adversarial:
            # Use wave_slice from generator computation (already matches y_hat shape)
            # Detach both from generator graph for discriminator training
            wave_slice_detached = mx.stop_gradient(wave_slice)
            y_hat_detached = mx.stop_gradient(y_hat)

            # Discriminator forward and loss
            def d_loss_fn(net_d):
                return self._compute_discriminator_loss(wave_slice_detached, y_hat_detached)

            # Compute discriminator gradients
            loss_and_grad_fn_d = nn.value_and_grad(self.net_d, d_loss_fn)
            loss_d, grads_d = loss_and_grad_fn_d(self.net_d)

            # Evaluate loss_d first to check threshold
            mx.eval(loss_d)
            d_loss_val = float(loss_d.item())

            # Skip D update if discriminator is winning too much (prevents collapse)
            if d_loss_val >= self.config.d_loss_threshold:
                # Sanitize and clip discriminator gradients
                grads_d, _ = sanitize_gradients(grads_d)
                grads_d, d_norm = clip_grad_norm(grads_d, max_norm=self.config.max_grad_norm)

                # Update discriminator
                self.optim_d.update(self.net_d, grads_d)

                # CRITICAL: Evaluate discriminator update immediately
                mx.eval(self.net_d.parameters(), self.optim_d.state)
                d_updated = True

            # Free discriminator gradients
            del grads_d

        # Evaluate all tensors to trigger computation and free graph
        # Note: D params/state only need eval if D was updated
        tensors_to_eval = [
            self.net_g.parameters(),
            self.optim_g.state,
            loss_g,
            loss_dict_g["mel"],
            loss_dict_g["kl"],
            loss_dict_g["fm"],
            loss_dict_g["gen"],
        ]
        if d_updated:
            tensors_to_eval.extend([
                self.net_d.parameters(),
                self.optim_d.state,
            ])
        mx.eval(*tensors_to_eval)

        # Extract scalar values AFTER eval
        # Use d_loss_val computed earlier (already evaluated)
        losses = {
            "g_total": float(loss_g.item()),
            "d_total": d_loss_val,  # Already computed above (0.0 if warmup)
            "mel": float(loss_dict_g["mel"].item()),
            "kl": float(loss_dict_g["kl"].item()),
            "fm": float(loss_dict_g["fm"].item()),
            "gen": float(loss_dict_g["gen"].item()),
        }

        # Free intermediate tensors
        del loss_g, loss_dict_g, y_hat, wave_slice

        # Clear Metal cache to free GPU memory
        mx.metal.clear_cache()

        self.global_step += 1

        return losses

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_losses = {k: [] for k in self.loss_history.keys()}

        # Progress bar for steps within epoch
        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {self.epoch + 1}",
            leave=False,
            ncols=100,
        )

        for batch_idx, batch in pbar:
            losses = self.train_step(batch)

            for k, v in losses.items():
                if k in epoch_losses:
                    epoch_losses[k].append(v)

            # Update progress bar with current losses
            pbar.set_postfix({
                "g": f"{losses['g_total']:.1f}",
                "d": f"{losses['d_total']:.2f}",
                "mel": f"{losses['mel']:.1f}",
            })

        pbar.close()

        # Compute epoch averages
        avg_losses = {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}

        # Update history
        for k, v in avg_losses.items():
            self.loss_history[k].append(v)

        return avg_losses

    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}

        val_losses = {k: [] for k in ["g_total", "mel"]}

        for batch in self.val_loader:
            loss_total, loss_dict, _, _ = self._compute_generator_loss(batch)
            val_losses["g_total"].append(float(loss_total.item()))
            val_losses["mel"].append(float(loss_dict["mel"].item()))

        return {k: np.mean(v) if v else 0.0 for k, v in val_losses.items()}

    def _flatten_params(self, params: Dict, prefix: str = "") -> Dict[str, mx.array]:
        """Flatten nested parameter dict to flat dict with dotted keys."""
        flat = {}
        for k, v in params.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, mx.array):
                flat[key] = v
            elif isinstance(v, dict):
                flat.update(self._flatten_params(v, key))
        return flat

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        # Save generator weights (flatten nested dict)
        g_weights = self._flatten_params(self.net_g.parameters())
        for v in g_weights.values():
            mx.eval(v)  # Ensure evaluated
        mx.savez(f"{path}_G.npz", **g_weights)

        # Save discriminator weights (flatten nested dict)
        d_weights = self._flatten_params(self.net_d.parameters())
        for v in d_weights.values():
            mx.eval(v)  # Ensure evaluated
        mx.savez(f"{path}_D.npz", **d_weights)

        # Save training state
        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "loss_history": self.loss_history,
            "config": {
                "sample_rate": self.config.sample_rate,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
            },
        }
        with open(f"{path}_state.json", "w") as f:
            json.dump(state, f, indent=2)

        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        # Load generator weights
        g_path = f"{path}_G.npz"
        if os.path.exists(g_path):
            g_weights = dict(mx.load(g_path))
            self.net_g.load_weights(list(g_weights.items()))
            print(f"Loaded generator from {g_path}")

        # Load discriminator weights
        d_path = f"{path}_D.npz"
        if os.path.exists(d_path):
            d_weights = dict(mx.load(d_path))
            self.net_d.load_weights(list(d_weights.items()))
            print(f"Loaded discriminator from {d_path}")

        # Load training state
        state_path = f"{path}_state.json"
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                state = json.load(f)
            self.epoch = state.get("epoch", 0)
            self.global_step = state.get("global_step", 0)
            self.best_loss = state.get("best_loss", float("inf"))
            self.loss_history = state.get("loss_history", self.loss_history)
            print(f"Loaded training state from {state_path}")

    def train(
        self,
        epochs: Optional[int] = None,
        save_every: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Main training loop.

        Args:
            epochs: Number of epochs (default from config)
            save_every: Save checkpoint every N epochs (default from config)
            checkpoint_dir: Directory for checkpoints (default from config)
        """
        epochs = epochs or self.config.epochs
        save_every = save_every or self.config.save_every_epoch
        checkpoint_dir = checkpoint_dir or self.config.checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  RVC MLX Training")
        print(f"  Epochs: {epochs} | Batch: {self.config.batch_size}")
        print(f"  Initial G LR: {self.current_lr_g:.2e} | D LR: {self.current_lr_d:.2e}")
        print(f"  LR Decay: {self.config.lr_decay} | D LR Scale: {self.config.d_lr_scale}")
        print(f"  Max Grad Norm: {self.config.max_grad_norm}")
        print(f"  Steps/epoch: {len(self.train_loader)}")
        if self.config.is_finetuning:
            print(f"  Mode: Fine-tuning (pretrained weights)")
        print(f"{'='*60}\n")

        start_epoch = self.epoch

        # Overall epoch progress bar
        epoch_pbar = tqdm(
            range(start_epoch, epochs),
            desc="Training",
            unit="epoch",
            ncols=100,
        )

        for epoch in epoch_pbar:
            self.epoch = epoch
            epoch_start = time.time()

            # Train
            train_losses = self.train_epoch()
            epoch_time = time.time() - epoch_start

            # Update epoch progress bar
            epoch_pbar.set_postfix({
                "g_loss": f"{train_losses['g_total']:.1f}",
                "mel": f"{train_losses['mel']:.1f}",
                "time": f"{epoch_time:.0f}s",
            })

            # Validate
            if self.val_loader:
                val_losses = self.validate()
                tqdm.write(f"  Val: g_loss={val_losses.get('g_total', 0):.4f}, mel={val_losses.get('mel', 0):.4f}")

            # Overtraining detection (based on AI Hub recommendations)
            if self.overtraining_detector is not None:
                status = self.overtraining_detector.update(train_losses['g_total'])

                # Log status on significant events
                if status.trend == "improving" and status.best_loss == train_losses['g_total']:
                    tqdm.write(f"  ✓ New best: {status.best_loss:.2f}")
                elif status.trend == "plateau" and status.epochs_since_best % 5 == 0:
                    tqdm.write(f"  ⚠ Plateau: {status.epochs_since_best} epochs since best")
                elif status.trend == "rising":
                    tqdm.write(f"  ⚠ Loss rising!")

                # Check if should stop
                if status.is_overtraining and self.config.auto_stop_on_overtraining:
                    tqdm.write(f"\n  STOPPING: {status.reason}")
                    tqdm.write(f"  Best model was at epoch {self.overtraining_detector.get_best_epoch()}")
                    tqdm.write(f"  Recommendation: Use checkpoint from that epoch")
                    break

            # Learning rate decay - update optimizer lr in-place (preserves momentum state)
            self.current_lr_g = self.current_lr_g * self.config.lr_decay
            self.current_lr_d = self.current_lr_d * self.config.lr_decay
            self.optim_g.learning_rate = self.current_lr_g
            self.optim_d.learning_rate = self.current_lr_d

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1:04d}")
                self.save_checkpoint(ckpt_path)
                tqdm.write(f"  💾 Saved checkpoint: epoch_{epoch + 1:04d}")

            # Track best
            if train_losses["g_total"] < self.best_loss:
                self.best_loss = train_losses["g_total"]
                best_path = os.path.join(checkpoint_dir, "best")
                self.save_checkpoint(best_path)

        epoch_pbar.close()
        print(f"\n{'='*60}")
        print(f"  Training complete!")
        print(f"  Best loss: {self.best_loss:.4f}")
        print(f"{'='*60}")
