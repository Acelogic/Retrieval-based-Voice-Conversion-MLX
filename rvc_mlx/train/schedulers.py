"""
Learning Rate Schedulers for RVC MLX Training
"""

from typing import Optional
import numpy as np


class LRScheduler:
    """Base class for learning rate schedulers."""

    def __init__(self, initial_lr: float):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.step_count = 0

    def step(self) -> float:
        """Perform a scheduler step and return new learning rate."""
        self.step_count += 1
        self.current_lr = self._compute_lr()
        return self.current_lr

    def _compute_lr(self) -> float:
        """Compute learning rate for current step. Override in subclass."""
        return self.initial_lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr


class ExponentialLR(LRScheduler):
    """
    Exponential learning rate decay.

    lr = initial_lr * decay^epoch
    """

    def __init__(self, initial_lr: float, decay: float = 0.999875):
        super().__init__(initial_lr)
        self.decay = decay

    def _compute_lr(self) -> float:
        return self.initial_lr * (self.decay ** self.step_count)


class CosineAnnealingLR(LRScheduler):
    """
    Cosine annealing learning rate.

    lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * step / T_max))
    """

    def __init__(
        self,
        initial_lr: float,
        T_max: int,
        min_lr: float = 1e-6,
    ):
        super().__init__(initial_lr)
        self.T_max = T_max
        self.min_lr = min_lr

    def _compute_lr(self) -> float:
        progress = min(self.step_count / self.T_max, 1.0)
        return self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + np.cos(np.pi * progress))


class WarmupLR(LRScheduler):
    """
    Linear warmup followed by another scheduler.

    During warmup: lr = initial_lr * step / warmup_steps
    After warmup: uses base scheduler
    """

    def __init__(
        self,
        initial_lr: float,
        warmup_steps: int,
        base_scheduler: Optional[LRScheduler] = None,
    ):
        super().__init__(initial_lr)
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler or LRScheduler(initial_lr)

    def _compute_lr(self) -> float:
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            return self.initial_lr * (self.step_count / self.warmup_steps)
        else:
            # Use base scheduler after warmup
            return self.base_scheduler._compute_lr()

    def step(self) -> float:
        self.step_count += 1
        if self.step_count > self.warmup_steps:
            self.base_scheduler.step_count = self.step_count - self.warmup_steps
        self.current_lr = self._compute_lr()
        return self.current_lr


class ReduceOnPlateauLR(LRScheduler):
    """
    Reduce learning rate when metric plateaus.
    """

    def __init__(
        self,
        initial_lr: float,
        factor: float = 0.5,
        patience: int = 10,
        min_lr: float = 1e-6,
    ):
        super().__init__(initial_lr)
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_metric = None
        self.counter = 0

    def step_with_metric(self, metric: float) -> float:
        """
        Step with metric value.

        Args:
            metric: Metric to monitor (lower is better)

        Returns:
            New learning rate
        """
        self.step_count += 1

        if self.best_metric is None:
            self.best_metric = metric
        elif metric < self.best_metric:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.current_lr = max(self.current_lr * self.factor, self.min_lr)
                self.counter = 0
                print(f"Reducing LR to {self.current_lr}")

        return self.current_lr
