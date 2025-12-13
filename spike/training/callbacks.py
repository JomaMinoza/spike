"""
Training Callbacks
EarlyStopping, ModelCheckpoint, LRScheduler
"""

import torch
import os
from typing import Optional, Callable
from abc import ABC, abstractmethod


class Callback(ABC):
    """Base callback class."""

    def on_epoch_start(self, epoch: int, trainer) -> None:
        pass

    def on_epoch_end(self, epoch: int, trainer, logs: dict) -> None:
        pass

    def on_train_start(self, trainer) -> None:
        pass

    def on_train_end(self, trainer) -> None:
        pass

    def on_batch_end(self, batch: int, trainer, logs: dict) -> None:
        pass


class EarlyStopping(Callback):
    """
    Stop training when a monitored metric stops improving.

    Args:
        monitor: Metric to monitor (default: 'loss')
        patience: Epochs to wait after last improvement (default: 10)
        min_delta: Minimum change to qualify as improvement (default: 1e-6)
        mode: 'min' or 'max' (default: 'min')
        verbose: Print messages (default: True)
    """

    def __init__(
        self,
        monitor: str = 'loss',
        patience: int = 10,
        min_delta: float = 1e-6,
        mode: str = 'min',
        verbose: bool = True
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.stop_training = False

    def on_epoch_end(self, epoch: int, trainer, logs: dict) -> None:
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.mode == 'min':
            improved = current < self.best - self.min_delta
        else:
            improved = current > self.best + self.min_delta

        if improved:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}")


class ModelCheckpoint(Callback):
    """
    Save model when a monitored metric improves.

    Args:
        filepath: Path to save model (use {epoch} for epoch number)
        monitor: Metric to monitor (default: 'loss')
        mode: 'min' or 'max' (default: 'min')
        save_best_only: Only save when metric improves (default: True)
        verbose: Print messages (default: True)
    """

    def __init__(
        self,
        filepath: str,
        monitor: str = 'loss',
        mode: str = 'min',
        save_best_only: bool = True,
        verbose: bool = True
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best = float('inf') if mode == 'min' else float('-inf')

    def on_epoch_end(self, epoch: int, trainer, logs: dict) -> None:
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.save_best_only:
            if self.mode == 'min':
                improved = current < self.best
            else:
                improved = current > self.best

            if improved:
                self.best = current
                self._save(trainer, epoch)
        else:
            self._save(trainer, epoch)

    def _save(self, trainer, epoch: int) -> None:
        filepath = self.filepath.format(epoch=epoch)
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        torch.save({
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'best_metric': self.best,
        }, filepath)

        if self.verbose:
            print(f"Saved checkpoint to {filepath}")


class LRScheduler(Callback):
    """
    Learning rate scheduler callback.

    Args:
        scheduler: torch.optim.lr_scheduler instance
        monitor: Metric for ReduceLROnPlateau (optional)
    """

    def __init__(
        self,
        scheduler,
        monitor: Optional[str] = None
    ):
        self.scheduler = scheduler
        self.monitor = monitor

    def on_epoch_end(self, epoch: int, trainer, logs: dict) -> None:
        if self.monitor and hasattr(self.scheduler, 'step'):
            # ReduceLROnPlateau needs the metric
            metric = logs.get(self.monitor)
            if metric is not None:
                self.scheduler.step(metric)
        else:
            self.scheduler.step()


class TensorBoardCallback(Callback):
    """
    Log metrics to TensorBoard.

    Args:
        log_dir: Directory for TensorBoard logs
    """

    def __init__(self, log_dir: str = './runs'):
        self.log_dir = log_dir
        self.writer = None

    def on_train_start(self, trainer) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")

    def on_epoch_end(self, epoch: int, trainer, logs: dict) -> None:
        if self.writer:
            for key, value in logs.items():
                self.writer.add_scalar(key, value, epoch)

    def on_train_end(self, trainer) -> None:
        if self.writer:
            self.writer.close()


class ProgressCallback(Callback):
    """
    Print training progress.

    Args:
        print_every: Print every N epochs (default: 100)
    """

    def __init__(self, print_every: int = 100):
        self.print_every = print_every

    def on_epoch_end(self, epoch: int, trainer, logs: dict) -> None:
        if (epoch + 1) % self.print_every == 0:
            msg = f"Epoch {epoch + 1}"
            for key, value in logs.items():
                msg += f" | {key}: {value:.6f}"
            print(msg)
