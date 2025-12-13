"""
SPIKE Training Module

Components:
- Trainer: Main training loop with physics-informed losses
- Callbacks: EarlyStopping, ModelCheckpoint, LRScheduler, ProgressCallback
- Samplers: Collocation point sampling strategies
"""

from .trainer import Trainer
from .callbacks import EarlyStopping, ModelCheckpoint, LRScheduler, ProgressCallback
from .samplers import UniformSampler, LatinHypercubeSampler, AdaptiveSampler

__all__ = [
    'Trainer',
    'EarlyStopping',
    'ModelCheckpoint',
    'LRScheduler',
    'ProgressCallback',
    'UniformSampler',
    'LatinHypercubeSampler',
    'AdaptiveSampler',
]
