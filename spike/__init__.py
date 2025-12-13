"""
SPIKE: Sparse Physics-Informed Koopman-Enhanced Neural Networks

A modular library for discovering sparse, interpretable representations
of PDE dynamics using Koopman operator theory.

Modules:
- models: PINN, Koopman, PIKE, SPIKE
- integrators: Euler, RK4, ExpmIntegrator
- losses: PhysicsLoss, KoopmanLoss, SparsityLoss, CombinedLoss
- diffeq: PDEs (Burgers, Heat, ...) and ODEs (Lorenz, SEIR, ...)
- training: Trainer, callbacks, samplers
- evaluation: metrics, residuals, Koopman analysis
"""

__version__ = "0.1.0"

from .models import PINN, Koopman, PIKE, SPIKE, SparseEmbedding
from .integrators import get_integrator, EulerIntegrator, RK4Integrator, ExpmIntegrator
from .losses import PhysicsLoss, KoopmanLoss, SparsityLoss, CombinedLoss
from .diffeq import BasePDE, BaseODE, BurgersEquation, HeatEquation, LorenzSystem, SEIRModel
from .training import Trainer, EarlyStopping, ModelCheckpoint
from .evaluation import compute_l2_error, compute_mse, compute_residual

__all__ = [
    # Models
    'PINN',
    'Koopman',
    'PIKE',
    'SPIKE',
    'SparseEmbedding',
    # Integrators
    'get_integrator',
    'EulerIntegrator',
    'RK4Integrator',
    'ExpmIntegrator',
    # Losses
    'PhysicsLoss',
    'KoopmanLoss',
    'SparsityLoss',
    'CombinedLoss',
    # DiffEq
    'BasePDE',
    'BaseODE',
    'BurgersEquation',
    'HeatEquation',
    'LorenzSystem',
    'SEIRModel',
    # Training
    'Trainer',
    'EarlyStopping',
    'ModelCheckpoint',
    # Evaluation
    'compute_l2_error',
    'compute_mse',
    'compute_residual',
]
