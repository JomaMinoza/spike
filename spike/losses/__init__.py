"""
SPIKE Loss Functions

- PhysicsLoss: PDE residual loss
- KoopmanLoss: Linearity in embedding space
- SparsityLoss: L1 regularization on Koopman matrix
- CombinedLoss: Weighted combination of all losses
"""

from .physics import PhysicsLoss
from .koopman import KoopmanLoss
from .sparsity import SparsityLoss
from .combined import CombinedLoss

__all__ = [
    'PhysicsLoss',
    'KoopmanLoss',
    'SparsityLoss',
    'CombinedLoss',
]
