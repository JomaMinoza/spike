"""
SPIKE Models

Model hierarchy:
- PINN: Base encoder (x,t) → u
- Koopman: Embedding + linear dynamics u → g(u) → A @ g(u)
- PIKE: PINN + Koopman combined
- SPIKE: PIKE + L1 sparsity regularization
"""

from .pinn import PINN
from .embedding import SparseEmbedding
from .koopman import Koopman
from .pike import PIKE
from .spike import SPIKE

__all__ = [
    'PINN',
    'SparseEmbedding',
    'Koopman',
    'PIKE',
    'SPIKE',
]
