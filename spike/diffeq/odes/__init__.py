"""
SPIKE ODEs

Available ODE systems:
- LorenzSystem: Classic chaotic attractor
- SEIRModel: Epidemic compartmental model (S-E-I-R)
"""

from .lorenz import LorenzSystem
from .seir import SEIRModel

__all__ = [
    'LorenzSystem',
    'SEIRModel',
]
