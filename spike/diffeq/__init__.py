"""
SPIKE Differential Equations

Base classes:
- BasePDE: Abstract base for PDEs
- BaseODE: Abstract base for ODEs

1D PDEs (spike.diffeq.pdes):
- BurgersEquation, HeatEquation, AdvectionEquation, WaveEquation
- KdVEquation, AllenCahnEquation, CahnHilliardEquation
- KuramotoSivashinskyEquation, ReactionDiffusionEquation

2D PDEs (spike.diffeq.pdes):
- NavierStokes2D, LidDrivenCavity2D, ChannelFlow2D

ODEs (spike.diffeq.odes):
- LorenzSystem, SEIRModel
"""

from .base import BaseDiffEq, BasePDE, BaseODE
from .pdes import (
    BurgersEquation,
    HeatEquation,
    AdvectionEquation,
    WaveEquation,
    KdVEquation,
    AllenCahnEquation,
    CahnHilliardEquation,
    KuramotoSivashinskyEquation,
    ReactionDiffusionEquation,
    NavierStokes2D,
    LidDrivenCavity2D,
    ChannelFlow2D,
)
from .odes import LorenzSystem, SEIRModel

__all__ = [
    # Base classes
    'BaseDiffEq',
    'BasePDE',
    'BaseODE',
    # 1D PDEs
    'BurgersEquation',
    'HeatEquation',
    'AdvectionEquation',
    'WaveEquation',
    'KdVEquation',
    'AllenCahnEquation',
    'CahnHilliardEquation',
    'KuramotoSivashinskyEquation',
    'ReactionDiffusionEquation',
    # 2D PDEs
    'NavierStokes2D',
    'LidDrivenCavity2D',
    'ChannelFlow2D',
    # ODEs
    'LorenzSystem',
    'SEIRModel',
]
