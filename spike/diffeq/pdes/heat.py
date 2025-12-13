"""
Heat Equation
1D Heat/Diffusion: u_t = alpha * u_xx
"""

import torch
import numpy as np
from ..base import BasePDE


class HeatEquation(BasePDE):
    """
    1D Heat (Diffusion) Equation.

    PDE: u_t = alpha * u_xx

    Simplest parabolic PDE. Models heat conduction.

    Args:
        alpha: Thermal diffusivity (default: 0.1)
        domain_x: Spatial domain (default: (0, 1))
        domain_t: Temporal domain (default: (0, 1))
    """

    def __init__(
        self,
        alpha: float = 0.01,
        domain_x=(0.0, 1.0),
        domain_t=(0.0, 1.0)
    ):
        super().__init__(domain_x, domain_t)
        self.alpha = alpha
        self.name = "HeatEquation"

    def residual(self, u, x):
        """Compute residual: u_t - alpha * u_xx = 0"""
        derivs = self.compute_derivatives(u, x)
        return derivs['u_t'] - self.alpha * derivs['u_xx']

    def initial_condition(self, x):
        """Initial condition: u(x, 0) = sin(pi * x)"""
        return torch.sin(np.pi * x)

    def boundary_condition(self, t, boundary='left'):
        """Dirichlet BCs: u(0, t) = u(1, t) = 0"""
        return torch.zeros_like(t)

    def exact_solution(self, inputs):
        """Exact: u(x,t) = exp(-alpha * pi^2 * t) * sin(pi * x)"""
        x_vals = inputs[:, 0:1]
        t_vals = inputs[:, 1:2]
        return torch.exp(-self.alpha * np.pi**2 * t_vals) * torch.sin(np.pi * x_vals)

    def get_params(self):
        return {'alpha': self.alpha}
