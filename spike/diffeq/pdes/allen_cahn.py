"""
Allen-Cahn Equation
Phase field: u_t = eps^2 * u_xx + u - u^3
"""

import torch
import numpy as np
from ..base import BasePDE


class AllenCahnEquation(BasePDE):
    """
    Allen-Cahn Equation (Phase Field Model).

    PDE: u_t = eps^2 * u_xx + u - u^3

    Benchmark for phase separation and interface dynamics.

    Args:
        eps: Interface width parameter (default: 0.01)
        domain_x: Spatial domain (default: (-1, 1))
        domain_t: Temporal domain (default: (0, 1))
    """

    def __init__(self, eps=0.01, domain_x=(-1.0, 1.0), domain_t=(0.0, 1.0)):
        super().__init__(domain_x, domain_t)
        self.eps = eps
        self.name = "AllenCahnEquation"

    def residual(self, u, x):
        """Residual: u_t - eps^2 * u_xx - u + u^3 = 0"""
        derivs = self.compute_derivatives(u, x)
        return derivs['u_t'] - self.eps**2 * derivs['u_xx'] - u + u**3

    def initial_condition(self, x):
        """IC: u(x, 0) = x^2 * cos(pi * x)"""
        return x**2 * torch.cos(np.pi * x)

    def boundary_condition(self, t, boundary='left'):
        """Dirichlet: u(-1, t) = u(1, t) = -1"""
        return -torch.ones_like(t)

    def get_params(self):
        return {'eps': self.eps}
