"""
Advection Equation
Linear transport: u_t + c * u_x = 0
"""

import torch
import numpy as np
from ..base import BasePDE


class AdvectionEquation(BasePDE):
    """
    1D Linear Advection Equation.

    PDE: u_t + c * u_x = 0

    Simplest hyperbolic PDE describing wave propagation.

    Args:
        c: Wave speed (default: 1.0)
        domain_x: Spatial domain (default: (0, 2*pi))
        domain_t: Temporal domain (default: (0, 1))
    """

    def __init__(self, c=1.0, domain_x=(0.0, 2*np.pi), domain_t=(0.0, 1.0)):
        super().__init__(domain_x, domain_t)
        self.c = c
        self.name = "AdvectionEquation"

    def residual(self, u, x):
        """Residual: u_t + c * u_x = 0"""
        derivs = self.compute_derivatives(u, x)
        return derivs['u_t'] + self.c * derivs['u_x']

    def initial_condition(self, x):
        """IC: u(x, 0) = sin(x)"""
        return torch.sin(x)

    def boundary_condition(self, t, boundary='left'):
        return torch.sin(-self.c * t)

    def exact_solution(self, x):
        """Exact: u(x, t) = sin(x - c*t)"""
        return torch.sin(x[:, 0:1] - self.c * x[:, 1:2])

    def get_params(self):
        return {'c': self.c}
