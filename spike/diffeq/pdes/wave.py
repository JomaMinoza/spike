"""
Wave Equation
u_tt = c^2 * u_xx
"""

import torch
import numpy as np
from ..base import BasePDE


class WaveEquation(BasePDE):
    """
    1D Wave Equation.

    PDE: u_tt = c^2 * u_xx

    Second-order in time PDE for wave propagation.

    Args:
        c: Wave speed (default: 1.0)
        domain_x: Spatial domain (default: (0, 1))
        domain_t: Temporal domain (default: (0, 1))
    """

    def __init__(self, c=1.0, domain_x=(0.0, 1.0), domain_t=(0.0, 1.0)):
        super().__init__(domain_x, domain_t)
        self.c = c
        self.name = "WaveEquation"

    def residual(self, u, x):
        """Residual: u_tt - c^2 * u_xx = 0"""
        grad_u = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        u_x = grad_u[:, 0:1]
        u_t = grad_u[:, 1:2]

        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0][:, 0:1]

        u_tt = torch.autograd.grad(
            u_t, x, grad_outputs=torch.ones_like(u_t),
            create_graph=True, retain_graph=True
        )[0][:, 1:2]

        return u_tt - self.c**2 * u_xx

    def initial_condition(self, x):
        """IC: u(x, 0) = sin(pi * x)"""
        return torch.sin(np.pi * x)

    def initial_velocity(self, x):
        """Initial velocity: u_t(x, 0) = 0"""
        return torch.zeros_like(x)

    def boundary_condition(self, t, boundary='left'):
        return torch.zeros_like(t)

    def exact_solution(self, x):
        """Exact: u(x, t) = sin(pi * x) * cos(c * pi * t)"""
        return torch.sin(np.pi * x[:, 0:1]) * torch.cos(self.c * np.pi * x[:, 1:2])

    def get_params(self):
        return {'c': self.c}
