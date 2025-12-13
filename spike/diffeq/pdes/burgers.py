"""
Burgers Equation
1D Viscous Burgers: u_t + u * u_x = nu * u_xx
"""

import torch
import numpy as np
from ..base import BasePDE


class BurgersEquation(BasePDE):
    """
    1D Viscous Burgers Equation.

    PDE: u_t + u * u_x = nu * u_xx

    Combines nonlinear advection with diffusion.
    Classic test case for PINN methods.

    Args:
        nu: Viscosity coefficient (default: 0.01/pi)
        domain_x: Spatial domain (default: (-1, 1))
        domain_t: Temporal domain (default: (0, 1))
    """

    def __init__(
        self,
        nu: float = 0.01,
        domain_x=(-1.0, 1.0),
        domain_t=(0.0, 1.0)
    ):
        super().__init__(domain_x, domain_t)
        self.nu = nu
        self.name = "BurgersEquation"

    def residual(self, u, x):
        """
        Compute residual: u_t + u * u_x - nu * u_xx = 0
        """
        derivs = self.compute_derivatives(u, x)

        u_t = derivs['u_t']
        u_x = derivs['u_x']
        u_xx = derivs['u_xx']

        return u_t + u * u_x - self.nu * u_xx

    def initial_condition(self, x):
        """Initial condition: u(x, 0) = -sin(pi * x)"""
        return -torch.sin(np.pi * x)

    def boundary_condition(self, t, boundary='left'):
        """Boundary conditions: u(-1, t) = u(1, t) = 0"""
        return torch.zeros_like(t)

    def get_params(self):
        return {'nu': self.nu}
