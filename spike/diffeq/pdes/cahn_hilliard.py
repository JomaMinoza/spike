"""
Cahn-Hilliard Equation
Phase separation: u_t = M * nabla^2 * (u^3 - u - eps^2 * nabla^2 u)
"""

import torch
import numpy as np
from ..base import BasePDE


class CahnHilliardEquation(BasePDE):
    """
    Cahn-Hilliard Equation (Phase Separation).

    PDE: u_t = M * nabla^2 * (u^3 - u - eps^2 * nabla^2 u)
    In 1D: u_t = M * (3*u^2*u_xx + 6*u*u_x^2 - u_xx - eps^2 * u_xxxx)

    Fourth-order PDE for phase separation and spinodal decomposition.

    Args:
        M: Mobility coefficient (default: 1.0)
        eps: Interface width parameter (default: 0.01)
        domain_x: Spatial domain (default: (0, 1))
        domain_t: Temporal domain (default: (0, 1))
    """

    def __init__(self, M=1.0, eps=0.01, domain_x=(0.0, 1.0), domain_t=(0.0, 1.0)):
        super().__init__(domain_x, domain_t)
        self.M = M
        self.eps = eps
        self.name = "CahnHilliardEquation"

    def residual(self, u, x):
        """Residual: u_t - M * (3*u^2*u_xx + 6*u*u_x^2 - u_xx - eps^2*u_xxxx) = 0"""
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

        u_xxx = torch.autograd.grad(
            u_xx, x, grad_outputs=torch.ones_like(u_xx),
            create_graph=True, retain_graph=True
        )[0][:, 0:1]

        u_xxxx = torch.autograd.grad(
            u_xxx, x, grad_outputs=torch.ones_like(u_xxx),
            create_graph=True, retain_graph=True
        )[0][:, 0:1]

        rhs = self.M * (3 * u**2 * u_xx + 6 * u * u_x**2 - u_xx - self.eps**2 * u_xxxx)
        return u_t - rhs

    def initial_condition(self, x):
        """IC: 0.1 * cos(2*pi*x) + 0.05 * cos(4*pi*x)"""
        return 0.1 * torch.cos(2 * np.pi * x) + 0.05 * torch.cos(4 * np.pi * x)

    def boundary_condition(self, t, boundary='left'):
        return torch.zeros_like(t)

    def get_params(self):
        return {'M': self.M, 'eps': self.eps}
