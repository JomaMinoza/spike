"""
Kuramoto-Sivashinsky Equation
Chaotic PDE: u_t + u*u_x + u_xx + u_xxxx = 0
"""

import torch
import numpy as np
from ..base import BasePDE


class KuramotoSivashinskyEquation(BasePDE):
    """
    Kuramoto-Sivashinsky Equation.

    PDE: u_t + u * u_x + u_xx + u_xxxx = 0

    Canonical chaotic PDE exhibiting spatiotemporal chaos.
    Arises in flame fronts, thin films, and plasma physics.

    Args:
        domain_x: Spatial domain (default: (0, 32*pi))
        domain_t: Temporal domain (default: (0, 100))
    """

    def __init__(self, domain_x=(0.0, 32*np.pi), domain_t=(0.0, 100.0)):
        super().__init__(domain_x, domain_t)
        self.name = "KuramotoSivashinskyEquation"

    def residual(self, u, x):
        """Residual: u_t + u * u_x + u_xx + u_xxxx = 0"""
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

        return u_t + u * u_x + u_xx + u_xxxx

    def initial_condition(self, x):
        """IC: cos(x/16) * (1 + sin(x/16))"""
        return torch.cos(x / 16) * (1 + torch.sin(x / 16))

    def boundary_condition(self, t, boundary='left'):
        return torch.zeros_like(t)

    def get_params(self):
        return {}
