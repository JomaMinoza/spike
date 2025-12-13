"""
Korteweg-de Vries (KdV) Equation
Nonlinear dispersive wave: u_t + u * u_x + u_xxx = 0
"""

import torch
from ..base import BasePDE


class KdVEquation(BasePDE):
    """
    Korteweg-de Vries (KdV) Equation.

    PDE: u_t + u * u_x + u_xxx = 0

    Fundamental nonlinear dispersive wave equation with soliton solutions.

    Args:
        domain_x: Spatial domain (default: (-10, 10))
        domain_t: Temporal domain (default: (0, 1))
    """

    def __init__(self, domain_x=(-10.0, 10.0), domain_t=(0.0, 1.0)):
        super().__init__(domain_x, domain_t)
        self.name = "KdVEquation"

    def residual(self, u, x):
        """Compute residual: u_t + u * u_x + u_xxx = 0"""
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

        return u_t + u * u_x + u_xxx

    def initial_condition(self, x):
        """Soliton IC: u(x, 0) = 0.5 * sech^2(x/2)"""
        sech = 1.0 / torch.cosh(x / 2)
        return 0.5 * sech ** 2

    def boundary_condition(self, t, boundary='left'):
        return torch.zeros_like(t)

    def exact_solution(self, x):
        """Exact soliton: u(x, t) = 0.5 * sech^2((x - t) / 2)"""
        x_coord = x[:, 0:1]
        t_coord = x[:, 1:2]
        xi = x_coord - t_coord
        sech = 1.0 / torch.cosh(xi / 2)
        return 0.5 * sech ** 2

    def get_params(self):
        return {}
