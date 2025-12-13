"""
2D Wave Equation
u_tt = c^2 * (u_xx + u_yy)
"""

import torch
import numpy as np
from ..base import BasePDE


class Wave2D(BasePDE):
    """
    2D Wave Equation.

    PDE: ∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²)

    Second-order in time PDE for wave propagation in 2D.

    Args:
        c: Wave speed (default: 1.0)
        domain_x: Spatial domain in x (default: (0, 1))
        domain_y: Spatial domain in y (default: (0, 1))
        domain_t: Time domain (default: (0, 1))
    """

    def __init__(
        self,
        c: float = 1.0,
        domain_x=(0.0, 1.0),
        domain_y=(0.0, 1.0),
        domain_t=(0.0, 1.0)
    ):
        super().__init__(
            domain_x=domain_x,
            domain_t=domain_t
        )
        self.c = c
        self.domain_y = domain_y
        self.output_dim = 1  # u
        self.input_dim = 3  # x, y, t
        self.name = "Wave2D"

    def get_domain(self):
        """Return full 2D+t domain."""
        return {
            'x_min': self.domain_x[0],
            'x_max': self.domain_x[1],
            'y_min': self.domain_y[0],
            'y_max': self.domain_y[1],
            't_min': self.domain_t[0],
            't_max': self.domain_t[1]
        }

    def residual(self, u, inputs):
        """
        Compute 2D Wave equation residual.

        Args:
            u: Output [batch_size, 1]
            inputs: Input [batch_size, 3] as [x, y, t], requires_grad=True

        Returns:
            Residual [batch_size, 1]: u_tt - c^2*(u_xx + u_yy)
        """
        # First derivatives
        grad_u = torch.autograd.grad(
            u, inputs, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]
        u_t = grad_u[:, 2:3]

        # Second derivative in x
        grad_u_x = torch.autograd.grad(
            u_x, inputs, grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0]
        u_xx = grad_u_x[:, 0:1]

        # Second derivative in y
        grad_u_y = torch.autograd.grad(
            u_y, inputs, grad_outputs=torch.ones_like(u_y),
            create_graph=True, retain_graph=True
        )[0]
        u_yy = grad_u_y[:, 1:2]

        # Second derivative in t
        grad_u_t = torch.autograd.grad(
            u_t, inputs, grad_outputs=torch.ones_like(u_t),
            create_graph=True, retain_graph=True
        )[0]
        u_tt = grad_u_t[:, 2:3]

        # Wave equation: u_tt = c^2 * (u_xx + u_yy)
        return u_tt - self.c**2 * (u_xx + u_yy)

    def get_params(self):
        return {'c': self.c}

    def initial_condition(self, x, y):
        """Default IC: u(x, y, 0) = sin(pi*x) * sin(pi*y)"""
        if isinstance(x, torch.Tensor):
            return torch.sin(np.pi * x) * torch.sin(np.pi * y)
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def initial_velocity(self, x, y):
        """Initial velocity: u_t(x, y, 0) = 0"""
        if isinstance(x, torch.Tensor):
            return torch.zeros_like(x)
        return 0.0

    def boundary_condition(self, x, y, t):
        """Default: zero Dirichlet boundaries."""
        if isinstance(x, torch.Tensor):
            return torch.zeros_like(x)
        return 0.0

    def exact_solution(self, x, y, t):
        """
        Exact solution for sin(pi*x)*sin(pi*y) IC with zero velocity and BCs:
        u(x,y,t) = sin(pi*x) * sin(pi*y) * cos(sqrt(2)*c*pi*t)
        """
        omega = np.sqrt(2) * self.c * np.pi
        if isinstance(x, torch.Tensor):
            return torch.sin(np.pi * x) * torch.sin(np.pi * y) * torch.cos(omega * t)
        return np.sin(np.pi * x) * np.sin(np.pi * y) * np.cos(omega * t)
