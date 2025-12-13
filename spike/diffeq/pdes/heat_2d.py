"""
2D Heat Equation
Heat/Diffusion in 2D: u_t = alpha * (u_xx + u_yy)
"""

import torch
import numpy as np
from ..base import BasePDE


class Heat2D(BasePDE):
    """
    2D Heat (Diffusion) Equation.

    PDE: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)

    Models heat conduction in 2D domains.

    Args:
        alpha: Thermal diffusivity (default: 0.1)
        domain_x: Spatial domain in x (default: (0, 1))
        domain_y: Spatial domain in y (default: (0, 1))
        domain_t: Time domain (default: (0, 1))
    """

    def __init__(
        self,
        alpha: float = 0.01,
        domain_x=(0.0, 1.0),
        domain_y=(0.0, 1.0),
        domain_t=(0.0, 1.0)
    ):
        super().__init__(
            domain_x=domain_x,
            domain_t=domain_t
        )
        self.alpha = alpha
        self.domain_y = domain_y
        self.output_dim = 1  # u
        self.input_dim = 3  # x, y, t
        self.name = "Heat2D"

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
        Compute 2D Heat equation residual.

        Args:
            u: Output [batch_size, 1]
            inputs: Input [batch_size, 3] as [x, y, t], requires_grad=True

        Returns:
            Residual [batch_size, 1]: u_t - alpha*(u_xx + u_yy)
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

        # Heat equation: u_t = alpha*(u_xx + u_yy)
        return u_t - self.alpha * (u_xx + u_yy)

    def get_params(self):
        return {'alpha': self.alpha}

    def initial_condition(self, x, y):
        """Default IC: u(x, y, 0) = sin(pi*x) * sin(pi*y)"""
        if isinstance(x, torch.Tensor):
            return torch.sin(np.pi * x) * torch.sin(np.pi * y)
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def boundary_condition(self, x, y, t):
        """Default: zero Dirichlet boundaries."""
        if isinstance(x, torch.Tensor):
            return torch.zeros_like(x)
        return 0.0

    def exact_solution(self, x, y, t):
        """
        Exact solution for sin(pi*x)*sin(pi*y) IC with zero BCs:
        u(x,y,t) = exp(-2*alpha*pi^2*t) * sin(pi*x) * sin(pi*y)
        """
        decay = np.exp(-2 * self.alpha * np.pi**2 * t)
        if isinstance(x, torch.Tensor):
            return decay * torch.sin(np.pi * x) * torch.sin(np.pi * y)
        return decay * np.sin(np.pi * x) * np.sin(np.pi * y)
