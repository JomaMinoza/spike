"""
2D Burgers Equations
Nonlinear convection-diffusion in 2D
"""

import torch
from ..base import BasePDE


class Burgers2D(BasePDE):
    """
    2D Burgers Equations.

    PDEs:
    - ∂u/∂t + u·∂u/∂x + v·∂u/∂y = ν(∂²u/∂x² + ∂²u/∂y²)
    - ∂v/∂t + u·∂v/∂x + v·∂v/∂y = ν(∂²v/∂x² + ∂²v/∂y²)

    Where:
    - u, v = velocity components
    - ν = viscosity coefficient

    Args:
        nu: Viscosity coefficient (default: 0.01)
        domain_x: Spatial domain in x (default: (0, 1))
        domain_y: Spatial domain in y (default: (0, 1))
        domain_t: Time domain (default: (0, 1))
    """

    def __init__(
        self,
        nu: float = 0.01,
        domain_x=(0.0, 1.0),
        domain_y=(0.0, 1.0),
        domain_t=(0.0, 1.0)
    ):
        super().__init__(
            domain_x=domain_x,
            domain_t=domain_t
        )
        self.nu = nu
        self.domain_y = domain_y
        self.output_dim = 2  # u, v
        self.input_dim = 3  # x, y, t
        self.name = "Burgers2D"

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

    def residual(self, uv, inputs):
        """
        Compute 2D Burgers residuals.

        Args:
            uv: Output [batch_size, 2] as [u, v]
            inputs: Input [batch_size, 3] as [x, y, t], requires_grad=True

        Returns:
            Residual [batch_size, 2] for u and v equations
        """
        u = uv[:, 0:1]
        v = uv[:, 1:2]

        # First derivatives of u
        grad_u = torch.autograd.grad(
            u, inputs, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]
        u_t = grad_u[:, 2:3]

        # First derivatives of v
        grad_v = torch.autograd.grad(
            v, inputs, grad_outputs=torch.ones_like(v),
            create_graph=True, retain_graph=True
        )[0]
        v_x = grad_v[:, 0:1]
        v_y = grad_v[:, 1:2]
        v_t = grad_v[:, 2:3]

        # Second derivatives of u
        grad_u_x = torch.autograd.grad(
            u_x, inputs, grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0]
        u_xx = grad_u_x[:, 0:1]

        grad_u_y = torch.autograd.grad(
            u_y, inputs, grad_outputs=torch.ones_like(u_y),
            create_graph=True, retain_graph=True
        )[0]
        u_yy = grad_u_y[:, 1:2]

        # Second derivatives of v
        grad_v_x = torch.autograd.grad(
            v_x, inputs, grad_outputs=torch.ones_like(v_x),
            create_graph=True, retain_graph=True
        )[0]
        v_xx = grad_v_x[:, 0:1]

        grad_v_y = torch.autograd.grad(
            v_y, inputs, grad_outputs=torch.ones_like(v_y),
            create_graph=True, retain_graph=True
        )[0]
        v_yy = grad_v_y[:, 1:2]

        # Burgers equations:
        # u_t + u*u_x + v*u_y = nu*(u_xx + u_yy)
        # v_t + u*v_x + v*v_y = nu*(v_xx + v_yy)
        res_u = u_t + u * u_x + v * u_y - self.nu * (u_xx + u_yy)
        res_v = v_t + u * v_x + v * v_y - self.nu * (v_xx + v_yy)

        return torch.cat([res_u, res_v], dim=1)

    def get_params(self):
        return {'nu': self.nu}

    def initial_condition(self, x, y):
        """Default initial condition: sinusoidal."""
        import numpy as np
        if isinstance(x, torch.Tensor):
            u0 = torch.sin(np.pi * x) * torch.sin(np.pi * y)
            v0 = torch.sin(np.pi * x) * torch.sin(np.pi * y)
        else:
            u0 = np.sin(np.pi * x) * np.sin(np.pi * y)
            v0 = np.sin(np.pi * x) * np.sin(np.pi * y)
        return u0, v0

    def boundary_condition(self, x, y, t):
        """Default: zero Dirichlet boundaries."""
        batch_size = x.shape[0] if isinstance(x, torch.Tensor) else 1
        zeros = torch.zeros(batch_size, 1) if isinstance(x, torch.Tensor) else 0.0
        return {'u': zeros, 'v': zeros}
