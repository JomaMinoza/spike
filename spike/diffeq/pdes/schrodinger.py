"""
Nonlinear Schrödinger Equation
Complex dispersive PDE with cubic nonlinearity
"""

import torch
import numpy as np
from ..base import BasePDE


class SchrodingerEquation(BasePDE):
    """
    Nonlinear Schrödinger Equation (NLS).

    PDE: i·∂u/∂t + ∂²u/∂x² + |u|²·u = 0

    Split into real and imaginary parts (u = u_re + i·u_im):
    - ∂u_re/∂t = -∂²u_im/∂x² - |u|²·u_im
    - ∂u_im/∂t = ∂²u_re/∂x² + |u|²·u_re

    Where |u|² = u_re² + u_im²

    This equation describes:
    - Nonlinear wave propagation in optical fibers
    - Bose-Einstein condensates
    - Deep water waves

    Args:
        domain_x: Spatial domain (default: (-5, 5))
        domain_t: Time domain (default: (0, π/2))
    """

    def __init__(
        self,
        domain_x=(-5.0, 5.0),
        domain_t=(0.0, np.pi / 2)
    ):
        super().__init__(
            domain_x=domain_x,
            domain_t=domain_t
        )
        self.output_dim = 2  # u_re, u_im
        self.input_dim = 2  # x, t
        self.name = "Schrodinger"

    def residual(self, u, inputs):
        """
        Compute Schrödinger residuals.

        Args:
            u: Output [batch_size, 2] as [u_re, u_im]
            inputs: Input [batch_size, 2] as [x, t], requires_grad=True

        Returns:
            Residual [batch_size, 2] for real and imaginary equations
        """
        u_re = u[:, 0:1]
        u_im = u[:, 1:2]

        # |u|^2 = u_re^2 + u_im^2
        u_sq = u_re ** 2 + u_im ** 2

        # Gradients of real part
        grad_re = torch.autograd.grad(
            u_re, inputs, grad_outputs=torch.ones_like(u_re),
            create_graph=True, retain_graph=True
        )[0]
        u_re_x = grad_re[:, 0:1]
        u_re_t = grad_re[:, 1:2]

        # Second derivative of real part
        grad_re_x = torch.autograd.grad(
            u_re_x, inputs, grad_outputs=torch.ones_like(u_re_x),
            create_graph=True, retain_graph=True
        )[0]
        u_re_xx = grad_re_x[:, 0:1]

        # Gradients of imaginary part
        grad_im = torch.autograd.grad(
            u_im, inputs, grad_outputs=torch.ones_like(u_im),
            create_graph=True, retain_graph=True
        )[0]
        u_im_x = grad_im[:, 0:1]
        u_im_t = grad_im[:, 1:2]

        # Second derivative of imaginary part
        grad_im_x = torch.autograd.grad(
            u_im_x, inputs, grad_outputs=torch.ones_like(u_im_x),
            create_graph=True, retain_graph=True
        )[0]
        u_im_xx = grad_im_x[:, 0:1]

        # NLS: i·u_t + u_xx + |u|^2·u = 0
        # Real part: u_re_t + u_im_xx + |u|^2·u_im = 0
        # Imag part: u_im_t - u_re_xx - |u|^2·u_re = 0
        res_re = u_re_t + u_im_xx + u_sq * u_im
        res_im = u_im_t - u_re_xx - u_sq * u_re

        return torch.cat([res_re, res_im], dim=1)

    def get_params(self):
        return {}

    def initial_condition(self, x):
        """
        Soliton initial condition: u(x,0) = 2·sech(x)

        Returns real and imaginary parts.
        """
        if isinstance(x, torch.Tensor):
            sech = 1.0 / torch.cosh(x)
            u_re = 2.0 * sech
            u_im = torch.zeros_like(x)
        else:
            sech = 1.0 / np.cosh(x)
            u_re = 2.0 * sech
            u_im = np.zeros_like(x)
        return u_re, u_im

    def analytical_solution(self, x, t):
        """
        Analytical soliton solution:
        u(x,t) = 2·sech(x)·exp(i·t)

        Returns real and imaginary parts.
        """
        if isinstance(x, torch.Tensor):
            sech = 1.0 / torch.cosh(x)
            u_re = 2.0 * sech * torch.cos(t)
            u_im = 2.0 * sech * torch.sin(t)
        else:
            sech = 1.0 / np.cosh(x)
            u_re = 2.0 * sech * np.cos(t)
            u_im = 2.0 * sech * np.sin(t)
        return u_re, u_im

    def boundary_condition(self, t, boundary='left'):
        """Decaying boundary conditions (soliton vanishes at infinity)."""
        if isinstance(t, torch.Tensor):
            return torch.zeros_like(t), torch.zeros_like(t)
        return 0.0, 0.0
