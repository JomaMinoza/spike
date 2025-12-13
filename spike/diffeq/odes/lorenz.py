"""
Lorenz System
Classic chaotic dynamical system
"""

import torch
from ..base import BaseODE


class LorenzSystem(BaseODE):
    """
    Lorenz System (Chaotic ODEs).

    ODEs:
    - dx/dt = sigma * (y - x)
    - dy/dt = x * (rho - z) - y
    - dz/dt = x * y - beta * z

    The iconic chaotic attractor, demonstrating sensitivity to initial conditions.
    Residual uses time derivative formulation for training stability.

    Args:
        sigma: Prandtl number (default: 10.0)
        rho: Rayleigh number (default: 28.0)
        beta: Geometric factor (default: 8/3)
        domain_t: Time domain (default: (0, 25))
    """

    def __init__(
        self,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0/3.0,
        domain_t=(0.0, 25.0)
    ):
        super().__init__(domain_t=domain_t, output_dim=3)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.name = "LorenzSystem"

    def residual(self, xyz, t):
        """
        Compute Lorenz system residual: d(x+y+z)/dt

        Args:
            xyz: State [batch_size, 3] where columns are [x, y, z]
            t: Time [batch_size, 1], requires_grad=True

        Returns:
            Residual [batch_size, 1]
        """
        # Sum of all components, then take time derivative
        u_sum = xyz.sum(dim=1, keepdim=True)
        u_t = torch.autograd.grad(
            u_sum, t, grad_outputs=torch.ones_like(u_sum),
            create_graph=True, retain_graph=True
        )[0]
        return u_t

    def initial_condition(self, dummy=None):
        """Initial state: classic starting point near attractor."""
        return torch.tensor([[1.0, 1.0, 1.0]])

    def get_params(self):
        return {
            'sigma': self.sigma,
            'rho': self.rho,
            'beta': self.beta
        }

    def is_chaotic(self) -> bool:
        """Check if parameters are in chaotic regime."""
        return self.rho > 24.74

    def get_ode_func(self):
        """Get ODE function for scipy.integrate.solve_ivp."""
        sigma, rho, beta = self.sigma, self.rho, self.beta

        def lorenz_ode(t, state):
            x, y, z = state
            dx_dt = sigma * (y - x)
            dy_dt = x * (rho - z) - y
            dz_dt = x * y - beta * z
            return [dx_dt, dy_dt, dz_dt]

        return lorenz_ode
