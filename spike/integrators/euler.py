"""
Euler Integrator - 1st order explicit method.

z_next = z + dt * A @ z

Fast but may be unstable for stiff systems.
"""

import torch
from .base import BaseIntegrator


class EulerIntegrator(BaseIntegrator):
    """Forward Euler integrator (1st order)."""

    def step(self, z: torch.Tensor, A: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Euler step: z_next = z + dt * (z @ A^T)

        Args:
            z: Current state [batch_size, dim]
            A: Linear operator [dim, dim] (weight matrix)
            dt: Time step

        Returns:
            z_next: State at t + dt
        """
        dz_dt = z @ A.T
        return z + dt * dz_dt

    @property
    def name(self) -> str:
        return "euler"

    @property
    def order(self) -> int:
        return 1
