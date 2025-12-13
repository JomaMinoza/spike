"""
RK4 Integrator - 4th order Runge-Kutta method.

For linear system dz/dt = Az:
    k1 = A @ z
    k2 = A @ (z + dt/2 * k1)
    k3 = A @ (z + dt/2 * k2)
    k4 = A @ (z + dt * k3)
    z_next = z + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

More accurate than Euler, still explicit.
"""

import torch
from .base import BaseIntegrator


class RK4Integrator(BaseIntegrator):
    """4th order Runge-Kutta integrator."""

    def step(self, z: torch.Tensor, A: torch.Tensor, dt: float) -> torch.Tensor:
        """
        RK4 step for dz/dt = Az.

        Args:
            z: Current state [batch_size, dim]
            A: Linear operator [dim, dim] (weight matrix)
            dt: Time step

        Returns:
            z_next: State at t + dt
        """
        k1 = z @ A.T
        k2 = (z + 0.5 * dt * k1) @ A.T
        k3 = (z + 0.5 * dt * k2) @ A.T
        k4 = (z + dt * k3) @ A.T
        return z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    @property
    def name(self) -> str:
        return "rk4"

    @property
    def order(self) -> int:
        return 4
