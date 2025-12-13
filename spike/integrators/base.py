"""
Base Integrator ABC for Koopman time-stepping.

All integrators solve dz/dt = Az for linear dynamics.
"""

from abc import ABC, abstractmethod
import torch


class BaseIntegrator(ABC):
    """Abstract base class for time integrators."""

    @abstractmethod
    def step(self, z: torch.Tensor, A: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Advance state z by dt using dynamics dz/dt = Az.

        Args:
            z: Current state [batch_size, dim]
            A: Linear operator [dim, dim]
            dt: Time step

        Returns:
            z_next: State at t + dt
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return integrator name."""
        pass

    @property
    def order(self) -> int:
        """Return order of accuracy (1 for Euler, 4 for RK4, inf for expm)."""
        return 1
