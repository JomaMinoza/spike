"""
Base Differential Equation Classes

Abstract base classes for PDEs and ODEs.
"""

from abc import ABC, abstractmethod
import torch
from typing import Dict, Tuple, Optional


class BaseDiffEq(ABC):
    """
    Abstract base class for all differential equations.

    Parent class for both PDEs and ODEs.
    """

    def __init__(self):
        self.name = "BaseDiffEq"

    @abstractmethod
    def residual(self, u, inputs):
        """Compute equation residual for physics loss."""
        pass

    @abstractmethod
    def initial_condition(self, x):
        """Compute initial condition/state."""
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """Get equation parameters."""
        pass

    def exact_solution(self, inputs):
        """Compute exact solution if available."""
        return None


class BasePDE(BaseDiffEq):
    """
    Abstract base class for 1D PDEs.

    PDEs have spatial domain (x) and temporal domain (t).
    Input dimension is 2: (x, t)

    Args:
        domain_x: Tuple (x_min, x_max) for spatial domain
        domain_t: Tuple (t_min, t_max) for temporal domain
    """

    def __init__(
        self,
        domain_x: Tuple[float, float] = (-1.0, 1.0),
        domain_t: Tuple[float, float] = (0.0, 1.0)
    ):
        super().__init__()
        self.domain_x = domain_x
        self.domain_t = domain_t
        self.input_dim = 2
        self.output_dim = 1
        self.name = "BasePDE"

    @abstractmethod
    def residual(self, u, x):
        """
        Compute PDE residual.

        Args:
            u: Solution tensor [batch_size, 1]
            x: Input coordinates [batch_size, 2] (x, t), requires_grad=True

        Returns:
            Residual values [batch_size, 1]
        """
        pass

    @abstractmethod
    def initial_condition(self, x):
        """
        Compute initial condition u(x, t=0).

        Args:
            x: Spatial coordinates [batch_size, 1]

        Returns:
            Initial values [batch_size, 1]
        """
        pass

    @abstractmethod
    def boundary_condition(self, t, boundary='left'):
        """
        Compute boundary condition.

        Args:
            t: Time values [batch_size, 1]
            boundary: 'left' or 'right'

        Returns:
            Boundary values [batch_size, 1]
        """
        pass

    def compute_derivatives(self, u: torch.Tensor, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute spatial and temporal derivatives using autograd.

        Args:
            u: Solution [batch_size, 1]
            x: Input coordinates [batch_size, 2], requires_grad=True

        Returns:
            Dict with u_x, u_t, u_xx
        """
        grad_u = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]

        u_x = grad_u[:, 0:1]
        u_t = grad_u[:, 1:2]

        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]

        return {'u_x': u_x, 'u_t': u_t, 'u_xx': u_xx}

    def get_domain(self) -> dict:
        """Get domain bounds."""
        return {
            'x_min': self.domain_x[0],
            'x_max': self.domain_x[1],
            't_min': self.domain_t[0],
            't_max': self.domain_t[1]
        }

    def __repr__(self):
        return f"{self.name}(x={self.domain_x}, t={self.domain_t})"


class BaseODE(BaseDiffEq):
    """
    Abstract base class for ODE systems.

    ODEs have temporal dynamics only (no spatial domain).
    Input dimension is 1: (t)

    Args:
        domain_t: Tuple (t_min, t_max) for temporal domain
        output_dim: Dimension of state vector
    """

    def __init__(
        self,
        domain_t: Tuple[float, float] = (0.0, 100.0),
        output_dim: int = 1
    ):
        super().__init__()
        self.domain_t = domain_t
        self.output_dim = output_dim
        self.input_dim = 1
        self.name = "BaseODE"

    @abstractmethod
    def residual(self, state, t):
        """
        Compute ODE residual.

        Args:
            state: State tensor [batch_size, output_dim]
            t: Time tensor [batch_size, 1], requires_grad=True

        Returns:
            Residual values [batch_size, output_dim]
        """
        pass

    @abstractmethod
    def initial_condition(self, dummy=None):
        """
        Return initial state vector.

        Returns:
            Initial state [1, output_dim]
        """
        pass

    def boundary_condition(self, t, boundary='left'):
        """ODEs don't have spatial boundary conditions."""
        return None

    def compute_time_derivative(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute time derivative dy/dt using autograd.

        Args:
            y: State component [batch_size, 1]
            t: Time tensor [batch_size, 1], requires_grad=True

        Returns:
            dy/dt [batch_size, 1]
        """
        return torch.autograd.grad(
            y, t,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True
        )[0]

    def get_domain(self) -> dict:
        """Get temporal domain bounds."""
        return {'t_min': self.domain_t[0], 't_max': self.domain_t[1]}

    @abstractmethod
    def get_ode_func(self):
        """
        Get the ODE function for numerical integration.

        Returns:
            Callable: Function f(t, y) -> dy/dt
        """
        pass

    def __repr__(self):
        return f"{self.name}(t={self.domain_t}, dim={self.output_dim})"
