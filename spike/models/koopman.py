"""
Koopman Module
Embedding layer + linear operator for discovering Koopman observables

This module maps solutions to observable space and provides linear dynamics.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple

from ..integrators import get_integrator, BaseIntegrator
from .embedding import SparseEmbedding


class Koopman(nn.Module):
    """
    Koopman operator module: embedding + linear dynamics.

    Architecture:
        u(t) → g(u) → z ∈ R^d
        dz/dt = A @ z  (continuous-time)

    The embedding g() maps solutions to observable space where dynamics are linear.
    The operator A is the infinitesimal generator of the Koopman semigroup.

    Args:
        solution_dim: Dimension of solution (input to embedding)
        embedding_dim: Dimension of Koopman observable space
        embedding_type: Type of embedding ('library', 'learned', 'augmented')
        poly_degree: Polynomial degree for library embedding
        mlp_hidden: Hidden layer size for learned embedding branch
        activation: Activation function for embedding MLP
        use_skip: Include identity skip connection in embedding
        integrator: Time integration method ('euler', 'rk4', 'expm')
        derivative_terms: List of derivative terms for library

    Example:
        >>> koopman = Koopman(solution_dim=1, embedding_dim=64)
        >>> u = torch.randn(100, 1)
        >>> z = koopman.embed(u)  # Embed solution
        >>> z_next = koopman.step(z, dt=0.01)  # Advance in time
    """

    def __init__(
        self,
        solution_dim: int = 1,
        embedding_dim: int = 64,
        embedding_type: str = 'augmented',
        poly_degree: int = 2,
        mlp_hidden: int = 64,
        activation: str = 'tanh',
        use_skip: bool = False,
        integrator: str = 'euler',
        derivative_terms: Optional[List[str]] = None
    ):
        super().__init__()
        self.solution_dim = solution_dim
        self.embedding_dim = embedding_dim
        self.integrator_name = integrator

        # Embedding layer: solution → observable space
        self.embedding = SparseEmbedding(
            input_dim=solution_dim,
            embedding_dim=embedding_dim,
            embedding_type=embedding_type,
            poly_degree=poly_degree,
            mlp_hidden=mlp_hidden,
            activation=activation,
            use_skip=use_skip,
            derivative_terms=derivative_terms
        )

        # Koopman operator A: linear dynamics in observable space
        # dz/dt = A @ z, so z(t+dt) ≈ z(t) + dt * A @ z(t)
        self.A = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Time integrator
        self._integrator = get_integrator(integrator)

    @property
    def integrator(self) -> BaseIntegrator:
        """Get the time integrator."""
        return self._integrator

    def embed(
        self,
        u: torch.Tensor,
        derivatives: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Embed solution into Koopman observable space.

        Args:
            u: Solution values [batch_size, solution_dim]
            derivatives: Optional dict of derivative tensors

        Returns:
            z: Koopman embedding [batch_size, embedding_dim]
        """
        return self.embedding(u, derivatives=derivatives)

    def dynamics(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute time derivative dz/dt = A @ z.

        Args:
            z: Current embedding [batch_size, embedding_dim]

        Returns:
            dz_dt: Time derivative [batch_size, embedding_dim]
        """
        return self.A(z)

    def step(self, z: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """
        Advance embedding by dt using the integrator.

        Args:
            z: Current embedding [batch_size, embedding_dim]
            dt: Time step

        Returns:
            z_next: Embedding at t + dt
        """
        return self._integrator.step(z, self.A.weight, dt)

    def multi_step(
        self,
        z: torch.Tensor,
        dt: float,
        n_steps: int
    ) -> torch.Tensor:
        """
        Advance embedding by multiple steps.

        Args:
            z: Initial embedding [batch_size, embedding_dim]
            dt: Time step per step
            n_steps: Number of steps

        Returns:
            z_final: Embedding after n_steps
        """
        for _ in range(n_steps):
            z = self.step(z, dt)
        return z

    def get_matrix(self) -> torch.Tensor:
        """Get the Koopman matrix A."""
        return self.A.weight.data

    def get_eigenvalues(self) -> torch.Tensor:
        """Compute eigenvalues of Koopman matrix A."""
        A = self.get_matrix()
        return torch.linalg.eigvals(A)

    def get_spectral_radius(self) -> float:
        """
        Compute spectral radius (max absolute eigenvalue).

        For continuous-time A, stability requires Re(λ) < 0 for all eigenvalues.
        """
        eigenvalues = self.get_eigenvalues()
        return torch.max(torch.abs(eigenvalues)).item()

    def get_max_real_part(self) -> float:
        """
        Get maximum real part of eigenvalues.

        For continuous-time systems, stability requires max(Re(λ)) < 0.
        """
        eigenvalues = self.get_eigenvalues()
        return torch.max(eigenvalues.real).item()

    def is_stable(self) -> bool:
        """
        Check if Koopman dynamics are stable.

        For continuous-time A: stable if max(Re(λ)) < 0
        """
        return self.get_max_real_part() < 0

    def get_sparsity(self, threshold: float = 1e-4) -> dict:
        """
        Get sparsity information of Koopman matrix A.

        Args:
            threshold: Values below this are considered zero

        Returns:
            dict with sparsity metrics
        """
        A = self.get_matrix()
        total = A.numel()
        near_zero = (torch.abs(A) < threshold).sum().item()

        return {
            'sparsity_percent': (near_zero / total) * 100,
            'l1_norm': torch.abs(A).sum().item(),
            'active_entries': total - near_zero,
            'total_entries': total
        }

    def l1_norm(self) -> torch.Tensor:
        """Get L1 norm of Koopman matrix A (for regularization)."""
        return torch.abs(self.A.weight).sum()

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"Koopman(solution_dim={self.solution_dim}, "
            f"embedding_dim={self.embedding_dim}, "
            f"integrator='{self.integrator_name}')"
        )
