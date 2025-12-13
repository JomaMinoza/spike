"""
Matrix Exponential Integrator - exact solution for linear systems.

z_next = expm(dt * A) @ z

This is EXACT for linear dz/dt = Az. Best for stiff PDEs (Cahn-Hilliard, KS, etc.)
Uses Taylor series with scaling and squaring for numerical stability.
"""

import torch
from .base import BaseIntegrator


def matrix_exponential(A: torch.Tensor, num_terms: int = 20) -> torch.Tensor:
    """
    Compute matrix exponential using Taylor series with scaling and squaring.

    exp(A) = I + A + A²/2! + A³/3! + ...

    For large ||A||, uses scaling and squaring for stability.

    Args:
        A: Square matrix [n, n]
        num_terms: Number of Taylor series terms

    Returns:
        exp(A): Matrix exponential [n, n]
    """
    n = A.shape[0]
    device = A.device
    dtype = A.dtype

    # Scaling: find s such that ||A/2^s|| < 1
    norm_A = torch.linalg.norm(A, ord=2)
    s = max(0, int(torch.ceil(torch.log2(norm_A + 1e-8)).item()))
    A_scaled = A / (2 ** s)

    # Taylor series for exp(A_scaled)
    result = torch.eye(n, device=device, dtype=dtype)
    term = torch.eye(n, device=device, dtype=dtype)

    for k in range(1, num_terms):
        term = term @ A_scaled / k
        result = result + term

        # Early stopping if converged
        if torch.linalg.norm(term, ord='fro') < 1e-10:
            break

    # Squaring: exp(A) = exp(A_scaled)^(2^s)
    for _ in range(s):
        result = result @ result

    return result


class ExpmIntegrator(BaseIntegrator):
    """Matrix exponential integrator (exact for linear systems)."""

    def __init__(self, num_terms: int = 20):
        """
        Args:
            num_terms: Number of Taylor series terms for matrix exponential
        """
        self.num_terms = num_terms

    def step(self, z: torch.Tensor, A: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Exact step using matrix exponential: z_next = expm(dt*A) @ z

        Args:
            z: Current state [batch_size, dim]
            A: Linear operator [dim, dim] (weight matrix)
            dt: Time step

        Returns:
            z_next: State at t + dt
        """
        exp_dtA = matrix_exponential(dt * A, self.num_terms)
        return z @ exp_dtA.T

    @property
    def name(self) -> str:
        return "expm"

    @property
    def order(self) -> int:
        return float('inf')  # Exact for linear systems
