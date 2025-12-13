"""
Physics Loss
PDE residual loss to enforce physics constraints
"""

import torch


class PhysicsLoss:
    """
    Physics-informed loss that enforces PDE satisfaction.

    Computes the residual of the PDE at collocation points and
    penalizes deviations from zero.

    Args:
        pde: PDE object with residual(u, x) method

    Example:
        >>> from spike.diffeq.pdes import BurgersEquation
        >>> pde = BurgersEquation(nu=0.01)
        >>> loss_fn = PhysicsLoss(pde)
        >>> loss = loss_fn(model, x_collocation)
    """

    def __init__(self, pde):
        self.pde = pde

    def __call__(self, model, x: torch.Tensor) -> torch.Tensor:
        """
        Compute physics loss at given points.

        Args:
            model: SPIKE/PIKE model
            x: Collocation points [batch_size, input_dim], requires_grad=True

        Returns:
            Mean squared PDE residual
        """
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)

        # Forward pass (get solution u)
        u, _ = model(x)

        # Compute PDE residual
        residual = self.pde.residual(u, x)

        # Mean squared error
        return torch.mean(residual ** 2)

    def compute_stats(self, model, x: torch.Tensor) -> dict:
        """
        Compute detailed residual statistics.

        Args:
            model: SPIKE/PIKE model
            x: Collocation points

        Returns:
            dict with mean, std, max, min, rms of residuals
        """
        with torch.no_grad():
            if not x.requires_grad:
                x = x.clone().detach().requires_grad_(True)

            u, _ = model(x)
            residual = self.pde.residual(u, x)

            return {
                'mean': residual.mean().item(),
                'std': residual.std().item(),
                'max': residual.max().item(),
                'min': residual.min().item(),
                'rms': torch.sqrt(torch.mean(residual ** 2)).item()
            }
