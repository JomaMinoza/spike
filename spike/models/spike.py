"""
SPIKE: Sparse Physics-Informed Koopman-Enhanced Neural Network

PIKE + L1 sparsity regularization on Koopman matrix for interpretability.
    Input (x,t) → PINN → u(x,t) → Koopman → g(u) → sparse A @ g(u)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List

from .pike import PIKE


class SPIKE(PIKE):
    """
    Sparse Physics-Informed Koopman-Enhanced Neural Network.

    Extends PIKE with L1 sparsity regularization on the Koopman matrix A
    for discovering sparse, interpretable representations of PDE dynamics.

    Key insight: A sparse Koopman matrix reveals which observable interactions
    matter for the dynamics, potentially recovering governing equations.

    Architecture:
        Same as PIKE: (x,t) → PINN → u → Koopman → sparse A @ g(u)

    Sparsity:
        The Koopman matrix A is regularized with L1 penalty during training.
        This encourages sparse dynamics where only essential interactions remain.

    Args:
        All PIKE arguments, plus:
        sparsity_target: Target sparsity percentage (for reporting only)

    Example:
        >>> spike = SPIKE(input_dim=2, output_dim=1, embedding_dim=64)
        >>> x = torch.randn(100, 2)
        >>> u, z = spike(x)

        >>> # Get sparsity info
        >>> info = spike.get_sparsity_info()
        >>> print(f"Sparsity: {info['sparsity_percent']:.1f}%")

        >>> # Get L1 norm for loss computation
        >>> l1 = spike.l1_norm()
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 1,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        embedding_type: str = 'augmented',
        poly_degree: int = 2,
        mlp_hidden: int = 64,
        activation: str = 'tanh',
        use_skip: bool = False,
        integrator: str = 'euler',
        normalize_embedding: bool = True,
        derivative_terms: Optional[List[str]] = None,
        sparsity_target: float = 80.0  # Target sparsity % (informational)
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            embedding_type=embedding_type,
            poly_degree=poly_degree,
            mlp_hidden=mlp_hidden,
            activation=activation,
            use_skip=use_skip,
            integrator=integrator,
            normalize_embedding=normalize_embedding,
            derivative_terms=derivative_terms
        )
        self.sparsity_target = sparsity_target

    def l1_norm(self) -> torch.Tensor:
        """
        Get L1 norm of Koopman matrix A.

        Use this in your loss function:
            loss = physics_loss + koopman_loss + lambda_sparse * spike.l1_norm()

        Returns:
            L1 norm as scalar tensor
        """
        return self.koopman.l1_norm()

    def get_sparsity_info(self, threshold: float = 1e-4) -> dict:
        """
        Get comprehensive sparsity information.

        Args:
            threshold: Values below this are considered zero

        Returns:
            dict with:
                - sparsity_percent: % of near-zero entries
                - l1_norm: L1 norm of A
                - active_entries: Number of non-zero entries
                - total_entries: Total entries in A
                - compression_ratio: active/total
                - target_sparsity: Target sparsity %
        """
        info = self.koopman.get_sparsity(threshold)
        info['compression_ratio'] = info['active_entries'] / info['total_entries']
        info['target_sparsity'] = self.sparsity_target
        return info

    def get_koopman_info(self) -> dict:
        """
        Get information about Koopman operator.

        Returns:
            dict with:
                - spectral_radius: Max |eigenvalue|
                - max_real_part: Max Re(eigenvalue)
                - is_stable: Whether dynamics are stable
                - eigenvalues: All eigenvalues
        """
        eigenvalues = self.koopman.get_eigenvalues()
        return {
            'spectral_radius': self.koopman.get_spectral_radius(),
            'max_real_part': self.koopman.get_max_real_part(),
            'is_stable': self.koopman.is_stable(),
            'eigenvalues': eigenvalues.cpu().numpy()
        }

    def prune(self, threshold: float = 1e-4, inplace: bool = True) -> Optional['SPIKE']:
        """
        Prune small weights in Koopman matrix to exactly zero.

        Args:
            threshold: Values below this become zero
            inplace: If True, modify this model. If False, return pruned copy.

        Returns:
            Pruned model (if inplace=False) or None (if inplace=True)
        """
        if inplace:
            model = self
        else:
            import copy
            model = copy.deepcopy(self)

        with torch.no_grad():
            A = model.koopman.A.weight
            mask = torch.abs(A) < threshold
            A[mask] = 0.0

        return None if inplace else model

    def get_active_structure(self, threshold: float = 1e-4) -> torch.Tensor:
        """
        Get binary mask showing active (non-zero) structure of A.

        Args:
            threshold: Values below this are considered zero

        Returns:
            Binary mask [embedding_dim, embedding_dim]
        """
        A = self.koopman.get_matrix()
        return (torch.abs(A) >= threshold).float()

    def summary(self):
        """Print model summary with sparsity information."""
        print("=" * 70)
        print("SPIKE (Sparse Physics-Informed Koopman-Enhanced) Model")
        print("=" * 70)
        print(f"Input dimension:      {self.input_dim}")
        print(f"Output dimension:     {self.output_dim}")
        print(f"Embedding dimension:  {self.embedding_dim}")
        print(f"Total parameters:     {self.count_parameters():,}")
        print()
        print("Architecture: (x,t) → PINN → u → Koopman → sparse A @ g(u)")
        print()

        sparsity = self.get_sparsity_info()
        print("Sparsity Information:")
        print(f"  Current sparsity:   {sparsity['sparsity_percent']:.1f}%")
        print(f"  Target sparsity:    {self.sparsity_target:.1f}%")
        print(f"  Active entries:     {sparsity['active_entries']}/{sparsity['total_entries']}")
        print(f"  L1 norm:            {sparsity['l1_norm']:.4f}")
        print()

        koopman = self.get_koopman_info()
        print("Koopman Information:")
        print(f"  Spectral radius:    {koopman['spectral_radius']:.4f}")
        print(f"  Max Re(eigenvalue): {koopman['max_real_part']:.4f}")
        print(f"  Stable dynamics:    {koopman['is_stable']}")
        print("=" * 70)

    def __repr__(self) -> str:
        return (
            f"SPIKE(input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"embedding_dim={self.embedding_dim}, target_sparsity={self.sparsity_target}%)"
        )
