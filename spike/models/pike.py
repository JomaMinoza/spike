"""
PIKE: Physics-Informed Koopman-Enhanced Neural Network

Combines PINN encoder with Koopman dynamics:
    Input (x,t) → PINN → u(x,t) → Koopman → g(u) → A @ g(u)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List

from .pinn import PINN
from .koopman import Koopman


class PIKE(nn.Module):
    """
    Physics-Informed Koopman-Enhanced Neural Network.

    Combines a PINN encoder with Koopman dynamics for learning
    linear representations of nonlinear PDE solutions.

    Architecture:
        1. PINN: (x, t) → u(x, t)
        2. Koopman: u → g(u) → dz/dt = A @ z

    Args:
        input_dim: Dimension of input coordinates
        output_dim: Dimension of solution
        embedding_dim: Dimension of Koopman observable space
        hidden_dim: Width of PINN hidden layers
        num_layers: Number of PINN hidden layers
        embedding_type: Type of Koopman embedding
        poly_degree: Polynomial degree for library
        mlp_hidden: Hidden size for learned embedding
        activation: Activation function
        integrator: Time integrator ('euler', 'rk4', 'expm')
        normalize_embedding: Normalize embedding during training
        derivative_terms: Derivative terms for library

    Example:
        >>> pike = PIKE(input_dim=2, output_dim=1, embedding_dim=64)
        >>> x = torch.randn(100, 2)
        >>> u, z = pike(x)
        >>> assert u.shape == (100, 1) and z.shape == (100, 64)
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
        derivative_terms: Optional[List[str]] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.normalize_embedding = normalize_embedding
        self.derivative_terms = derivative_terms or []
        self.use_derivatives = len(self.derivative_terms) > 0

        # Component 1: PINN encoder
        self.pinn = PINN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation
        )

        # Component 2: Koopman module
        self.koopman = Koopman(
            solution_dim=output_dim,
            embedding_dim=embedding_dim,
            embedding_type=embedding_type,
            poly_degree=poly_degree,
            mlp_hidden=mlp_hidden,
            activation=activation,
            use_skip=use_skip,
            integrator=integrator,
            derivative_terms=derivative_terms
        )

    def forward(
        self,
        x: torch.Tensor,
        compute_derivatives: bool = True
    ) -> tuple:
        """
        Forward pass through PIKE.

        Args:
            x: Input coordinates [batch_size, input_dim]
            compute_derivatives: Whether to compute derivatives for embedding

        Returns:
            u: Predicted solution [batch_size, output_dim]
            z: Koopman embedding [batch_size, embedding_dim]
        """
        # Ensure requires_grad for derivative computation
        if self.use_derivatives and compute_derivatives and not x.requires_grad:
            x = x.detach().requires_grad_(True)

        # Step 1: PINN forward pass
        u = self.pinn(x)

        # Step 2: Compute derivatives if needed
        derivatives = None
        if self.use_derivatives and compute_derivatives:
            derivatives = self._compute_derivatives(u, x)

        # Step 3: Koopman embedding
        z = self.koopman.embed(u, derivatives=derivatives)

        # Step 4: Normalize during training
        if self.normalize_embedding and self.training:
            z = z / (z.norm(dim=1, keepdim=True) + 1e-8)

        return u, z

    def _compute_derivatives(
        self,
        u: torch.Tensor,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute spatial derivatives of solution u w.r.t. input x."""
        derivatives = {}
        is_2d = self.input_dim == 3

        # First derivatives
        grad_u = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]

        if any(t in self.derivative_terms for t in ['u_x', 'u_ux', 'u_xx', 'u_xxx']):
            derivatives['u_x'] = grad_u[:, 0:1]

        if is_2d and 'u_y' in self.derivative_terms:
            derivatives['u_y'] = grad_u[:, 1:2]

        if 'u_t' in self.derivative_terms:
            t_idx = 2 if is_2d else 1
            derivatives['u_t'] = grad_u[:, t_idx:t_idx+1]

        # Second derivatives
        if any(t in self.derivative_terms for t in ['u_xx', 'u_uxx', 'u_xxx']):
            u_xx = torch.autograd.grad(
                derivatives['u_x'], x,
                grad_outputs=torch.ones_like(derivatives['u_x']),
                create_graph=True,
                retain_graph=True
            )[0][:, 0:1]
            derivatives['u_xx'] = u_xx

        if is_2d and 'u_yy' in self.derivative_terms:
            u_yy = torch.autograd.grad(
                derivatives['u_y'], x,
                grad_outputs=torch.ones_like(derivatives['u_y']),
                create_graph=True,
                retain_graph=True
            )[0][:, 1:2]
            derivatives['u_yy'] = u_yy

        # Third derivatives
        if 'u_xxx' in self.derivative_terms:
            u_xxx = torch.autograd.grad(
                derivatives['u_xx'], x,
                grad_outputs=torch.ones_like(derivatives['u_xx']),
                create_graph=True,
                retain_graph=True
            )[0][:, 0:1]
            derivatives['u_xxx'] = u_xxx

        return derivatives

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get solution only (PINN forward pass)."""
        return self.pinn(x)

    def embed(
        self,
        u: torch.Tensor,
        derivatives: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Embed solution into Koopman space."""
        return self.koopman.embed(u, derivatives=derivatives)

    def step(self, z: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """Advance embedding by dt using Koopman dynamics."""
        return self.koopman.step(z, dt)

    def get_koopman_matrix(self) -> torch.Tensor:
        """Get Koopman matrix A."""
        return self.koopman.get_matrix()

    def get_eigenvalues(self) -> torch.Tensor:
        """Get eigenvalues of Koopman matrix."""
        return self.koopman.get_eigenvalues()

    def is_stable(self) -> bool:
        """Check if Koopman dynamics are stable."""
        return self.koopman.is_stable()

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        """Print model summary."""
        print("=" * 60)
        print("PIKE (Physics-Informed Koopman-Enhanced) Model")
        print("=" * 60)
        print(f"Input dimension:      {self.input_dim}")
        print(f"Output dimension:     {self.output_dim}")
        print(f"Embedding dimension:  {self.embedding_dim}")
        print(f"Total parameters:     {self.count_parameters():,}")
        print()
        print("Architecture: (x,t) → PINN → u → Koopman → g(u) → A @ g(u)")
        print("=" * 60)

    def __repr__(self) -> str:
        return (
            f"PIKE(input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"embedding_dim={self.embedding_dim})"
        )
