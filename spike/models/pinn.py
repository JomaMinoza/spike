"""
Base PINN Module
Encoder network that maps input coordinates (x,t) to solution u(x,t)
"""

import torch
import torch.nn as nn
from typing import Optional


class PINN(nn.Module):
    """
    Base Physics-Informed Neural Network.

    Simple encoder that maps input coordinates to solution values.
    This is the foundation that can be extended with Koopman dynamics.

    Architecture:
        Input (x, t) → MLP → u(x, t)

    Args:
        input_dim: Dimension of input coordinates (e.g., 2 for [x, t], 3 for [x, y, t])
        output_dim: Dimension of solution (e.g., 1 for scalar, 3 for NS velocity)
        hidden_dim: Width of hidden layers
        num_layers: Number of hidden layers
        activation: Activation function ('tanh', 'relu', 'silu')

    Example:
        >>> pinn = PINN(input_dim=2, output_dim=1)
        >>> x = torch.randn(100, 2)
        >>> u = pinn(x)
        >>> assert u.shape == (100, 1)
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 3,
        activation: str = 'tanh'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Activation function
        if activation == 'tanh':
            act = nn.Tanh()
        elif activation == 'relu':
            act = nn.ReLU()
        elif activation == 'silu':
            act = nn.SiLU()
        else:
            act = nn.Tanh()

        # Build encoder network
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(act)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act)
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: input coords → solution.

        Args:
            x: Input coordinates [batch_size, input_dim]

        Returns:
            u: Predicted solution [batch_size, output_dim]
        """
        return self.encoder(x)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"PINN(input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"hidden_dim={self.hidden_dim}, num_layers={self.num_layers})"
        )
