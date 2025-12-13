"""
Combined Loss
Weighted combination of physics, Koopman, and sparsity losses

Total = λ_physics * Physics + λ_koopman * Koopman + λ_sparse * Sparsity
"""

from typing import Optional, Dict, Tuple
import torch

from .physics import PhysicsLoss
from .koopman import KoopmanLoss
from .sparsity import SparsityLoss


class CombinedLoss:
    """
    Weighted combination of all SPIKE loss components.

    Total Loss = λ_physics * Physics + λ_koopman * Koopman + λ_sparse * Sparsity

    Args:
        pde: PDE object with residual() method
        lambda_physics: Weight for physics loss (default: 1.0)
        lambda_koopman: Weight for Koopman loss (default: 0.1)
        lambda_sparse: Weight for sparsity loss (default: 0.001)
        koopman_norm: Norm for Koopman loss ('mse', 'l1', 'huber')
        sparsity_norm: Norm for sparsity ('l1', 'l0_approx', 'l1_l2')

    Example:
        >>> from spike.diffeq.pdes import BurgersEquation
        >>> pde = BurgersEquation(nu=0.01)
        >>> loss_fn = CombinedLoss(
        ...     pde,
        ...     lambda_physics=1.0,
        ...     lambda_koopman=0.1,
        ...     lambda_sparse=0.001
        ... )
        >>> total, components = loss_fn(model, x_phys, x_k, x_k_next, dt=0.01)
    """

    def __init__(
        self,
        pde,
        lambda_physics: float = 1.0,
        lambda_koopman: float = 0.1,
        lambda_sparse: float = 0.001,
        koopman_norm: str = 'mse',
        sparsity_norm: str = 'l1'
    ):
        self.physics_loss = PhysicsLoss(pde)
        self.koopman_loss = KoopmanLoss(norm=koopman_norm)
        self.sparsity_loss = SparsityLoss(norm_type=sparsity_norm)

        self.lambda_physics = lambda_physics
        self.lambda_koopman = lambda_koopman
        self.lambda_sparse = lambda_sparse

    def __call__(
        self,
        model,
        x_physics: torch.Tensor,
        x_koopman: torch.Tensor,
        x_koopman_next: torch.Tensor,
        dt: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total weighted loss.

        Args:
            model: SPIKE/PIKE model
            x_physics: Collocation points for physics loss
            x_koopman: Current state points for Koopman
            x_koopman_next: Next state points for Koopman
            dt: Time step for Koopman prediction

        Returns:
            total_loss: Weighted sum
            components: Dict with individual loss values
        """
        loss_p = self.physics_loss(model, x_physics)
        loss_k = self.koopman_loss(model, x_koopman, x_koopman_next, dt)
        loss_s = self.sparsity_loss(model)

        total = (
            self.lambda_physics * loss_p +
            self.lambda_koopman * loss_k +
            self.lambda_sparse * loss_s
        )

        components = {
            'physics': loss_p.item(),
            'koopman': loss_k.item(),
            'sparsity': loss_s.item(),
            'total': total.item()
        }

        return total, components

    def update_weights(
        self,
        lambda_physics: Optional[float] = None,
        lambda_koopman: Optional[float] = None,
        lambda_sparse: Optional[float] = None
    ):
        """Update loss weights dynamically."""
        if lambda_physics is not None:
            self.lambda_physics = lambda_physics
        if lambda_koopman is not None:
            self.lambda_koopman = lambda_koopman
        if lambda_sparse is not None:
            self.lambda_sparse = lambda_sparse

    def get_weights(self) -> dict:
        """Get current loss weights."""
        return {
            'lambda_physics': self.lambda_physics,
            'lambda_koopman': self.lambda_koopman,
            'lambda_sparse': self.lambda_sparse
        }

    def compute_metrics(
        self,
        model,
        x_physics: torch.Tensor,
        x_koopman: torch.Tensor,
        x_koopman_next: torch.Tensor,
        dt: float = 1.0
    ) -> dict:
        """Compute detailed metrics for all loss components."""
        return {
            'physics': self.physics_loss.compute_stats(model, x_physics),
            'koopman': self.koopman_loss.compute_error(
                model, x_koopman, x_koopman_next, dt
            ),
            'sparsity': self.sparsity_loss.get_metrics(model)
        }
