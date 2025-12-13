"""
Sparsity Loss
L1 regularization on Koopman matrix A to encourage sparse dynamics
"""

import torch


class SparsityLoss:
    """
    L1 regularization on Koopman matrix A for sparsity.

    Penalizes the L1 norm of the Koopman operator matrix A, which
    encourages sparse linear dynamics in the embedding space.

    Args:
        norm_type: Type of sparsity penalty ('l1', 'l0_approx', 'l1_l2')

    Example:
        >>> loss_fn = SparsityLoss()
        >>> loss = loss_fn(model)
    """

    def __init__(self, norm_type: str = 'l1'):
        self.norm_type = norm_type

    def __call__(self, model) -> torch.Tensor:
        """
        Compute sparsity loss for Koopman matrix A.

        Args:
            model: SPIKE/PIKE model with koopman.l1_norm() method

        Returns:
            Sparsity penalty
        """
        if self.norm_type == 'l1':
            return model.koopman.l1_norm()

        A = model.koopman.get_matrix()

        if self.norm_type == 'l0_approx':
            # Smooth approximation to L0 norm
            sigma = 0.1
            return torch.sum(1 - torch.exp(-torch.abs(A) / sigma))

        elif self.norm_type == 'l1_l2':
            # Elastic net
            alpha = 0.5
            l1 = torch.norm(A, p=1)
            l2 = torch.norm(A, p=2)
            return alpha * l1 + (1 - alpha) * l2

        else:
            raise ValueError(f"Unknown norm type: {self.norm_type}")

    def get_metrics(self, model, threshold: float = 0.01) -> dict:
        """
        Compute detailed sparsity metrics.

        Returns:
            dict with sparsity statistics
        """
        with torch.no_grad():
            A = model.koopman.get_matrix()

            total = A.numel()
            near_zero = (torch.abs(A) < threshold).sum().item()
            l1 = torch.norm(A, p=1).item()
            l2 = torch.norm(A, p=2).item()
            linf = torch.max(torch.abs(A)).item()

            return {
                'sparsity_percent': (near_zero / total) * 100,
                'l1_norm': l1,
                'l2_norm': l2,
                'linf_norm': linf,
                'active_entries': total - near_zero,
                'total_entries': total
            }
