"""
Koopman Loss
Enforces linearity in embedding space via Koopman operator

Loss: ||K @ g(u_t) - g(u_{t+dt})||²
"""

import torch


class KoopmanLoss:
    """
    Koopman loss that enforces linear dynamics in solution embedding space.

    The Koopman operator should satisfy: g(u_{t+dt}) ≈ step(g(u_t), dt)
    where u is the PDE solution and g is the observable function.

    Args:
        norm: Type of norm ('mse', 'l1', 'huber')

    Example:
        >>> loss_fn = KoopmanLoss()
        >>> loss = loss_fn(model, x_current, x_next, dt=0.01)
    """

    def __init__(self, norm: str = 'mse'):
        self.norm = norm

        if norm == 'mse':
            self.loss_fn = lambda diff: torch.mean(diff ** 2)
        elif norm == 'l1':
            self.loss_fn = lambda diff: torch.mean(torch.abs(diff))
        elif norm == 'huber':
            self.loss_fn = torch.nn.HuberLoss()
        else:
            raise ValueError(f"Unknown norm type: {norm}")

    def __call__(
        self,
        model,
        x_current: torch.Tensor,
        x_next: torch.Tensor,
        dt: float = 1.0
    ) -> torch.Tensor:
        """
        Compute Koopman linearity loss.

        Args:
            model: SPIKE/PIKE model
            x_current: Coordinates at time t [batch_size, input_dim]
            x_next: Coordinates at time t+dt [batch_size, input_dim]
            dt: Time step for Koopman prediction

        Returns:
            Linearity error in observable space
        """
        # Get embeddings at current and next time
        _, z_current = model(x_current)
        _, z_next_true = model(x_next)

        # Predict next embedding using Koopman dynamics
        z_next_pred = model.koopman.step(z_current, dt)

        # Compute loss
        diff = z_next_pred - z_next_true
        return self.loss_fn(diff)

    def compute_error(
        self,
        model,
        x_current: torch.Tensor,
        x_next: torch.Tensor,
        dt: float = 1.0
    ) -> dict:
        """
        Compute detailed prediction error statistics.

        Returns:
            dict with mse, rmse, mae, max_error
        """
        with torch.no_grad():
            _, z_current = model(x_current)
            _, z_next_true = model(x_next)
            z_next_pred = model.koopman.step(z_current, dt)

            diff = z_next_pred - z_next_true

            return {
                'mse': torch.mean(diff ** 2).item(),
                'rmse': torch.sqrt(torch.mean(diff ** 2)).item(),
                'mae': torch.mean(torch.abs(diff)).item(),
                'max_error': torch.max(torch.abs(diff)).item()
            }
