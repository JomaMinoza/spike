"""
SPIKE Trainer
Main training loop for physics-informed neural networks
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Callable, Dict, Any
from tqdm import tqdm

from .callbacks import Callback, EarlyStopping, ProgressCallback
from .samplers import UniformSampler, BaseSampler


class Trainer:
    """
    Physics-Informed Neural Network Trainer.

    Handles:
    - PDE/ODE residual loss
    - Initial/boundary condition losses
    - Koopman losses (if applicable)
    - Sparsity regularization (if applicable)

    Args:
        model: SPIKE/PIKE/PINN model
        pde: PDE/ODE object with residual() method
        optimizer: torch optimizer (default: Adam)
        lr: Learning rate (default: 1e-3)
        n_collocation: Number of collocation points (default: 2000)
        n_boundary: Number of boundary points (default: 200)
        n_initial: Number of initial condition points (default: 200)
        weights: Loss weights dict (default: equal weights)
        sampler: Collocation point sampler (default: UniformSampler)
        device: torch device (default: cpu)
    """

    def __init__(
        self,
        model: nn.Module,
        pde,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 1e-3,
        n_collocation: int = 2000,
        n_boundary: int = 200,
        n_initial: int = 200,
        weights: Optional[Dict[str, float]] = None,
        sampler: Optional[BaseSampler] = None,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.pde = pde
        self.device = device

        # Optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer

        # Sampling
        self.n_collocation = n_collocation
        self.n_boundary = n_boundary
        self.n_initial = n_initial

        # Domain for sampler
        domain = pde.get_domain()
        self.domain = {
            'x': (domain.get('x_min', 0), domain.get('x_max', 1)),
            't': (domain.get('t_min', 0), domain.get('t_max', 1))
        }

        if sampler is None:
            self.sampler = UniformSampler(self.domain)
        else:
            self.sampler = sampler

        # Loss weights
        self.weights = weights or {
            'physics': 1.0,
            'ic': 10.0,
            'bc': 1.0,
            'koopman': 0.1,
            'sparsity': 0.01
        }

        self.history = {'loss': [], 'physics': [], 'ic': [], 'bc': []}
        self.callbacks: List[Callback] = []

    def add_callback(self, callback: Callback) -> None:
        """Add a training callback."""
        self.callbacks.append(callback)

    def compute_physics_loss(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute PDE/ODE residual loss."""
        out = self.model(inputs)
        u = out[0] if isinstance(out, tuple) else out
        residual = self.pde.residual(u, inputs)
        return (residual ** 2).mean()

    def compute_ic_loss(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute initial condition loss."""
        out = self.model(inputs)
        u = out[0] if isinstance(out, tuple) else out

        # Get IC from PDE
        if hasattr(self.pde, 'initial_condition'):
            x = inputs[:, 0:1] if inputs.shape[1] > 1 else inputs
            u_ic = self.pde.initial_condition(x)
            return ((u - u_ic) ** 2).mean()

        return torch.tensor(0.0, device=self.device)

    def compute_bc_loss(
        self,
        boundaries: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute boundary condition loss."""
        bc_loss = torch.tensor(0.0, device=self.device)

        if not hasattr(self.pde, 'boundary_condition'):
            return bc_loss

        for key, pts in boundaries.items():
            out = self.model(pts)
            u = out[0] if isinstance(out, tuple) else out

            # Get BC from PDE
            if hasattr(self.pde, 'boundary_condition'):
                u_bc = self.pde.boundary_condition(pts)
                if u_bc is not None:
                    bc_loss = bc_loss + ((u - u_bc) ** 2).mean()

        return bc_loss

    def compute_koopman_loss(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute Koopman consistency loss using finite differences.

        Loss = ||z(t+dt) - exp(dt*A) @ z(t)||²
        Simplified to: ||z(t+dt) - (I + dt*A) @ z(t)||² for small dt
        """
        if not hasattr(self.model, 'koopman'):
            return torch.tensor(0.0, device=self.device)

        dt = 0.01  # Small time step

        # Points at t
        out_t = self.model(inputs)
        if not isinstance(out_t, tuple) or len(out_t) < 2:
            return torch.tensor(0.0, device=self.device)
        _, z_t = out_t

        # Points at t + dt
        inputs_dt = inputs.clone()
        inputs_dt[:, -1] = inputs_dt[:, -1] + dt
        out_dt = self.model(inputs_dt)
        _, z_dt = out_dt

        # Linear Koopman prediction: z(t+dt) ≈ (I + dt*A) @ z(t)
        # A is nn.Linear, so use A.weight for the matrix
        A_weight = self.model.koopman.A.weight
        z_pred = z_t + dt * (z_t @ A_weight.T)

        return ((z_dt - z_pred) ** 2).mean()

    def compute_sparsity_loss(self) -> torch.Tensor:
        """Compute L1 sparsity loss on Koopman matrix."""
        if hasattr(self.model, 'l1_norm'):
            return self.model.l1_norm()
        elif hasattr(self.model, 'koopman') and hasattr(self.model.koopman, 'l1_norm'):
            return self.model.koopman.l1_norm()
        return torch.tensor(0.0, device=self.device)

    def train_step(self) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Sample points
        colloc = self.sampler.sample(self.n_collocation).to(self.device)
        ic_pts = self.sampler.sample_initial(self.n_initial).to(self.device)
        bc_pts = self.sampler.sample_boundary(self.n_boundary)
        bc_pts = {k: v.to(self.device) for k, v in bc_pts.items()}

        # Compute losses
        physics_loss = self.compute_physics_loss(colloc)
        ic_loss = self.compute_ic_loss(ic_pts)
        bc_loss = self.compute_bc_loss(bc_pts)
        koopman_loss = self.compute_koopman_loss(colloc)
        sparsity_loss = self.compute_sparsity_loss()

        # Weighted sum
        total_loss = (
            self.weights['physics'] * physics_loss +
            self.weights['ic'] * ic_loss +
            self.weights['bc'] * bc_loss +
            self.weights.get('koopman', 0) * koopman_loss +
            self.weights.get('sparsity', 0) * sparsity_loss
        )

        total_loss.backward()
        self.optimizer.step()

        return {
            'loss': total_loss.item(),
            'physics': physics_loss.item(),
            'ic': ic_loss.item(),
            'bc': bc_loss.item(),
            'koopman': koopman_loss.item(),
            'sparsity': sparsity_loss.item()
        }

    def train(
        self,
        epochs: int,
        verbose: bool = True,
        progress_bar: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            epochs: Number of training epochs
            verbose: Print progress (default: True)
            progress_bar: Show tqdm progress bar (default: True)

        Returns:
            Training history dict
        """
        # Add default progress callback if verbose
        if verbose and not any(isinstance(c, ProgressCallback) for c in self.callbacks):
            self.add_callback(ProgressCallback(print_every=100))

        # Notify callbacks
        for cb in self.callbacks:
            cb.on_train_start(self)

        iterator = range(epochs)
        if progress_bar:
            iterator = tqdm(iterator, desc='Training')

        for epoch in iterator:
            # Epoch start callbacks
            for cb in self.callbacks:
                cb.on_epoch_start(epoch, self)

            # Training step
            logs = self.train_step()

            # Record history
            for key, value in logs.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)

            # Epoch end callbacks
            for cb in self.callbacks:
                cb.on_epoch_end(epoch, self, logs)

            # Check early stopping
            for cb in self.callbacks:
                if isinstance(cb, EarlyStopping) and cb.stop_training:
                    break

            if any(isinstance(cb, EarlyStopping) and cb.stop_training for cb in self.callbacks):
                break

            # Update progress bar
            if progress_bar:
                iterator.set_postfix(loss=logs['loss'], physics=logs['physics'])

        # Notify callbacks
        for cb in self.callbacks:
            cb.on_train_end(self)

        return self.history

    def evaluate(self, n_points: int = 1000) -> Dict[str, float]:
        """
        Evaluate model on random test points.

        Returns:
            Dict with physics residual and other metrics
        """
        self.model.eval()

        with torch.no_grad():
            test_pts = self.sampler.sample(n_points).to(self.device)
            test_pts.requires_grad_(True)

            with torch.enable_grad():
                physics_loss = self.compute_physics_loss(test_pts)

            return {
                'physics_residual': physics_loss.item(),
            }

    def save(self, filepath: str) -> None:
        """Save model and training state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'weights': self.weights
        }, filepath)

    def load(self, filepath: str) -> None:
        """Load model and training state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', {})
        self.weights = checkpoint.get('weights', self.weights)
