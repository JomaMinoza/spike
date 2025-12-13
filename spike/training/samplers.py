"""
Collocation Point Samplers
Strategies for sampling training points in PINN training
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from abc import ABC, abstractmethod


class BaseSampler(ABC):
    """Base class for collocation point samplers."""

    def __init__(self, domain: dict, seed: int = 42):
        """
        Args:
            domain: Dict with domain bounds e.g. {'x': (0, 1), 't': (0, 1)}
            seed: Random seed
        """
        self.domain = domain
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    @abstractmethod
    def sample(self, n_points: int) -> torch.Tensor:
        """Sample collocation points."""
        pass

    def sample_boundary(self, n_points: int) -> Dict[str, torch.Tensor]:
        """Sample boundary points."""
        pass

    def sample_initial(self, n_points: int) -> torch.Tensor:
        """Sample initial condition points."""
        pass


class UniformSampler(BaseSampler):
    """
    Uniform random sampling.

    Simple but effective for most problems.
    """

    def sample(self, n_points: int) -> torch.Tensor:
        """Sample uniformly within domain."""
        points = []
        for key, (low, high) in self.domain.items():
            pts = np.random.uniform(low, high, n_points)
            points.append(pts)

        return torch.tensor(
            np.stack(points, axis=1),
            dtype=torch.float32,
            requires_grad=True
        )

    def sample_boundary(self, n_points: int) -> Dict[str, torch.Tensor]:
        """
        Sample boundary points for each dimension.

        Returns dict with keys like 'x_low', 'x_high', etc.
        """
        boundaries = {}
        dims = list(self.domain.keys())

        for i, (key, (low, high)) in enumerate(self.domain.items()):
            # Sample other dimensions uniformly
            other_points = []
            for j, (k, (l, h)) in enumerate(self.domain.items()):
                if j != i:
                    other_points.append(np.random.uniform(l, h, n_points))
                else:
                    other_points.append(np.full(n_points, low))

            boundaries[f'{key}_low'] = torch.tensor(
                np.stack(other_points, axis=1),
                dtype=torch.float32,
                requires_grad=True
            )

            # High boundary
            other_points[i] = np.full(n_points, high)
            boundaries[f'{key}_high'] = torch.tensor(
                np.stack(other_points, axis=1),
                dtype=torch.float32,
                requires_grad=True
            )

        return boundaries

    def sample_initial(self, n_points: int) -> torch.Tensor:
        """Sample at t=t_min for initial conditions."""
        points = []
        for key, (low, high) in self.domain.items():
            if key == 't':
                pts = np.full(n_points, low)  # t = t_min
            else:
                pts = np.random.uniform(low, high, n_points)
            points.append(pts)

        return torch.tensor(
            np.stack(points, axis=1),
            dtype=torch.float32,
            requires_grad=True
        )


class LatinHypercubeSampler(BaseSampler):
    """
    Latin Hypercube Sampling (LHS).

    Better space-filling properties than uniform sampling.
    Recommended for initial training.
    """

    def sample(self, n_points: int) -> torch.Tensor:
        """Sample using Latin Hypercube."""
        n_dims = len(self.domain)
        samples = np.zeros((n_points, n_dims))

        for i, (key, (low, high)) in enumerate(self.domain.items()):
            # Create stratified samples
            cut = np.linspace(0, 1, n_points + 1)
            u = np.random.uniform(cut[:-1], cut[1:])
            np.random.shuffle(u)
            samples[:, i] = low + u * (high - low)

        return torch.tensor(samples, dtype=torch.float32, requires_grad=True)

    def sample_boundary(self, n_points: int) -> Dict[str, torch.Tensor]:
        """LHS on boundary surfaces."""
        # Fall back to uniform for boundaries
        uniform = UniformSampler(self.domain, self.seed)
        return uniform.sample_boundary(n_points)

    def sample_initial(self, n_points: int) -> torch.Tensor:
        """LHS at initial time."""
        n_dims = len(self.domain)
        samples = np.zeros((n_points, n_dims))

        for i, (key, (low, high)) in enumerate(self.domain.items()):
            if key == 't':
                samples[:, i] = low  # t = t_min
            else:
                cut = np.linspace(0, 1, n_points + 1)
                u = np.random.uniform(cut[:-1], cut[1:])
                np.random.shuffle(u)
                samples[:, i] = low + u * (high - low)

        return torch.tensor(samples, dtype=torch.float32, requires_grad=True)


class AdaptiveSampler(BaseSampler):
    """
    Adaptive residual-based sampling.

    Concentrates points where PDE residual is high.
    Use after initial training with uniform/LHS.
    """

    def __init__(
        self,
        domain: dict,
        seed: int = 42,
        base_ratio: float = 0.5
    ):
        """
        Args:
            domain: Domain bounds
            seed: Random seed
            base_ratio: Ratio of points from uniform sampling (vs adaptive)
        """
        super().__init__(domain, seed)
        self.base_ratio = base_ratio
        self.residual_points = None
        self.residual_values = None

    def sample(self, n_points: int) -> torch.Tensor:
        """Sample with adaptation based on residuals."""
        n_base = int(n_points * self.base_ratio)
        n_adaptive = n_points - n_base

        # Base uniform samples
        uniform = UniformSampler(self.domain, self.seed)
        base_points = uniform.sample(n_base)

        if self.residual_points is None or n_adaptive == 0:
            # No residual info yet, return uniform
            if n_adaptive > 0:
                extra = uniform.sample(n_adaptive)
                return torch.cat([base_points, extra], dim=0)
            return base_points

        # Adaptive samples based on residual magnitude
        weights = torch.abs(self.residual_values).squeeze()
        weights = weights / weights.sum()

        indices = torch.multinomial(weights, n_adaptive, replacement=True)
        adaptive_points = self.residual_points[indices].clone()

        # Add small perturbation
        noise_scale = 0.01
        for i, (key, (low, high)) in enumerate(self.domain.items()):
            noise = torch.randn(n_adaptive) * noise_scale * (high - low)
            adaptive_points[:, i] += noise
            adaptive_points[:, i].clamp_(low, high)

        adaptive_points.requires_grad_(True)

        return torch.cat([base_points, adaptive_points], dim=0)

    def update_residuals(
        self,
        points: torch.Tensor,
        residuals: torch.Tensor
    ) -> None:
        """
        Update residual information for adaptive sampling.

        Args:
            points: Collocation points [N, dim]
            residuals: PDE residuals at points [N, 1]
        """
        self.residual_points = points.detach()
        self.residual_values = residuals.detach()

    def sample_boundary(self, n_points: int) -> Dict[str, torch.Tensor]:
        uniform = UniformSampler(self.domain, self.seed)
        return uniform.sample_boundary(n_points)

    def sample_initial(self, n_points: int) -> torch.Tensor:
        uniform = UniformSampler(self.domain, self.seed)
        return uniform.sample_initial(n_points)


class GridSampler(BaseSampler):
    """
    Regular grid sampling.

    Useful for evaluation and visualization.
    """

    def sample(self, n_points_per_dim: int) -> torch.Tensor:
        """Create regular grid."""
        grids = []
        for key, (low, high) in self.domain.items():
            grids.append(np.linspace(low, high, n_points_per_dim))

        mesh = np.meshgrid(*grids, indexing='ij')
        points = np.stack([m.ravel() for m in mesh], axis=1)

        return torch.tensor(points, dtype=torch.float32, requires_grad=True)

    def sample_boundary(self, n_points: int) -> Dict[str, torch.Tensor]:
        uniform = UniformSampler(self.domain, self.seed)
        return uniform.sample_boundary(n_points)

    def sample_initial(self, n_points_per_dim: int) -> torch.Tensor:
        """Grid at initial time."""
        grids = []
        for key, (low, high) in self.domain.items():
            if key == 't':
                grids.append(np.array([low]))
            else:
                grids.append(np.linspace(low, high, n_points_per_dim))

        mesh = np.meshgrid(*grids, indexing='ij')
        points = np.stack([m.ravel() for m in mesh], axis=1)

        return torch.tensor(points, dtype=torch.float32, requires_grad=True)
