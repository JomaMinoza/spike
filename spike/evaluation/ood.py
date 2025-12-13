"""
Out-of-Distribution (OOD) Evaluation
Spatial and temporal extrapolation metrics
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


def compute_ood_space_mse(
    model,
    pde,
    x_ranges: List[Tuple[float, float]] = None,
    n_points: int = 1000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Compute OOD-Space MSE (spatial extrapolation).

    Evaluates physics residual in spatial regions outside training domain.

    Args:
        model: SPIKE/PIKE/PINN model
        pde: PDE object
        x_ranges: List of (x_min, x_max) tuples for OOD regions
                  Default: [(-5, 0), (1, 3), (3, 5)] assuming training on [0, 1]
        n_points: Number of evaluation points per region
        seed: Random seed

    Returns:
        Dict with MSE for each x_range
    """
    np.random.seed(seed)

    domain = pde.get_domain()
    t_min, t_max = domain.get('t_min', 0), domain.get('t_max', 1)

    if x_ranges is None:
        x_ranges = [(-5, 0), (1, 3), (3, 5)]

    results = {}

    for x_min, x_max in x_ranges:
        x = np.random.uniform(x_min, x_max, n_points)
        t = np.random.uniform(t_min, t_max, n_points)

        inputs = torch.tensor(
            np.stack([x, t], axis=1),
            dtype=torch.float32,
            requires_grad=True
        )

        try:
            with torch.enable_grad():
                out = model(inputs)
                u = out[0] if isinstance(out, tuple) else out
                residual = pde.residual(u, inputs)
                mse = (residual ** 2).mean().item()

            key = f"x_[{x_min},{x_max}]"
            results[key] = mse
        except Exception as e:
            key = f"x_[{x_min},{x_max}]"
            results[key] = float('nan')

    return results


def compute_ood_time_mse(
    model,
    pde,
    t_ranges: List[Tuple[float, float]] = None,
    n_points: int = 1000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Compute OOD-Time MSE (temporal extrapolation).

    Evaluates physics residual in temporal regions outside training domain.

    Args:
        model: SPIKE/PIKE/PINN model
        pde: PDE object
        t_ranges: List of (t_min, t_max) tuples for OOD regions
                  Default: [(1, 3), (3, 5)] assuming training on [0, 1]
        n_points: Number of evaluation points per region
        seed: Random seed

    Returns:
        Dict with MSE for each t_range
    """
    np.random.seed(seed)

    domain = pde.get_domain()
    x_min, x_max = domain.get('x_min', 0), domain.get('x_max', 1)

    if t_ranges is None:
        t_ranges = [(1, 3), (3, 5)]

    results = {}

    for t_lo, t_hi in t_ranges:
        x = np.random.uniform(x_min, x_max, n_points)
        t = np.random.uniform(t_lo, t_hi, n_points)

        inputs = torch.tensor(
            np.stack([x, t], axis=1),
            dtype=torch.float32,
            requires_grad=True
        )

        try:
            with torch.enable_grad():
                out = model(inputs)
                u = out[0] if isinstance(out, tuple) else out
                residual = pde.residual(u, inputs)
                mse = (residual ** 2).mean().item()

            key = f"t_[{t_lo},{t_hi}]"
            results[key] = mse
        except Exception as e:
            key = f"t_[{t_lo},{t_hi}]"
            results[key] = float('nan')

    return results


def compute_ood_2d_space_mse(
    model,
    pde,
    xy_ranges: List[Tuple[float, float]] = None,
    n_points: int = 1000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Compute OOD-Space MSE for 2D PDEs.

    Args:
        model: SPIKE/PIKE/PINN model
        pde: 2D PDE object (input_dim=3: x, y, t)
        xy_ranges: List of (min, max) tuples for OOD spatial regions
        n_points: Number of evaluation points per region
        seed: Random seed

    Returns:
        Dict with MSE for each region
    """
    np.random.seed(seed)

    domain = pde.get_domain()
    t_min = domain.get('t_min', 0)
    t_max = domain.get('t_max', 1)

    if xy_ranges is None:
        xy_ranges = [(1, 2), (2, 3)]

    results = {}

    for xy_min, xy_max in xy_ranges:
        x = np.random.uniform(xy_min, xy_max, n_points)
        y = np.random.uniform(xy_min, xy_max, n_points)
        t = np.random.uniform(t_min, t_max, n_points)

        inputs = torch.tensor(
            np.stack([x, y, t], axis=1),
            dtype=torch.float32,
            requires_grad=True
        )

        try:
            with torch.enable_grad():
                out = model(inputs)
                u = out[0] if isinstance(out, tuple) else out
                residual = pde.residual(u, inputs)
                mse = (residual ** 2).mean().item()

            key = f"xy_[{xy_min},{xy_max}]"
            results[key] = mse
        except Exception as e:
            key = f"xy_[{xy_min},{xy_max}]"
            results[key] = float('nan')

    return results


def compute_full_ood_metrics(
    model,
    pde,
    n_points: int = 1000,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Compute all OOD metrics (space + time).

    Returns:
        Dict with 'space' and 'time' sub-dicts
    """
    return {
        'space': compute_ood_space_mse(model, pde, n_points=n_points, seed=seed),
        'time': compute_ood_time_mse(model, pde, n_points=n_points, seed=seed)
    }
