"""
Residual Computation
Compute PDE/ODE residuals for physics loss evaluation
"""

import torch
import numpy as np
from typing import Optional, Tuple


def compute_residual(
    model,
    pde,
    n_points: int = 2000,
    domain: Optional[dict] = None,
    seed: int = 42
) -> float:
    """
    Compute PDE residual for model evaluation.

    Supports 1D PDEs (x, t), 2D PDEs (x, y, t), and ODEs (t only).

    Args:
        model: SPIKE/PIKE model
        pde: PDE object with residual() method
        n_points: Number of collocation points
        domain: Optional domain override
        seed: Random seed for reproducibility

    Returns:
        Mean squared residual (float)
    """
    np.random.seed(seed)

    # Get domain from PDE
    pde_domain = pde.get_domain()

    # Detect dimensionality
    has_y = 'y_min' in pde_domain
    has_x = 'x_min' in pde_domain

    if domain is None:
        x_range = (pde_domain.get('x_min', 0), pde_domain.get('x_max', 1))
        y_range = (pde_domain.get('y_min', 0), pde_domain.get('y_max', 1))
        t_range = (pde_domain.get('t_min', 0), pde_domain.get('t_max', 1))
    else:
        x_range = domain.get('x', (0, 1))
        y_range = domain.get('y', (0, 1))
        t_range = domain.get('t', (0, 1))

    # Generate random points based on dimensionality
    if has_y:
        # 2D PDE: (x, y, t)
        x = np.random.uniform(*x_range, n_points)
        y = np.random.uniform(*y_range, n_points)
        t = np.random.uniform(*t_range, n_points)
        inputs = torch.tensor(
            np.stack([x, y, t], axis=1),
            dtype=torch.float32,
            requires_grad=True
        )
    elif has_x:
        # 1D PDE: (x, t)
        x = np.random.uniform(*x_range, n_points)
        t = np.random.uniform(*t_range, n_points)
        inputs = torch.tensor(
            np.stack([x, t], axis=1),
            dtype=torch.float32,
            requires_grad=True
        )
    else:
        # ODE: (t) only
        t = np.random.uniform(*t_range, n_points)
        inputs = torch.tensor(
            t.reshape(-1, 1),
            dtype=torch.float32,
            requires_grad=True
        )

    try:
        with torch.enable_grad():
            # Forward pass
            out = model(inputs)
            u = out[0] if isinstance(out, tuple) else out

            # Compute residual
            residual = pde.residual(u, inputs)

            return (residual ** 2).mean().item()
    except Exception as e:
        return float('nan')


def compute_residual_stats(
    model,
    pde,
    n_points: int = 2000,
    seed: int = 42
) -> dict:
    """
    Compute detailed residual statistics.

    Returns:
        dict with mean, std, max, min, rms of residuals
    """
    np.random.seed(seed)

    pde_domain = pde.get_domain()
    x_range = (pde_domain.get('x_min', 0), pde_domain.get('x_max', 1))
    t_range = (pde_domain.get('t_min', 0), pde_domain.get('t_max', 1))

    x = torch.tensor(
        np.random.uniform(*x_range, n_points),
        dtype=torch.float32
    ).unsqueeze(1)
    t = torch.tensor(
        np.random.uniform(*t_range, n_points),
        dtype=torch.float32
    ).unsqueeze(1)

    x.requires_grad_(True)
    t.requires_grad_(True)
    inputs = torch.cat([x, t], dim=1)

    try:
        with torch.enable_grad():
            out = model(inputs)
            u = out[0] if isinstance(out, tuple) else out
            residual = pde.residual(u, inputs)

            with torch.no_grad():
                return {
                    'mean': residual.mean().item(),
                    'std': residual.std().item(),
                    'max': residual.max().item(),
                    'min': residual.min().item(),
                    'rms': torch.sqrt((residual ** 2).mean()).item()
                }
    except Exception as e:
        return {'mean': float('nan'), 'std': float('nan'),
                'max': float('nan'), 'min': float('nan'), 'rms': float('nan')}
