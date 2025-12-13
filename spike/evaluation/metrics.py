"""
Error Metrics
L2, MSE, MAE, relative errors, IC/BC errors, generalization gap
"""

import torch
import numpy as np
from typing import Union, Optional, Dict


def compute_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute Mean Squared Error.

    Args:
        pred: Predicted values
        target: Target values
        reduction: 'mean', 'sum', or 'none'

    Returns:
        MSE value
    """
    diff_sq = (pred - target) ** 2
    if reduction == 'mean':
        return diff_sq.mean()
    elif reduction == 'sum':
        return diff_sq.sum()
    return diff_sq


def compute_l2_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    relative: bool = False
) -> float:
    """
    Compute L2 error norm.

    Args:
        pred: Predicted values
        target: Target values
        relative: If True, compute relative L2 error

    Returns:
        L2 error (float)
    """
    with torch.no_grad():
        l2 = torch.norm(pred - target, p=2).item()
        if relative:
            target_norm = torch.norm(target, p=2).item()
            return l2 / (target_norm + 1e-10)
        return l2


def compute_relative_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-10
) -> float:
    """
    Compute relative error: ||pred - target|| / ||target||

    Args:
        pred: Predicted values
        target: Target values
        eps: Small value to prevent division by zero

    Returns:
        Relative error (float)
    """
    with torch.no_grad():
        diff_norm = torch.norm(pred - target).item()
        target_norm = torch.norm(target).item()
        return diff_norm / (target_norm + eps)


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Mean Absolute Error."""
    with torch.no_grad():
        return torch.mean(torch.abs(pred - target)).item()


def compute_max_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute maximum absolute error."""
    with torch.no_grad():
        return torch.max(torch.abs(pred - target)).item()


def compute_r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute R² score (coefficient of determination).

    R² = 1 - SS_res / SS_tot
    """
    with torch.no_grad():
        ss_res = torch.sum((target - pred) ** 2).item()
        ss_tot = torch.sum((target - target.mean()) ** 2).item()
        return 1 - ss_res / (ss_tot + 1e-10)


def compute_ic_mse(
    model,
    pde,
    n_points: int = 500,
    seed: int = 42
) -> float:
    """
    Compute Initial Condition MSE.

    Evaluates how well the model satisfies u(x, t=0) = u_0(x).

    Args:
        model: SPIKE/PIKE/PINN model
        pde: PDE object with initial_condition() method
        n_points: Number of evaluation points
        seed: Random seed

    Returns:
        IC MSE (float)
    """
    np.random.seed(seed)

    domain = pde.get_domain()
    x_min, x_max = domain.get('x_min', 0), domain.get('x_max', 1)
    t_min = domain.get('t_min', 0)

    # Sample x at t=0
    x = np.random.uniform(x_min, x_max, n_points)
    t = np.full(n_points, t_min)

    inputs = torch.tensor(
        np.stack([x, t], axis=1),
        dtype=torch.float32
    )

    try:
        with torch.no_grad():
            out = model(inputs)
            u_pred = out[0] if isinstance(out, tuple) else out

            # Get true IC
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
            u_true = pde.initial_condition(x_tensor)

            return ((u_pred - u_true) ** 2).mean().item()
    except Exception as e:
        return float('nan')


def compute_bc_mse(
    model,
    pde,
    n_points: int = 500,
    seed: int = 42
) -> Dict[str, float]:
    """
    Compute Boundary Condition MSE.

    Evaluates how well the model satisfies boundary conditions.

    Args:
        model: SPIKE/PIKE/PINN model
        pde: PDE object with boundary_condition() method
        n_points: Number of evaluation points per boundary
        seed: Random seed

    Returns:
        Dict with 'left', 'right', 'total' BC MSE
    """
    np.random.seed(seed)

    domain = pde.get_domain()
    x_min, x_max = domain.get('x_min', 0), domain.get('x_max', 1)
    t_min, t_max = domain.get('t_min', 0), domain.get('t_max', 1)

    results = {}

    try:
        with torch.no_grad():
            # Left boundary (x = x_min)
            t_left = np.random.uniform(t_min, t_max, n_points)
            x_left = np.full(n_points, x_min)
            inputs_left = torch.tensor(
                np.stack([x_left, t_left], axis=1),
                dtype=torch.float32
            )

            out_left = model(inputs_left)
            u_left = out_left[0] if isinstance(out_left, tuple) else out_left

            t_tensor = torch.tensor(t_left, dtype=torch.float32).unsqueeze(1)
            u_bc_left = pde.boundary_condition(t_tensor, boundary='left')

            if u_bc_left is not None:
                results['left'] = ((u_left - u_bc_left) ** 2).mean().item()
            else:
                results['left'] = 0.0

            # Right boundary (x = x_max)
            t_right = np.random.uniform(t_min, t_max, n_points)
            x_right = np.full(n_points, x_max)
            inputs_right = torch.tensor(
                np.stack([x_right, t_right], axis=1),
                dtype=torch.float32
            )

            out_right = model(inputs_right)
            u_right = out_right[0] if isinstance(out_right, tuple) else out_right

            u_bc_right = pde.boundary_condition(t_tensor, boundary='right')

            if u_bc_right is not None:
                results['right'] = ((u_right - u_bc_right) ** 2).mean().item()
            else:
                results['right'] = 0.0

            results['total'] = (results['left'] + results['right']) / 2

            return results
    except Exception as e:
        return {'left': float('nan'), 'right': float('nan'), 'total': float('nan')}


def compute_generalization_gap(
    model,
    pde_in_domain,
    pde_out_domain,
    n_points: int = 1000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Compute Generalization Gap.

    Gap = |Residual(in-domain) - Residual(out-domain)|

    This measures how well the model generalizes to unseen domains.

    Args:
        model: SPIKE/PIKE/PINN model
        pde_in_domain: PDE object for training domain
        pde_out_domain: PDE object for test domain (different geometry)
        n_points: Number of evaluation points
        seed: Random seed

    Returns:
        Dict with 'in_domain', 'out_domain', 'gap', 'ratio'
    """
    from .residuals import compute_residual

    residual_in = compute_residual(model, pde_in_domain, n_points=n_points, seed=seed)
    residual_out = compute_residual(model, pde_out_domain, n_points=n_points, seed=seed)

    gap = abs(residual_out - residual_in)
    ratio = residual_out / (residual_in + 1e-10)

    return {
        'in_domain': residual_in,
        'out_domain': residual_out,
        'gap': gap,
        'ratio': ratio  # >1 means worse on out-domain
    }


def compute_all_metrics(
    model,
    pde,
    n_points: int = 1000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for a trained model.

    Returns comprehensive dict with:
    - physics_mse: PDE residual
    - ic_mse: Initial condition error
    - bc_mse: Boundary condition error
    - sparsity_percent: Koopman matrix sparsity (if applicable)
    - max_real_eigenvalue: Stability indicator (if applicable)
    - koopman_r2: Koopman prediction R² (if applicable)

    Args:
        model: SPIKE/PIKE/PINN model
        pde: PDE object
        n_points: Number of evaluation points
        seed: Random seed

    Returns:
        Dict with all metrics
    """
    from .residuals import compute_residual
    from .koopman import compute_koopman_r2, check_stability, get_sparsity_metrics

    results = {}

    # Physics residual
    results['physics_mse'] = compute_residual(model, pde, n_points=n_points, seed=seed)

    # IC error
    results['ic_mse'] = compute_ic_mse(model, pde, n_points=n_points, seed=seed)

    # BC error
    bc = compute_bc_mse(model, pde, n_points=n_points, seed=seed)
    results['bc_mse'] = bc['total']

    # Koopman metrics (if applicable)
    if hasattr(model, 'koopman'):
        results['koopman_r2'] = compute_koopman_r2(model, pde, n_points=n_points, seed=seed)

        is_stable, max_real = check_stability(model)
        results['is_stable'] = is_stable
        results['max_real_eigenvalue'] = max_real

        sparsity = get_sparsity_metrics(model)
        results['sparsity_percent'] = sparsity['sparsity_percent']
        results['l1_norm'] = sparsity['l1_norm']

    return results
