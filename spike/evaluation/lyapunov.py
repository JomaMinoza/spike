"""
Lyapunov Analysis for Chaotic Systems
Valid time, tau ratio, prediction horizons
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple


# Known Lyapunov times for common chaotic systems
LYAPUNOV_TIMES = {
    'lorenz': 1.1,  # ~1.1 seconds
    'rossler': 5.4,
    'double_pendulum': 1.0,
}


def compute_lyapunov_metrics(
    model,
    ode,
    t_train: Tuple[float, float] = (0, 25),
    t_ood: Tuple[float, float] = (25, 35),
    tau_lambda: Optional[float] = None,
    n_points: int = 1000,
    error_threshold: float = 0.5,
    seed: int = 42
) -> Dict[str, float]:
    """
    Compute Lyapunov analysis metrics for chaotic systems.

    Metrics:
    - In-Domain MSE: Error for t in training domain
    - OOD MSE: Error for t in out-of-distribution
    - Short-term MSE: Error within one Lyapunov time (t < tau_lambda)
    - Valid Time: Prediction horizon until relative error > threshold
    - tau ratio: Valid time / tau_lambda (higher = better)

    Args:
        model: SPIKE/PIKE/PINN model
        ode: ODE object (e.g., LorenzSystem)
        t_train: Training time domain (default: 0-25)
        t_ood: OOD time domain (default: 25-35)
        tau_lambda: Lyapunov time (auto-detected if None)
        n_points: Number of evaluation points
        error_threshold: Relative error threshold for valid time (default: 0.5)
        seed: Random seed

    Returns:
        Dict with Lyapunov metrics
    """
    np.random.seed(seed)

    # Auto-detect Lyapunov time
    if tau_lambda is None:
        ode_name = getattr(ode, 'name', '').lower()
        tau_lambda = LYAPUNOV_TIMES.get(ode_name, 1.0)

    results = {
        'tau_lambda': tau_lambda,
    }

    # Get reference solution using scipy
    try:
        from scipy.integrate import solve_ivp

        ode_func = ode.get_ode_func()
        y0 = ode.initial_condition().squeeze().numpy()
        t_span = (t_train[0], t_ood[1])
        t_eval = np.linspace(t_span[0], t_span[1], n_points)

        sol = solve_ivp(ode_func, t_span, y0, t_eval=t_eval, method='RK45')
        t_ref = sol.t
        y_ref = sol.y.T  # Shape: (n_points, output_dim)
    except Exception as e:
        return {'error': str(e)}

    # Model predictions
    t_tensor = torch.tensor(t_ref, dtype=torch.float32).unsqueeze(1)

    try:
        with torch.no_grad():
            out = model(t_tensor)
            y_pred = out[0] if isinstance(out, tuple) else out
            y_pred = y_pred.numpy()
    except Exception as e:
        return {'error': str(e)}

    y_ref_tensor = torch.tensor(y_ref, dtype=torch.float32)
    y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)

    # In-Domain MSE (t in t_train)
    in_domain_mask = (t_ref >= t_train[0]) & (t_ref <= t_train[1])
    if in_domain_mask.sum() > 0:
        in_domain_mse = ((y_pred_tensor[in_domain_mask] - y_ref_tensor[in_domain_mask]) ** 2).mean().item()
        results['in_domain_mse'] = in_domain_mse
    else:
        results['in_domain_mse'] = float('nan')

    # OOD MSE (t in t_ood)
    ood_mask = (t_ref >= t_ood[0]) & (t_ref <= t_ood[1])
    if ood_mask.sum() > 0:
        ood_mse = ((y_pred_tensor[ood_mask] - y_ref_tensor[ood_mask]) ** 2).mean().item()
        results['ood_mse'] = ood_mse
    else:
        results['ood_mse'] = float('nan')

    # Short-term MSE (t < tau_lambda from start)
    short_term_mask = (t_ref >= t_train[0]) & (t_ref <= t_train[0] + tau_lambda)
    if short_term_mask.sum() > 0:
        short_term_mse = ((y_pred_tensor[short_term_mask] - y_ref_tensor[short_term_mask]) ** 2).mean().item()
        results['short_term_mse'] = short_term_mse
    else:
        results['short_term_mse'] = float('nan')

    # Valid Time: prediction horizon until relative error > threshold
    ref_norm = np.linalg.norm(y_ref, axis=1)
    errors = np.linalg.norm(y_pred - y_ref, axis=1)
    relative_errors = errors / (ref_norm + 1e-10)

    valid_indices = np.where(relative_errors > error_threshold)[0]
    if len(valid_indices) > 0:
        valid_time = t_ref[valid_indices[0]] - t_ref[0]
    else:
        valid_time = t_ref[-1] - t_ref[0]

    results['valid_time'] = valid_time
    results['tau_ratio'] = valid_time / tau_lambda

    return results


def compute_trajectory_divergence(
    model,
    ode,
    perturbation: float = 1e-6,
    t_max: float = 10.0,
    n_steps: int = 1000,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Compute trajectory divergence for Lyapunov exponent estimation.

    Args:
        model: SPIKE/PIKE/PINN model
        ode: ODE object
        perturbation: Initial perturbation magnitude
        t_max: Maximum time
        n_steps: Number of time steps
        seed: Random seed

    Returns:
        Dict with time array and divergence curve
    """
    np.random.seed(seed)

    t = np.linspace(0, t_max, n_steps)
    t_tensor = torch.tensor(t, dtype=torch.float32).unsqueeze(1)

    # Get base trajectory
    with torch.no_grad():
        out = model(t_tensor)
        y_base = out[0] if isinstance(out, tuple) else out
        y_base = y_base.numpy()

    # Get perturbed trajectory (add perturbation to initial embedding)
    # This is approximate - true Lyapunov requires perturbed IC
    divergence = np.zeros(n_steps)

    # For now, estimate from model behavior
    # A more sophisticated approach would use variational equations

    return {
        'time': t,
        'divergence': divergence,
        'estimated_lambda': float('nan')  # Would need proper calculation
    }
