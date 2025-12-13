"""
Conservation and Invariance Tests
Mass conservation, energy conservation, temporal consistency
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from scipy.linalg import expm


def compute_mass_conservation(
    model,
    pde,
    n_x: int = 100,
    n_t: int = 50,
    seed: int = 42
) -> Dict[str, float]:
    """
    Compute mass conservation metric.

    Mass = integral(u) over spatial domain
    Conservation = relative std of mass over time (lower = better)

    Args:
        model: SPIKE/PIKE/PINN model
        pde: PDE object
        n_x: Number of spatial points
        n_t: Number of time points
        seed: Random seed

    Returns:
        Dict with mass conservation metrics
    """
    np.random.seed(seed)

    domain = pde.get_domain()
    x_min, x_max = domain.get('x_min', 0), domain.get('x_max', 1)
    t_min, t_max = domain.get('t_min', 0), domain.get('t_max', 1)

    x = np.linspace(x_min, x_max, n_x)
    t = np.linspace(t_min, t_max, n_t)
    dx = (x_max - x_min) / (n_x - 1)

    masses = []

    try:
        with torch.no_grad():
            for ti in t:
                t_arr = np.full(n_x, ti)
                inputs = torch.tensor(
                    np.stack([x, t_arr], axis=1),
                    dtype=torch.float32
                )

                out = model(inputs)
                u = out[0] if isinstance(out, tuple) else out
                u = u.numpy().flatten()

                # Trapezoidal integration
                mass = np.trapz(u, dx=dx)
                masses.append(mass)

        masses = np.array(masses)
        mean_mass = np.mean(masses)
        std_mass = np.std(masses)
        rel_std = std_mass / (np.abs(mean_mass) + 1e-10)

        return {
            'mean_mass': mean_mass,
            'std_mass': std_mass,
            'rel_std': rel_std,
            'masses': masses.tolist()
        }
    except Exception as e:
        return {'rel_std': float('nan'), 'error': str(e)}


def compute_energy_conservation(
    model,
    pde,
    n_x: int = 100,
    n_t: int = 50,
    seed: int = 42
) -> Dict[str, float]:
    """
    Compute energy conservation metric.

    Energy = integral(u^2) over spatial domain
    Conservation = relative std of energy over time (lower = better)

    Args:
        model: SPIKE/PIKE/PINN model
        pde: PDE object
        n_x: Number of spatial points
        n_t: Number of time points
        seed: Random seed

    Returns:
        Dict with energy conservation metrics
    """
    np.random.seed(seed)

    domain = pde.get_domain()
    x_min, x_max = domain.get('x_min', 0), domain.get('x_max', 1)
    t_min, t_max = domain.get('t_min', 0), domain.get('t_max', 1)

    x = np.linspace(x_min, x_max, n_x)
    t = np.linspace(t_min, t_max, n_t)
    dx = (x_max - x_min) / (n_x - 1)

    energies = []

    try:
        with torch.no_grad():
            for ti in t:
                t_arr = np.full(n_x, ti)
                inputs = torch.tensor(
                    np.stack([x, t_arr], axis=1),
                    dtype=torch.float32
                )

                out = model(inputs)
                u = out[0] if isinstance(out, tuple) else out
                u = u.numpy().flatten()

                # Energy = integral(u^2)
                energy = np.trapz(u ** 2, dx=dx)
                energies.append(energy)

        energies = np.array(energies)
        mean_energy = np.mean(energies)
        std_energy = np.std(energies)
        rel_std = std_energy / (np.abs(mean_energy) + 1e-10)

        return {
            'mean_energy': mean_energy,
            'std_energy': std_energy,
            'rel_std': rel_std,
            'energies': energies.tolist()
        }
    except Exception as e:
        return {'rel_std': float('nan'), 'error': str(e)}


def compute_temporal_consistency(
    model,
    pde,
    dt: float = 0.1,
    n_steps: int = 10,
    n_points: int = 500,
    seed: int = 42
) -> Dict[str, float]:
    """
    Compute temporal consistency metric.

    Compares Koopman rollout (z -> exp(dt*A) @ z -> ...) vs direct evaluation.
    Lower = more consistent dynamics.

    Args:
        model: SPIKE/PIKE model with Koopman
        pde: PDE object
        dt: Time step for rollout
        n_steps: Number of rollout steps
        n_points: Number of test points
        seed: Random seed

    Returns:
        Dict with temporal consistency metrics
    """
    if not hasattr(model, 'koopman'):
        return {'rel_error': float('nan'), 'error': 'Model has no Koopman module'}

    np.random.seed(seed)

    domain = pde.get_domain()
    x_min, x_max = domain.get('x_min', 0), domain.get('x_max', 1)
    t_min = domain.get('t_min', 0)

    # Initial points at t=t_min
    x = np.random.uniform(x_min, x_max, n_points)
    t0 = np.full(n_points, t_min)

    inputs_t0 = torch.tensor(
        np.stack([x, t0], axis=1),
        dtype=torch.float32
    )

    try:
        with torch.no_grad():
            # Get initial embedding
            out_t0 = model(inputs_t0)
            _, z_t0 = out_t0

            # Koopman rollout: z(t+n*dt) = exp(n*dt*A) @ z(t)
            A = model.koopman.get_matrix().numpy()
            exp_ndt_A = expm(n_steps * dt * A)
            z_rollout = torch.tensor(
                (exp_ndt_A @ z_t0.numpy().T).T,
                dtype=torch.float32
            )

            # Direct evaluation at t = t_min + n_steps * dt
            t_final = np.full(n_points, t_min + n_steps * dt)
            inputs_tf = torch.tensor(
                np.stack([x, t_final], axis=1),
                dtype=torch.float32
            )

            out_tf = model(inputs_tf)
            _, z_direct = out_tf

            # Relative error
            diff_norm = torch.norm(z_rollout - z_direct).item()
            direct_norm = torch.norm(z_direct).item()
            rel_error = diff_norm / (direct_norm + 1e-10)

            return {
                'rel_error': rel_error,
                'abs_error': diff_norm,
                'n_steps': n_steps,
                'dt': dt,
                't_final': t_min + n_steps * dt
            }
    except Exception as e:
        return {'rel_error': float('nan'), 'error': str(e)}


def compute_all_conservation_metrics(
    model,
    pde,
    n_x: int = 100,
    n_t: int = 50,
    seed: int = 42
) -> Dict[str, Dict]:
    """
    Compute all conservation and consistency metrics.

    Returns:
        Dict with 'mass', 'energy', 'temporal_consistency' sub-dicts
    """
    results = {
        'mass': compute_mass_conservation(model, pde, n_x=n_x, n_t=n_t, seed=seed),
        'energy': compute_energy_conservation(model, pde, n_x=n_x, n_t=n_t, seed=seed),
    }

    if hasattr(model, 'koopman'):
        results['temporal_consistency'] = compute_temporal_consistency(model, pde, seed=seed)

    return results
