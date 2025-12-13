"""
Koopman Evaluation
R², stability, eigenvalue analysis, sparsity metrics
"""

import torch
import numpy as np
from typing import Optional, Tuple
from scipy.linalg import expm


def compute_koopman_r2(
    model,
    pde,
    n_points: int = 1000,
    dt: float = 0.1,
    seed: int = 42
) -> float:
    """
    Compute Koopman prediction R².

    Measures how well the Koopman operator predicts the next embedding:
    R² = 1 - ||g(u_{t+dt}) - exp(dt*A) @ g(u_t)||² / ||g(u_{t+dt}) - mean||²

    Args:
        model: SPIKE/PIKE model
        pde: PDE object for domain info
        n_points: Number of test points
        dt: Time step for prediction
        seed: Random seed

    Returns:
        R² score (float)
    """
    np.random.seed(seed)

    # Check if model has Koopman
    if not hasattr(model, 'koopman'):
        return float('nan')

    pde_domain = pde.get_domain()
    x_range = (pde_domain.get('x_min', 0) + 0.1,
               pde_domain.get('x_max', 1) - 0.1)
    t_range = (pde_domain.get('t_min', 0) + 0.1,
               pde_domain.get('t_max', 1) - 0.1 - dt)

    x = np.random.uniform(*x_range, n_points)
    t = np.random.uniform(*t_range, n_points)

    inputs_t = torch.tensor(np.stack([x, t], axis=1), dtype=torch.float32)
    inputs_t_dt = torch.tensor(np.stack([x, t + dt], axis=1), dtype=torch.float32)

    try:
        with torch.no_grad():
            # Get embeddings at t and t+dt
            _, z_t = model(inputs_t)
            _, z_t_dt = model(inputs_t_dt)

            # Koopman prediction: z(t+dt) = exp(dt*A) @ z(t)
            A = model.koopman.get_matrix().numpy()
            K = expm(dt * A)
            z_pred = torch.tensor((K @ z_t.numpy().T).T, dtype=torch.float32)

            # R²
            ss_res = ((z_t_dt - z_pred) ** 2).sum().item()
            ss_tot = ((z_t_dt - z_t_dt.mean(0)) ** 2).sum().item()
            r2 = 1 - ss_res / (ss_tot + 1e-10)

            return r2
    except Exception as e:
        return float('nan')


def check_stability(model) -> Tuple[Optional[bool], float]:
    """
    Check eigenvalue stability of Koopman operator.

    For continuous-time A, stability requires max(Re(λ)) < 0.

    Args:
        model: SPIKE/PIKE model

    Returns:
        (is_stable, max_real_eigenvalue)
    """
    if not hasattr(model, 'koopman'):
        return None, float('nan')

    try:
        A = model.koopman.get_matrix().numpy()
        eigenvalues = np.linalg.eigvals(A)
        max_real = np.max(np.real(eigenvalues))

        # Stable if all real parts are negative (with small tolerance)
        is_stable = max_real <= 0.01

        return is_stable, max_real
    except:
        return None, float('nan')


def get_eigenvalue_spectrum(model) -> Optional[np.ndarray]:
    """
    Get eigenvalues of Koopman matrix.

    Args:
        model: SPIKE/PIKE model

    Returns:
        Array of complex eigenvalues
    """
    if not hasattr(model, 'koopman'):
        return None

    try:
        A = model.koopman.get_matrix().numpy()
        return np.linalg.eigvals(A)
    except:
        return None


def get_sparsity_metrics(model, threshold: float = 1e-4) -> dict:
    """
    Get sparsity metrics for Koopman matrix.

    Args:
        model: SPIKE/PIKE model
        threshold: Values below this are considered zero

    Returns:
        dict with sparsity metrics
    """
    if not hasattr(model, 'koopman'):
        return {'sparsity_percent': float('nan')}

    try:
        A = model.koopman.get_matrix()
        total = A.numel()
        near_zero = (torch.abs(A) < threshold).sum().item()

        return {
            'sparsity_percent': (near_zero / total) * 100,
            'l1_norm': torch.abs(A).sum().item(),
            'l2_norm': torch.norm(A).item(),
            'active_entries': total - near_zero,
            'total_entries': total,
            'compression_ratio': (total - near_zero) / total
        }
    except:
        return {'sparsity_percent': float('nan')}


def analyze_dominant_modes(model, top_k: int = 5) -> Optional[dict]:
    """
    Analyze dominant modes (eigenvectors) of Koopman operator.

    Args:
        model: SPIKE/PIKE model
        top_k: Number of top modes to analyze

    Returns:
        dict with dominant eigenvalues and modes
    """
    if not hasattr(model, 'koopman'):
        return None

    try:
        A = model.koopman.get_matrix().numpy()
        eigenvalues, eigenvectors = np.linalg.eig(A)

        # Sort by magnitude
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        top_eigenvalues = eigenvalues[idx[:top_k]]
        top_modes = eigenvectors[:, idx[:top_k]]

        return {
            'eigenvalues': top_eigenvalues,
            'modes': top_modes,
            'frequencies': np.imag(top_eigenvalues),
            'decay_rates': np.real(top_eigenvalues)
        }
    except:
        return None
