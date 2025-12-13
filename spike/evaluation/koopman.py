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
    diffeq=None,
    n_points: int = 1000,
    dt: float = 0.1,
    seed: int = 42,
    input_dim: int = None
) -> float:
    """
    Compute Koopman prediction R².

    Measures how well the Koopman operator predicts the next embedding:
    R² = 1 - ||g(u_{t+dt}) - exp(dt*A) @ g(u_t)||² / ||g(u_{t+dt}) - mean||²

    Args:
        model: SPIKE/PIKE model
        diffeq: PDE/ODE object (optional, used to infer input_dim)
        n_points: Number of test points
        dt: Time step for prediction
        seed: Random seed
        input_dim: Input dimension (1 for ODE, 2 for 1D PDE, 3 for 2D PDE)

    Returns:
        R² score (float)
    """
    np.random.seed(seed)

    # Check if model has Koopman
    if not hasattr(model, 'koopman'):
        return float('nan')

    # Infer input_dim from model or diffeq
    if input_dim is None:
        if hasattr(model, 'input_dim'):
            input_dim = model.input_dim
        elif diffeq is not None:
            diffeq_domain = diffeq.get_domain()
            if 'y_min' in diffeq_domain:
                input_dim = 3  # 2D PDE
            elif 'x_min' in diffeq_domain:
                input_dim = 2  # 1D PDE
            else:
                input_dim = 1  # ODE
        else:
            input_dim = 2  # Default to 1D PDE

    # Use fixed domain [0.1, 0.9] for consistency with original verify script
    if input_dim == 1:
        t = np.random.uniform(0.1, 0.9, n_points)
        inputs_t = torch.tensor(t.reshape(-1, 1), dtype=torch.float32)
        inputs_t_dt = torch.tensor((t + dt).reshape(-1, 1), dtype=torch.float32)
    elif input_dim == 2:
        x = np.random.uniform(0.1, 0.9, n_points)
        t = np.random.uniform(0.1, 0.9, n_points)
        inputs_t = torch.tensor(np.stack([x, t], axis=1), dtype=torch.float32)
        inputs_t_dt = torch.tensor(np.stack([x, t + dt], axis=1), dtype=torch.float32)
    else:  # input_dim == 3
        x = np.random.uniform(0.1, 0.9, n_points)
        y = np.random.uniform(0.1, 0.9, n_points)
        t = np.random.uniform(0.1, 0.9, n_points)
        inputs_t = torch.tensor(np.stack([x, y, t], axis=1), dtype=torch.float32)
        inputs_t_dt = torch.tensor(np.stack([x, y, t + dt], axis=1), dtype=torch.float32)

    try:
        with torch.no_grad():
            # Use model.pinn + model.koopman.embed directly (matching verify script)
            u_t = model.pinn(inputs_t)
            u_t_dt = model.pinn(inputs_t_dt)
            g_t = model.koopman.embed(u_t)
            g_t_dt = model.koopman.embed(u_t_dt)

            # Koopman prediction: g(t+dt) = exp(dt*A) @ g(t)
            A = model.koopman.A.weight.detach().numpy()
            K = expm(dt * A)
            g_pred = torch.tensor((K @ g_t.numpy().T).T, dtype=torch.float32)

            # R²
            ss_res = ((g_t_dt - g_pred) ** 2).sum().item()
            ss_tot = ((g_t_dt - g_t_dt.mean(0)) ** 2).sum().item()
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
        A = model.koopman.A.weight.detach().numpy()
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
