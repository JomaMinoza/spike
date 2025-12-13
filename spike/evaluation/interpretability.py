"""
Interpretability Metrics
Library-latent decomposition, symbolic dynamics, coefficient recovery
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_interpretable_ratio(
    model,
    library_dim: Optional[int] = None,
    threshold: float = 1e-4
) -> Dict[str, float]:
    """
    Compute interpretable ratio of Koopman matrix.

    The embedding is decomposed into:
    - Library (interpretable): Polynomial observables [1, u, u^2, ...]
    - Latent (neural): Learned features

    Interpretable ratio = norm of library->library quadrant / total norm

    Args:
        model: SPIKE/PIKE model with Koopman
        library_dim: Dimension of library portion (auto-detect if None)
        threshold: Threshold for near-zero entries

    Returns:
        Dict with interpretability metrics
    """
    if not hasattr(model, 'koopman'):
        return {'interpretable_ratio': float('nan')}

    try:
        A = model.koopman.get_matrix().numpy()
        total_dim = A.shape[0]

        # Auto-detect library dimension from embedding
        if library_dim is None:
            if hasattr(model.koopman, 'embedding'):
                emb = model.koopman.embedding
                if hasattr(emb, 'library_dim'):
                    library_dim = emb.library_dim
                else:
                    # Default: assume polynomial [1, u, u^2] = 3 terms
                    library_dim = 3
            else:
                library_dim = 3

        latent_dim = total_dim - library_dim

        # Extract quadrants
        A_lib_lib = A[:library_dim, :library_dim]  # Library -> Library
        A_lib_lat = A[:library_dim, library_dim:]  # Latent -> Library
        A_lat_lib = A[library_dim:, :library_dim]  # Library -> Latent
        A_lat_lat = A[library_dim:, library_dim:]  # Latent -> Latent

        # Compute norms
        norm_lib_lib = np.linalg.norm(A_lib_lib)
        norm_total = np.linalg.norm(A)

        interpretable_ratio = (norm_lib_lib / (norm_total + 1e-10)) * 100

        # Sparsity in each quadrant
        def sparsity(M):
            return (np.abs(M) < threshold).sum() / M.size * 100

        return {
            'interpretable_ratio': interpretable_ratio,
            'library_dim': library_dim,
            'latent_dim': latent_dim,
            'norm_lib_lib': norm_lib_lib,
            'norm_lib_lat': np.linalg.norm(A_lib_lat),
            'norm_lat_lib': np.linalg.norm(A_lat_lib),
            'norm_lat_lat': np.linalg.norm(A_lat_lat),
            'sparsity_lib_lib': sparsity(A_lib_lib),
            'sparsity_total': sparsity(A),
        }
    except Exception as e:
        return {'interpretable_ratio': float('nan'), 'error': str(e)}


def get_embedding_dimensions(model) -> Dict[str, int]:
    """
    Get embedding dimension breakdown.

    Returns:
        Dict with library_dim, latent_dim, total_dim
    """
    if not hasattr(model, 'koopman'):
        return {'total_dim': 0}

    try:
        total_dim = model.koopman.embedding_dim

        if hasattr(model.koopman, 'embedding'):
            emb = model.koopman.embedding
            library_dim = getattr(emb, 'library_dim', 3)
        else:
            library_dim = 3

        return {
            'library_dim': library_dim,
            'latent_dim': total_dim - library_dim,
            'total_dim': total_dim
        }
    except:
        return {'total_dim': 0}


def extract_symbolic_dynamics(
    model,
    library_terms: List[str] = None,
    threshold: float = 0.01
) -> Dict[str, str]:
    """
    Extract symbolic dynamics from library portion of Koopman matrix.

    Args:
        model: SPIKE/PIKE model
        library_terms: Names of library terms (default: ['1', 'u', 'u^2'])
        threshold: Coefficient threshold for inclusion

    Returns:
        Dict mapping observable to its dynamics equation
    """
    if not hasattr(model, 'koopman'):
        return {}

    if library_terms is None:
        library_terms = ['1', 'u', 'u^2']

    try:
        A = model.koopman.get_matrix().numpy()
        library_dim = len(library_terms)
        A_lib = A[:library_dim, :library_dim]

        equations = {}

        for i, term in enumerate(library_terms):
            # Row i gives d(term)/dt in terms of all library terms
            coeffs = A_lib[i, :]

            terms_str = []
            for j, (coeff, lib_term) in enumerate(zip(coeffs, library_terms)):
                if abs(coeff) > threshold:
                    if lib_term == '1':
                        terms_str.append(f"{coeff:.4f}")
                    else:
                        terms_str.append(f"{coeff:.4f}*{lib_term}")

            if terms_str:
                eq = " + ".join(terms_str)
            else:
                eq = "0"

            equations[f"d({term})/dt"] = eq

        return equations
    except Exception as e:
        return {'error': str(e)}


def compute_pde_coefficient_recovery(
    model,
    pde,
    true_coefficients: Dict[str, float],
    n_points: int = 1000,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Recover PDE coefficients via least squares regression.

    Given model derivatives and true PDE form, estimate coefficients.

    Args:
        model: SPIKE/PIKE/PINN model
        pde: PDE object
        true_coefficients: Dict of true coefficient values
                          e.g., {'u_xx': 0.01, 'u*u_x': -1.0}
        n_points: Number of points for regression
        seed: Random seed

    Returns:
        Dict with recovered coefficients, errors, and R^2
    """
    np.random.seed(seed)

    domain = pde.get_domain()
    x_min, x_max = domain.get('x_min', 0), domain.get('x_max', 1)
    t_min, t_max = domain.get('t_min', 0), domain.get('t_max', 1)

    # Sample points
    x = np.random.uniform(x_min, x_max, n_points)
    t = np.random.uniform(t_min, t_max, n_points)

    inputs = torch.tensor(
        np.stack([x, t], axis=1),
        dtype=torch.float32,
        requires_grad=True
    )

    try:
        # Forward pass
        out = model(inputs)
        u = out[0] if isinstance(out, tuple) else out

        # Compute derivatives
        grad_u = torch.autograd.grad(
            u, inputs,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]

        u_x = grad_u[:, 0:1]
        u_t = grad_u[:, 1:2]

        # Second derivatives
        u_xx = torch.autograd.grad(
            u_x, inputs,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True
        )[0][:, 0:1]

        # Build design matrix based on PDE terms
        with torch.no_grad():
            u_np = u.numpy().flatten()
            u_x_np = u_x.numpy().flatten()
            u_t_np = u_t.numpy().flatten()
            u_xx_np = u_xx.numpy().flatten()

        # Common PDE terms
        terms = {}
        terms['u_x'] = u_x_np
        terms['u_xx'] = u_xx_np
        terms['u'] = u_np
        terms['u^2'] = u_np ** 2
        terms['u^3'] = u_np ** 3
        terms['u*u_x'] = u_np * u_x_np

        # For KdV: need u_xxx
        try:
            u_xxx = torch.autograd.grad(
                u_xx, inputs,
                grad_outputs=torch.ones_like(u_xx),
                create_graph=True
            )[0][:, 0:1]
            terms['u_xxx'] = u_xxx.detach().numpy().flatten()
        except:
            pass

        # Build design matrix for requested coefficients
        coef_names = list(true_coefficients.keys())
        X = np.column_stack([terms[name] for name in coef_names if name in terms])
        y = u_t_np  # Target: u_t

        if X.shape[1] == 0:
            return {'error': 'No matching terms found'}

        # Least squares regression
        from numpy.linalg import lstsq
        coeffs, residuals, rank, s = lstsq(X, y, rcond=None)

        # Compute R^2
        y_pred = X @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)

        # Build results
        recovered = {}
        errors = {}

        for i, name in enumerate([n for n in coef_names if n in terms]):
            recovered[name] = coeffs[i]
            true_val = true_coefficients[name]
            rel_error = abs(coeffs[i] - true_val) / (abs(true_val) + 1e-10) * 100
            errors[name] = rel_error

        return {
            'recovered': recovered,
            'true': true_coefficients,
            'rel_error_percent': errors,
            'r2': r2
        }
    except Exception as e:
        return {'error': str(e)}


# Common PDE coefficient dictionaries
PDE_COEFFICIENTS = {
    'heat': {'u_xx': 0.01},
    'advection': {'u_x': -1.0},
    'burgers': {'u*u_x': -1.0, 'u_xx': 0.01},
    'allen_cahn': {'u_xx': 0.0001, 'u': 1.0, 'u^3': -1.0},
    'kdv': {'u*u_x': -1.0, 'u_xxx': -1.0},
    'reaction_diffusion': {'u_xx': 0.01, 'u': 1.0, 'u^2': -1.0},
}

ODE_COEFFICIENTS = {
    'lorenz': {'sigma': 10.0, 'rho': 28.0, 'beta': 2.6666666666666665},
    'seir': {'beta': 0.4, 'sigma': 0.2, 'gamma': 0.1},
}
