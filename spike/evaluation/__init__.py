"""
SPIKE Evaluation Module

Submodules:
- metrics: L2, MSE, relative errors, IC/BC errors
- residuals: PDE/ODE residual computation
- koopman: R², stability, sparsity analysis
- ood: Out-of-distribution evaluation (space/time extrapolation)
- conservation: Mass, energy, temporal consistency tests
- lyapunov: Chaotic system analysis
- interpretability: Symbolic dynamics, coefficient recovery
"""

# Basic metrics
from .metrics import (
    compute_l2_error,
    compute_mse,
    compute_relative_error,
    compute_mae,
    compute_max_error,
    compute_r2_score,
    compute_ic_mse,
    compute_bc_mse,
    compute_generalization_gap,
    compute_all_metrics,
)

# Residuals
from .residuals import compute_residual, compute_residual_stats

# Koopman analysis
from .koopman import (
    compute_koopman_r2,
    check_stability,
    get_sparsity_metrics,
    get_eigenvalue_spectrum,
    analyze_dominant_modes,
)

# OOD metrics
from .ood import (
    compute_ood_space_mse,
    compute_ood_time_mse,
    compute_ood_2d_space_mse,
    compute_full_ood_metrics,
)

# Conservation metrics
from .conservation import (
    compute_mass_conservation,
    compute_energy_conservation,
    compute_temporal_consistency,
    compute_all_conservation_metrics,
)

# Lyapunov analysis
from .lyapunov import (
    compute_lyapunov_metrics,
    LYAPUNOV_TIMES,
)

# Interpretability
from .interpretability import (
    compute_interpretable_ratio,
    get_embedding_dimensions,
    extract_symbolic_dynamics,
    compute_pde_coefficient_recovery,
    PDE_COEFFICIENTS,
    ODE_COEFFICIENTS,
)

__all__ = [
    # Basic metrics
    'compute_l2_error',
    'compute_mse',
    'compute_relative_error',
    'compute_mae',
    'compute_max_error',
    'compute_r2_score',
    'compute_ic_mse',
    'compute_bc_mse',
    'compute_generalization_gap',
    'compute_all_metrics',
    # Residuals
    'compute_residual',
    'compute_residual_stats',
    # Koopman
    'compute_koopman_r2',
    'check_stability',
    'get_sparsity_metrics',
    'get_eigenvalue_spectrum',
    'analyze_dominant_modes',
    # OOD
    'compute_ood_space_mse',
    'compute_ood_time_mse',
    'compute_ood_2d_space_mse',
    'compute_full_ood_metrics',
    # Conservation
    'compute_mass_conservation',
    'compute_energy_conservation',
    'compute_temporal_consistency',
    'compute_all_conservation_metrics',
    # Lyapunov
    'compute_lyapunov_metrics',
    'LYAPUNOV_TIMES',
    # Interpretability
    'compute_interpretable_ratio',
    'get_embedding_dimensions',
    'extract_symbolic_dynamics',
    'compute_pde_coefficient_recovery',
    'PDE_COEFFICIENTS',
    'ODE_COEFFICIENTS',
]
