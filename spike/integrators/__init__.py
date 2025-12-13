"""
Time integrators for Koopman dynamics dz/dt = Az.

Available integrators:
- euler: 1st order, fast but may be unstable for stiff systems
- rk4: 4th order Runge-Kutta, more accurate
- expm: Matrix exponential, exact for linear systems, best for stiff PDEs
"""

from .base import BaseIntegrator
from .euler import EulerIntegrator
from .rk4 import RK4Integrator
from .expm import ExpmIntegrator, matrix_exponential

_INTEGRATORS = {
    'euler': EulerIntegrator,
    'rk4': RK4Integrator,
    'expm': ExpmIntegrator,
}


def get_integrator(name: str, **kwargs) -> BaseIntegrator:
    """
    Factory function to get integrator by name.

    Args:
        name: Integrator name ('euler', 'rk4', 'expm')
        **kwargs: Additional arguments passed to integrator constructor

    Returns:
        BaseIntegrator instance

    Example:
        >>> integrator = get_integrator('rk4')
        >>> z_next = integrator.step(z, A, dt=0.01)
    """
    if name not in _INTEGRATORS:
        raise ValueError(
            f"Unknown integrator: {name}. "
            f"Choose from {list(_INTEGRATORS.keys())}"
        )
    return _INTEGRATORS[name](**kwargs)


__all__ = [
    'BaseIntegrator',
    'EulerIntegrator',
    'RK4Integrator',
    'ExpmIntegrator',
    'get_integrator',
    'matrix_exponential',
]
