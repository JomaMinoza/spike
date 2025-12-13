"""
Reaction-Diffusion Equation
u_t = D * u_xx + R(u)
"""

import torch
from ..base import BasePDE


class ReactionDiffusionEquation(BasePDE):
    """
    1D Reaction-Diffusion Equation.

    PDE: u_t = D * u_xx + R(u)

    Reaction types:
    - Fisher-KPP: R(u) = r * u * (1 - u)
    - Bistable: R(u) = u * (1 - u) * (u - a)
    - Linear: R(u) = r * u

    Args:
        D: Diffusion coefficient (default: 0.01)
        r: Reaction rate (default: 1.0)
        reaction_type: 'fisher', 'bistable', 'linear' (default: 'fisher')
        a: Bistable threshold (default: 0.5)
        domain_x: Spatial domain (default: (0, 1))
        domain_t: Temporal domain (default: (0, 1))
    """

    def __init__(self, D=0.01, r=1.0, reaction_type='fisher', a=0.5,
                 domain_x=(0.0, 1.0), domain_t=(0.0, 1.0)):
        super().__init__(domain_x, domain_t)
        self.D = D
        self.r = r
        self.reaction_type = reaction_type
        self.a = a
        self.name = "ReactionDiffusionEquation"

    def reaction_term(self, u):
        """Compute reaction term R(u)."""
        if self.reaction_type == 'fisher':
            return self.r * u * (1 - u)
        elif self.reaction_type == 'bistable':
            return u * (1 - u) * (u - self.a)
        elif self.reaction_type == 'linear':
            return self.r * u
        else:
            raise ValueError(f"Unknown reaction type: {self.reaction_type}")

    def residual(self, u, x):
        """Residual: u_t - D * u_xx - R(u) = 0"""
        derivs = self.compute_derivatives(u, x)
        R = self.reaction_term(u)
        return derivs['u_t'] - self.D * derivs['u_xx'] - R

    def initial_condition(self, x):
        """IC: Gaussian pulse"""
        x_mid = (self.domain_x[0] + self.domain_x[1]) / 2
        sigma = 0.1
        return torch.exp(-((x - x_mid) ** 2) / (2 * sigma ** 2))

    def boundary_condition(self, t, boundary='left'):
        return torch.zeros_like(t)

    def get_params(self):
        return {'D': self.D, 'r': self.r, 'reaction_type': self.reaction_type, 'a': self.a}
