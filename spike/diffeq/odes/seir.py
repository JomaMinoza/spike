"""
SEIR Epidemic Model
Compartmental model for infectious disease dynamics
"""

import torch
from ..base import BaseODE


class SEIRModel(BaseODE):
    """
    SEIR Epidemic Model (Compartmental ODEs).

    ODEs:
    - dS/dt = -beta * S * I / N
    - dE/dt = beta * S * I / N - sigma * E
    - dI/dt = sigma * E - gamma * I
    - dR/dt = gamma * I

    Where:
    - S = Susceptible
    - E = Exposed (incubating)
    - I = Infected (infectious)
    - R = Recovered

    Residual uses time derivative formulation for training stability.

    Args:
        beta: Transmission rate (default: 0.4)
        sigma: 1/incubation_period (default: 0.2, ~5 days)
        gamma: 1/infectious_period (default: 0.1, ~10 days)
        N: Total population (default: 1000)
        domain_t: Time domain in days (default: (0, 160))
    """

    def __init__(
        self,
        beta: float = 0.4,
        sigma: float = 0.2,
        gamma: float = 0.1,
        N: float = 1000.0,
        domain_t=(0.0, 160.0)
    ):
        super().__init__(domain_t=domain_t, output_dim=4)
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.N = N
        self.name = "SEIRModel"

    def residual(self, seir, t):
        """
        Compute SEIR residual: d(S+E+I+R)/dt

        Args:
            seir: State [batch_size, 4] as [S, E, I, R]
            t: Time [batch_size, 1], requires_grad=True

        Returns:
            Residual [batch_size, 1]
        """
        # Sum of all components, then take time derivative
        u_sum = seir.sum(dim=1, keepdim=True)
        u_t = torch.autograd.grad(
            u_sum, t, grad_outputs=torch.ones_like(u_sum),
            create_graph=True, retain_graph=True
        )[0]
        return u_t

    def initial_condition(self, dummy=None):
        """Initial: one infected in population."""
        S0 = self.N - 1.0
        E0 = 0.0
        I0 = 1.0
        R0 = 0.0
        return torch.tensor([[S0, E0, I0, R0]])

    def get_params(self):
        return {
            'beta': self.beta,
            'sigma': self.sigma,
            'gamma': self.gamma,
            'N': self.N,
            'R0': self.beta / self.gamma
        }

    def get_R0(self) -> float:
        """Basic reproduction number."""
        return self.beta / self.gamma

    def is_epidemic(self) -> bool:
        """Check if R0 > 1 (epidemic spreads)."""
        return self.get_R0() > 1.0

    def get_ode_func(self):
        """Get ODE function for scipy.integrate.solve_ivp."""
        beta, sigma, gamma, N = self.beta, self.sigma, self.gamma, self.N

        def seir_ode(t, state):
            S, E, I, R = state
            dS_dt = -beta * S * I / N
            dE_dt = beta * S * I / N - sigma * E
            dI_dt = sigma * E - gamma * I
            dR_dt = gamma * I
            return [dS_dt, dE_dt, dI_dt, dR_dt]

        return seir_ode
