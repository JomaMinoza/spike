"""
Sparse Embedding Module
Embedding layer with L1 regularization for discovering sparse Koopman observables

Three embedding types:
- 'library': Polynomial dictionary (EDMD-style) - interpretable, fixed basis
- 'learned': MLP encoder (Deep Koopman style) - flexible, black-box
- 'augmented': Library + Learned (our contribution) - interpretable + flexible
"""

import torch
import torch.nn as nn
import itertools
from typing import List, Tuple, Optional, Dict


def compute_polynomial_features(x: torch.Tensor, degree: int) -> torch.Tensor:
    """
    Compute polynomial features up to given degree.

    For input [x1, x2, ..., xn], generates:
    - degree 1: [x1, x2, ..., xn]
    - degree 2: [x1², x2², ..., x1*x2, x1*x3, ...]
    - etc.

    Args:
        x: Input tensor [batch_size, input_dim]
        degree: Maximum polynomial degree

    Returns:
        Polynomial features [batch_size, n_features]
    """
    batch_size, input_dim = x.shape
    features = [torch.ones(batch_size, 1, device=x.device)]  # Constant term

    for d in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(input_dim), d):
            term = torch.ones(batch_size, device=x.device)
            for idx in combo:
                term = term * x[:, idx]
            features.append(term.unsqueeze(1))

    return torch.cat(features, dim=1)


def get_polynomial_feature_count(input_dim: int, degree: int) -> int:
    """Calculate number of polynomial features for given input dim and degree."""
    from math import comb
    count = 0
    for d in range(degree + 1):
        count += comb(input_dim + d - 1, d)
    return count


def get_polynomial_feature_names(
    input_dim: int,
    degree: int,
    var_names: Optional[List[str]] = None
) -> List[str]:
    """Get human-readable names for polynomial features."""
    if var_names is None:
        var_names = [f'x{i}' for i in range(input_dim)]

    names = ['1']  # Constant term

    for d in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(input_dim), d):
            counts = {}
            for idx in combo:
                counts[idx] = counts.get(idx, 0) + 1

            parts = []
            for idx, count in sorted(counts.items()):
                if count == 1:
                    parts.append(var_names[idx])
                else:
                    parts.append(f'{var_names[idx]}^{count}')
            names.append('*'.join(parts) if len(parts) > 1 else parts[0])

    return names


def compute_derivative_features(
    u: torch.Tensor,
    derivatives: Dict[str, torch.Tensor],
    derivative_terms: List[str]
) -> Tuple[torch.Tensor, List[str]]:
    """
    Compute derivative-based features for the library.

    Args:
        u: Solution tensor [batch_size, output_dim]
        derivatives: Dict with keys like 'u_x', 'u_xx', 'u_t'
        derivative_terms: List of terms to include

    Returns:
        features: Tensor [batch_size, n_derivative_features]
        names: List of feature names
    """
    batch_size = u.shape[0]
    features = []
    names = []

    for term in derivative_terms:
        # Pure derivatives
        if term == 'u_x' and 'u_x' in derivatives:
            features.append(derivatives['u_x'])
            names.append('u_x')
        elif term == 'u_xx' and 'u_xx' in derivatives:
            features.append(derivatives['u_xx'])
            names.append('u_xx')
        elif term == 'u_xxx' and 'u_xxx' in derivatives:
            features.append(derivatives['u_xxx'])
            names.append('u_xxx')
        elif term == 'u_xxxx' and 'u_xxxx' in derivatives:
            features.append(derivatives['u_xxxx'])
            names.append('u_xxxx')
        elif term == 'u_t' and 'u_t' in derivatives:
            features.append(derivatives['u_t'])
            names.append('u_t')

        # u * derivative combinations
        elif term == 'u_ux' and 'u_x' in derivatives:
            features.append(u * derivatives['u_x'])
            names.append('u·u_x')
        elif term == 'u_uxx' and 'u_xx' in derivatives:
            features.append(u * derivatives['u_xx'])
            names.append('u·u_xx')

        # u² * derivative combinations
        elif term == 'u2_ux' and 'u_x' in derivatives:
            features.append(u**2 * derivatives['u_x'])
            names.append('u²·u_x')
        elif term == 'u2_uxx' and 'u_xx' in derivatives:
            features.append(u**2 * derivatives['u_xx'])
            names.append('u²·u_xx')

        # Cross-derivative products
        elif term == 'ux_uxx' and 'u_x' in derivatives and 'u_xx' in derivatives:
            features.append(derivatives['u_x'] * derivatives['u_xx'])
            names.append('u_x·u_xx')

        # 2D terms
        elif term == 'u_y' and 'u_y' in derivatives:
            features.append(derivatives['u_y'])
            names.append('u_y')
        elif term == 'u_yy' and 'u_yy' in derivatives:
            features.append(derivatives['u_yy'])
            names.append('u_yy')
        elif term == 'u_uy' and 'u_y' in derivatives:
            features.append(u * derivatives['u_y'])
            names.append('u·u_y')

    if len(features) == 0:
        return torch.zeros(batch_size, 0, device=u.device), []

    features = [f.view(batch_size, -1) if f.dim() == 1 else f for f in features]
    return torch.cat(features, dim=1), names


def get_derivative_feature_count(derivative_terms: List[str]) -> int:
    """Get the number of derivative features based on term list."""
    return len(derivative_terms)


class SparseEmbedding(nn.Module):
    """
    Sparse embedding layer for Koopman observable discovery.

    Three embedding types:
    - 'library': Polynomial dictionary (EDMD-style)
    - 'learned': MLP encoder (Deep Koopman style)
    - 'augmented': Library + Learned (best of both)

    Args:
        input_dim: Dimension of input (solution u)
        embedding_dim: Total dimension of embedding space
        embedding_type: 'library', 'learned', or 'augmented'
        poly_degree: Polynomial degree for library
        mlp_hidden: Hidden layer size for learned branch
        activation: Activation function
        use_skip: Include identity skip connection
        derivative_terms: List of derivative terms for library
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        embedding_type: str = 'augmented',
        poly_degree: int = 2,
        mlp_hidden: int = 64,
        activation: str = 'tanh',
        use_skip: bool = False,
        derivative_terms: Optional[List[str]] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.embedding_type = embedding_type
        self.poly_degree = poly_degree
        self.use_skip = use_skip
        self.derivative_terms = derivative_terms or []
        self.use_derivatives = len(self.derivative_terms) > 0

        # Activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.Identity()

        # Calculate library dimensions
        self.poly_library_dim = get_polynomial_feature_count(input_dim, poly_degree)
        self.deriv_library_dim = get_derivative_feature_count(self.derivative_terms)
        self.library_dim = self.poly_library_dim + self.deriv_library_dim

        # Library feature names
        self.poly_names = get_polynomial_feature_names(input_dim, poly_degree)
        self.library_names = self.poly_names + self.derivative_terms

        # Reserve input_dim for skip if enabled
        self.skip_dim = input_dim if use_skip else 0
        remaining_dim = embedding_dim - self.skip_dim

        if use_skip and remaining_dim <= 0:
            raise ValueError(f"embedding_dim must be > input_dim when use_skip=True")

        if embedding_type == 'library':
            self.library_projection = nn.Linear(self.library_dim, remaining_dim)
            self.learned_dim = 0

        elif embedding_type == 'learned':
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, mlp_hidden),
                self.activation,
                nn.Linear(mlp_hidden, mlp_hidden),
                self.activation,
                nn.Linear(mlp_hidden, remaining_dim)
            )
            self.learned_dim = remaining_dim

        elif embedding_type == 'augmented':
            self.library_output_dim = min(self.library_dim, remaining_dim // 2)
            self.learned_dim = remaining_dim - self.library_output_dim

            self.library_projection = nn.Linear(self.library_dim, self.library_output_dim)
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, mlp_hidden),
                self.activation,
                nn.Linear(mlp_hidden, mlp_hidden),
                self.activation,
                nn.Linear(mlp_hidden, self.learned_dim)
            )
        else:
            raise ValueError(f"Unknown embedding_type: {embedding_type}")

    def forward(
        self,
        x: torch.Tensor,
        derivatives: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass through embedding layer.

        Args:
            x: Input tensor [batch_size, input_dim]
            derivatives: Optional dict of derivative tensors

        Returns:
            z: Embedded tensor [batch_size, embedding_dim]
        """
        poly_features = compute_polynomial_features(x, self.poly_degree)

        if self.use_derivatives:
            if derivatives is not None:
                deriv_features, _ = compute_derivative_features(
                    x, derivatives, self.derivative_terms
                )
            else:
                batch_size = x.shape[0]
                deriv_features = torch.zeros(
                    batch_size, self.deriv_library_dim, device=x.device
                )
            all_library_features = torch.cat([poly_features, deriv_features], dim=1)
        else:
            all_library_features = poly_features

        if self.embedding_type == 'library':
            z_other = self.library_projection(all_library_features)

        elif self.embedding_type == 'learned':
            z_other = self.mlp(x)

        elif self.embedding_type == 'augmented':
            z_library = self.library_projection(all_library_features)
            z_learned = self.mlp(x)
            z_other = torch.cat([z_library, z_learned], dim=1)

        if self.use_skip:
            z = torch.cat([x, z_other], dim=1)
        else:
            z = z_other

        return z

    def get_library_features(
        self,
        x: torch.Tensor,
        derivatives: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Get raw library features before projection."""
        poly_features = compute_polynomial_features(x, self.poly_degree)

        if self.use_derivatives and derivatives is not None:
            deriv_features, _ = compute_derivative_features(
                x, derivatives, self.derivative_terms
            )
            return torch.cat([poly_features, deriv_features], dim=1)

        return poly_features

    def get_sparsity(self, threshold: float = 0.01) -> float:
        """Compute percentage of weights below threshold."""
        total_params = 0
        near_zero = 0

        for param in self.parameters():
            total_params += param.numel()
            near_zero += (torch.abs(param.data) < threshold).sum().item()

        return (near_zero / total_params) * 100.0 if total_params > 0 else 0.0

    def get_l1_norm(self) -> float:
        """Get L1 norm of all embedding weights."""
        l1 = 0.0
        for param in self.parameters():
            l1 += torch.norm(param, p=1).item()
        return l1

    def __repr__(self) -> str:
        return (
            f"SparseEmbedding(input_dim={self.input_dim}, "
            f"embedding_dim={self.embedding_dim}, "
            f"type='{self.embedding_type}', "
            f"poly_degree={self.poly_degree})"
        )
