#!/usr/bin/env python
"""
SPIKE Evaluation Script

Usage:
    python evaluate.py --checkpoint results/burgers_spike/best_model.pt --pde burgers
    python evaluate.py --checkpoint model.pt --pde lorenz --n_points 5000
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from spike.models import PINN, PIKE, SPIKE
from spike.diffeq import (
    BurgersEquation, HeatEquation, AdvectionEquation, WaveEquation,
    KdVEquation, AllenCahnEquation, KuramotoSivashinskyEquation,
    LorenzSystem, SEIRModel
)
from spike.evaluation import (
    compute_residual, compute_l2_error, compute_mse,
    compute_koopman_r2, check_stability, get_sparsity_metrics
)
from spike.evaluation.residuals import compute_residual_stats
from spike.evaluation.koopman import analyze_dominant_modes


EQUATIONS = {
    'burgers': BurgersEquation,
    'heat': HeatEquation,
    'advection': AdvectionEquation,
    'wave': WaveEquation,
    'kdv': KdVEquation,
    'allen_cahn': AllenCahnEquation,
    'ks': KuramotoSivashinskyEquation,
    'lorenz': LorenzSystem,
    'seir': SEIRModel,
}


def load_model(checkpoint_path: str, model_type: str, input_dim: int, output_dim: int, config: dict):
    """Load model from checkpoint."""
    if model_type == 'pinn':
        model = PINN(
            input_dim=input_dim,
            hidden_dim=config.get('hidden_dim', 64),
            output_dim=output_dim,
            n_layers=config.get('n_layers', 4)
        )
    elif model_type == 'pike':
        model = PIKE(
            input_dim=input_dim,
            hidden_dim=config.get('hidden_dim', 64),
            output_dim=output_dim,
            n_layers=config.get('n_layers', 4),
            embedding_dim=config.get('embedding_dim', 32)
        )
    elif model_type == 'spike':
        model = SPIKE(
            input_dim=input_dim,
            hidden_dim=config.get('hidden_dim', 64),
            output_dim=output_dim,
            n_layers=config.get('n_layers', 4),
            embedding_dim=config.get('embedding_dim', 32)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser(description='Evaluate SPIKE models')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--pde', type=str, required=True,
                        choices=list(EQUATIONS.keys()),
                        help='PDE/ODE type')
    parser.add_argument('--model', type=str, default='spike',
                        choices=['pinn', 'pike', 'spike'],
                        help='Model type')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.json (optional)')
    parser.add_argument('--n_points', type=int, default=2000,
                        help='Number of evaluation points')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results (optional)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots')

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load config if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Try to find config in same directory as checkpoint
        config_path = Path(args.checkpoint).parent / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            args.model = config.get('model', args.model)

    # Create equation
    pde = EQUATIONS[args.pde]()
    print(f"Equation: {pde.name}")

    # Determine dims
    domain = pde.get_domain()
    input_dim = 2
    if args.pde in ['lorenz', 'seir']:
        input_dim = 1
    output_dim = getattr(pde, 'output_dim', 1)

    # Load model
    model = load_model(args.checkpoint, args.model, input_dim, output_dim, config)
    print(f"Model: {args.model.upper()}")
    print(f"Loaded from: {args.checkpoint}")

    results = {}

    # Physics residual
    print("\n--- Physics Evaluation ---")
    residual = compute_residual(model, pde, n_points=args.n_points, seed=args.seed)
    results['physics_residual_mse'] = residual
    print(f"Physics Residual (MSE): {residual:.6e}")

    residual_stats = compute_residual_stats(model, pde, n_points=args.n_points, seed=args.seed)
    results['residual_stats'] = residual_stats
    print(f"  Mean: {residual_stats['mean']:.6e}")
    print(f"  Std:  {residual_stats['std']:.6e}")
    print(f"  Max:  {residual_stats['max']:.6e}")
    print(f"  RMS:  {residual_stats['rms']:.6e}")

    # Koopman analysis (if applicable)
    if args.model in ['pike', 'spike'] and hasattr(model, 'koopman'):
        print("\n--- Koopman Analysis ---")

        # R²
        r2 = compute_koopman_r2(model, pde, n_points=args.n_points, seed=args.seed)
        results['koopman_r2'] = r2
        print(f"Koopman R²: {r2:.4f}")

        # Stability
        is_stable, max_real = check_stability(model)
        results['is_stable'] = is_stable
        results['max_real_eigenvalue'] = max_real
        print(f"Stability: {'Stable' if is_stable else 'Unstable'} (max Re(λ) = {max_real:.4f})")

        # Sparsity
        if args.model == 'spike':
            sparsity = get_sparsity_metrics(model)
            results['sparsity'] = sparsity
            print(f"Sparsity: {sparsity['sparsity_percent']:.1f}%")
            print(f"  Active entries: {sparsity['active_entries']}/{sparsity['total_entries']}")
            print(f"  L1 norm: {sparsity['l1_norm']:.4f}")

        # Dominant modes
        modes = analyze_dominant_modes(model, top_k=5)
        if modes:
            print("\nDominant Eigenvalues:")
            for i, (ev, decay, freq) in enumerate(zip(
                modes['eigenvalues'],
                modes['decay_rates'],
                modes['frequencies']
            )):
                print(f"  λ_{i+1}: {ev:.4f} (decay={decay:.4f}, freq={freq:.4f})")

    # Save results
    if args.output:
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj

        results = convert(results)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    # Plots
    if args.plot:
        print("\nGenerating plots...")
        plot_dir = Path(args.checkpoint).parent / 'plots'
        plot_dir.mkdir(exist_ok=True)

        # Solution visualization (for 1D+t PDEs)
        if input_dim == 2 and output_dim == 1:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Generate grid
            x = np.linspace(domain['x_min'], domain['x_max'], 100)
            t = np.linspace(domain['t_min'], domain['t_max'], 100)
            X, T = np.meshgrid(x, t)
            inputs = torch.tensor(
                np.stack([X.ravel(), T.ravel()], axis=1),
                dtype=torch.float32
            )

            with torch.no_grad():
                out = model(inputs)
                u = out[0] if isinstance(out, tuple) else out
                U = u.numpy().reshape(100, 100)

            # Solution
            c1 = axes[0].contourf(X, T, U, levels=50, cmap='viridis')
            plt.colorbar(c1, ax=axes[0])
            axes[0].set_xlabel('x')
            axes[0].set_ylabel('t')
            axes[0].set_title(f'{pde.name} Solution')

            # Residual
            inputs.requires_grad_(True)
            with torch.enable_grad():
                out = model(inputs)
                u = out[0] if isinstance(out, tuple) else out
                res = pde.residual(u, inputs)
                R = res.detach().numpy().reshape(100, 100)

            c2 = axes[1].contourf(X, T, np.abs(R), levels=50, cmap='hot')
            plt.colorbar(c2, ax=axes[1])
            axes[1].set_xlabel('x')
            axes[1].set_ylabel('t')
            axes[1].set_title('|Residual|')

            plt.tight_layout()
            plt.savefig(plot_dir / 'solution.png', dpi=150)
            print(f"  Saved: {plot_dir / 'solution.png'}")

        # Koopman matrix (if applicable)
        if args.model in ['pike', 'spike'] and hasattr(model, 'koopman'):
            A = model.koopman.get_matrix().numpy()

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Matrix visualization
            im = axes[0].imshow(A, cmap='RdBu', aspect='auto')
            plt.colorbar(im, ax=axes[0])
            axes[0].set_title('Koopman Matrix A')

            # Eigenvalue spectrum
            eigenvalues = np.linalg.eigvals(A)
            axes[1].scatter(np.real(eigenvalues), np.imag(eigenvalues), s=50, alpha=0.7)
            axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axes[1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
            axes[1].set_xlabel('Re(λ)')
            axes[1].set_ylabel('Im(λ)')
            axes[1].set_title('Eigenvalue Spectrum')

            plt.tight_layout()
            plt.savefig(plot_dir / 'koopman.png', dpi=150)
            print(f"  Saved: {plot_dir / 'koopman.png'}")

        plt.close('all')

    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
