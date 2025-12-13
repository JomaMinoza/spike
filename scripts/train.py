#!/usr/bin/env python
"""
SPIKE Training Script

Usage:
    python train.py --pde burgers --model spike --epochs 5000
    python train.py --pde lorenz --model pike --epochs 10000 --lr 1e-4
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from spike.models import PINN, PIKE, SPIKE
from spike.diffeq import (
    BurgersEquation, HeatEquation, AdvectionEquation, WaveEquation,
    KdVEquation, AllenCahnEquation, KuramotoSivashinskyEquation,
    LorenzSystem, SEIRModel
)
from spike.training import Trainer, EarlyStopping, ModelCheckpoint, ProgressCallback
from spike.training.samplers import UniformSampler, LatinHypercubeSampler


# PDE/ODE registry
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


def get_model(model_type: str, input_dim: int, output_dim: int, args):
    """Create model based on type."""
    if model_type == 'pinn':
        return PINN(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=output_dim,
            num_layers=args.n_layers,
            activation=args.activation
        )
    elif model_type == 'pike':
        return PIKE(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=output_dim,
            num_layers=args.n_layers,
            embedding_dim=args.embedding_dim,
            activation=args.activation
        )
    elif model_type == 'spike':
        return SPIKE(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=output_dim,
            num_layers=args.n_layers,
            embedding_dim=args.embedding_dim,
            activation=args.activation
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description='Train SPIKE models')

    # Equation
    parser.add_argument('--pde', type=str, default='burgers',
                        choices=list(EQUATIONS.keys()),
                        help='PDE/ODE to solve')

    # Model
    parser.add_argument('--model', type=str, default='spike',
                        choices=['pinn', 'pike', 'spike'],
                        help='Model type')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden layer dimension')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of hidden layers')
    parser.add_argument('--embedding_dim', type=int, default=32,
                        help='Koopman embedding dimension')
    parser.add_argument('--activation', type=str, default='tanh',
                        help='Activation function')

    # Training
    parser.add_argument('--epochs', type=int, default=5000,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--n_collocation', type=int, default=2000,
                        help='Number of collocation points')
    parser.add_argument('--n_boundary', type=int, default=200,
                        help='Number of boundary points')
    parser.add_argument('--n_initial', type=int, default=200,
                        help='Number of initial condition points')

    # Loss weights
    parser.add_argument('--w_physics', type=float, default=1.0,
                        help='Physics loss weight')
    parser.add_argument('--w_ic', type=float, default=10.0,
                        help='Initial condition loss weight')
    parser.add_argument('--w_bc', type=float, default=1.0,
                        help='Boundary condition loss weight')
    parser.add_argument('--w_koopman', type=float, default=0.1,
                        help='Koopman loss weight')
    parser.add_argument('--w_sparsity', type=float, default=0.01,
                        help='Sparsity (L1) loss weight')

    # Callbacks
    parser.add_argument('--patience', type=int, default=500,
                        help='Early stopping patience')
    parser.add_argument('--sampler', type=str, default='uniform',
                        choices=['uniform', 'lhs'],
                        help='Collocation point sampler')

    # Output
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create equation
    pde = EQUATIONS[args.pde]()
    print(f"Equation: {pde.name}")

    # Determine input/output dims
    domain = pde.get_domain()
    input_dim = 2  # (x, t) for PDEs, (t,) for ODEs
    if hasattr(pde, 'output_dim'):
        output_dim = pde.output_dim
    else:
        output_dim = 1

    # For ODEs, input is just time
    if args.pde in ['lorenz', 'seir']:
        input_dim = 1

    # Create model
    model = get_model(args.model, input_dim, output_dim, args)
    print(f"Model: {args.model.upper()} ({sum(p.numel() for p in model.parameters())} params)")

    # Create sampler
    sampler_domain = {
        'x': (domain.get('x_min', 0), domain.get('x_max', 1)),
        't': (domain.get('t_min', 0), domain.get('t_max', 1))
    }

    if args.sampler == 'uniform':
        sampler = UniformSampler(sampler_domain, seed=args.seed)
    else:
        sampler = LatinHypercubeSampler(sampler_domain, seed=args.seed)

    # Loss weights
    weights = {
        'physics': args.w_physics,
        'ic': args.w_ic,
        'bc': args.w_bc,
        'koopman': args.w_koopman,
        'sparsity': args.w_sparsity
    }

    # Create trainer
    trainer = Trainer(
        model=model,
        pde=pde,
        lr=args.lr,
        n_collocation=args.n_collocation,
        n_boundary=args.n_boundary,
        n_initial=args.n_initial,
        weights=weights,
        sampler=sampler,
        device=args.device
    )

    # Add callbacks
    trainer.add_callback(EarlyStopping(patience=args.patience))
    trainer.add_callback(ProgressCallback(print_every=100))

    # Output directory
    output_dir = Path(args.output_dir) / f"{args.pde}_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer.add_callback(ModelCheckpoint(
        filepath=str(output_dir / 'best_model.pt'),
        monitor='loss',
        save_best_only=True
    ))

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"Output: {output_dir}")
    print(f"Training for {args.epochs} epochs...")

    # Train
    history = trainer.train(epochs=args.epochs, progress_bar=True)

    # Save final model
    trainer.save(str(output_dir / 'final_model.pt'))

    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f)

    print(f"\nTraining complete!")
    print(f"Final loss: {history['loss'][-1]:.6f}")
    print(f"Final physics residual: {history['physics'][-1]:.6f}")

    if args.model in ['pike', 'spike']:
        print(f"Final Koopman loss: {history.get('koopman', [0])[-1]:.6f}")
        if args.model == 'spike':
            print(f"Final sparsity loss: {history.get('sparsity', [0])[-1]:.6f}")


if __name__ == '__main__':
    main()
