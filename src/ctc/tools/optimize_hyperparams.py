#!/usr/bin/env python3
"""
Hyperparameter optimization for CTC model using Optuna

Usage:
    python src/ctc/tools/optimize_hyperparams.py
"""

import optuna
import subprocess
import json
import os
from datetime import datetime


def objective(trial):
    """Optuna objective function"""

    # Define hyperparameter search space
    params = {
        "lr": trial.suggest_float("lr", 1e-4, 5e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
        "gradient_clip": trial.suggest_float("gradient_clip", 1.0, 5.0),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256]),
    }

    # Create unique run name for this trial
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"optuna_trial_{trial.number}_{timestamp}"

    # Run training with suggested parameters
    cmd = [
        "python",
        "src/ctc/train.py",
        "--lr",
        str(params["lr"]),
        "--batch_size",
        str(params["batch_size"]),
        "--gradient_clip",
        str(params["gradient_clip"]),
        "--dropout",
        str(params["dropout"]),
        "--hidden_dim",
        str(params["hidden_dim"]),
        "--epochs",
        "10",  # Use fewer epochs for optimization
        "--device",
        "mps",
        "--save_dir",
        f"generated/tuning/ctc/{run_name}",
    ]

    print(f"\nTrial {trial.number}: {params}")

    try:
        # Run training and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        if result.returncode != 0:
            print(f"Trial {trial.number} failed: {result.stderr}")
            return float("inf")

        # Parse output to get validation loss
        # Look for "Best validation loss:" in output
        for line in result.stdout.split("\n"):
            if "Best validation loss:" in line:
                val_loss = float(line.split(":")[-1].strip())
                print(f"Trial {trial.number} completed. Val Loss: {val_loss}")
                return val_loss

        # If we can't find the validation loss, return infinity
        print(f"Trial {trial.number}: Could not parse validation loss")
        return float("inf")

    except subprocess.TimeoutExpired:
        print(f"Trial {trial.number} timed out")
        return float("inf")
    except Exception as e:
        print(f"Trial {trial.number} error: {e}")
        return float("inf")


def main():
    # Create study with TPE (Tree-structured Parzen Estimator) sampler
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )

    # Run optimization
    study.optimize(
        objective,
        n_trials=15,  # Number of trials
        timeout=54000,  # 15 hours total timeout
    )

    # Print results
    print("\n" + "=" * 50)
    print("Optimization Results")
    print("=" * 50)

    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (Val Loss): {study.best_value:.4f}")

    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Create output directory
    output_dir = "generated/tuning/ctc"
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    results = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "n_trials": len(study.trials),
    }

    results_path = os.path.join(output_dir, "hyperparameter_optimization_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")

    # Generate visualization if available
    try:
        import optuna.visualization as vis

        # Plot optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(os.path.join(output_dir, "optimization_history.html"))

        # Plot parameter importances
        fig = vis.plot_param_importances(study)
        fig.write_html(os.path.join(output_dir, "param_importances.html"))

        print(f"Visualizations saved to {output_dir}/")

    except ImportError:
        print("Install plotly for visualizations: pip install plotly")


if __name__ == "__main__":
    main()
