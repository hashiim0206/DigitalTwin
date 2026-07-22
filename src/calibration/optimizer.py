"""
optimizer.py
Runs the Optuna Bayesian Optimization loop to find the best SUMO car-following parameters.
"""

import optuna
import yaml
import os
import sys
import pandas as pd

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(base_dir, 'src'))

from calibration.objective import evaluate_params

def load_config():
    config_path = os.path.join(base_dir, 'experiments', 'calibration_bayesopt.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def objective_wrapper(trial, config):
    params_config = config['parameters']
    
    # Suggest parameters for this trial based on YAML bounds
    car_params = {}
    for param_name, specs in params_config.items():
        car_params[param_name] = str(round(trial.suggest_float(
            param_name, 
            specs['low'], 
            specs['high']
        ), 2))
        
    # Evaluate the objective function (runs SUMO -> metrics)
    score = evaluate_params(car_params)
    
    return score

def main():
    config = load_config()
    n_trials = config['optimizer'].get('n_trials', 50)
    timeout = config['optimizer'].get('timeout', 3600)
    
    print(f"Starting ML-Driven Calibration ({n_trials} trials)...")
    
    # Use SQLite storage for reproducibility and stopping/resuming
    db_path = os.path.join(base_dir, 'outputs', 'optuna_study.db')
    storage_name = f"sqlite:///{db_path}"
    
    study = optuna.create_study(
        study_name="sumo_micro_calibration", 
        direction="minimize",
        storage=storage_name,
        load_if_exists=True
    )
    
    study.optimize(
        lambda trial: objective_wrapper(trial, config), 
        n_trials=n_trials, 
        timeout=timeout
    )
    
    print("\n--- OPTIMIZATION COMPLETE ---")
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial

    print(f"  Score: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Save best params to JSON for the simulation to use permanently
    best_params_path = os.path.join(base_dir, 'outputs', 'best_params.json')
    import json
    with open(best_params_path, 'w') as f:
        json.dump(trial.params, f, indent=4)
        
    # Generate optimization history plot
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances
        fig1 = plot_optimization_history(study)
        fig1.write_image(os.path.join(base_dir, 'outputs', 'opt_history.png'))
        
        fig2 = plot_param_importances(study)
        fig2.write_image(os.path.join(base_dir, 'outputs', 'opt_importances.png'))
        print("Saved optimization plots to outputs/")
    except ImportError:
        print("Note: Install 'plotly' and 'kaleido' to generate optimization visualization plots.")

if __name__ == "__main__":
    main()
