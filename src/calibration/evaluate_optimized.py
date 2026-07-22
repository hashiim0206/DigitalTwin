"""
evaluate_optimized.py
Evaluates the best parameters found by Bayesian Optimization and generates the optimized_metrics_summary.csv
"""

import os
import sys
import json
import pandas as pd

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(base_dir, 'src'))

from calibration.objective import evaluate_params
from calibration.metrics import calculate_all_metrics

def main():
    best_params_path = os.path.join(base_dir, 'outputs', 'best_params.json')
    if not os.path.exists(best_params_path):
        print("Error: best_params.json not found.")
        return
        
    with open(best_params_path, 'r') as f:
        best_params = json.load(f)
        
    print(f"Evaluating Best Params: {best_params}")
    
    # evaluate_params runs SUMO and writes outputs/matched_trajectories.json
    score = evaluate_params(best_params)
    print(f"Objective Score: {score}")
    
    # Calculate metrics
    matched_data_path = os.path.join(base_dir, 'outputs', 'matched_trajectories.json')
    metrics_data = calculate_all_metrics(matched_data_path)
    
    summary = metrics_data['summary']
    
    df_summary = pd.DataFrame([{
        'Experiment': 'Bayesian Optimized',
        'Vehicles Evaluated': summary['total_evaluated'],
        'Median Spatial RMSE (norm)': round(summary['median_rmse'], 4),
        '95th Pctl RMSE (norm)': round(summary['p95_rmse'], 4),
        'Median Timing Deviation': round(summary['median_td'], 4),
        'Lane Agreement Rate': round(summary['lane_agreement_rate'], 4)
    }])
    
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    csv_path = os.path.join(results_dir, 'optimized_metrics_summary.csv')
    df_summary.to_csv(csv_path, index=False)
    
    print(f"Generated optimized metrics: {csv_path}")
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    main()
