"""
evaluate_baseline.py
Evaluates the baseline parameters and generates baseline_metrics_summary.csv
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
    baseline_params = {'accel': 3.0, 'decel': 4.5, 'sigma': 0.5, 'minGap': 2.5, 'tau': 1.0}
        
    print(f"Evaluating Baseline Params: {baseline_params}")
    
    score = evaluate_params(baseline_params)
    print(f"Objective Score: {score}")
    
    matched_data_path = os.path.join(base_dir, 'outputs', 'matched_trajectories.json')
    metrics_data = calculate_all_metrics(matched_data_path)
    
    summary = metrics_data['summary']
    
    df_summary = pd.DataFrame([{
        'Experiment': 'Baseline (Krauss Defaults)',
        'Vehicles Evaluated': summary['total_evaluated'],
        'Median Spatial RMSE (norm)': round(summary['median_rmse'], 4),
        '95th Pctl RMSE (norm)': round(summary['p95_rmse'], 4),
        'Median Timing Deviation': round(summary['median_td'], 4),
        'Lane Agreement Rate': round(summary['lane_agreement_rate'], 4)
    }])
    
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    csv_path = os.path.join(results_dir, 'baseline_metrics_summary.csv')
    df_summary.to_csv(csv_path, index=False)
    
    print(f"Generated baseline metrics: {csv_path}")
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    main()
