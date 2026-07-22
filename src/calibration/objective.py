"""
objective.py
Defines the objective function for the Bayesian Optimizer.
Runs headless SUMO, evaluates metrics, returns a single weighted fitness score.
"""

import sys
import os
import json

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(base_dir, 'src'))

from traci_runner import run_sync
from calibration.match_real_sim import load_real_data, extract_sim_trajectories, match_trajectories
from calibration.metrics import calculate_all_metrics

def evaluate_params(car_params):
    """
    Evaluates a set of SUMO car-following parameters.
    Returns: float (the objective cost to minimize)
    """
    # 1. Run Headless Simulation
    try:
        run_sync(gui=False, car_params=car_params)
    except Exception as e:
        print(f"Simulation failed for params {car_params}: {e}")
        return float('inf')
        
    # 2. Extract and Match Trajectories
    real_path = os.path.join(base_dir, 'outputs', 'traffic_data.json')
    sim_path = os.path.join(base_dir, 'outputs', 'fcd.xml')
    matched_out_path = os.path.join(base_dir, 'outputs', 'matched_trajectories.json')
    
    real_data = load_real_data(real_path)
    sim_data = extract_sim_trajectories(sim_path)
    matched, unmatched = match_trajectories(real_data, sim_data)
    
    with open(matched_out_path, 'w') as f:
        json.dump(matched, f)
        
    # 3. Calculate Metrics
    metrics = calculate_all_metrics(matched_out_path)
    
    if not metrics.get('summary'):
        return float('inf')
        
    summary = metrics['summary']
    
    # 4. Compute Weighted Objective Score
    # We want to MINIMIZE this score.
    # Weights prioritize Median RMSE, but penalize highly if 95th percentile is bad.
    w_median_rmse = 0.5
    w_p95_rmse = 0.2
    w_median_td = 0.3
    
    median_rmse = summary['median_rmse']
    p95_rmse = summary['p95_rmse']
    median_td = summary['median_td']
    
    score = (w_median_rmse * median_rmse) + (w_p95_rmse * p95_rmse) + (w_median_td * median_td)
    
    # Optional: Penalty for low vehicle throughput (if bad params cause traffic jams)
    # The baseline evaluates ~118 vehicles. If fewer than 100 get through, penalize heavily.
    num_eval = summary['total_evaluated']
    if num_eval < 100:
        score += (100 - num_eval) * 0.1
        
    return score

if __name__ == "__main__":
    test_params = {"accel": "3.0", "decel": "4.5", "sigma": "0.5", "minGap": "2.5", "tau": "1.0"}
    score = evaluate_params(test_params)
    print(f"Test Score: {score}")
