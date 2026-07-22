"""
evaluate.py
Evaluates and compares the three control strategies:
1. Fixed-Time (Default SUMO)
2. Standard Delay-Only RL Agent
3. Digital-Twin-in-the-Loop RL Agent (Delay + Fidelity)
"""

import os
import sys
import yaml
import json
import pandas as pd
import numpy as np
from stable_baselines3 import PPO

# Import our custom environment
from env import SumoDTEnv

def evaluate_agent(env, model, num_episodes=3):
    """Evaluates an agent (or None for fixed-time) and returns average metrics."""
    avg_delay = 0.0
    avg_fidelity = 0.0
    avg_throughput = 0.0
    avg_emissions = 0.0
    avg_safety_conflicts = 0.0
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        
        ep_delay = 0.0
        ep_fidelity = 0.0
        ep_emissions = 0.0
        ep_safety_conflicts = 0
        
        while not done:
            if model is None:
                # Fixed time: don't change lights, let SUMO default programs run
                # Wait, our _apply_phase logic overwrites the default.
                # If we want fixed time, we need to pass a specific action cycle.
                # E.g., cycle through 0, 1, 2, 3 every step.
                action = int((env.curr_time // env.action_step) % 4)
            else:
                action, _states = model.predict(obs, deterministic=True)
                
            obs, reward, terminated, truncated, _ = env.step(int(action))
            
            # Since reward = -(w_delay * delay) - (w_fidelity * penalty)
            # We can recompute raw delay and penalty directly.
            # We already have queue length in obs!
            ep_delay += np.sum(obs)
            # Fidelity is computed in step, we can grab it from a custom env method
            ep_fidelity += env._compute_fidelity_penalty()
            
            import traci
            active_vehicles = traci.vehicle.getIDList()
            for v in active_vehicles:
                # 1. Emissions (sampled per action step)
                ep_emissions += traci.vehicle.getCO2Emission(v)
                
                # 2. Safety (Hard Braking)
                if traci.vehicle.getAcceleration(v) < -4.5:
                    ep_safety_conflicts += 1
                    
                # 3. Safety (Dangerous TTC)
                leader_info = traci.vehicle.getLeader(v, 100.0)
                if leader_info is not None:
                    l_id, dist = leader_info
                    v_speed = traci.vehicle.getSpeed(v)
                    l_speed = traci.vehicle.getSpeed(l_id)
                    if v_speed > l_speed and (v_speed - l_speed) > 0.1:
                        ttc = dist / (v_speed - l_speed)
                        if ttc < 1.5:
                            ep_safety_conflicts += 1
            
            done = terminated or truncated
            
        import traci
        active_veh = len(traci.vehicle.getIDList())
        total_veh = env.total_data
        throughput = total_veh - active_veh # Approximate completed
        
        avg_delay += ep_delay
        avg_fidelity += ep_fidelity
        avg_throughput += throughput
        avg_emissions += ep_emissions
        avg_safety_conflicts += ep_safety_conflicts
        
    return {
        "Total Delay (veh-s)": avg_delay / num_episodes,
        "Spatial RMSE (Fidelity)": avg_fidelity / num_episodes,
        "Throughput (veh)": avg_throughput / num_episodes,
        "CO2 Emissions (g)": (avg_emissions / num_episodes) / 1000.0,
        "Safety Conflicts": avg_safety_conflicts / num_episodes
    }

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, 'experiments', 'rl_config.yaml')
    
    print("--- Evaluating Control Strategies ---")
    
    # 1. Setup Environment (We use the baseline config for evaluation so we can measure raw metrics)
    env = SumoDTEnv(config_path)
    
    results = []
    
    # 1. Fixed-Time Strategy
    print("Evaluating Fixed-Time Controller...")
    ft_metrics = evaluate_agent(env, None)
    ft_metrics["Strategy"] = "Fixed-Time"
    results.append(ft_metrics)
    
    # 2. Standard RL Agent
    print("Evaluating Standard Delay-Only Agent...")
    baseline_model_path = os.path.join(base_dir, 'outputs', 'rl_logs_baseline', 'final_ppo_model.zip')
    if os.path.exists(baseline_model_path):
        baseline_model = PPO.load(baseline_model_path, env=env)
        bl_metrics = evaluate_agent(env, baseline_model)
        bl_metrics["Strategy"] = "Delay-Only RL"
        results.append(bl_metrics)
    else:
        print("Standard RL model not found.")
        
    # 3. Digital-Twin RL Agent
    print("Evaluating Digital-Twin RL Agent...")
    dt_model_path = os.path.join(base_dir, 'outputs', 'rl_logs', 'final_ppo_model.zip')
    if os.path.exists(dt_model_path):
        dt_model = PPO.load(dt_model_path, env=env)
        dt_metrics = evaluate_agent(env, dt_model)
        dt_metrics["Strategy"] = "Digital-Twin RL"
        results.append(dt_metrics)
    else:
        print("Digital-Twin RL model not found.")
        
    env.close()
    
    # Save Results
    df = pd.DataFrame(results)
    # Reorder columns
    df = df[["Strategy", "Total Delay (veh-s)", "Spatial RMSE (Fidelity)", "Throughput (veh)", "CO2 Emissions (g)", "Safety Conflicts"]]
    
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, 'evaluation_metrics.csv')
    df.to_csv(out_csv, index=False)
    
    print(f"\nEvaluation Complete! Results saved to {out_csv}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
