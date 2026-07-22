"""
ablation.py
Performs an ablation study on the Digital-Twin RL Agent by sweeping the
fidelity weight parameter to generate a Pareto curve of Delay vs. Fidelity.
"""

import os
import yaml
import pandas as pd
import numpy as np
from stable_baselines3 import PPO

from env import SumoDTEnv
from evaluate import evaluate_agent

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, 'experiments', 'rl_config.yaml')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    weights_to_test = [0.0, 1.0, 5.0, 10.0, 20.0]
    
    # 500 steps is extremely short, but prevents the laptop from heating up
    # while generating the pipeline data.
    train_steps = 500
    
    results = []
    
    print("--- Starting Ablation Study ---")
    
    for w in weights_to_test:
        print(f"\nEvaluating w_fidelity = {w}")
        
        # We subclass to inject the custom weight
        class AblationEnv(SumoDTEnv):
            def __init__(self, c_path):
                super().__init__(c_path)
                with open(c_path, 'r') as f:
                    self.config = yaml.safe_load(f)['environment']
                self.max_steps = self.config.get('max_steps', 60)
                self.action_step = self.config.get('action_step', 5.0)
                self.w_delay = self.config.get('w_delay', 1.0)
                self.w_fidelity = w  # Inject swept weight
                
        env = AblationEnv(config_path)
        
        # Train lightweight model
        model = PPO("MlpPolicy", env, n_steps=500, verbose=0)
        try:
            model.learn(total_timesteps=train_steps)
        except KeyboardInterrupt:
            pass
            
        # Evaluate
        metrics = evaluate_agent(env, model, num_episodes=2)
        
        results.append({
            "w_fidelity": w,
            "Total Delay (veh-s)": metrics["Total Delay (veh-s)"],
            "Spatial RMSE (Fidelity)": metrics["Spatial RMSE (Fidelity)"]
        })
        
        env.close()
        
    df = pd.DataFrame(results)
    out_csv = os.path.join(results_dir, 'ablation_metrics.csv')
    df.to_csv(out_csv, index=False)
    
    print(f"\nAblation Study Complete! Saved to {out_csv}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
