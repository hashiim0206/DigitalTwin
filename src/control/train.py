"""
train.py
Trains a PPO Agent on the Digital-Twin-in-the-Loop SUMO environment.
"""

import os
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from env import SumoDTEnv

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, 'experiments', 'rl_config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['agent']
        
    print("--- Digital-Twin-in-the-Loop RL Training ---")
    print("Initializing environment...")
    
    # Initialize and wrap environment
    raw_env = SumoDTEnv(config_path)
    # check_env(raw_env)  # Validate the Gymnasium spec
    env = Monitor(raw_env)
    
    # Create logs dir
    log_dir = os.path.join(base_dir, 'outputs', 'rl_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        env, 
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=100, # Eval every 100 steps
        deterministic=True,
        render=False
    )
    
    print("Creating PPO Agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.get('learning_rate', 0.0003),
        n_steps=config.get('n_steps', 2048),
        batch_size=config.get('batch_size', 64),
        n_epochs=config.get('n_epochs', 10),
        gamma=config.get('gamma', 0.99),
        gae_lambda=config.get('gae_lambda', 0.95),
        clip_range=config.get('clip_range', 0.2),
        ent_coef=config.get('ent_coef', 0.01),
        verbose=1,
        tensorboard_log=log_dir
    )
    
    print("Starting Training...")
    # Train for enough timesteps to demonstrate learning convergence
    try:
        model.learn(total_timesteps=1500, callback=eval_callback)
    except KeyboardInterrupt:
        print("Training interrupted.")
        
    print("Saving model...")
    model.save(os.path.join(log_dir, "final_ppo_model"))
    env.close()
    
    print("Training Complete!")

if __name__ == "__main__":
    main()
