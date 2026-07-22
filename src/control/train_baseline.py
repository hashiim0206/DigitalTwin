"""
train_baseline.py
Trains a Standard Delay-Only PPO Agent for comparison.
"""

import os
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from env import SumoDTEnv

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, 'experiments', 'rl_config.yaml')
    
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
        agent_config = full_config['agent']
        
    print("--- Standard Baseline RL Training ---")
    
    # Initialize environment, overriding the env config key internally or by modifying the env.py
    # Since env.py loads 'environment' by default, we can temporarily modify it or pass a flag.
    # Actually, env.py doesn't take a flag for the config key. Let's subclass or patch it.
    
    class BaselineEnv(SumoDTEnv):
        def __init__(self, config_path=None):
            super().__init__(config_path)
            # Override with baseline settings
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)['environment_baseline']
            self.max_steps = self.config.get('max_steps', 60)
            self.action_step = self.config.get('action_step', 5.0)
            self.w_delay = self.config.get('w_delay', 1.0)
            self.w_fidelity = self.config.get('w_fidelity', 0.0) # NO FIDELITY PENALTY

    raw_env = BaselineEnv(config_path)
    env = Monitor(raw_env)
    
    log_dir = os.path.join(base_dir, 'outputs', 'rl_logs_baseline')
    os.makedirs(log_dir, exist_ok=True)
    
    eval_callback = EvalCallback(
        env, 
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=100,
        deterministic=True,
        render=False
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=agent_config.get('learning_rate', 0.0003),
        n_steps=agent_config.get('n_steps', 2048),
        batch_size=agent_config.get('batch_size', 64),
        n_epochs=agent_config.get('n_epochs', 10),
        gamma=agent_config.get('gamma', 0.99),
        gae_lambda=agent_config.get('gae_lambda', 0.95),
        clip_range=agent_config.get('clip_range', 0.2),
        ent_coef=agent_config.get('ent_coef', 0.01),
        verbose=1,
        tensorboard_log=log_dir
    )
    
    print("Starting Training (Baseline Agent)...")
    try:
        model.learn(total_timesteps=1500, callback=eval_callback)
    except KeyboardInterrupt:
        pass
        
    model.save(os.path.join(log_dir, "final_ppo_model"))
    env.close()

if __name__ == "__main__":
    main()
