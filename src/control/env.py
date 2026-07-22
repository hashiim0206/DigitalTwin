"""
env.py
Gymnasium environment for SUMO Digital-Twin-in-the-Loop RL Signal Control.
Computes in-memory spatial RMSE against real video trajectories as a penalty.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import os
import sys
import json
import yaml

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(base_dir, 'src'))

class SumoDTEnv(gym.Env):
    def __init__(self, config_path=None):
        super(SumoDTEnv, self).__init__()
        
        # Load Config
        if config_path is None:
            config_path = os.path.join(base_dir, 'experiments', 'rl_config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['environment']
            
        self.max_steps = self.config.get('max_steps', 60)
        self.action_step = self.config.get('action_step', 5.0)
        self.w_delay = self.config.get('w_delay', 1.0)
        self.w_fidelity = self.config.get('w_fidelity', 5.0)
        
        # Action Space: 4 discrete phases
        # 0: North-South Green
        # 1: East-West Green
        # 2: North-South Left Turn Green
        # 3: East-West Left Turn Green
        self.action_space = spaces.Discrete(4)
        
        # State Space: Queue lengths on 4 approaches (N, S, E, W)
        self.observation_space = spaces.Box(low=0, high=200, shape=(4,), dtype=np.float32)
        
        # Load Reference Traffic Data (Digital Twin Target)
        data_path = os.path.join(base_dir, 'outputs', 'traffic_data.json')
        with open(data_path, 'r') as f:
            self.traffic_data = json.load(f)
        self.traffic_data.sort(key=lambda x: x['depart'])
        
        # Build lookup for fast RMSE penalty calculation
        self.real_trajectories = {}
        for veh in self.traffic_data:
            vid = f"v_{veh['id']}"
            pos_array = np.array(veh['positions'])
            if len(pos_array) > 0:
                if pos_array.shape[1] == 2:
                    self.real_trajectories[vid] = pos_array
                else:
                    self.real_trajectories[vid] = pos_array[:, 1:] # Drop time if present
                
        # Load Optimized Car Params
        params_path = os.path.join(base_dir, 'outputs', 'best_params.json')
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                self.car_params = json.load(f)
        else:
            self.car_params = {"accel": "3.0", "decel": "4.5", "sigma": "0.5", "minGap": "2.5", "tau": "1.0"}
            
        self.edges = {
            'north': {'in': 'N', 'out': 'NN'},
            'south': {'in': 'S', 'out': 'SS'},
            'east':  {'in': 'E', 'out': 'EE'},
            'west':  {'in': 'W', 'out': 'WW'}
        }
        
        self.curr_time = 0.0
        self.data_idx = 0
        self.total_data = len(self.traffic_data)
        self.tls_id = "clusterJ1_J2_J4_J6"

    def _apply_phase(self, action):
        """Maps discrete actions to SUMO traffic light states."""
        s_list = ["r"] * 28
        
        if action == 0:
            # N-S Green
            for i in [0, 1, 2, 3, 4, 5]: s_list[i] = 'G' # N
            for i in [13, 14, 15, 16, 17]: s_list[i] = 'G' # S
            s_list[12] = 'g' # S Free Right
            s_list[18] = 'g' # W Free Right
            
        elif action == 1:
            # E-W Green
            for i in range(6, 12): s_list[i] = 'G'
            for i in range(18, 24): s_list[i] = 'G'
            s_list[12] = 'g'
            
        elif action == 2:
            # N-S Left
            # Just an example mapping based on standard phase layouts
            for i in [4, 5]: s_list[i] = 'G' # N Left
            for i in [16, 17]: s_list[i] = 'G' # S Left
            s_list[12] = 'g'
            s_list[18] = 'g'
            
        elif action == 3:
            # E-W Left
            for i in [10, 11]: s_list[i] = 'G' # E Left
            for i in [22, 23]: s_list[i] = 'G' # W Left
            s_list[12] = 'g'
            s_list[18] = 'g'
            
        traci.trafficlight.setRedYellowGreenState(self.tls_id, "".join(s_list))

    def _inject_vehicles(self):
        """Injects vehicles from traffic_data.json at their respective depart times."""
        while self.data_idx < self.total_data and self.traffic_data[self.data_idx]['depart'] <= self.curr_time:
            event = self.traffic_data[self.data_idx]
            origin, dest = event['origin'], event['dest']
            
            if origin in self.edges and dest in self.edges:
                veh_id = f"v_{event['id']}"
                route_id = f"r_{event['id']}"
                
                try:
                    route_info = traci.simulation.findRoute(self.edges[origin]['in'], self.edges[dest]['out'])
                    if route_info.edges:
                        traci.route.add(route_id, route_info.edges)
                        vtype = "static_car" if event['is_static'] else "car"
                        
                        l_idx = int(event.get('origin_lane', 1))
                        if l_idx == 0: l_idx = 1
                        
                        traci.vehicle.add(veh_id, route_id, typeID=vtype, departLane=str(l_idx), departPos="0")
                except Exception:
                    pass
            self.data_idx += 1

    def _get_obs(self):
        """Returns the queue lengths on the 4 approaches."""
        q_n = traci.edge.getLastStepHaltingNumber('N')
        q_s = traci.edge.getLastStepHaltingNumber('S')
        q_e = traci.edge.getLastStepHaltingNumber('E')
        q_w = traci.edge.getLastStepHaltingNumber('W')
        return np.array([q_n, q_s, q_e, q_w], dtype=np.float32)

    def _compute_fidelity_penalty(self):
        """
        The Novelty: Computes in-memory spatial RMSE between active SUMO vehicles
        and their real-world video trajectory counterparts to constrain the RL agent.
        """
        penalty = 0.0
        active_vehicles = traci.vehicle.getIDList()
        count = 0
        
        for vid in active_vehicles:
            if vid in self.real_trajectories:
                # Get SUMO position
                sx, sy = traci.vehicle.getPosition(vid)
                # Normalize SUMO coords (roughly -150 to 150)
                snx = (sx + 150) / 300
                sny = (sy + 150) / 300
                
                # We need to find the closest point on the real trajectory
                real_traj = self.real_trajectories[vid]
                
                # Normalize Video coords (0-1920, 0-1080)
                rnx = real_traj[:, 0] / 1920.0
                rny = real_traj[:, 1] / 1080.0
                
                # Compute distances to all points in the real trajectory
                dx = rnx - snx
                dy = rny - sny
                distances = np.sqrt(dx**2 + dy**2)
                
                # The minimum distance represents the spatial error regardless of exact timing
                min_dist = np.min(distances)
                penalty += min_dist
                count += 1
                
        if count > 0:
            return penalty / count
        return 0.0

    def step(self, action):
        self._apply_phase(action)
        
        step_penalty = 0.0
        step_delay = 0.0
        
        # Step simulation forward by action_step seconds
        target_time = self.curr_time + self.action_step
        while self.curr_time < target_time:
            self._inject_vehicles()
            traci.simulationStep()
            self.curr_time = traci.simulation.getTime()
            
            # Accumulate penalties at every tick
            step_penalty += self._compute_fidelity_penalty()
            obs = self._get_obs()
            step_delay += np.sum(obs)
            
            if self.curr_time >= self.max_steps:
                break
                
        # Final Reward
        reward = -(self.w_delay * step_delay) - (self.w_fidelity * step_penalty)
        
        terminated = bool(self.curr_time >= self.max_steps)
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        try:
            traci.close()
        except:
            pass
            
        net_file = os.path.join(base_dir, 'configs', 'version.net.xml')
        empty_route = os.path.join(base_dir, 'configs', 'empty.rou.xml')
        
        # Write route file with optimized params
        with open(empty_route, "w") as f:
            f.write('<routes>\n')
            vtype_str = f'    <vType id="car" accel="{self.car_params.get("accel", 3.0)}" ' \
                        f'decel="{self.car_params.get("decel", 4.5)}" ' \
                        f'sigma="{self.car_params.get("sigma", 0.5)}" ' \
                        f'minGap="{self.car_params.get("minGap", 2.5)}" ' \
                        f'tau="{self.car_params.get("tau", 1.0)}" ' \
                        f'length="5" maxSpeed="70"/>\n'
            f.write(vtype_str)
            f.write('    <vType id="static_car" accel="1.0" decel="2.0" length="5" color="1,0,0"/>\n')
            f.write('</routes>')
            
        sumo_cmd = [
            "sumo", "-n", net_file, "-r", empty_route,
            "--start", "--no-warnings", "--delay", "0",
            "--random"
        ]
        
        traci.start(sumo_cmd)
        
        self.curr_time = 0.0
        self.data_idx = 0
        
        return self._get_obs(), {}

    def close(self):
        try:
            traci.close()
        except:
            pass
