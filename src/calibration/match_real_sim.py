"""
match_real_sim.py
Matches real vehicle trajectories (from traffic_data.json) to simulated
trajectories (from fcd.xml) based on entry time and approach.
"""

import json
import os
import xml.etree.ElementTree as ET
import numpy as np

def load_real_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_sim_trajectories(fcd_path):
    """
    Parses fcd.xml and groups coordinates by vehicle ID over time.
    Returns: dict {veh_id: {'depart': float, 'positions': [(time, x, y), ...], 'edge': start_edge}}
    """
    tree = ET.parse(fcd_path)
    root = tree.getroot()
    
    sim_data = {}
    for timestep in root.findall('timestep'):
        t = float(timestep.get('time'))
        for vehicle in timestep.findall('vehicle'):
            vid = vehicle.get('id')
            x = float(vehicle.get('x'))
            y = float(vehicle.get('y'))
            lane = vehicle.get('lane')
            edge = lane.split('_')[0]
            
            if vid not in sim_data:
                sim_data[vid] = {
                    'id': vid,
                    'depart': t,
                    'start_edge': edge,
                    'positions': []
                }
            sim_data[vid]['positions'].append((t, x, y))
            
    return sim_data

def edge_to_direction(edge):
    mapping = {
        'N': 'north', 'NN': 'north',
        'S': 'south', 'SS': 'south',
        'E': 'east', 'EE': 'east',
        'W': 'west', 'WW': 'west'
    }
    return mapping.get(edge, 'center')

def match_trajectories(real_data, sim_data, time_tolerance=5.0):
    """
    Matches real and sim vehicles by depart time and origin direction.
    """
    matched = []
    unmatched_real = []
    
    # Sort both lists by depart time
    real_sorted = sorted(real_data, key=lambda x: x['depart'])
    
    # Convert sim_data dict to list and sort
    sim_list = list(sim_data.values())
    sim_list.sort(key=lambda x: x['depart'])
    
    sim_used = set()
    
    for real_veh in real_sorted:
        r_depart = real_veh['depart']
        r_origin = real_veh['origin'].lower()
        r_id = real_veh['id']
        
        # In this specific project, v_{id} is the exact match for baseline.
        # However, we must implement robust matching for free-running SUMO cases.
        direct_match_id = f"v_{r_id}"
        
        best_match = None
        min_time_diff = float('inf')
        
        # 1. Try strict ID match first (if it exists)
        if direct_match_id in sim_data and direct_match_id not in sim_used:
            best_match = sim_data[direct_match_id]
        else:
            # 2. Robust matching by time and approach
            for sim_veh in sim_list:
                if sim_veh['id'] in sim_used:
                    continue
                    
                s_origin = edge_to_direction(sim_veh['start_edge'])
                time_diff = abs(r_depart - sim_veh['depart'])
                
                if s_origin == r_origin and time_diff <= time_tolerance:
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        best_match = sim_veh
                        
        if best_match:
            sim_used.add(best_match['id'])
            matched.append({
                'real_id': r_id,
                'sim_id': best_match['id'],
                'real_depart': r_depart,
                'sim_depart': best_match['depart'],
                'origin': r_origin,
                'dest': real_veh['dest'].lower(),
                'real_positions': real_veh['positions'],
                'sim_positions': best_match['positions']
            })
        else:
            unmatched_real.append(r_id)
            
    return matched, unmatched_real

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    real_path = os.path.join(base_dir, 'outputs', 'traffic_data.json')
    sim_path = os.path.join(base_dir, 'outputs', 'fcd.xml')
    output_path = os.path.join(base_dir, 'outputs', 'matched_trajectories.json')
    
    if not os.path.exists(sim_path):
        print(f"Error: {sim_path} not found. Make sure FCD output is enabled in SUMO.")
        return
        
    print("Loading real data...")
    real_data = load_real_data(real_path)
    
    print("Extracting SUMO FCD data...")
    sim_data = extract_sim_trajectories(sim_path)
    
    print("Matching trajectories...")
    matched, unmatched = match_trajectories(real_data, sim_data)
    
    print(f"Successfully matched {len(matched)} / {len(real_data)} vehicles.")
    if unmatched:
        print(f"Unmatched real vehicles: {len(unmatched)}")
        
    with open(output_path, 'w') as f:
        json.dump(matched, f, indent=4)
    print(f"Matched data saved to {output_path}")

if __name__ == "__main__":
    main()
