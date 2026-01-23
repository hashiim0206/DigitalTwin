"""
Comprehensive Comparison Script

Generates two key comparisons for conference paper:
1. Video AI vs Ground Truth (gg.csv) - Validates AI accuracy
2. Video AI vs SUMO State - Proves digital twin fidelity
"""

import json
import csv
import numpy as np
from collections import defaultdict
import os

def load_ground_truth(csv_path):
    """Load manual ground truth from gg.csv"""
    data = []
    if not os.path.exists(csv_path):
        return data
    
    # Map SUMO edge names to directions
    edge_map = {
        'N': 'NORTH', 'NN': 'NORTH',
        'S': 'SOUTH', 'SS': 'SOUTH',
        'E': 'EAST', 'EE': 'EAST',
        'W': 'WEST', 'WW': 'WEST'
    }
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_str = row['start_time']
            if ':' in t_str:
                m, s = map(int, t_str.split(':'))
                depart = m * 60 + s
            else:
                depart = float(t_str)
            
            if depart <= 60:
                origin = edge_map.get(row['start_edge'], row['start_edge'])
                dest = edge_map.get(row['end_edge'], row['end_edge'])
                data.append({
                    'origin': origin,
                    'dest': dest,
                    'depart': depart
                })
    return data

def load_json_data(json_path):
    """Load JSON data (video or SUMO)"""
    data = []
    if not os.path.exists(json_path):
        return data
    
    with open(json_path, 'r') as f:
        events = json.load(f)
        for event in events:
            if event['depart'] <= 60:
                data.append({
                    'origin': event['origin'].upper(),
                    'dest': event['dest'].upper(),
                    'depart': event['depart']
                })
    return data

def analyze_turns(data):
    """Categorize vehicles by turn type"""
    turn_types = {
        'left': [('NORTH', 'EAST'), ('SOUTH', 'WEST'), ('EAST', 'SOUTH'), ('WEST', 'NORTH')],
        'right': [('NORTH', 'WEST'), ('SOUTH', 'EAST'), ('EAST', 'NORTH'), ('WEST', 'SOUTH')],
        'straight': [('NORTH', 'SOUTH'), ('SOUTH', 'NORTH'), ('EAST', 'WEST'), ('WEST', 'EAST')]
    }
    
    counts = {'left': 0, 'right': 0, 'straight': 0}
    details = {'left': [], 'right': [], 'straight': []}
    
    for vehicle in data:
        flow = (vehicle['origin'], vehicle['dest'])
        for turn_type, flows in turn_types.items():
            if flow in flows:
                counts[turn_type] += 1
                details[turn_type].append(flow)
                break
    
    return counts, details

def compare_datasets(name1, data1, name2, data2):
    """Compare two datasets and print results"""
    print(f"\n{'='*60}")
    print(f"COMPARISON: {name1} vs {name2}")
    print(f"{'='*60}\n")
    
    # Volume
    print(f"Total Vehicles:")
    print(f"  {name1}: {len(data1)} vehicles")
    print(f"  {name2}: {len(data2)} vehicles")
    vol_acc = (min(len(data1), len(data2)) / max(len(data1), len(data2))) * 100
    print(f"  Volume Accuracy: {vol_acc:.2f}%\n")
    
    # Turn analysis
    counts1, details1 = analyze_turns(data1)
    counts2, details2 = analyze_turns(data2)
    
    print(f"Turn Type Breakdown:")
    print(f"{'Type':<12} | {name1:<15} | {name2:<15} | Deviation")
    print("-" * 60)
    
    total_deviation = 0
    for turn_type in ['straight', 'left', 'right']:
        c1, c2 = counts1[turn_type], counts2[turn_type]
        dev = abs(c1 - c2)
        total_deviation += dev
        acc = (min(c1, c2) / max(c1, c2)) * 100 if max(c1, c2) > 0 else 100
        print(f"{turn_type.capitalize():<12} | {c1:<15} | {c2:<15} | {dev} ({acc:.1f}%)")
    
    # RMSE calculation
    max_time = 61
    bins1 = np.zeros(max_time)
    bins2 = np.zeros(max_time)
    
    for v in data1:
        idx = int(v['depart'])
        if idx < max_time:
            bins1[idx] += 1
    
    for v in data2:
        idx = int(v['depart'])
        if idx < max_time:
            bins2[idx] += 1
    
    rmse = np.sqrt(np.mean((bins1 - bins2) ** 2))
    
    print(f"\nTemporal Metrics:")
    print(f"  RMSE: {rmse:.3f} vehicles/second")
    print(f"  Total Turn Deviation: {total_deviation} vehicles")
    
    return {
        'volume_accuracy': vol_acc,
        'rmse': rmse,
        'turn_counts_1': counts1,
        'turn_counts_2': counts2,
        'total_deviation': total_deviation
    }

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load all datasets
    gt_path = os.path.join(base_dir, 'data', 'gg.csv')
    video_path = os.path.join(base_dir, 'outputs', 'traffic_data.json')
    sumo_path = os.path.join(base_dir, 'outputs', 'sumo_state.json')
    
    print("Loading datasets...")
    ground_truth = load_ground_truth(gt_path)
    video_ai = load_json_data(video_path)
    sumo_state = load_json_data(sumo_path)
    
    print(f"Loaded: GT={len(ground_truth)}, Video={len(video_ai)}, SUMO={len(sumo_state)}")
    
    # Comparison 1: Video AI vs Ground Truth (Validates AI accuracy)
    results_ai_vs_gt = compare_datasets("Ground Truth", ground_truth, "Video AI", video_ai)
    
    # Comparison 2: Video AI vs SUMO (Proves digital twin fidelity)
    results_ai_vs_sumo = compare_datasets("Video AI", video_ai, "SUMO State", sumo_state)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY FOR CONFERENCE PAPER")
    print(f"{'='*60}\n")
    print("1. AI Detection Accuracy (Video AI vs Ground Truth):")
    print(f"   - Volume Accuracy: {results_ai_vs_gt['volume_accuracy']:.2f}%")
    print(f"   - RMSE: {results_ai_vs_gt['rmse']:.3f}")
    print(f"   - Turn Detection Deviation: {results_ai_vs_gt['total_deviation']} vehicles")
    
    print("\n2. Digital Twin Fidelity (Video AI vs SUMO):")
    print(f"   - Volume Accuracy: {results_ai_vs_sumo['volume_accuracy']:.2f}%")
    print(f"   - RMSE: {results_ai_vs_sumo['rmse']:.3f}")
    print(f"   - Perfect Replication: {results_ai_vs_sumo['total_deviation'] == 0}")

if __name__ == "__main__":
    main()
