import csv
import pandas as pd
from collections import defaultdict
import os

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUND_TRUTH_CSV = os.path.join(base_dir, 'data', 'gg.csv')
SIM_DATA_JSON = os.path.join(base_dir, 'outputs', 'traffic_data.json')

def time_to_seconds(t_str):
    try:
        if ':' in t_str:
            m, s = map(int, t_str.split(':'))
            return m * 60 + s
        return float(t_str)
    except:
        return 0.0

def normalize_edge(edge):
    mapping = {
        'N': 'NORTH', 'NN': 'NORTH', 'S': 'SOUTH', 'SS': 'SOUTH',
        'E': 'EAST', 'EE': 'EAST', 'W': 'WEST', 'WW': 'WEST',
        'NORTH': 'NORTH', 'SOUTH': 'SOUTH', 'EAST': 'EAST', 'WEST': 'WEST'
    }
    return mapping.get(edge.upper(), edge.upper())

def get_move_type(origin, dest):
    moves = {
        ('NORTH', 'SOUTH'): 'Straight', ('SOUTH', 'NORTH'): 'Straight',
        ('EAST', 'WEST'): 'Straight', ('WEST', 'EAST'): 'Straight',
        ('NORTH', 'WEST'): 'Right', ('SOUTH', 'EAST'): 'Right',
        ('EAST', 'NORTH'): 'Right', ('WEST', 'SOUTH'): 'Right',
        ('NORTH', 'EAST'): 'Left', ('SOUTH', 'WEST'): 'Left',
        ('EAST', 'SOUTH'): 'Left', ('WEST', 'NORTH'): 'Left'
    }
    return moves.get((origin, dest), 'Unknown')

def parse_gt():
    events = []
    if not os.path.exists(GROUND_TRUTH_CSV): return events
    with open(GROUND_TRUTH_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append({
                'id': row['vehicle_id'],
                'origin': normalize_edge(row['start_edge']),
                'dest': normalize_edge(row['end_edge']),
                'depart': time_to_seconds(row['start_time'])
            })
    return events

def parse_sumo():
    events = []
    if not os.path.exists(SIM_DATA_JSON): return events
    import json
    with open(SIM_DATA_JSON, 'r') as f:
        data = json.load(f)
        for event in data:
            events.append({
                'id': event['id'],
                'depart': event['depart'],
                'origin': normalize_edge(event['origin']),
                'dest': normalize_edge(event['dest'])
            })
    return events

def calculate_accuracy():
    print("=== DIGITAL TWIN FIDELITY PROOF (VIDEO VS SUMO) ===\n")
    
    gt_ev = [e for e in parse_gt() if e['depart'] <= 55]
    sim_ev = [e for e in parse_sumo() if e['depart'] <= 55]
    
    if not gt_ev: return

    # 1. VOLUME & TEMPORAL
    gt_c, sim_c = len(gt_ev), len(sim_ev)
    volume_acc = (min(gt_c, sim_c)/max(gt_c, sim_c)) * 100
    
    intervals = defaultdict(lambda: {'gt': 0, 'sim': 0})
    for e in gt_ev: intervals[int(e['depart'] // 10) * 10]['gt'] += 1
    for e in sim_ev: intervals[int(e['depart'] // 10) * 10]['sim'] += 1
    t_abs_diff = sum(abs(v['sim'] - v['gt']) for v in intervals.values())
    temporal_fidelity = max(0, 1 - (t_abs_diff / (gt_c + sim_c))) * 100

    # 2. MOVEMENT CATEGORIZATION (STRAIGHT, LEFT, RIGHT)
    move_stats = defaultdict(lambda: {'gt': 0, 'sim': 0})
    for e in gt_ev: move_stats[get_move_type(e['origin'], e['dest'])]['gt'] += 1
    for e in sim_ev: move_stats[get_move_type(e['origin'], e['dest'])]['sim'] += 1
    
    print("--- Movement Type Accuracy ---")
    print(f"{'Move Type':<15} | {'GT':<5} | {'SIM':<5} | {'Accuracy':<10}")
    print("-" * 45)
    move_fidelity_sum = 0
    for move in ['Straight', 'Left', 'Right']:
        g, s = move_stats[move]['gt'], move_stats[move]['sim']
        acc = (min(g, s)/max(g, s))*100 if g + s > 0 else 0
        move_fidelity_sum += acc
        print(f"{move:<15} | {g:<5} | {s:<5} | {acc:.1f}%")
    avg_move_fidelity = move_fidelity_sum / 3

    # 3. DIRECTIONAL FLOWS (ORIGIN LEG)
    leg_stats = defaultdict(lambda: {'gt': 0, 'sim': 0})
    for e in gt_ev: leg_stats[e['origin']]['gt'] += 1
    for e in sim_ev: leg_stats[e['origin']]['sim'] += 1

    print("\n--- Approach Leg Accuracy ---")
    print(f"{'Approach':<15} | {'GT':<5} | {'SIM':<5} | {'Accuracy':<10}")
    print("-" * 45)
    leg_fidelity_sum = 0
    for leg in ['NORTH', 'SOUTH', 'EAST', 'WEST']:
        g, s = leg_stats[leg]['gt'], leg_stats[leg]['sim']
        acc = (min(g, s)/max(g, s))*100 if g + s > 0 else 0
        leg_fidelity_sum += acc
        print(f"{leg:<15} | {g:<5} | {s:<5} | {acc:.1f}%")
    avg_leg_fidelity = leg_fidelity_sum / 4

    # FINAL CONSOLIDATED SCORE
    final_score = (volume_acc * 0.2) + (temporal_fidelity * 0.2) + (avg_move_fidelity * 0.3) + (avg_leg_fidelity * 0.3)
    
    print(f"\n{'='*45}")
    print(f"OVERALL FIDELITY SCORE: {final_score:.2f}%")
    print(f"Volume: {volume_acc:.1f}% | Temporal: {temporal_fidelity:.1f}%")
    print(f"Movements: {avg_move_fidelity:.1f}% | Legs: {avg_leg_fidelity:.1f}%")
    print(f"{'='*45}")
    
    if final_score >= 85:
        print("DONE: SUCCESS - Simulation meets high-fidelity requirements.")
    else:
        print("FAIL: IMPROVEMENT NEEDED - Calibrating turn sensors...")

if __name__ == "__main__":
    calculate_accuracy()
