"""
SUMO State Extractor - Captures what actually happened in SUMO GUI

This script reads SUMO's simulation state to extract:
- Which vehicles were actually present in the simulation
- When they appeared (injection time)
- Their routes (origin -> destination)

This gives us the TRUE SUMO behavior to compare against video detections.
"""

import xml.etree.ElementTree as ET
import json
import os

def extract_sumo_state():
    """Extract actual SUMO simulation state from route file."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    route_file = os.path.join(base_dir, 'outputs', 'calibrated.rou.xml')
    
    sumo_vehicles = []
    
    if not os.path.exists(route_file):
        print(f"Error: {route_file} not found!")
        return sumo_vehicles
    
    tree = ET.parse(route_file)
    root = tree.getroot()
    
    # Map edge names to directions
    edge_to_dir = {
        'N': 'NORTH', 'NN': 'NORTH',
        'S': 'SOUTH', 'SS': 'SOUTH',
        'E': 'EAST', 'EE': 'EAST',
        'W': 'WEST', 'WW': 'WEST'
    }
    
    for vehicle in root.findall('vehicle'):
        veh_id = vehicle.get('id')
        depart = float(vehicle.get('depart'))
        
        # Only include vehicles within 60-second window
        if depart > 60:
            continue
        
        route = vehicle.find('route')
        if route is not None:
            edges = route.get('edges').split()
            if len(edges) >= 2:
                origin_edge = edges[0]
                dest_edge = edges[-1]
                
                origin = edge_to_dir.get(origin_edge, origin_edge)
                dest = edge_to_dir.get(dest_edge, dest_edge)
                
                # Extract vehicle ID number
                try:
                    vid_num = int(veh_id.split('_')[1])
                except:
                    vid_num = len(sumo_vehicles)
                
                sumo_vehicles.append({
                    'id': vid_num,
                    'origin': origin,
                    'dest': dest,
                    'depart': depart
                })
    
    return sumo_vehicles

def save_sumo_state():
    """Save SUMO state to JSON for comparison."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_file = os.path.join(base_dir, 'outputs', 'sumo_state.json')
    
    sumo_data = extract_sumo_state()
    
    with open(output_file, 'w') as f:
        json.dump(sumo_data, f, indent=4)
    
    print(f"Extracted {len(sumo_data)} vehicles from SUMO simulation (0-60s)")
    print(f"Saved to: {output_file}")
    
    return sumo_data

if __name__ == "__main__":
    save_sumo_state()
