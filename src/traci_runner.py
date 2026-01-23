import traci
import json
import time
import os
import sys

# Configuration
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET_FILE = os.path.join(base_dir, 'configs', 'version.net.xml')
DATA_FILE = os.path.join(base_dir, 'outputs', 'traffic_data.json')
TLS_ID = "clusterJ1_J2_J4_J6"

def setup_simulation():
    """Starts SUMO and initializes environment types."""
    # Ensure TraCI tools are found if SUMO_HOME is set
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    
    # Create a fresh empty route file to avoid pre-existing flow collisions
    empty_route = os.path.join(base_dir, 'configs', 'empty.rou.xml')
    with open(empty_route, "w") as f:
        f.write('<routes>\n')
        f.write('    <vType id="car" accel="3.0" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="70"/>\n')
        f.write('    <vType id="static_car" accel="1.0" decel="2.0" length="5" color="1,0,0"/>\n')
        f.write('</routes>')

    # Launch SUMO-GUI with EMPTY routes and generate trip statistics
    tripinfo_path = os.path.join(base_dir, 'outputs', 'tripinfo.xml')
    sumo_cmd = [
        "sumo-gui", "-n", NET_FILE, "-r", empty_route, 
        "--start", "--no-warnings", "--delay", "200",
        "--tripinfo-output", tripinfo_path
    ]
    
    print(f"--- Digital Twin Sync ---\nLaunching SUMO with 384 pre-loaded routes and 200ms delay...")
    traci.start(sumo_cmd)
    print("Connection established with TraCI.")

def run_sync():
    try:
        setup_simulation()
        
        # Load the 252 detected vehicles from input.mp4
        with open(DATA_FILE, 'r') as f:
            traffic_data = json.load(f)
        traffic_data.sort(key=lambda x: x['depart'])
        
        edges = {
            'north': {'in': 'N', 'out': 'NN'},
            'south': {'in': 'S', 'out': 'SS'},
            'east':  {'in': 'E', 'out': 'EE'},
            'west':  {'in': 'W', 'out': 'WW'}
        }
        
        data_idx = 0
        injected_count = 0
        total_data = len(traffic_data)
        
        print(f"Syncing {total_data} real-video vehicles...")

        while traci.simulation.getMinExpectedNumber() > 0 or data_idx < total_data:
            curr_time = traci.simulation.getTime()
            
            # 1. SIGNAL SYNC (Natural Phasing)
            # Link indices: N(0-5), E(6-11), S(12-17), W(18-23)
            s_list = ["r"] * 28

            if curr_time < 25:
                # Phase 1: North Green
                # North indices 0-5.
                for i in [0, 1, 2, 3, 4, 5]: s_list[i] = 'G'
                # Free Rights: South (12) and West (18)
                s_list[12] = 'g'
                s_list[18] = 'g'
            elif curr_time < 30:
                # Transition Phase (Yellow for North)
                for i in [0, 1, 2, 3, 4, 5]: s_list[i] = 'y'
                # Keep Free Rights yield-green
                s_list[12] = 'g'
                s_list[18] = 'g'
            elif curr_time < 45:
                # Phase 2: East/West PROTECTED LEFT
                # E-Left: 10,11. W-Left: 22,23. 
                for i in [10, 11, 22, 23]: s_list[i] = 'G'
                # Free Rights: South (12) and West (18)
                # West Right (18) stays 'g' while West Left (22,23) is 'G'
                s_list[12] = 'g'
                s_list[18] = 'g'
            else:
                # Phase 3: East/West TOTAL GREEN
                # Straight + Left + Right for East (6-11) and West (18-23)
                for i in range(6, 12): s_list[i] = 'G'
                for i in range(18, 24): s_list[i] = 'G'
                # South Right (12) can still be a free right during EW Green
                s_list[12] = 'g'
            
            traci.trafficlight.setRedYellowGreenState(TLS_ID, "".join(s_list))

            # 2. VEHICLE INJECTION
            while data_idx < total_data and traffic_data[data_idx]['depart'] <= curr_time:
                event = traffic_data[data_idx]
                origin, dest = event['origin'], event['dest']
                
                veh_id = f"v_{event['id']}"
                route_id = f"r_{event['id']}"
                
                try:
                    route_info = traci.simulation.findRoute(edges[origin]['in'], edges[dest]['out'])
                    if route_info.edges:
                        traci.route.add(route_id, route_info.edges)
                        
                        vtype = "car"
                        if event['is_static']: vtype = "static_car"
                        
                        depart_lane = str(event.get('origin_lane', 1))
                        l_idx = int(depart_lane)
                        if l_idx == 0: l_idx = 1
                        
                        # Every vehicle starts at the VERY BEGINNING of the edge
                        # This ensures natural movement and stopping behavior relative to the light.
                        # No more "artificial halting" or teleporting to the stop line.
                        traci.vehicle.add(veh_id, route_id, typeID=vtype, departLane=str(l_idx), departPos="0")
                        
                        # Removed setStop to allow vehicles to wait naturally for Red lights
                        # and move naturally when it turns Green.
                        
                        injected_count += 1
                except Exception: pass
                data_idx += 1

            traci.simulationStep()
            
            # Log progress every 5 seconds
            if int(curr_time) % 5 == 0 and curr_time == int(curr_time):
                print(f"  [T={curr_time:.1f}s] Injected: {injected_count}/{total_data} | Active: {traci.vehicle.getIDCount()}")
            
            # Sync stop at adjusted duration (60 steps)
            if curr_time > 60:
                break
                
        print(f"\nDigital Twin Sync Complete. Total Vehicles: {injected_count}")
        
    except Exception as e:
        print(f"Fatal Sync Error: {e}")
    finally:
        try:
            traci.close()
        except:
            pass

if __name__ == "__main__":
    run_sync()
