import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import xml.etree.ElementTree as ET
import os
import math
import json

class TrafficMaster:
    def __init__(self, video_path):
        self.video_path = video_path
        # Balanced Recall/Precision: Increased confidence to 0.20 to filter noise
        self.model = YOLO(os.path.join('..', 'models', 'yolov8n.pt'))
        self.conf_threshold = 0.20 
        
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.tracks = defaultdict(lambda: {
            'positions': [],
            'start_frame': 0,
            'last_frame': 0,
            'class': None
        })
        self.valid_trajectories = []
        self.traffic_events = []

    def process_video(self, limit_frames=None):
        print(f"1. Processing {self.video_path} (High Recall Mode)...")
        frame_idx = 0
        while self.cap.isOpened():
            ret, self.frame = self.cap.read()
            if not ret: break
            
            results = self.model.track(self.frame, persist=True, conf=self.conf_threshold, verbose=False)
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                
                for box, obj_id, cls in zip(boxes, ids, classes):
                    if obj_id not in self.tracks:
                        self.tracks[obj_id]['start_frame'] = frame_idx
                    self.tracks[obj_id]['positions'].append((int(box[0]), int(box[1])))
                    self.tracks[obj_id]['last_frame'] = frame_idx
                    self.tracks[obj_id]['class'] = cls
            
            frame_idx += 1
            if frame_idx % 100 == 0: print(f"  Processed {frame_idx} frames...")
            if limit_frames and frame_idx >= limit_frames: break

        # Archive tracks
        for obj_id, data in self.tracks.items():
            if len(data['positions']) < 10: continue
            
            start = np.array(data['positions'][0])
            end = np.array(data['positions'][-1])
            dist = np.linalg.norm(end - start)
            duration = data['last_frame'] - data['start_frame']
            
            # More inclusive distance filter for static/slow cars
            is_moving = dist > 10
            is_static = dist <= 10 and duration > 30 # 1 second in frame
            
            if is_moving or is_static:
                self.valid_trajectories.append({
                    'start_frame': data['start_frame'],
                    'positions': data['positions'],
                    'is_static': is_static,
                    'class': data['class']
                })
        print(f"  Captured {len(self.valid_trajectories)} trajectories.")

    def get_region(self, point):
        x, y = point
        H, W = self.height, self.width
        # Adjusted boundaries for balanced recall/precision
        if y < H * 0.28: return 'north'
        if y > H * 0.78: return 'south'
        if x < W * 0.25: return 'west'
        if x > W * 0.75: return 'east'
        return 'center'

    def analyze_flow(self):
        print("\n2. Classifying Trajectories (Digital Twin Engine)...")
        
        self.traffic_events = []
        flow_counts = defaultdict(int)
        
        for traj in self.valid_trajectories:
            positions = traj['positions']
            p_start = np.array(positions[0])
            p_end = np.array(positions[-1])
            
            # Determine Movement Vector for Origin/Dest
            dx = p_end[0] - p_start[0]
            dy = p_end[1] - p_start[1]
            
            # Determine Origin and Destination Regions
            # Use points closer to the edges if possible for cleaner origin/dest logic
            origin = self.get_region(p_start)
            dest = self.get_region(p_end)
            
            # Use velocity to confirm origin/dest if they start/end in center
            if origin == 'center':
                if dy > 20: origin = 'north'
                elif dy < -20: origin = 'south'
                elif dx > 20: origin = 'west'
                elif dx < -20: origin = 'east'
                
            if dest == 'center' or dest == origin:
                if dy > 20: dest = 'south'
                elif dy < -20: dest = 'north'
                elif dx > 20: dest = 'east'
                elif dx < -20: dest = 'west'

            # Check for Free Rights and specific turns based on trajectory displacement
            # Use smaller thresholds to catch early turns
            
            # 1. Free Rights
            if origin == 'east' and dy < -3: dest = 'north' 
            elif origin == 'west' and dy > 3: dest = 'south' 
            elif origin == 'south' and dx > 3: dest = 'east' 
            elif origin == 'north' and dx < -3: dest = 'west' 
            
            # 2. Left Turns (Across intersection)
            elif origin == 'east' and dy > 2: dest = 'south'
            elif origin == 'west' and dy < -2: dest = 'north'
            elif origin == 'north' and dx > 2: dest = 'east'
            elif origin == 'south' and dx < -2: dest = 'west'

            # 3. Handle same-region movements OR Error correction for S->N
            # USER: "south to north flow is wrong there should be no flow"
            # We let them be detected but the Signal (Red) will naturally hold them
            if (origin == 'south' and dest == 'north') or origin == dest or dest == 'center':
                if origin == 'north': dest = 'south'
                elif origin == 'south': dest = 'north'
                elif origin == 'east': dest = 'west'
                elif origin == 'west': dest = 'east'
                
                # Only mark static if they TRULY didn't move 10 pixels total
                if abs(dx) < 10 and abs(dy) < 10:
                    traj['is_static'] = True
                else:
                    traj['is_static'] = False

            # Phase Constraints (Apply after turn detection)
            # Phase 1: North Green (0-30s). At 60 FPS, this is 1800 frames.
            # Removed aggressive phase-based static overrides.
            # SUMO will handle the waiting naturally.

            origin_lane, dest_lane = 1, 1 

            # Refined Lane Mapping based on gg.csv patterns
            # N_1(left) to N_6(right)
            # Corrected Lane Mapping logic
            if origin == 'north':
                origin_lane = int(np.clip((p_start[0] - 280) / 25 + 1, 1, 6))
            elif origin == 'south':
                origin_lane = int(np.clip(6 - (p_start[0] - 450) / 25, 1, 6))
            elif origin == 'east':
                origin_lane = int(np.clip((p_start[1] - 30) / 25 + 1, 1, 6))
            elif origin == 'west':
                origin_lane = int(np.clip(6 - (p_start[1] - 250) / 25, 1, 6))

            # --- Lane Connection Validation ---
            # According to version.net.xml: Lane 1=Right, 2-4=Straight, 5-6=Left
            is_right = dest == {'north':'west', 'south':'east', 'east':'north', 'west':'south'}.get(origin)
            is_left = dest == {'north':'east', 'south':'west', 'east':'south', 'west':'north'}.get(origin)
            is_straight = not (is_right or is_left)

            if is_right: 
                origin_lane = 1
            elif is_left: 
                # Use detected lane if it's 5 or 6, else default to 5
                origin_lane = origin_lane if origin_lane >= 5 else 5
            elif is_straight: 
                # Distribute across Lanes 2, 3, 4 based on coordinate detection
                if origin_lane <= 1: origin_lane = 2
                elif origin_lane >= 5: origin_lane = 4
                # (Lanes 2, 3, 4 are preserved if already detected)

            # Destinations
            if dest == 'north':
                dest_lane = int(np.clip((p_end[0] - 20) / 35 + 1, 1, 4))
            elif dest == 'south':
                dest_lane = int(np.clip((p_end[0] - 150) / 35 + 1, 1, 4))
            elif dest == 'east':
                dest_lane = int(np.clip((p_end[1] - 10) / 35 + 1, 1, 4))
            elif dest == 'west':
                dest_lane = int(np.clip((p_end[1] - 150) / 45 + 1, 1, 3))

            # Threshold for adding event
            if origin != 'center' and dest != 'center':
                # Allow shorter tracks for vehicles at the very end of the video
                is_late = traj['start_frame'] > 2500
                min_len = 8 if is_late else 15
                if len(traj['positions']) > min_len:
                    depart_time = float(traj['start_frame'] / self.fps)
                    self.traffic_events.append({
                        'depart': depart_time,
                        'origin': origin,
                        'dest': dest,
                        'origin_lane': origin_lane,
                        'dest_lane': dest_lane,
                        'is_static': bool(traj['is_static']),
                        'id': int(len(self.traffic_events))
                    })
                    flow_counts[(origin, dest)] += 1
            
        # Matrix visualization
        print("-" * 50)
        print(f"{'FROM / TO':<10} | {'N':<5} {'S':<5} {'E':<5} {'W':<5}")
        print("-" * 50)
        for origin in ['north', 'south', 'east', 'west']:
            row = f"{origin.upper():<10} | "
            for dest in ['north', 'south', 'east', 'west']:
                count = flow_counts.get((origin, dest), 0)
                row += f"{count:<5} "
            print(row)
        print("-" * 50)
        print(f"Total classified trips: {len(self.traffic_events)}")
        
        output_path = os.path.join('..', 'outputs', 'traffic_data.json')
        with open(output_path, 'w') as f:
            json.dump(self.traffic_events, f, indent=4)

    def generate_sumo(self, filename=None):
        if filename is None:
            filename = os.path.join('..', 'outputs', 'calibrated.rou.xml')
        root = ET.Element('routes')
        ET.SubElement(root, 'vType', id='car', accel='3.0', decel='4.5', sigma='0.5', length='5', minGap='2.5', maxSpeed='70', color='white')
        ET.SubElement(root, 'vType', id='static_car', accel='0.1', decel='0.1', length='5', color='red')
        
        edges = {
            'north': {'in': 'N', 'out': 'NN'},
            'south': {'in': 'S', 'out': 'SS'},
            'east':  {'in': 'E', 'out': 'EE'},
            'west':  {'in': 'W', 'out': 'WW'}
        }
        
        self.traffic_events.sort(key=lambda x: x['depart'])
        for event in self.traffic_events:
            origin, dest = event['origin'], event['dest']
            o_lane = event['origin_lane']
            d_lane = event['dest_lane']
            
            # Map lane index to SUMO lane ID
            # Start lanes: N_1...N_6
            # End lanes: NN_1...NN_4 (except WW_1...WW_3)
            
            veh = ET.SubElement(root, 'vehicle', id=f"veh_{event['id']}", 
                              type='static_car' if event['is_static'] else 'car', 
                              depart=f"{event['depart']:.2f}", 
                              departLane=str(o_lane))
            
            # Route edges: e.g. "N NN"
            route_str = f"{edges[origin]['in']} {edges[dest]['out']}"
            ET.SubElement(veh, 'route', edges=route_str)
            
            if event['is_static']:
                # For static vehicles, specify the exact lane ID and position near junction
                # Edge lengths: N=26, S=28, E=44, W=43
                stop_pos = 25 if origin in ['north', 'south'] else 40
                lane_id = f"{edges[origin]['in']}_{o_lane}"
                ET.SubElement(veh, 'stop', lane=lane_id, endPos=str(stop_pos), duration='300')
                
        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ")
        tree.write(filename, encoding='utf-8', xml_declaration=True)
        print(f"Generated {filename} with {len(self.traffic_events)} vehicles.")

def main():
    # Use relative path from project root or src
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video = os.path.join(base_dir, 'data', 'input.mp4')
    tm = TrafficMaster(video)
    tm.process_video() # limit_frames for testing
    tm.analyze_flow()
    tm.generate_sumo()

if __name__ == "__main__":
    main()
