import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import xml.etree.ElementTree as ET
import os
import math
import json
from filterpy.kalman import KalmanFilter
from sklearn.cluster import KMeans

class TrafficMaster:
    def __init__(self, video_path, conf_threshold=0.20, use_kalman=True, use_kmeans=True, min_dist=15, min_frames=15):
        self.video_path = video_path
        # Load model with explicit path handling
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'yolov8n.pt')
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.use_kalman = use_kalman
        self.use_kmeans = use_kmeans
        self.min_dist = min_dist
        self.min_frames = min_frames
        
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.tracks = defaultdict(lambda: {
            'positions': [],
            'start_frame': 0,
            'last_frame': 0,
            'class': None,
            'kf': self.init_kalman() if self.use_kalman else None
        })
        self.valid_trajectories = []
        self.traffic_events = []
        self.kmeans_model = None
        self.lane_clusters = {}

    def init_kalman(self):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 0, 1, 0],  # state transition matrix
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],  # measurement function
                         [0, 1, 0, 0]])
        kf.P *= 10.0                    # covariance matrix
        kf.R = 5                        # measurement noise
        kf.Q = np.eye(4) * 0.1          # process noise
        return kf

    def process_video(self, limit_frames=None):
        print(f"1. Processing {self.video_path} (Conf={self.conf_threshold}, Kalman={self.use_kalman})...", flush=True)
        frame_idx = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break
            
            # Run YOLO directly
            results = self.model.track(frame, persist=True, conf=self.conf_threshold, verbose=False, iou=0.5)
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                
                for box, obj_id, cls in zip(boxes, ids, classes):
                    center_x, center_y = int(box[0]), int(box[1])
                    
                    if obj_id not in self.tracks:
                        self.tracks[obj_id]['start_frame'] = frame_idx
                    
                    # Kalman Filter Update
                    if self.use_kalman:
                        kf = self.tracks[obj_id]['kf']
                        if len(self.tracks[obj_id]['positions']) == 0:
                            kf.x[:2] = np.array([[center_x], [center_y]])
                        
                        kf.predict()
                        kf.update([center_x, center_y])
                        filetered_x, filetered_y = kf.x[0], kf.x[1]
                        self.tracks[obj_id]['positions'].append((float(filetered_x), float(filetered_y)))
                    else:
                        self.tracks[obj_id]['positions'].append((center_x, center_y))
                        
                    self.tracks[obj_id]['last_frame'] = frame_idx
                    self.tracks[obj_id]['class'] = cls
            
            frame_idx += 1
            if frame_idx % 100 == 0: print(f"  Processed {frame_idx} frames...", flush=True)
            if limit_frames and frame_idx >= limit_frames: break

        # Post-Processing: cleanup tracks
        self.cleanup_tracks()

    def cleanup_tracks(self):
        for obj_id, data in self.tracks.items():
            if len(data['positions']) < 10: continue
            
            start = np.array(data['positions'][0])
            end = np.array(data['positions'][-1])
            dist = np.linalg.norm(end - start)
            duration = data['last_frame'] - data['start_frame']
            
            # Filter short/static tracks
            is_moving = dist > self.min_dist
            is_static = dist <= self.min_dist and duration > (self.min_frames * 2) 
            
            if (is_moving or is_static) and duration > self.min_frames:
                self.valid_trajectories.append({
                    'id': obj_id,
                    'start_frame': data['start_frame'],
                    'positions': data['positions'],
                    'is_static': is_static,
                    'class': data['class']
                })
        print(f"  Captured {len(self.valid_trajectories)} trajectories.")

    def train_kmeans_lanes(self):
        # Gather all start points to learn lane clusters
        start_points = []
        for t in self.valid_trajectories:
            start_points.append(t['positions'][0])
        
        if len(start_points) < 4: return # Not enough data
        
        start_points = np.array(start_points)
        # Learn 4 main approach clusters (N, S, E, W)
        kmeans = KMeans(n_clusters=4, random_state=0, n_init=10).fit(start_points)
        self.kmeans_model = kmeans
        
        # Identify which cluster is which based on centroids
        centroids = kmeans.cluster_centers_
        # N=Top(min y), S=Bottom(max y), W=Left(min x), E=Right(max x)
        # We assign labels logic later
        self.centroids = centroids

    def get_region_ml(self, point):
        # Predict using KMeans model
        if self.kmeans_model is None: return self.get_region_heuristic(point)
        
        cluster_idx = self.kmeans_model.predict([point])[0]
        cx, cy = self.centroids[cluster_idx]
        
        H, W = self.height, self.width
        # Map centroid to region
        if cy < H * 0.3: return 'north'
        if cy > H * 0.7: return 'south'
        if cx < W * 0.3: return 'west'
        if cx > W * 0.7: return 'east'
        return 'center'

    def get_region_heuristic(self, point):
        x, y = point
        H, W = self.height, self.width
        if y < H * 0.28: return 'north'
        if y > H * 0.78: return 'south'
        if x < W * 0.25: return 'west'
        if x > W * 0.75: return 'east'
        return 'center'

    def analyze_flow(self):
        print("\n2. Classifying Trajectories (Digital Twin Engine)...", flush=True)
        
        if self.use_kmeans:
            self.train_kmeans_lanes()
            
        self.traffic_events = []
        
        for traj in self.valid_trajectories:
            positions = traj['positions']
            p_start = positions[0]
            p_end = positions[-1]
            
            # --- Improved Heading-Based Logic ---
            # Calculate start and end vectors (using first/last 10% of points for stability)
            n_pts = len(positions)
            window = max(3, int(n_pts * 0.15))
            
            p_start_avg = np.mean(positions[:window], axis=0)
            p_end_avg = np.mean(positions[-window:], axis=0)
            
            # Overall movement vector
            dx = p_end_avg[0] - p_start_avg[0]
            dy = p_end_avg[1] - p_start_avg[1]
            
            # Determine dominant direction map
            # 0: East (+x), 90: South (+y), 180: West (-x), 270: North (-y)
            angle = np.degrees(np.arctan2(dy, dx)) % 360
            
            if self.use_kmeans:
                origin = self.get_region_ml(p_start)
                dest = self.get_region_ml(p_end)
            else:
                origin = self.get_region_heuristic(p_start)
                dest = self.get_region_heuristic(p_end)

            # Heuristic Correction based on dominant movement
            if origin == 'center':
                # If moving South (45-135 deg) -> came from North
                if 45 <= angle < 135: origin = 'north'
                # If moving North (225-315 deg) -> came from South
                elif 225 <= angle < 315: origin = 'south'
                # If moving East (315-45 deg) -> came from West
                elif angle >= 315 or angle < 45: origin = 'west'
                # If moving West (135-225 deg) -> came from East
                elif 135 <= angle < 225: origin = 'east'
            
            if dest == 'center' or dest == origin or (origin in ['north','south'] and dest in ['north','south']) or (origin in ['east','west'] and dest in ['east','west']):
                # Recalculate dest based on trajectory curve
                # Compare start heading vs end heading
                if n_pts > 10:
                    v_start = np.array(positions[min(5, n_pts-1)]) - np.array(positions[0])
                    v_end = np.array(positions[-1]) - np.array(positions[max(0, n_pts-5)])
                    
                    angle_start = np.degrees(np.arctan2(v_start[1], v_start[0]))
                    angle_end = np.degrees(np.arctan2(v_end[1], v_end[0]))
                    diff = (angle_end - angle_start + 180) % 360 - 180
                    
                    # Refinement: Narrow Straight to 25 degrees to force more Turns
                    # Previous Stats: Straight Mean=-1.2, Std=18. Misclassified turns likely in 25-45 range.
                    
                    # Assign Dest based on Origin + Turn
                    if abs(diff) <= 25: # Narrow Straight
                        if origin == 'north': dest = 'south'
                        elif origin == 'south': dest = 'north'
                        elif origin == 'east': dest = 'west'
                        elif origin == 'west': dest = 'east'
                    elif diff < -25: # Turning Left (Wider range including shallow turns)
                        if origin == 'north': dest = 'east' # Down -> Right
                        elif origin == 'south': dest = 'west' # Up -> Left
                        elif origin == 'east': dest = 'south' # Left -> Down
                        elif origin == 'west': dest = 'north' # Right -> Up
                    elif diff > 25: # Turning Right (Wider range)
                         if origin == 'north': dest = 'west'
                         elif origin == 'south': dest = 'east'
                         elif origin == 'east': dest = 'north'
                         elif origin == 'west': dest = 'south'
                
                # Fallback: Smart Logic for Diagonals (favor turns)
                if dest == 'center' or dest == origin:
                     # 0: East, 90: South, 180: West, 270: North
                     if origin == 'north': # Coming from Top
                         if 340 <= angle or angle < 60: dest = 'east' # Left Turn (User View)
                         elif 120 <= angle < 200: dest = 'west' # Right Turn
                         elif 60 <= angle < 120: dest = 'south' # Straight
                     elif origin == 'south': # Coming from Bottom
                         if 160 <= angle < 240: dest = 'west' # Left Turn
                         elif 300 <= angle < 360 or angle < 20: dest = 'east' # Right Turn
                         elif 240 <= angle < 300: dest = 'north' # Straight
                     elif origin == 'east': # Coming from Right
                         if 200 <= angle < 280: dest = 'south'
                         elif 80 <= angle < 160: dest = 'north'
                         elif 160 <= angle < 200: dest = 'west'
                     elif origin == 'west': # Coming from Left
                         if 260 <= angle < 340: dest = 'north' # Left
                         elif 20 <= angle < 100: dest = 'south' # Right
                         elif 340 <= angle or angle < 20: dest = 'east'

            # Lane Mapping Logic (Can be optimized later, hardcoded for now)
            origin_lane, dest_lane = 1, 1
            if origin == 'north': origin_lane = int(np.clip((p_start[0] - 280) / 25 + 1, 1, 6))
            elif origin == 'south': origin_lane = int(np.clip(6 - (p_start[0] - 450) / 25, 1, 6))
            elif origin == 'east': origin_lane = int(np.clip((p_start[1] - 30) / 25 + 1, 1, 6))
            elif origin == 'west': origin_lane = int(np.clip(6 - (p_start[1] - 250) / 25, 1, 6))

            # Final check before adding
            if origin != 'center' and dest != 'center':
                 self.traffic_events.append({
                    'depart': float(traj['start_frame'] / self.fps),
                    'origin': origin,
                    'dest': dest,
                    'origin_lane': origin_lane,
                    'dest_lane': dest_lane,
                    'is_static': bool(traj['is_static']),
                    'id': int(traj['id']),
                    'positions': traj['positions'] # Export full path for Space-Time Diagram
                })
        
        # Sort by departure time
        self.traffic_events.sort(key=lambda x: x['depart'])
        
        # Save output
        output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'traffic_data.json')
        with open(output_path, 'w') as f:
            json.dump(self.traffic_events, f, indent=4)

    def generate_sumo(self, filename=None):
        if filename is None:
            filename = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'calibrated.rou.xml')
        root = ET.Element('routes')
        ET.SubElement(root, 'vType', id='car', accel='3.0', decel='4.5', sigma='0.5', length='5', minGap='2.5', maxSpeed='70', color='white')
        ET.SubElement(root, 'vType', id='static_car', accel='0.1', decel='0.1', length='5', color='red')
        
        edges = {
            'north': {'in': 'N', 'out': 'NN'},
            'south': {'in': 'S', 'out': 'SS'},
            'east':  {'in': 'E', 'out': 'EE'},
            'west':  {'in': 'W', 'out': 'WW'}
        }
        
        for event in self.traffic_events:
            if event['origin'] not in edges or event['dest'] not in edges: continue
            
            origin, dest = event['origin'], event['dest']
            o_lane = event['origin_lane']
            
            veh = ET.SubElement(root, 'vehicle', id=f"veh_{event['id']}", 
                              type='static_car' if event['is_static'] else 'car', 
                              depart=f"{event['depart']:.2f}", 
                              departLane=str(o_lane))
            
            route_str = f"{edges[origin]['in']} {edges[dest]['out']}"
            ET.SubElement(veh, 'route', edges=route_str)
            
            if event['is_static']:
                stop_pos = 25 if origin in ['north', 'south'] else 40
                lane_id = f"{edges[origin]['in']}_{o_lane}"
                ET.SubElement(veh, 'stop', lane=lane_id, endPos=str(stop_pos), duration='300')
                
        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ")
        tree.write(filename, encoding='utf-8', xml_declaration=True)
        print(f"Generated {filename} with {len(self.traffic_events)} vehicles.", flush=True)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    video = os.path.join(base_dir, 'data', 'input.mp4')
    
    # Run in Full Production Mode
    print("Running Full Video Processing with Optimized Parameters...", flush=True)
    tm = TrafficMaster(video, conf_threshold=0.18, use_kalman=True, use_kmeans=False)
    tm.process_video() 
    
    tm.analyze_flow()
    tm.generate_sumo()

if __name__ == "__main__":
    main()
