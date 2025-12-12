import cv2
import json
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from ultralytics import YOLO

VIDEO_PATH = "input.mp4"
LANES_JSON = "lanes.json"
OUTPUT_CSV = "trajectories.csv"

# Vehicle classes YOLO detects
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}

def load_lanes():
    with open(LANES_JSON, "r") as f:
        data = json.load(f)
    lanes = []
    for lane in data:
        polygon = Polygon(lane["points"])
        lanes.append({
            "lane_id": lane["lane_id"],
            "polygon": polygon
        })
    return lanes

def main():
    lanes = load_lanes()
    print(f"Loaded {len(lanes)} lanes from {LANES_JSON}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS =", fps)
    cap.release()

    # Load YOLO (Nanomodel for speed)
    model = YOLO("yolov8n.pt")

    trajectories = []
    frame_idx = 0

    # YOLO tracker
    for result in model.track(
        source=VIDEO_PATH,
        stream=True,
        persist=True,
        verbose=False,
        agnostic_nms=True
    ):
        boxes = result.boxes
        if boxes is None:
            frame_idx += 1
            continue

        t = frame_idx / fps
        frame = result.orig_img.copy()

        for box in boxes:
            # Check class
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            if cls_name not in VEHICLE_CLASSES:
                continue

            # Track ID
            if box.id is None:
                continue
            vid = int(box.id[0])

            # Bounding box center
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            point = Point(cx, cy)

            # Determine lane_id
            lane_id = "none"
            for lane in lanes:
                if lane["polygon"].contains(point):
                    lane_id = lane["lane_id"]
                    break

            # Save trajectory
            trajectories.append({
                "veh_id": vid,
                "time_s": t,
                "cx": cx,
                "cy": cy,
                "lane_id": lane_id
            })

        # Optional: show overlay
        cv2.imshow("Trajectories", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cv2.destroyAllWindows()

    # SAVE CSV
    df = pd.DataFrame(trajectories)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved trajectories to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
