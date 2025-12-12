import cv2
import json
import os

VIDEO_PATH = "input.mp4"
OUTPUT_JSON = "lanes.json"

lanes = []
current_points = []
current_lane_name = None
img = None
clone = None

def click_event(event, x, y, flags, param):
    global current_points, img

    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((int(x), int(y)))
        # draw small circle
        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
        # draw lines between points
        if len(current_points) > 1:
            cv2.line(img, current_points[-2], current_points[-1], (0, 0, 255), 2)
        cv2.imshow("Define Lanes", img)

def main():
    global img, clone, current_points, current_lane_name, lanes

    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: {VIDEO_PATH} not found.")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("ERROR: Could not read first frame from video.")
        return

    img = frame.copy()
    clone = frame.copy()

    cv2.namedWindow("Define Lanes")
    cv2.setMouseCallback("Define Lanes", click_event)

    print("Instructions:")
    print("- We will define one lane at a time.")
    print("- Left-click to add polygon points around the lane area.")
    print("- When done for a lane, press 'n' to save it and move to the next lane.")
    print("- Press 'q' when you are completely finished.")

    lane_index = 1

    while True:
        cv2.imshow("Define Lanes", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('n'):
            if len(current_points) < 3:
                print("Need at least 3 points to form a polygon. Keep clicking.")
                continue

            lane_name = input(f"Enter lane ID for this polygon (e.g., N_in_L{lane_index}): ")
            if not lane_name:
                lane_name = f"lane_{lane_index}"

            lanes.append({
                "lane_id": lane_name,
                "points": current_points.copy()
            })
            print(f"Saved lane '{lane_name}' with {len(current_points)} points.")

            lane_index += 1
            current_points = []
            img = clone.copy()

        elif key == ord('q'):
            print("Finished defining lanes.")
            break

    cv2.destroyAllWindows()

    if lanes:
        with open(OUTPUT_JSON, "w") as f:
            json.dump(lanes, f, indent=2)
        print(f"Saved {len(lanes)} lanes to {OUTPUT_JSON}")
    else:
        print("No lanes defined. Nothing saved.")

if __name__ == "__main__":
    main()
