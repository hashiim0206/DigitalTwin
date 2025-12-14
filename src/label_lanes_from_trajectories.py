# label_lanes_from_trajectories.py

import pandas as pd

IN_CSV = "trajectories.csv"
OUT_CSV = "trajectories_labeled.csv"

def main():
    df = pd.read_csv(IN_CSV)

    # Robust sort
    df = df.sort_values(["veh_id", "time_s"])

    # Intersection center = median position of all vehicles
    cx_center = df["cx"].median()
    cy_center = df["cy"].median()
    print(f"Center approx at (cx={cx_center:.2f}, cy={cy_center:.2f})")

    def classify_dir(x, y):
        dx = x - cx_center
        dy = y - cy_center
        # Decide whether movement is more horizontal or vertical
        if abs(dx) > abs(dy):
            # Comes from left or right
            return "W" if dx < 0 else "E"
        else:
            # Comes from top or bottom
            return "N" if dy < 0 else "S"

    def classify_lane(origin, dest):
        # origin, dest are one of 'W','E','N','S'
        if origin == "W":
            if dest == "E":
                return "W_through"
            elif dest == "N":
                return "W_left_turn"
            elif dest == "S":
                return "W_right_turn"
        elif origin == "E":
            if dest == "W":
                return "E_through"
            elif dest == "S":
                return "E_left_turn"
            elif dest == "N":
                return "E_right_turn"
        elif origin == "N":
            if dest == "S":
                return "N_through"
            elif dest == "W":
                return "N_left_turn"
            elif dest == "E":
                return "N_right_turn"
        elif origin == "S":
            if dest == "N":
                return "S_through"
            elif dest == "E":
                return "S_left_turn"
            elif dest == "W":
                return "S_right_turn"

        # Weird / U-turn / stuck in intersection
        return "unknown"

    # Compute lane label per vehicle
    lane_by_veh = {}
    for vid, g in df.groupby("veh_id"):
        g = g.sort_values("time_s")
        x0, y0 = g.iloc[0][["cx", "cy"]]
        x1, y1 = g.iloc[-1][["cx", "cy"]]

        origin = classify_dir(x0, y0)
        dest = classify_dir(x1, y1)
        lane_label = classify_lane(origin, dest)
        lane_by_veh[vid] = lane_label

    # Attach new labels
    df["lane_from_geom"] = df["veh_id"].map(lane_by_veh)

    print("Lane label counts (geometry-based):")
    print(df["lane_from_geom"].value_counts())

    # Overwrite old lane_id with the new one (optional but clean)
    df["lane_id"] = df["lane_from_geom"]

    df.to_csv(OUT_CSV, index=False)
    print(f"Saved labeled trajectories to {OUT_CSV}")

if __name__ == "__main__":
    main()
