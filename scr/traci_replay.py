import traci
import pandas as pd
import time

SUMO_SECONDS_PER_FRAME = 0.033     # video FPS ~30
SUMO_STEP = 0.1                    # your simulation step

def replay():
    df = pd.read_csv("trajectories.csv")

    df = df.sort_values(["time", "vehicle_id"])

    traci.start([
        "sumo-gui",
        "-c", "traffic.sumocfg",
        "--start",
        "--quit-on-end"
    ])

    current_sumo_time = 0.0
    next_frame_time = 0.0

    while True:
        traci.simulationStep(current_sumo_time)

        # Get all trajectory points for this simulation time
        slice_df = df[(df.time >= next_frame_time - 0.02) &
                      (df.time < next_frame_time + 0.02)]

        for _, row in slice_df.iterrows():
            vid = f"veh{int(row.vehicle_id)}"

            if vid not in traci.vehicle.getIDList():
                traci.vehicle.add(vid, routeID="dummy_route", typeID="car")

            try:
                traci.vehicle.moveToXY(
                    vid,
                    edgeID="",
                    lane=0,
                    x=row.x,
                    y=row.y,
                    angle=180,
                    keepRoute=2
                )
            except Exception:
                pass

        current_sumo_time += SUMO_STEP
        next_frame_time += SUMO_SECONDS_PER_FRAME

        if next_frame_time > df.time.max():
            break

    traci.close()

if __name__ == "__main__":
    replay()
