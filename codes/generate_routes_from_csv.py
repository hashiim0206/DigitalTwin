import pandas as pd
from xml.etree.ElementTree import Element, SubElement, ElementTree

CSV = "trajectories.csv"
OUT = "routes.xml"

lane_map = {
    "W_through": ("W", "EE"),
    "W_left_turn": ("W", "NN"),
    "W_right_turn": ("W", "SS"),

    "E_through": ("E", "WW"),
    "E_left_turn": ("E", "SS"),
    "E_right_turn": ("E", "NN"),

    "N_through": ("N", "SS"),
    "N_left_turn": ("N", "EE"),
    "N_right_turn": ("N", "WW"),

    "S_through": ("S", "NN"),
    "S_left_turn": ("S", "WW"),
    "S_right_turn": ("S", "EE")
}

def build_routes():
    df = pd.read_csv(CSV)
    routes = Element("routes")

    grouped = df.groupby("veh_id")

    for vid, group in grouped:
        lane_id = group.iloc[0]["lane_id"]

        if lane_id not in lane_map:
            print(f"Skipping: {vid} lane={lane_id}")
            continue

        edge_in, edge_out = lane_map[lane_id]

        route_id = f"r{vid}"
        SubElement(routes, "route", id=route_id, edges=f"{edge_in} {edge_out}")

        veh_el = SubElement(routes, "vehicle", id=str(vid), route=route_id)
        depart_t = float(group.iloc[0]["time_s"])
        veh_el.set("depart", str(depart_t))

    ElementTree(routes).write(OUT)
    print("Wrote:", OUT)

build_routes()
