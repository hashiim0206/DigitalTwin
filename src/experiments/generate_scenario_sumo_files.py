"""
generate_scenario_sumo_files.py
================================
Generates SUMO route files (.rou.xml) and config files (.sumocfg) for
three Week-5 scenarios so each can be opened in SUMO-GUI independently:

  1. Baseline     — original 279 vehicles, no changes
  2. Demand Surge — +50% extra vehicles on the NORTH approach (+38 vehicles)
  3. Lane Closure — West->South right-turn movement removed (24 vehicles blocked)

Output files (under outputs/scenarios/):
  scenario_baseline.rou.xml    + scenario_baseline.sumocfg
  scenario_surge.rou.xml       + scenario_surge.sumocfg
  scenario_closure.rou.xml     + scenario_closure.sumocfg

To open in SUMO-GUI:
  sumo-gui -c outputs/scenarios/scenario_baseline.sumocfg
  sumo-gui -c outputs/scenarios/scenario_surge.sumocfg
  sumo-gui -c outputs/scenarios/scenario_closure.sumocfg
"""

import json, os, copy, random, textwrap
import numpy as np

random.seed(42)
np.random.seed(42)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILE   = os.path.join(BASE_DIR, "outputs", "traffic_data.json")
NET_FILE    = os.path.join(BASE_DIR, "configs", "version.net.xml")
OUT_DIR     = os.path.join(BASE_DIR, "outputs", "scenarios")
os.makedirs(OUT_DIR, exist_ok=True)

SIM_DURATION = 70  # seconds (extra 10s buffer so vehicles clear the intersection)

# Edge mapping (from traci_runner.py)
EDGES = {
    "north": {"in": "N",  "out": "NN"},
    "south": {"in": "S",  "out": "SS"},
    "east":  {"in": "E",  "out": "EE"},
    "west":  {"in": "W",  "out": "WW"},
}

# Visual colours per scenario
VEHICLE_COLORS = {
    "baseline": "0.4,0.7,1.0",    # blue
    "surge":    "1.0,0.4,0.2",    # orange/red  (surge vehicles: bright red)
    "closure":  "0.5,0.2,0.8",    # purple
}
SURGE_NEW_COLOR  = "1.0,0.0,0.0"   # bright red for the ADDED surge vehicles
CLOSURE_REMOVED_COLOR = "0.8,0.8,0.8"  # grey (not used — those vehicles are simply absent)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def load_vehicles():
    with open(DATA_FILE) as f:
        raw = json.load(f)
    return [
        {
            "id":          v["id"],
            "origin":      v["origin"].lower(),
            "dest":        v["dest"].lower(),
            "depart":      float(v["depart"]),
            "origin_lane": int(v.get("origin_lane", 1)),
            "is_static":   v.get("is_static", False),
        }
        for v in raw
        if float(v["depart"]) < SIM_DURATION
    ]

def scenario_baseline(vehicles):
    return copy.deepcopy(vehicles)

def scenario_demand_surge(vehicles, approach="north", factor=1.5):
    modified  = copy.deepcopy(vehicles)
    target    = [v for v in modified if v["origin"] == approach]
    n_extra   = int(len(target) * (factor - 1.0))
    max_id    = max(v["id"] for v in modified) + 1
    for i in range(n_extra):
        tmpl = copy.deepcopy(random.choice(target))
        tmpl["id"]     = max_id + i
        tmpl["depart"] = float(np.clip(
            tmpl["depart"] + np.random.uniform(-3, 3), 0, SIM_DURATION - 2
        ))
        tmpl["_added"] = True          # flag: this is a new vehicle
        modified.append(tmpl)
    modified.sort(key=lambda v: v["depart"])
    return modified

def scenario_lane_closure(vehicles, blocked_origin="west", blocked_dest="south"):
    return [
        v for v in copy.deepcopy(vehicles)
        if not (v["origin"] == blocked_origin and v["dest"] == blocked_dest)
    ]

# ─────────────────────────────────────────────
# Route XML generator
# ─────────────────────────────────────────────
def vehicle_color(v, scenario):
    if v.get("is_static"):
        return "0.9,0.1,0.1"          # red for queued/static vehicles
    if v.get("_added"):
        return SURGE_NEW_COLOR         # bright red = added surge vehicle
    return VEHICLE_COLORS.get(scenario, "1.0,1.0,1.0")

def write_route_xml(vehicles, scenario_name, out_path):
    """Write a .rou.xml file for the given vehicle list."""
    lines = ["<?xml version='1.0' encoding='utf-8'?>",
             "<routes>",
             "    <!-- Vehicle types -->",
             "    <vType id=\"car\" accel=\"3.0\" decel=\"4.5\" sigma=\"0.5\" length=\"5\" minGap=\"2.5\" maxSpeed=\"70\"/>",
             "    <vType id=\"static_car\" accel=\"1.0\" decel=\"2.0\" length=\"5\" color=\"0.9,0.1,0.1\"/>",
             ""]

    # Traffic light program — replicate traci_runner.py phases as a static phase
    # (SUMO will use the net's own TLS; we just set a phase in the sumocfg)
    lines.append("    <!-- Vehicles -->")
    for v in vehicles:
        origin = v["origin"]
        dest   = v["dest"]
        if origin not in EDGES or dest not in EDGES:
            continue
        in_edge  = EDGES[origin]["in"]
        out_edge = EDGES[dest]["out"]
        lane     = max(1, int(v.get("origin_lane", 1)))
        depart   = v["depart"]
        vtype    = "static_car" if v.get("is_static") else "car"
        color    = vehicle_color(v, scenario_name)
        vid      = f"v_{v['id']}"

        route_line = (
            f'    <vehicle id="{vid}" type="{vtype}" depart="{depart:.2f}" '
            f'departLane="{lane}" color="{color}">'
        )
        lines.append(route_line)
        lines.append(f'        <route edges="{in_edge} {out_edge}"/>')
        if v.get("is_static"):
            # Park near the stop line
            lines.append(f'        <stop lane="{in_edge}_{lane}" endPos="40" duration="300"/>')
        lines.append("    </vehicle>")

    lines.append("</routes>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ─────────────────────────────────────────────
# SUMO config (.sumocfg) generator
# ─────────────────────────────────────────────
TLS_PROGRAM = textwrap.dedent("""\
    <!-- Signal timing replicates traci_runner.py:
         Phase 1 (0-25s):  North Green
         Phase 2 (25-30s): Yellow (all red)
         Phase 3 (30-45s): E/W Protected Left
         Phase 4 (45-60s): E/W Full Green   -->
""")

def write_sumocfg(scenario_name, rou_file, out_path):
    net_rel = os.path.relpath(NET_FILE, OUT_DIR).replace("\\", "/")
    rou_rel = os.path.basename(rou_file)
    tripinfo = f"tripinfo_{scenario_name}.xml"

    cfg = textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <configuration>
            <input>
                <net-file value="{net_rel}"/>
                <route-files value="{rou_rel}"/>
            </input>
            <time>
                <begin value="0"/>
                <end value="{SIM_DURATION}"/>
                <step-length value="0.5"/>
            </time>
            <processing>
                <collision.action value="warn"/>
                <time-to-teleport value="-1"/>
            </processing>
            <report>
                <no-warnings value="true"/>
                <tripinfo-output value="{tripinfo}"/>
            </report>
            <gui_only>
                <delay value="150"/>
                <start value="true"/>
            </gui_only>
        </configuration>
    """)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(cfg)

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("\n=== Generating SUMO-GUI Scenario Files ===\n")
    all_vehicles = load_vehicles()
    print(f"  Loaded {len(all_vehicles)} vehicles from traffic_data.json")

    scenarios = {
        "baseline": scenario_baseline(all_vehicles),
        "surge":    scenario_demand_surge(all_vehicles, approach="north", factor=1.5),
        "closure":  scenario_lane_closure(all_vehicles, blocked_origin="west", blocked_dest="south"),
    }

    descriptions = {
        "baseline": "Original 279 vehicles, unchanged",
        "surge":    f"+{len(scenarios['surge'])-len(scenarios['baseline'])} extra NORTH vehicles (demand surge +50%)",
        "closure":  f"-{len(scenarios['baseline'])-len(scenarios['closure'])} West->South vehicles removed (lane closure)",
    }

    for name, vehicles in scenarios.items():
        rou_path = os.path.join(OUT_DIR, f"scenario_{name}.rou.xml")
        cfg_path = os.path.join(OUT_DIR, f"scenario_{name}.sumocfg")

        write_route_xml(vehicles, name, rou_path)
        write_sumocfg(name, rou_path, cfg_path)

        print(f"\n  [{name.upper()}] {descriptions[name]}")
        print(f"    Vehicles : {len(vehicles)}")
        print(f"    Route    : {rou_path}")
        print(f"    Config   : {cfg_path}")

    print("\n" + "=" * 55)
    print("  HOW TO OPEN IN SUMO-GUI:")
    print("=" * 55)
    for name in scenarios:
        cfg = os.path.join(OUT_DIR, f"scenario_{name}.sumocfg")
        print(f"\n  {name.upper()}:")
        print(f"    sumo-gui -c \"{cfg}\"")

    print("\n  OR double-click the .sumocfg file in File Explorer.")
    print("\n  COLOUR CODING (for professor demo):")
    print("  - Blue vehicles   = normal baseline traffic")
    print("  - Orange/Red      = surge vehicles (existing North ones)")
    print("  - Bright Red      = ADDED extra vehicles (demand surge only)")
    print("  - Purple          = normal vehicles in lane closure scenario")
    print("  - Dark Red        = queued/static vehicles")
    print()


if __name__ == "__main__":
    main()
