#!/usr/bin/env python3
# csv_to_routes.py (robust)
import os, sys, argparse, re
import pandas as pd

# --- SUMO tools bootstrap ---
if "SUMO_HOME" not in os.environ:
    sys.exit("ERROR: SUMO_HOME not set. Install SUMO and set SUMO_HOME environment variable.")
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import sumolib  # noqa

REQUIRED_COLS = ["vehicle_id","start_time","start_edge","start_lane","end_edge","end_lane"]
OPTIONAL_COLS = ["vtype","depart_pos","depart_speed","route_edges","speed_factor"]

BASE_VTYPE_DEFAULTS = dict(
    accel="2.6", decel="4.5", sigma="0.0", speedDev="0.0",
    length="4.5", maxSpeed="13.9", guiShape="passenger"
)

def fail(msg): sys.exit(f"ERROR: {msg}")

def parse_time_to_seconds(x):
    if pd.isna(x): return None
    s = str(x).strip()
    if s == "": return None
    # plain number
    try: return float(s)
    except Exception: pass
    # HH:MM:SS or MM:SS
    if ":" in s:
        parts = [p for p in s.split(":") if p != ""]
        try: parts = list(map(float, parts))
        except Exception: return None
        if   len(parts) == 3: h,m,sec = parts
        elif len(parts) == 2: h,m,sec = 0.0, parts[0], parts[1]
        else: return None
        return h*3600 + m*60 + sec
    # with suffix
    ls = s.lower()
    for suf in ("seconds","secs","sec","s"):
        if ls.endswith(suf):
            core = s[:-len(suf)].strip()
            try: return float(core)
            except Exception: return None
    return None

def parse_start_time_series(series: pd.Series) -> pd.Series:
    out = series.apply(parse_time_to_seconds)
    if out.isna().any():
        bad_idx = out[out.isna()].index.tolist()[:10]
        raise SystemExit(
            f"start_time has non-parsable values at rows { [i+2 for i in bad_idx] } "
            f"(1-based including header). Examples: {series.loc[bad_idx].tolist()}"
        )
    return out

def lane_belongs_to_edge(net, lane_id, edge_id) -> bool:
    try:
        return net.getLane(lane_id).getEdge().getID() == edge_id
    except Exception:
        return False

def shortest_path_edges(net, start_edge, end_edge):
    p = net.getShortestPath(net.getEdge(start_edge), net.getEdge(end_edge))
    if not p or not p[0]: return None
    return [e.getID() for e in p[0]]

def normalize_lane_token(tok: str) -> str:
    # map "E 5" -> "E 5"; "E_5" -> "E 5"; trim/multiple spaces
    t = tok.replace("_", " ").strip()
    t = re.sub(r"\s+", " ", t)
    return t

def resolve_lane_index_from_human(edge, token_normalized: str, n_lanes: int, leftmost_is_1: bool) -> int | None:
    """
    token_normalized like "E 5" or just "5". Convert to SUMO lane index.
    SUMO: 0 = rightmost lane. If your labeling has rightmost=1, then index = number-1.
    If leftmost=1, then index = (n_lanes - number).
    """
    m = re.match(r"^([A-Z]+)\s+(\d+)$", token_normalized)
    if m:
        # has edge letter and number, e.g. "E 5" -> use the number
        number = int(m.group(2))
    else:
        # bare number?
        m2 = re.match(r"^(\d+)$", token_normalized)
        if not m2: return None
        number = int(m2.group(1))

    if number < 1 or number > n_lanes:
        return None

    if leftmost_is_1:
        # leftmost=1 ... rightmost=n => SUMO index = n - number
        return n_lanes - number
    else:
        # rightmost=1 ... leftmost=n => SUMO index = number - 1
        return number - 1

def lane_index_on_edge(net, edge_id, lane_token, leftmost_is_1: bool):
    """
    Accepts: full lane id (e.g., 'E_3'), human 'E 5', bare '5'.
    Returns SUMO lane index (int) on that edge, or None if cannot resolve.
    """
    edge = net.getEdge(edge_id)
    lanes = edge.getLanes()
    lane_ids = [ln.getID() for ln in lanes]
    t = str(lane_token).strip()

    # exact SUMO lane id
    if t in lane_ids:
        # map to index
        for idx, ln in enumerate(lanes):
            if ln.getID() == t: return idx
        return None

    # human or bare
    tok = normalize_lane_token(t)
    idx = resolve_lane_index_from_human(edge_id, tok, len(lanes), leftmost_is_1)
    return idx

def validate_and_build(df, net, relax_lanes: bool, leftmost_is_1: bool):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing: fail(f"CSV missing required columns: {missing}")

    # defaults
    if "vtype"        not in df.columns: df["vtype"]        = "car"
    if "depart_pos"   not in df.columns: df["depart_pos"]   = "base"
    if "depart_speed" not in df.columns: df["depart_speed"] = "max"  # spawn-friendly
    if "speed_factor" not in df.columns: df["speed_factor"] = 1.0

    # parse casts
    df["start_time"] = parse_start_time_series(df["start_time"])
    try:
        df["speed_factor"] = pd.to_numeric(df["speed_factor"])
    except Exception:
        fail("speed_factor must be numeric (omit the column to default to 1.0).")
    if (df["speed_factor"] <= 0).any():
        bad = df[df["speed_factor"] <= 0]["vehicle_id"].tolist()[:5]
        fail(f"speed_factor must be > 0. Offenders: {bad}")

    # unique IDs
    if df["vehicle_id"].duplicated().any():
        dups = df[df["vehicle_id"].duplicated()]["vehicle_id"].tolist()
        fail(f"Duplicate vehicle_id(s): {dups[:5]}{'...' if len(dups)>5 else ''}")

    # lookups
    edges = {e.getID(): e for e in net.getEdges()}
    routes = {}
    lane_indexes = {}
    vtype_for_vehicle = {}

    hard_errors = []
    for _, row in df.iterrows():
        veid = str(row["vehicle_id"])
        se, sl = str(row["start_edge"]), str(row["start_lane"])
        ee, el = str(row["end_edge"]),   str(row["end_lane"])
        vtyp_base = str(row["vtype"])

        # edges must exist
        if se not in edges: hard_errors.append(f"{veid}: start_edge '{se}' not in net"); continue
        if ee not in edges: hard_errors.append(f"{veid}: end_edge '{ee}' not in net");   continue

        # route edges
        if "route_edges" in df.columns and isinstance(row.get("route_edges",""), str) and row["route_edges"].strip():
            edge_list = row["route_edges"].split()
        else:
            edge_list = shortest_path_edges(net, se, ee)
            if not edge_list:
                hard_errors.append(f"{veid}: no path from {se} → {ee}"); continue

        if edge_list[-1] != ee:
            hard_errors.append(f"{veid}: route must end on end_edge '{ee}', got '{edge_list[-1]}'"); continue

        # end lane presence (only check if not relaxing)
        if not relax_lanes:
            final_lane_ids = [ln.getID() for ln in edges[ee].getLanes()]
            if el not in final_lane_ids:
                # allow human label mapping here too
                idx_end = lane_index_on_edge(net, ee, el, leftmost_is_1)
                if idx_end is None:
                    hard_errors.append(f"{veid}: end_lane '{el}' not recognized on end_edge '{ee}'"); continue

        # depart lane index
        if relax_lanes:
            li = None  # SUMO will pick
        else:
            li = lane_index_on_edge(net, se, sl, leftmost_is_1)
            if li is None:
                hard_errors.append(f"{veid}: cannot derive lane index for start_lane '{sl}' on '{se}'"); continue

        routes[veid] = edge_list
        lane_indexes[veid] = li
        vtype_for_vehicle[veid] = f"{vtyp_base}_{veid}"

    if hard_errors:
        # show a concise sample + write full log
        log = "\n".join(hard_errors)
        with open("csv_to_routes_errors.txt","w",encoding="utf-8") as f:
            f.write(log)
        fail(f"{len(hard_errors)} vehicle(s) invalid. See csv_to_routes_errors.txt (first: {hard_errors[0]})")

    return routes, lane_indexes, vtype_for_vehicle

def write_routes_xml(out_path, df, routes, lane_indexes, vtype_for_vehicle, relax_lanes: bool):
    def esc(s):
        return (str(s).replace("&","&amp;").replace("<","&lt;")
                     .replace(">","&gt;").replace('"',"&quot;"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')

        # vTypes
        for _, row in df.iterrows():
            veid = str(row["vehicle_id"])
            vt_id = vtype_for_vehicle[veid]
            sf = float(row["speed_factor"])
            attrs = dict(id=vt_id, speedFactor=str(sf), **BASE_VTYPE_DEFAULTS)
            attrs_str = " ".join([f'{k}="{esc(v)}"' for k, v in attrs.items()])
            f.write(f'  <vType {attrs_str} />\n')

        # routes
        for veid, edges in routes.items():
            rid = f"r_{esc(veid)}"
            f.write(f'  <route id="{rid}" edges="{" ".join(esc(e) for e in edges)}"/>\n')

        # vehicles
        for _, row in df.iterrows():
            veid = esc(row["vehicle_id"])
            rid  = f"r_{veid}"
            vtyp = esc(vtype_for_vehicle[str(row["vehicle_id"])])
            dep  = float(row["start_time"])
            dpos = esc(row["depart_pos"])
            dspe = esc(row["depart_speed"])
            li   = lane_indexes[str(row["vehicle_id"])]

            if relax_lanes or li is None:
                lane_attr = 'departLane="best"'
            else:
                lane_attr = f'departLane="{li}"'

            f.write(
                f'  <vehicle id="{veid}" type="{vtyp}" route="{rid}" '
                f'depart="{dep:.3f}" {lane_attr} departPos="{dpos}" departSpeed="{dspe}"/>\n'
            )

        f.write("</routes>\n")

def main():
    ap = argparse.ArgumentParser(description="Build SUMO routes.rou.xml from CSV with robust lane/edge handling.")
    ap.add_argument("--net", required=True, help="Path to version.net.xml")
    ap.add_argument("--csv", required=True, help="Input CSV (vehicle_id,start_time,start_edge,start_lane,end_edge,end_lane,...)")
    ap.add_argument("--out", required=True, help="Output routes file (e.g., routes.rou.xml)")
    ap.add_argument("--expect-count", type=int, default=None, help="Expected number of vehicles (optional)")
    ap.add_argument("--relax-lanes", action="store_true", help="Do not force lane IDs; use departLane='best'")
    ap.add_argument("--leftmost-is-1", action="store_true", help="Interpret human lane numbers as leftmost=1 instead of rightmost=1")
    args = ap.parse_args()

    net = sumolib.net.readNet(args.net)
    df  = pd.read_csv(args.csv).copy()

    # Parse times *before* sorting; then sort by parsed seconds
    df["start_time"] = parse_start_time_series(df["start_time"])
    df = df.sort_values("start_time").reset_index(drop=True)

    if args.expect_count is not None and len(df) != args.expect_count:
        fail(f"CSV has {len(df)} rows; expected {args.expect_count}. Use --expect-count to override or fix the CSV.")

    routes, lane_indexes, vtype_for_vehicle = validate_and_build(
        df, net, relax_lanes=args.relax_lanes, leftmost_is_1=args.leftmost_is_1
    )
    write_routes_xml(args.out, df, routes, lane_indexes, vtype_for_vehicle, args.relax_lanes)
    print(f"OK: wrote {len(df)} vehicles to {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
