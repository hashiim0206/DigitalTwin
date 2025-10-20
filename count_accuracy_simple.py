#!/usr/bin/env python3
import os, sys, argparse, pandas as pd, numpy as np
from xml.etree import ElementTree as ET

def try_load_net(net_xml):
    try:
        import os
        if "SUMO_HOME" not in os.environ:
            return None
        sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
        import sumolib  # noqa
        return sumolib.net.readNet(net_xml)
    except Exception:
        return None

def nearest_edge_id(net, x, y, search=10.0):
    try:
        lane, pos, dist = net.getNeighboringLanes(x, y, search)[0]
        return lane.getEdge().getID()
    except Exception:
        return None

def movements_from_video(df_vid, net=None):
    if "movement" in df_vid.columns:
        mv = (df_vid.groupby("vehicle_id")["movement"]
              .agg(lambda s: str(s.dropna().iloc[0]) if not s.dropna().empty else "UNKNOWN"))
        return mv

    if net is None:
        raise SystemExit("Video has no 'movement' column and no --net provided for inference.")

    starts = (df_vid.sort_values(["vehicle_id","time_s"])
                   .groupby("vehicle_id")
                   .first()[["x_m","y_m"]])
    ends   = (df_vid.sort_values(["vehicle_id","time_s"])
                   .groupby("vehicle_id")
                   .last()[["x_m","y_m"]])

    def map_row(row):
        return nearest_edge_id(net, float(row["x_m"]), float(row["y_m"])) or "UNK"

    start_edges = starts.apply(map_row, axis=1)
    end_edges   = ends.apply(map_row, axis=1)
    mv = (start_edges.astype(str) + "->" + end_edges.astype(str))
    return mv

def movements_from_routes(rou_xml):
    rid2edges = {}
    veh2route = {}
    for ev, elem in ET.iterparse(rou_xml, events=("start",)):
        if elem.tag == "route":
            rid = elem.attrib.get("id")
            edges = elem.attrib.get("edges","").split()
            if rid:
                rid2edges[rid] = edges
        elif elem.tag == "vehicle":
            vid = elem.attrib.get("id")
            rid = elem.attrib.get("route")
            if vid and rid:
                veh2route[vid] = rid
    rows=[]
    for vid, rid in veh2route.items():
        edges = rid2edges.get(rid, [])
        if len(edges)==0:
            mv = "UNKNOWN"
        else:
            mv = f"{edges[0]}->{edges[-1]}"
        rows.append(dict(vehicle_id=vid, movement=mv))
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(description="Compute vehicle count accuracy (overall and per movement).")
    ap.add_argument("--video-csv", required=True, help="video_traj.csv with columns: vehicle_id,time_s,x_m,y_m[,movement]")
    ap.add_argument("--routes", required=True, help="routes.rou.xml used in SUMO")
    ap.add_argument("--net", default=None, help="version.net.xml (needed only if video CSV lacks 'movement')")
    ap.add_argument("--out", default="count_accuracy.csv", help="output CSV report")
    args = ap.parse_args()

    dfv = pd.read_csv(args.video_csv)
    need = ["vehicle_id","time_s","x_m","y_m"]
    miss = [c for c in need if c not in dfv.columns]
    if miss:
        sys.exit(f"ERROR: video CSV missing columns: {miss}")

    video_veh_ids = dfv["vehicle_id"].astype(str).unique().tolist()

    net = None
    if "movement" not in dfv.columns:
        if not args.net:
            sys.exit("ERROR: video CSV has no 'movement' column; provide --net to infer.")
        net = try_load_net(args.net)
        if net is None:
            sys.exit("ERROR: could not load SUMO net for movement inference.")
    vid_mv = movements_from_video(dfv, net=net)
    vid_mv.name = "movement"
    vid_mv = vid_mv.reset_index()
    vid_mv["vehicle_id"] = vid_mv["vehicle_id"].astype(str)

    dfs = movements_from_routes(args.routes)
    dfs["vehicle_id"] = dfs["vehicle_id"].astype(str)

    n_video = len(video_veh_ids)
    n_sumo  = len(dfs["vehicle_id"].unique())
    overall_acc = 1.0 - abs(n_sumo - n_video) / max(1, n_video)

    vc = vid_mv.groupby("movement")["vehicle_id"].nunique().rename("video_count")
    sc = dfs.groupby("movement")["vehicle_id"].nunique().rename("sumo_count")
    tbl = pd.concat([vc, sc], axis=1).fillna(0).astype(int).reset_index()
    tbl["diff"] = tbl["sumo_count"] - tbl["video_count"]

    summary = pd.DataFrame([
        dict(movement="__OVERALL__", video_count=int(n_video), sumo_count=int(n_sumo),
             diff=int(n_sumo-n_video), accuracy=overall_acc)
    ])
    out = pd.concat([tbl, summary], ignore_index=True)
    out.to_csv(args.out, index=False)

    print(f"Video vehicles: {n_video} | SUMO vehicles: {n_sumo}")
    print(f"Overall count accuracy: {overall_acc:.3f}")
    print(f"Wrote report -> {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
