import argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from lxml import etree

# ---------- parsing / helpers ----------
def parse_time_to_seconds(val):
    if pd.isna(val): return np.nan
    if isinstance(val, (int, float, np.number)): return float(val)
    s = str(val).strip()
    try: return float(s)
    except ValueError: pass
    parts = s.split(":")
    try: parts = [float(p) for p in parts]
    except ValueError: return np.nan
    if len(parts)==2: m, sec = parts; return 60*m + sec
    if len(parts)==3: h,m,sec = parts; return 3600*h + 60*m + sec
    return np.nan

def lane_norm(label: str) -> str:
    if label is None: return ""
    label = str(label).replace("_"," ").strip()
    return " ".join(label.split())

def edge_of_lane(lane: str) -> str:
    s = lane_norm(lane).upper()
    if not s: return ""
    return s[0] if s[0] in ("E","N","S","W") else ""

def resample_series(t, s, dt):
    if len(t) < 2: return None, None
    t0, t1 = float(np.min(t)), float(np.max(t))
    grid = np.arange(t0, t1+1e-9, dt)
    s_i = np.interp(grid, t, s)
    return grid, s_i

def best_time_shift(t_obs, s_obs, t_sim, s_sim, search_min, search_max, search_step):
    shifts = np.arange(search_min, search_max+1e-9, search_step)
    best = (0.0, float("inf"))
    for dt in shifts:
        s_sim_shifted = np.interp(t_obs, t_sim + dt, s_sim, left=np.nan, right=np.nan)
        mask = ~np.isnan(s_sim_shifted)
        if mask.sum() >= max(5, int(0.5*len(t_obs))):
            mse = float(np.mean((s_sim_shifted[mask] - s_obs[mask])**2))
            if mse < best[1]:
                best = (float(dt), mse)
    return best  # (shift_seconds, mse)

def time_deviation_seconds_at_checkpoints(t_grid, s_obs, s_sim_aligned, ds=0.05):
    """
    Compare arrival times at distance checkpoints using the SAME time grid.
    s_obs and s_sim_aligned are already aligned to t_grid and in [0,1].
    """
    m_obs = np.isfinite(t_grid) & np.isfinite(s_obs)
    m_sim = np.isfinite(t_grid) & np.isfinite(s_sim_aligned)
    m = m_obs & m_sim
    if m.sum() < 3:
        return np.nan

    t = t_grid[m]
    so = s_obs[m]
    ss = s_sim_aligned[m]

    def sort_by_s(s, t):
        order = np.argsort(s)
        s_sorted = s[order]
        t_sorted = t[order]
        uniq, idx = np.unique(s_sorted, return_index=True)
        return s_sorted[idx], t_sorted[idx]

    so_s, to_s = sort_by_s(so, t)
    ss_s, ts_s = sort_by_s(ss, t)

    lo = max(float(so_s.min()), float(ss_s.min()))
    hi = min(float(so_s.max()), float(ss_s.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 1e-9:
        return np.nan

    checkpoints = np.arange(lo, hi + 1e-9, ds)
    if len(checkpoints) < 3:
        return np.nan

    t_obs_at_s = np.interp(checkpoints, so_s, to_s)
    t_sim_at_s = np.interp(checkpoints, ss_s, ts_s)

    return float(np.mean(np.abs(t_sim_at_s - t_obs_at_s)))

# ---------- loaders ----------
def load_observed(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    req = {"vehicle_id","vehicle_type","start_lane","start_time","start_edge","end_lane","end_time","end_edge"}
    miss = req - set(df.columns)
    if miss: raise ValueError(f"Observed CSV missing columns: {miss}")
    df = df.dropna(subset=["vehicle_id","start_lane","start_time","end_time"]).copy()
    for c in ["vehicle_id","vehicle_type","start_lane","end_lane","start_edge","end_edge"]:
        df[c] = df[c].astype(str)
    df["entry_time"] = df["start_time"].apply(parse_time_to_seconds)
    df["exit_time"]  = df["end_time"].apply(parse_time_to_seconds)
    df = df[np.isfinite(df["entry_time"]) & np.isfinite(df["exit_time"])]
    df = df[df["exit_time"] >= df["entry_time"]].copy()
    df["lane_id"] = df["start_lane"].apply(lane_norm)
    df["veh_id"]  = df["vehicle_id"].astype(str)
    df["edge"]    = df["lane_id"].apply(edge_of_lane)
    return df[["veh_id","lane_id","edge","entry_time","exit_time"]]

def load_fcd(fcd_path: str) -> pd.DataFrame:
    rows = []
    ctx = etree.iterparse(fcd_path, events=("end",), tag="timestep")
    for _, step in ctx:
        t = float(step.get("time"))
        for v in step.iterfind("vehicle"):
            vid = str(v.get("id"))
            lane = lane_norm(v.get("lane") or "")
            pos_attr = v.get("pos")
            pos = float(pos_attr) if pos_attr is not None else np.nan
            x = float(v.get("x")) if v.get("x") is not None else np.nan
            y = float(v.get("y")) if v.get("y") is not None else np.nan
            rows.append((t, vid, lane, pos, x, y))
        step.clear()
    del ctx
    fcd = pd.DataFrame(rows, columns=["time_s","veh_id","lane_id","pos","x","y"])
    fcd["edge"] = fcd["lane_id"].apply(edge_of_lane)

    # If pos is missing, approximate with cumulative XY per (veh,lane)
    if fcd["pos"].isna().all():
        def add_s(g):
            g = g.sort_values("time_s").copy()
            dx = g["x"].diff().fillna(0.0); dy = g["y"].diff().fillna(0.0)
            g["pos"] = np.sqrt(dx*dx + dy*dy).cumsum()
            return g
        # include_groups kw only exists on newer pandas; safe default:
        fcd = fcd.groupby(["veh_id","lane_id"], group_keys=False).apply(add_s)

    # Normalize pos to [0,1] per (veh,lane)
    def norm_s(g):
        g = g.sort_values("time_s").copy()
        p0, p1 = float(g["pos"].min()), float(g["pos"].max())
        g["s_norm"] = np.nan if (not np.isfinite(p0) or not np.isfinite(p1) or p1 <= p0) else (g["pos"] - p0) / (p1 - p0)
        return g
    fcd = fcd.groupby(["veh_id","lane_id"], group_keys=False).apply(norm_s)
    return fcd

# ---------- plotting ----------
def plot_edge_overlay(edge, obs_df, fcd_df, outdir: Path):
    o = obs_df[obs_df["edge"] == edge]
    s = fcd_df[fcd_df["edge"] == edge]
    if o.empty and s.empty: return
    fig, ax = plt.subplots(figsize=(12,9), dpi=260)
    # observed: straight 0→1
    for _, r in o.iterrows():
        ax.plot([r["entry_time"], r["exit_time"]], [0.0,1.0], linestyle="-", linewidth=1.8)
    # sim: dashed curves
    for (vid, lane), g in s.groupby(["veh_id","lane_id"]):
        g = g.sort_values("time_s")
        if g["s_norm"].notna().sum() < 2: continue
        ax.plot(g["time_s"].values, g["s_norm"].values, linestyle="--", linewidth=1.0)
    ax.set_xlabel("time (s)"); ax.set_ylabel("normalized distance s (0..1)")
    ax.set_title(f"Space–time overlay (edge {edge})")
    import matplotlib.lines as mlines
    ax.legend([mlines.Line2D([0],[0],lw=2), mlines.Line2D([0],[0],lw=2,ls="--")],
              ["Observed (video intervals)","Simulated (FCD trajectories)"],
              loc="upper left")
    fig.tight_layout()
    fig.savefig(outdir / f"spacetime_edge_{edge}.png"); plt.close(fig)

# ---------- main analysis ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--fcd", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--shift_min", type=float, default=-15.0)
    ap.add_argument("--shift_max", type=float, default=15.0)
    ap.add_argument("--shift_step", type=float, default=0.05)
    ap.add_argument("--similarity_tol_s", type=float, default=15.0)
    ap.add_argument("--rmse_thresh", type=float, default=0.25)
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    obs = load_observed(args.csv)
    fcd = load_fcd(args.fcd)

    # plots per edge
    for edge in ["E","N","S","W"]:
        plot_edge_overlay(edge, obs, fcd, outdir)

    # per-vehicle metrics
    rows = []
    for (lane, vid), r in obs.groupby(["lane_id","veh_id"]):
        t0 = float(r["entry_time"].iloc[0]); t1 = float(r["exit_time"].iloc[0])
        if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0: continue
        t_obs = np.arange(t0, t1+1e-9, args.dt)
        s_obs = (t_obs - t0) / max(1e-9, (t1 - t0))  # linear 0->1

        gsim = fcd[(fcd["lane_id"]==lane) & (fcd["veh_id"]==vid)].sort_values("time_s")
        if gsim["s_norm"].notna().sum() < 2: continue
        ts, ss = resample_series(gsim["time_s"].values, gsim["s_norm"].values, args.dt)
        if ts is None: continue

        # best time shift to align sim to obs
        best_dt, best_mse = best_time_shift(t_obs, s_obs, ts, ss,
                                            args.shift_min, args.shift_max, args.shift_step)
        sim_aligned = np.interp(t_obs, ts + best_dt, ss, left=np.nan, right=np.nan)
        mask = ~np.isnan(sim_aligned)
        if mask.sum() < max(5, int(0.4*len(t_obs))): continue

        rmse = float(np.sqrt(np.mean((sim_aligned[mask] - s_obs[mask])**2)))
        # mean time deviation across distance checkpoints
        tdev = time_deviation_seconds_at_checkpoints(t_obs, s_obs, sim_aligned, ds=0.05)

        rows.append({
            "lane_id": lane,
            "veh_id": vid,
            "best_time_shift_s": round(best_dt,3),
            "rmse_snorm": round(rmse,4),
            "mean_time_deviation_s": (None if tdev is None or np.isnan(tdev) else round(tdev,3)),
            "obs_duration_s": round(t1-t0,3),
            "n_points": int(mask.sum())
        })

    if not rows:
        print("No matched vehicles with sufficient data.")
        return

    perveh = pd.DataFrame(rows).sort_values(["lane_id","veh_id"])
    perveh.to_csv(outdir / "vehicle_metrics.csv", index=False)

    # Similarity % (explicit & fair):
    # After per-vehicle best shift, BOTH start and end residuals must be within tolerance
    # AND the normalized RMSE must be <= threshold.
    merged = pd.merge(
        obs.rename(columns={"entry_time":"entry_obs","exit_time":"exit_obs"})[["veh_id","lane_id","entry_obs","exit_obs"]],
        perveh[["veh_id","lane_id","best_time_shift_s","rmse_snorm"]],
        on=["veh_id","lane_id"], how="inner"
    )
    sim_iv = (fcd.groupby(["veh_id","lane_id"])
                .agg(entry_sim=("time_s","min"), exit_sim=("time_s","max"))
                .reset_index())
    merged = pd.merge(merged, sim_iv, on=["veh_id","lane_id"], how="inner")

    merged["res_entry_err_s"] = (merged["entry_sim"] - merged["entry_obs"]) - merged["best_time_shift_s"]
    merged["res_exit_err_s"]  = (merged["exit_sim"]  - merged["exit_obs"])  - merged["best_time_shift_s"]

    tol = float(args.similarity_tol_s)
    rmse_th = float(args.rmse_thresh)
    merged["match"] = ((merged["res_entry_err_s"].abs() <= tol) &
                       (merged["res_exit_err_s"].abs()  <= tol) &
                       (merged["rmse_snorm"] <= rmse_th))

    sim_pct = 100.0 * float(merged["match"].mean())
    pd.DataFrame([{
        "vehicles_compared": int(len(merged)),
        "similarity_tolerance_s": tol,
        "rmse_threshold_norm": rmse_th,
        "similarity_percent": round(sim_pct,1)
    }]).to_csv(outdir / "summary_global.csv", index=False)

    merged["edge"] = merged["lane_id"].apply(edge_of_lane)
    by_edge = (merged.groupby("edge")
                .agg(vehicles=("veh_id","count"),
                     similarity_percent=("match", lambda x: 100.0*float(np.mean(x))))
                .reset_index())
    by_edge["similarity_percent"] = by_edge["similarity_percent"].round(1)
    by_edge.to_csv(outdir / "summary_by_edge.csv", index=False)

    print("Wrote:", outdir)

if __name__ == "__main__":
    main()
