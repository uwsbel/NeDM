"""Summary of the E2 rigid f104 check: Gator vs HMMWV on the same 12 designed routes."""
import json, sys
from pathlib import Path
import numpy as np

E = Path(__file__).resolve().parent / "rigid_f104"
rows = []
for d in sorted((E / "gator").iterdir()):
    if not d.is_dir():
        continue
    r = {"id": d.name}
    for v in ("gator", "hmmwv"):
        o = json.load(open(E / v / d.name / "outcome.json"))
        t = np.load(E / v / d.name / "trajectory.npz")
        iv = json.load(open(E / v / d.name / "initial_state_validation.json"))
        nh = json.load(open(E / v / d.name / "native_height_check.json"))
        s = t["state"]
        ri = np.load(E / v / d.name / "rich_intervals.npz")
        chassis = [k for k in ri.files if "chassis_contact" in k and "max" in k]
        r[v] = {"status": o["status"], "elapsed_s": o["elapsed_s"], "goal_time_s": o["goal_time_s"], "wall_s": o["wall_s"],
                "wall_per_sim_s": o["wall_s"] / max(o["elapsed_s"] + 0.8, 1e-9), "launch_ok": iv["passed"], "native_height_ok": nh["passed"],
                "anchor_vx": float(s[0, 0]), "min_vx_after_1s": float(s[20:, 0].min()) if len(s) > 20 else None,
                "max_abs_roll_deg": float(np.degrees(np.abs(s[:, 2]).max())), "max_abs_pitch_deg": float(np.degrees(np.abs(s[:, 3]).max())),
                "finite": bool(np.isfinite(s).all() and np.isfinite(t["terminal_state"]).all()),
                "max_chassis_contact_n": float(ri[chassis[0]].max()) if chassis else None,
                "sustained_near_stop": o["sustained_near_stop"]}
        if v == "gator":
            b = o["vehicle"]["belly"]
            r[v].update({"belly_min_m": b["min_clearance_m"], "belly_frames_below": b["frames_below_surface"],
                         "anchor_vz": b["anchor_chassis_vz_mps"], "anchor_belly_m": b["anchor_clearance_m"]})
    rows.append(r)
json.dump(rows, open(E / "summary.json", "w"), indent=1)
print(f"{'route':30s} {'gator':>36s} | {'hmmwv':>36s}")
for r in rows:
    g, h = r["gator"], r["hmmwv"]
    fmt = lambda x: f"{x['status'][:14]:14s} {x['elapsed_s']:6.1f}s vx0 {x['anchor_vx']:+.2f} w/s {x['wall_per_sim_s']:.2f}"
    print(f"{r['id']:30s} {fmt(g)} belly {g['belly_min_m']:+.3f} chas {g['max_chassis_contact_n'] or 0:7.0f} | {fmt(h)} chas {h['max_chassis_contact_n'] or 0:7.0f}")
for v in ("gator", "hmmwv"):
    x = [r[v] for r in rows]
    print(v, "goal", sum(y["status"] == "goal_reached" for y in x), "/", len(x), "launch ok", sum(y["launch_ok"] for y in x),
          "finite", sum(y["finite"] for y in x), "wall per sim s (median)", round(float(np.median([y["wall_per_sim_s"] for y in x])), 3),
          "total sim", round(sum(y["elapsed_s"] for y in x), 1), "total wall", round(sum(y["wall_s"] for y in x), 1),
          "anchor vx median", round(float(np.median([y["anchor_vx"] for y in x])), 2))
