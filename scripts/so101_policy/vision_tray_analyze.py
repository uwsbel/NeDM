"""Analysis of tray-student evaluations (build8_tray_student.md): success with Wilson intervals, failure causes, a
failure taxonomy from the per-step states (``run(record_states=True)``), paired comparisons (McNemar exact) against
the teacher or another student on the same tasks, NRD vs CPU MuJoCo agreement, command smoothness.

    PYTHONPATH=src python -P scripts/so101_policy/vision_tray_analyze.py EVAL.npz [--ref TEACHER_OR_OTHER.npz ...]
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np


def mcnemar(a, b):
    """Exact two-sided McNemar test on paired booleans: -> (n11, n10, n01, n00, p)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n11, n10, n01, n00 = int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())
    n = n10 + n01
    k = min(n10, n01)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return n11, n10, n01, n00, p


def taxonomy(r):
    """Per-episode class from the recorded per-step states (cube pose, q): 'success', 'no_grasp' (no firm lift),
    'holding' (cube still in the air at the end), 'not_released' (cube back on the table, jaw still closed < 0.3 rad),
    'dropped' (lifted, release_high / unsupported end), 'off_target' (cube released and resting, success rule not met:
    off the tray or the gripper not clear), 'other'."""
    from nedm.so101_policy.task import ENDS
    s = np.asarray(r["success"], bool)
    out = []
    cube = r.get("state_cube")
    for b in range(len(s)):
        if s[b]:
            out.append("success")
            continue
        end = ENDS[int(r["end"][b])]
        if not r["lifted"][b]:
            out.append("no_grasp" if end == "timeout" else f"no_grasp/{end}")
            continue
        if cube is not None:
            n = int(r["steps"][b])
            z = cube[b, n - 1, 2]
            if z > 0.02:
                out.append("holding" if end in ("timeout", "hold_cap") else f"holding/{end}")
                continue
            if "state_q" in r and r["state_q"][b, n - 1, 5] < 0.3:
                out.append("not_released" if end == "timeout" else f"not_released/{end}")
                continue
        if end in ("release_high", "unsupported"):
            out.append(f"dropped/{end}")
        elif end == "timeout":
            out.append("off_target")
        else:
            out.append(f"other/{end}")
    return out


def describe(path):
    from nedm.so101_policy.vision.tray import summarize
    r = dict(np.load(path, allow_pickle=False))
    s = summarize(r)
    tx = taxonomy(r)
    vals, cnt = np.unique(tx, return_counts=True)
    s["taxonomy"] = {str(v): int(c) for v, c in zip(vals, cnt)}
    ok = np.asarray(r["success"], bool)
    ex, ey = np.asarray(r["end_xy"]) * 1e3, np.degrees(np.asarray(r["end_yaw"]))
    if (~ok).any():
        s["fail_end_xy_mm_p50"] = float(np.nanmedian(ex[~ok]))
        s["fail_end_yaw_deg_p50"] = float(np.nanmedian(ey[~ok]))
    if "u_exe" in r and "ep_step_start" in r:
        du = []
        for b in range(len(ok)):
            s0, n = int(r["ep_step_start"][b]), int(r["steps"][b])
            u = r["u_exe"][s0:s0 + n]
            if n > 2:
                du.append(np.abs(np.diff(u[:, :5], 2, axis=0)).max(0))
        if du:
            s["cmd_accel_max_p50_rad"] = np.median(np.stack(du), 0).round(4).tolist()
        lab = np.abs(r["u_lab"] - r["u_exe"])
        s["exec_vs_shadow_label_mae_rad"] = lab.mean(0).round(4).tolist()
    for k in ("wall_s", "render_s", "model_s", "world_steps"):
        if k in r:
            s[k] = float(r[k])
    return r, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ref", nargs="*", default=[])
    a = ap.parse_args()
    refs = {p: dict(np.load(p, allow_pickle=False)) for p in a.ref}
    for p in a.files:
        r, s = describe(p)
        print(f"== {p}\n{json.dumps(s, indent=1)}")
        for q, rr in refs.items():
            if not np.array_equal(np.asarray(rr["seeds"]), np.asarray(r["seeds"])):
                print(f"  vs {q}: different tasks")
                continue
            n11, n10, n01, n00, pv = mcnemar(rr["success"], r["success"])
            agree = (n11 + n00) / max(1, len(r["success"]))
            print(f"  vs {q}: both {n11} / ref only {n10} / this only {n01} / neither {n00}; "
                  f"diff {(n01 - n10) / len(r['success']) * 100:+.1f} pts, McNemar p {pv:.3g}, agreement {agree * 100:.0f} %")


if __name__ == "__main__":
    main()
