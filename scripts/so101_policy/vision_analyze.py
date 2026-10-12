"""Phase analysis of closed-loop runs (``vision_eval.py`` / ``eval_policy.py run`` outputs with per-step records):
per run the share of episodes that started a GRASP, aborted one, closed the jaw, lifted, started a CARRY line, started
a PLACE, the GRASP / PLACE start times, and the end causes.

    PYTHONPATH=src python -P scripts/so101_policy/vision_analyze.py RUN1.npz [RUN2.npz ...] [--json out.json]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from nedm.so101_policy.decoder import M

ROOT = Path(__file__).resolve().parents[2]


def gate():
    spec = importlib.util.spec_from_file_location("gate_expert", ROOT / "scripts" / "so101_policy" / "gate_expert.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def phases(r):
    mode = r["mode"].astype(int)                         # [B, T] decoder mode per 20 ms step
    t_end = r["t_end"].astype(int)
    B, T = mode.shape
    live = np.arange(T)[None] < t_end[:, None]
    m = np.where(live, mode, -1)
    prev = np.concatenate([np.full((B, 1), M["free"]), m[:, :-1]], 1)
    g_start = (m >= M["g_brake"]) & (m <= M["g_lift"]) & (prev == M["free"])      # the brake is skipped at rest
    g_abort = (prev >= M["g_brake"]) & (prev <= M["g_settle"]) & (m == M["free"])
    closed = (m == M["g_close"]) & (prev != M["g_close"])
    held = (m == M["hold"]) & (prev == M["g_lift"])
    p_start = (m >= M["p_brake"]) & (prev == M["hold"])
    first = lambda x: np.where(x.any(1), x.argmax(1), -1)            # noqa: E731
    return dict(grasp_starts=g_start.sum(1), grasp_aborts=g_abort.sum(1), closes=closed.sum(1), lifts=held.sum(1),
                place_starts=p_start.sum(1), t_grasp=first(g_start) * 0.02, t_place=first(p_start) * 0.02)


def carry_progress(r, tasks):
    """Per episode: cube-goal xy distance (mm) at the first hold step and its minimum while in mode hold (nan: never held)."""
    idx = {int(s): i for i, s in enumerate(tasks["seeds"])}
    g = tasks["goal_xy"][[idx[int(s)] for s in r["seeds"]]]
    mode, cube = r["mode"].astype(int), r["cube"][..., :2]
    d = np.linalg.norm(cube - g[:, None], axis=-1) * 1e3
    hold = mode == M["hold"]
    first = np.where(hold.any(1), hold.argmax(1), 0)
    d0 = np.where(hold.any(1), d[np.arange(len(d)), first], np.nan)
    dmin = np.where(hold.any(1), np.where(hold, d, np.inf).min(1), np.nan)
    return d0, dmin


def summarize(path, G, tasks=None):
    r = dict(np.load(path, allow_pickle=False))
    p = phases(r)
    n = len(r["end"])
    pct = lambda x: round(float(np.mean(x)) * 100, 1)                 # noqa: E731
    tg, tp = p["t_grasp"], p["t_place"]
    causes = G.cause_names(r)
    u, c = np.unique(causes, return_counts=True)
    return dict(run=f"{Path(path).parent.name}/{Path(path).stem}", n=n, success=pct(r["success"]), lifted_rule=pct(r["lifted"]),
                any_grasp=pct(p["grasp_starts"] > 0), any_abort=pct(p["grasp_aborts"] > 0),
                grasp_starts_mean=round(float(p["grasp_starts"].mean()), 2), closed=pct(p["closes"] > 0),
                held=pct(p["lifts"] > 0), any_place=pct(p["place_starts"] > 0),
                t_grasp_p50=float(np.median(tg[tg >= 0])) if (tg >= 0).any() else None,
                t_place_p50=float(np.median(tp[tp >= 0])) if (tp >= 0).any() else None,
                end_xy_mm_p50_success=float(np.median(r["end_xy"][r["success"]]) * 1e3) if r["success"].any() else None,
                causes={str(k): int(v) for k, v in sorted(zip(u, c), key=lambda kv: -kv[1])},
                **(carry_stats(r, tasks) if tasks is not None else {}))


def carry_stats(r, tasks):
    d0, dmin = carry_progress(r, tasks)
    h = np.isfinite(d0)
    if not h.any():
        return {}
    return dict(hold_start_goal_mm_p50=float(np.median(d0[h])), hold_min_goal_mm_p50=float(np.median(dmin[h])),
                hold_reached_15mm=round(float(np.mean(dmin[h] <= 15)) * 100, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+"), ap.add_argument("--json")
    ap.add_argument("--tasks", help="task file of the runs (carry progress toward the goal)")
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"), help="paired 2 x 2 table of success (A rows, B columns), "
                    "exact McNemar p and the success difference with a paired bootstrap 95 %% interval (tasks resampled)")
    a = ap.parse_args()
    G = gate()
    tasks = dict(np.load(Path(a.tasks).expanduser())) if a.tasks else None
    rows = [summarize(p, G, tasks) for p in a.runs]
    for r in rows:
        print(json.dumps(r))
    if a.pair:
        x, y = (np.load(p)["success"].astype(bool) for p in a.pair)
        sx, sy = (np.load(p)["seeds"] for p in a.pair)
        if not np.array_equal(sx, sy):
            raise ValueError("the two runs have different tasks")
        t = G.table2x2(x, y)
        rng = np.random.default_rng(0)
        bs = [float(np.mean(y[i]) - np.mean(x[i])) for i in (rng.integers(0, len(x), len(x)) for _ in range(10000))]
        pr = dict(pair=[f"{Path(p).parent.name}/{Path(p).stem}" for p in a.pair], n=len(x), success_a=float(x.mean()), success_b=float(y.mean()),
                  diff_b_minus_a=float(y.mean() - x.mean()), diff_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                  **{k: t[k] for k in ("n11", "n10", "n01", "n00", "mcnemar_p")})
        print(json.dumps(pr))
        rows.append(pr)
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
