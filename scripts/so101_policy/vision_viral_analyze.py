"""Evaluation files of build9 (``vision_viral.py eval`` / ``run`` / ``vision_viral_teacher.py eval``) -> success with
Wilson intervals, failure classes, paired comparison (exact McNemar) against a reference file on the same tasks.

    PYTHONPATH=src python -P scripts/so101_policy/vision_viral_analyze.py EVAL.npz [EVAL2.npz ...] [--ref REF.npz]

Failure classes (episodes without lenient success; from the end code, the lifted flag and the last recorded decision
row): ``no lift`` (never firmly lifted), ``dropped`` (lifted, then unsupported / release_high), ``held too long``
(hold_cap), ``held at time-out`` (time-out with the cube held / in the air), ``off the block`` (lifted, the cube on the
table outside the block at the end), ``other`` (any other end).
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

ENDS = ("running", "success", "timeout", "world_invalid", "world_arm_table", "lowest_point", "link_overlap",
        "jaw_overlap", "base_zone", "grasp_box", "keep_out", "release_high", "unsupported", "hold_cap", "cube_slide",
        "cube_tilted", "arm_self", "moved_untouched")      # nedm.so101_policy.task.ENDS (no torch import here)
BLOCK_HALF = 0.025


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar(b, c):
    """Exact two-sided McNemar p for discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def classes(r):
    le = np.asarray(r["lenient"]).astype(bool)
    end = np.asarray(r["end"])
    lifted = np.asarray(r["lifted"]).astype(bool)
    out = {}
    last_cube = None
    if "row_ep" in r:
        ep = np.asarray(r["row_ep"])
        last = np.zeros(len(le), np.int64)
        np.maximum.at(last, ep, np.arange(len(ep)))
        last_cube = np.asarray(r["row_cube_pos"])[last]
    for e in np.flatnonzero(~le):
        name = ENDS[int(end[e])]
        if not lifted[e]:
            k = f"no lift ({name})"
        elif name in ("unsupported", "release_high"):
            k = "dropped"
        elif name == "hold_cap":
            k = "held too long"
        elif name == "timeout":
            k = "time-out"
            if last_cube is not None:
                c = last_cube[e]
                g, y = np.asarray(r["goal_xy"])[e], float(np.asarray(r["goal_yaw"])[e])
                d = c[:2] - g
                bx, by = math.cos(y) * d[0] + math.sin(y) * d[1], -math.sin(y) * d[0] + math.cos(y) * d[1]
                if c[2] > 0.017:
                    k = "held at time-out"
                elif max(abs(bx), abs(by)) > BLOCK_HALF:
                    k = "off the block (time-out)"
        else:
            k = f"other ({name})"
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def summary(r):
    le = np.asarray(r["lenient"]).astype(bool)
    st = np.asarray(r["success"]).astype(bool)
    n = len(le)
    return dict(n=n, lenient=float(le.mean()), lenient_ci=[round(x, 3) for x in wilson(int(le.sum()), n)],
                strict=float(st.mean()), strict_ci=[round(x, 3) for x in wilson(int(st.sum()), n)],
                lifted=float(np.mean(r["lifted"])), failure_classes=classes(r))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ref", default="")
    a = ap.parse_args()
    ref = dict(np.load(a.ref, allow_pickle=True)) if a.ref else None
    for f in a.files:
        r = dict(np.load(f, allow_pickle=True))
        s = summary(r)
        if ref is not None:
            if not np.array_equal(np.asarray(ref["seeds"]), np.asarray(r["seeds"])):
                raise ValueError(f"{f}: other tasks than {a.ref}")
            x, y = np.asarray(r["lenient"]).astype(bool), np.asarray(ref["lenient"]).astype(bool)
            b, c = int((x & ~y).sum()), int((~x & y).sum())
            s["paired_vs_ref_lenient"] = dict(both=int((x & y).sum()), this_only=b, ref_only=c,
                                              neither=int((~x & ~y).sum()), diff=float(x.mean() - y.mean()),
                                              mcnemar_p=mcnemar(b, c))
        print(f, json.dumps(s))


if __name__ == "__main__":
    main()
