"""Grasp / release frames per episode from the gripper signal, checked with the cube motion in the external view.

grasp_frame   = first frame of the holding plateau (gripper state settled at the cube width while the action keeps
                commanding a closed jaw; the cube still rests on the table at this frame)
release_frame = last frame of that plateau (the cube rests on the plate; the jaw starts to open at the next frame)
Output: ~/so101_r2s_claude/calib/segments.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calib_common import CALIB, DATASETS, FK, load_dataset, real_to_sim, save_json  # noqa: E402

# approximate map for the lowest-point search (signs from the camera fits; offsets do not move the argmin much)
JM = dict(sign=[1, 1, 1, 1, 1, 1], offset_rad=[0, 0, 0, 0, 0, 0], gripper_lin=[0.0, 0.0])
FKM = FK()


def segment(d, e):
    a, b = int(d["ep_from"][e]), int(d["ep_to"][e])
    s = d["state"][a:b, 5]
    ac = d["action"][a:b, 5]
    hold = (ac < s - 3) & (s > 8) & (s < 22)
    best, st = (0, 0, 0), None
    for i, h in enumerate(np.r_[hold, False]):
        if h and st is None:
            st = i
        if not h and st is not None:
            if i - st > best[0]:
                best = (i - st, st, i)
            st = None
    _, g0, g1 = best
    plateau = float(np.median(s[g0:g1]))
    idx = np.where(s[g0:g1] <= plateau + 0.5)[0] + g0
    grasp_plateau, release = int(idx[0]), int(idx[-1])
    # grasp frame: the lowest gripper position (FK) while the jaw is closed on the cube (gripper within 3 units of
    # the plateau), in the first 2.5 s of the hold: the cube is still on the table there. (The first plateau frame
    # can already be in the lift when the jaw keeps creeping closed, e.g. green_cube_red_plate ep 3.)
    win = np.arange(g0, min(g1, g0 + 75))
    win = win[s[win] <= plateau + 3.0]
    q = real_to_sim(d["state"][a + win], JM)
    z = FKM(q)["gripper"][0][:, 2]
    grasp = int(win[int(np.argmin(z))])
    return dict(a=a, b=b, grasp_frame=grasp, release_frame=release, plateau_grip=plateau,
                grasp_frame_first_plateau=grasp_plateau, hold_mask_first=int(g0), hold_mask_last=int(g1 - 1))


def track_stats(det, a, b, lo, hi):
    G = det["G"][a + lo:a + hi, 0]
    ok = G[:, 0] > 0
    if ok.sum() == 0:
        return None
    c = G[ok, 1:3]
    med = np.median(c, 0)
    return dict(n=int(ok.sum()), med=med.tolist(), mad=np.median(np.abs(c - med), 0).tolist(),
                area=float(np.median(G[ok, 0])))


def main():
    out = []
    for ds in DATASETS:
        d = load_dataset(ds)
        det = np.load(CALIB / "det" / f"{ds}_external.npz")
        for e in range(len(d["ep_from"])):
            sg = segment(d, e)
            a, b = sg["a"], sg["b"]
            n = b - a
            # cube 2D track (external) before the grasp and after the release
            sg["ext_pre"] = track_stats(det, a, b, 0, max(10, sg["grasp_frame"] - 40))
            sg["ext_post"] = track_stats(det, a, b, min(n - 10, sg["release_frame"] + 40), n)
            G = det["G"][a:b, 0]
            c = G[:, 1:3]
            # first frame the external cube centroid leaves its start position by > 3 px (lift-off check)
            if sg["ext_pre"]:
                dev = np.linalg.norm(c - np.array(sg["ext_pre"]["med"]), axis=1)
                mv = np.where((dev > 3) & (G[:, 0] > 0) & (np.arange(n) > sg["grasp_frame"] - 60))[0]
                sg["ext_first_move"] = int(mv[0]) if len(mv) else None
            sg.update(dataset=ds, episode=e, n=n)
            out.append(sg)
            print(ds[-2:], e, "n", n, "grasp", sg["grasp_frame"], "release", sg["release_frame"], "plateau %.1f" %
                  sg["plateau_grip"], "first move", sg.get("ext_first_move"), "pre", sg["ext_pre"] and
                  np.round(sg["ext_pre"]["med"], 1), sg["ext_pre"] and np.round(sg["ext_pre"]["mad"], 2), "post",
                  sg["ext_post"] and np.round(sg["ext_post"]["med"], 1), sg["ext_post"] and np.round(sg["ext_post"]["mad"], 2))
    save_json(CALIB / "segments.json", out)


if __name__ == "__main__":
    main()
