"""Final calibration runs: full fit, cross-camera holdout, camera-model comparison, export inputs.

  python calib_run.py full    --init X.npy --tag T [--emodel M] [--wmodel M]   -> fits/T_full.npy, T_full_report.json,
                                                                                    T_assoc.npz
  python calib_run.py holdout --init T_full.npy --tag T [--emodel/--wmodel]     -> fits/T_holdout.json
Holdout episodes (fixed before any result was looked at): green_cube_red_plate ep 2, green_cube_red_plate_1 ep 3, ep 7.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calib_common import CALIB, DATASETS, save_json  # noqa: E402
from calib_joint import Data, Joint, holdout_eval, load_x, stage_joint  # noqa: E402

HOLD = [(DATASETS[0], 2), (DATASETS[1], 3), (DATASETS[1], 7)]
NDIST = {"opencv_fisheye": 4, "opencv": 5, "opencv8": 8}


def make(emodel, wmodel, init, signs=(1, 1, 1, 1, 1)):
    data = Data()
    J = Joint(data, signs=signs, model=emodel, wmodel=wmodel)
    J.lines_used = data.lines
    J.ws_keep = np.ones(len(data.ws_uv), bool)
    J.wp_keep = np.ones(len(data.wp_uv), bool)
    x0 = np.load(init)
    try:
        x = load_x(J, init)
    except Exception:
        x = None
    if x is None or len(x0) != J.n:
        # different distortion lengths: copy every block by name from a fisheye/fisheye layout
        Jf = Joint(data)
        xf = load_x(Jf, init)
        x = np.zeros(J.n)
        for k, _ in J.GLOBAL:
            src = Jf.g(xf, k)
            dst = J.sl[k]
            n = dst.stop - dst.start
            if k in ("eD", "wD") and len(src) != n:
                continue  # distortion restarts at 0 for the other model
            x[dst] = src
        x[J.nglob:] = xf[Jf.nglob:]
    return data, J, x


def save_assoc(J, path):
    r, c, o = J.mat_assoc
    np.savez(path, ws_keep=J.ws_keep, wp_keep=J.wp_keep, ma_rows=r, ma_corner=c, ma_obs=o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["full", "holdout"])
    ap.add_argument("--init", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--emodel", default="opencv_fisheye")
    ap.add_argument("--wmodel", default="opencv_fisheye")
    ap.add_argument("--signs", default="1,1,1,1,1")
    a = ap.parse_args()
    data, J, x = make(a.emodel, a.wmodel, a.init, [int(v) for v in a.signs.split(",")])
    out = CALIB / "fits"
    if a.mode == "full":
        if a.emodel != "opencv_fisheye" or a.wmodel != "opencv_fisheye":
            # settle the new distortion model first with the rest fixed
            J.gate_wrist(x, 15.0, 0.4, 15.0)
            J.associate_mat(x, 20.0)
            gl = [k for k in ("eK", "eD") if a.emodel != "opencv_fisheye"] + \
                 [k for k in ("wK", "wD") if a.wmodel != "opencv_fisheye"]
            x = J.fit(x, J.free_mask(gl, []), max_nfev=100)
        x = stage_joint(J, x, rounds=((12.0, 15.0, 80), (8.0, 10.0, 200)))
        np.save(out / f"{a.tag}_full.npy", x)
        save_assoc(J, out / f"{a.tag}_assoc.npz")
        save_json(out / f"{a.tag}_full_report.json", J.report(x))
    else:
        H = [data.eps.index(e) for e in HOLD]
        J.train = np.ones(data.E, bool)
        J.train[H] = False
        x = stage_joint(J, x, rounds=((12.0, 15.0, 60), (8.0, 10.0, 150)))
        np.save(out / f"{a.tag}_holdtrain.npy", x)
        train_rep = J.report(x)
        # gates for the held-out episodes from the training-only solution (generous: gross detection failures only)
        J.gate_wrist(x, 15.0, 0.4, 15.0)
        J.associate_mat(x, 12.0)
        hold = holdout_eval(J, x, H)
        hold["train_report"] = train_rep
        hold["holdout_episodes"] = [list(e) for e in HOLD]
        save_json(out / f"{a.tag}_holdout.json", hold)
        print("HOLDOUT", {k: {kk: round(vv["rms"], 2) for kk, vv in v.items() if isinstance(vv, dict) and "rms" in vv}
                          for k, v in hold.items() if k in ("ext", "wrist", "ext_kin")})


if __name__ == "__main__":
    main()
