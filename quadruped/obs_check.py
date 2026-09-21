#!/usr/bin/env python3
"""Does finetune.py's ObsBuilder rebuild the observation the robot actually saw?

The fine-tune rolls the policy inside the NN-ROM on an observation rebuilt from the
propagated state. If that rebuild differs from what lib/policy.py builds in Chrono, the
fine-tune optimises a policy for an input it will never receive, and no amount of model
accuracy rescues it.

The corpus makes this checkable without Chrono: every row records the network's own 12
outputs (`policy_raw_*`, before injected noise), the command it was following, and the
state it acted on (one 1 ms substep later). So the check is

    policy( ObsBuilder(state_k, cmd_k, raw_{k-1}) )  ==  raw_k ?

and it is run with more than one candidate source for the angular-velocity block, because
the preset's channels (roll_rate, ang_vel_body_y, yaw_rate) and the deployed call
(GetAngVelLocal, body frame x/y/z) are not obviously the same quantity.

A faithful rebuild reproduces raw_k to a small fraction of its spread. A wrong one does
not, and says which block is wrong.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

LEGS = ("rr", "rl", "fr", "fl")
JP = [f"joint_{l}_{s}_pos_rad" for l in LEGS for s in ("hip", "thigh", "calf")]
JV = [f"joint_{l}_{s}_vel_radps" for l in LEGS for s in ("hip", "thigh", "calf")]
RAW = [f"policy_raw_{l}_{s}" for l in LEGS for s in ("hip", "thigh", "calf")]
ANG_SOURCES = {
    "preset (roll_rate, ang_vel_body_y, yaw_rate)":
        ["roll_rate_radps", "ang_vel_body_y_radps", "yaw_rate_radps"],
    "body frame (ang_vel_body_x/y/z)":
        ["ang_vel_body_x_radps", "ang_vel_body_y_radps", "ang_vel_body_z_radps"],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--segments", type=int, default=12)
    a = ap.parse_args()

    import torch
    import finetune as F
    import train as T

    pol_cfg = yaml.safe_load((HERE / "params" / "policy.yaml").read_text())
    import json as _json
    man = _json.loads((Path(a.corpus) / "manifest.json").read_text())
    capture = man.get("row_capture", "post_step")
    raw_order = man.get("policy_raw_order", "policy")
    p2c = list(pol_cfg["joints"]["policy_to_chrono"])
    print(f"corpus: row_capture={capture}  policy_raw_order={raw_order}")
    net = torch.jit.load(a.policy, map_location="cpu").eval()
    files = sorted((Path(a.corpus) / "episodes").glob("*.csv"))[: a.segments]

    # Channel identity first: are the preset's rate channels the body-frame rates?
    rows_all = []
    for f in files:
        with f.open() as fh:
            rows = list(csv.DictReader(fh))
        T.add_derived(rows)
        rows_all.append(rows)
    flat = [r for rows in rows_all for r in rows]
    for p, b in (("roll_rate_radps", "ang_vel_body_x_radps"),
                 ("yaw_rate_radps", "ang_vel_body_z_radps"),
                 ("yaw_rate_radps", "ang_vel_world_z_radps")):
        x = np.array([float(r[p]) for r in flat]); y = np.array([float(r[b]) for r in flat])
        print(f"{p:18s} vs {b:22s}  corr {np.corrcoef(x, y)[0, 1]:+.4f}  "
              f"max|diff| {np.abs(x - y).max():.4f}")
    dts = np.diff([float(r["time_s"]) for r in rows_all[0]])
    print(f"row spacing {np.median(dts):.4f} s (the policy acts every 0.02 s)\n")

    for label, ang in ANG_SOURCES.items():
        fields = ang + ["grav_body_x", "grav_body_y", "grav_body_z"] + JP + JV
        # ObsBuilder looks channels up by preset name; alias the candidate source onto
        # those names so the builder under test is the real one, unmodified.
        names = ["roll_rate_radps", "ang_vel_body_y_radps", "yaw_rate_radps",
                 "grav_body_x", "grav_body_y", "grav_body_z"] + JP + JV
        ob = F.ObsBuilder(torch, names, pol_cfg)
        errs, spread, n_ctl = [], [], []
        for rows in rows_all:
            S = torch.tensor([[float(r[c]) for c in fields] for r in rows], dtype=torch.float32)
            C = torch.tensor([[float(r["cmd_vx_mps"]), float(r["cmd_vy_mps"]),
                               float(r["cmd_wz_radps"])] for r in rows], dtype=torch.float32)
            R = torch.tensor([[float(r[c]) for c in RAW] for r in rows], dtype=torch.float32)
            if raw_order == "chrono":
                R = R[:, p2c]            # back to the network's own order
            # Rows are 100 Hz and the policy acts at 50 Hz, so raw is held for two rows.
            # Only rows where raw CHANGED are control instants; on those, the held value
            # in the previous row is exactly the last action the policy saw.
            if capture == "pre_step":
                # Control instants are the rows at multiples of 0.02 s; the row before
                # holds the previous output, which is the last action the policy saw.
                t = np.array([float(r["time_s"]) for r in rows]) / 0.02
                ctl = torch.tensor(np.nonzero(np.abs(t - np.round(t)) < 1e-6)[0])
                ctl = ctl[ctl > 0]
            else:
                ctl = torch.nonzero((R[1:] != R[:-1]).any(-1)).squeeze(-1) + 1
            with torch.no_grad():
                o = ob.observe(S[ctl], C[ctl], R[ctl - 1])
                pred = torch.clamp(net(o), *ob.act_clip)
            errs.append((pred - R[ctl]).numpy())
            spread.append(R[ctl].numpy())
            n_ctl.append(len(ctl) / max(len(rows) - 1, 1))
        e = np.concatenate(errs); s = np.concatenate(spread)
        rms_e = np.sqrt((e ** 2).mean()); rms_s = s.std()
        print(f"{label}  (control instants: {100 * np.mean(n_ctl):.0f}% of rows)")
        print(f"  reproduces raw_k to RMS {rms_e:.4f} against a raw spread of {rms_s:.4f} "
              f"({100 * rms_e / rms_s:.1f}%)")
        print(f"  per joint RMS: " + " ".join(f"{v:.3f}" for v in np.sqrt((e ** 2).mean(0))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
