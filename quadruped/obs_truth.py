#!/usr/bin/env python3
"""Record the observation the policy ACTUALLY received, and compare it with the rebuild.

obs_check.py found that ObsBuilder, fed recorded corpus rows, reproduces the policy's own
recorded outputs to only ~60% of their spread. That says the rebuild is wrong somewhere
but not where. This runs the real collector on rigid ground with Go2Policy.act wrapped to
stash every observation it builds, then rebuilds each one from the CSV row written at the
same instant and reports the disagreement block by block:

    ang(3) grav(3) cmd(3) dof_pos(12) dof_vel(12) last_action(12)

Rigid ground so it runs on CPU in about a minute; the observation code does not depend on
terrain.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

BLOCKS = [("ang", 0, 3), ("grav", 3, 6), ("cmd", 6, 9), ("dof_pos", 9, 21),
          ("dof_vel", 21, 33), ("last_action", 33, 45)]


def main() -> int:
    out_dir, policy, urdf = sys.argv[1], sys.argv[2], sys.argv[3]
    import torch
    from quadruped.lib.policy import Go2Policy
    import collect as COL
    import finetune as F
    import train as T

    seen = []
    orig_act = Go2Policy.act

    def act(self, robot):
        obs = self.observe(robot)
        r = orig_act(self, robot)
        seen.append((obs.copy(), self.last_actions.copy()))
        return r

    Go2Policy.act = act
    sys.argv = ["collect.py", "--corpus", "obs_truth", "--out", out_dir, "--policy", policy,
                "--urdf", urdf, "--episodes", "1", "--terrain", "rigid", "--duration-s", "4",
                "--pushes", "0", "--vx", "0.5", "--skip-doctor", "--val-fraction", "0",
                "--long-fraction", "0"]
    try:
        COL.main()
    except SystemExit as e:
        print(f"(collect exited: {e})")
    print(f"policy acted {len(seen)} times")

    files = sorted((Path(out_dir) / "obs_truth" / "episodes").glob("*.csv"))
    rows = []
    for f in files:
        with f.open() as fh:
            rows += list(csv.DictReader(fh))
    T.add_derived(rows)
    print(f"{len(rows)} rows recorded in {len(files)} segment(s)")

    # Match each recorded row to the observation the policy built at that instant by the
    # policy output it produced: raw is unique to the control step that computed it.
    raw_cols = [c for c in rows[0] if c.startswith("policy_raw_")]
    import json as _json
    man = _json.loads((Path(out_dir) / "obs_truth" / "manifest.json").read_text())
    order = man.get("policy_raw_order", "policy (unstamped, pre-fix)")
    p2c = yaml.safe_load((HERE / "params" / "policy.yaml").read_text())["joints"]["policy_to_chrono"]
    print(f"manifest: row_capture={man.get('row_capture', 'post_step (unstamped, pre-fix)')}  "
          f"policy_raw_order={order}  policy sha256={str(man.get('policy', {}).get('sha256'))[:12]}")
    # Nearest match with a tolerance: the CSV holds float64 text of float32 values, so
    # exact equality never holds.
    seen_raw = np.array([r for _o, r in seen], dtype=np.float64)
    pairs = []
    prev = None
    for row in rows:
        raw = np.array([float(row[c]) for c in raw_cols])
        if order == "chrono":
            raw = raw[p2c]           # back to the network's own order
        if prev is not None and np.array_equal(raw, prev):
            continue
        prev = raw
        d = np.abs(seen_raw - raw).max(axis=1)
        i = int(np.argmin(d))
        if d[i] < 1e-5:
            pairs.append((row, i))
    print(f"{len(pairs)} rows matched to the observation that produced them\n")

    pol_cfg = yaml.safe_load((HERE / "params" / "policy.yaml").read_text())
    legs = ("rr", "rl", "fr", "fl")
    jp = [f"joint_{l}_{s}_pos_rad" for l in legs for s in ("hip", "thigh", "calf")]
    jv = [f"joint_{l}_{s}_vel_radps" for l in legs for s in ("hip", "thigh", "calf")]
    names = ["roll_rate_radps", "ang_vel_body_y_radps", "yaw_rate_radps",
             "grav_body_x", "grav_body_y", "grav_body_z"] + jp + jv
    ob = F.ObsBuilder(torch, names, pol_cfg)
    true_o, rebuilt = [], []
    for row, i in pairs:
        if i == 0:
            continue
        s = torch.tensor([[float(row[c]) for c in names]], dtype=torch.float32)
        c = torch.tensor([[float(row["cmd_vx_mps"]), float(row["cmd_vy_mps"]),
                           float(row["cmd_wz_radps"])]], dtype=torch.float32)
        last = torch.tensor([seen[i - 1][1]], dtype=torch.float32)
        rebuilt.append(ob.observe(s, c, last)[0].numpy())
        true_o.append(seen[i][0])
    true_o = np.array(true_o); rebuilt = np.array(rebuilt)
    print(f"{'block':12s} {'true spread':>12s} {'RMS diff':>10s} {'ratio':>7s}")
    for name, lo, hi in BLOCKS:
        d = rebuilt[:, lo:hi] - true_o[:, lo:hi]
        sp = true_o[:, lo:hi].std()
        rms = np.sqrt((d ** 2).mean())
        print(f"{name:12s} {sp:12.4f} {rms:10.4f} {rms / max(sp, 1e-9):7.2f}")
        if rms > 0.05 * max(sp, 1e-9):
            k = np.argmax(np.sqrt((d ** 2).mean(0)))
            print(f"   worst element {lo + k}: true {true_o[:5, lo + k].round(3)} "
                  f"rebuilt {rebuilt[:5, lo + k].round(3)}")
    # What the mismatch does to the policy's OUTPUT, which is what the fine-tune acts on.
    net = torch.jit.load(policy, map_location="cpu").eval()
    with torch.no_grad():
        p_true = net(torch.tensor(true_o, dtype=torch.float32)).numpy()
        p_reb = net(torch.tensor(rebuilt, dtype=torch.float32)).numpy()
    e = p_reb - p_true
    print(f"\npolicy output from the rebuilt obs vs from the true obs: RMS "
          f"{np.sqrt((e ** 2).mean()):.4f} against an output spread of {p_true.std():.4f} "
          f"({100 * np.sqrt((e ** 2).mean()) / p_true.std():.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
