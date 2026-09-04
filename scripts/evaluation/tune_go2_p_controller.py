"""Tune a proportional controller on references the EVAL EIGHT DO NOT CONTAIN.

THE BASELINE THAT ANSWERS THE READER'S REAL QUESTION. Not "is the surrogate
accurate" but "would a simple controller have done this without any of the
framework". A baseline that cannot embarrass us is not a baseline.

TUNING SET: the 12 flat references in go2_flat_crm_ref40.npz that are NOT among
the evaluated eight. Tuning on the eight would hand the baseline the test set;
not tuning at all would strawman it.

AND THE ASYMMETRY IS IN THE BASELINE'S DISFAVOUR, WHICH MUST BE STATED WHEREVER
THE RESULT IS. The learned policy trained on all forty references INCLUDING the
eight it is evaluated on. This controller is tuned on twelve it is not evaluated
on. That is a harder deal, and if it wins anyway the result is stronger than it
looks; if it loses, some of the gap is the deal rather than the method.

Flat only: the evaluation arm is flat, and tuning on CRM would spend hours of SPH
to fit gains for a terrain the comparison does not use.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

EVAL_EIGHT = [16, 17, 10, 19, 12, 13, 6, 15]
FLAT = list(range(20))
TUNE_ON = [i for i in FLAT if i not in EVAL_EIGHT]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=str, default="0.5,1.0,2.0,4.0")
    ap.add_argument("--horizon-s", type=float, default=6.0)
    ap.add_argument("--out", type=Path, default=Path("artifacts/rl_eval/p_controller_tuning.json"))
    a = ap.parse_args()

    from nedm.rl.go2_chrono_tracking_env import Go2ChronoTrackingEnv, go2_default_chrono_env_cfg
    sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ev", REPO / "scripts/evaluation/eval_go2_rl_chrono_tracking.py")
    ev = importlib.util.module_from_spec(spec)
    sys.modules["ev"] = ev
    spec.loader.exec_module(ev)

    cfg = go2_default_chrono_env_cfg()
    cfg.update({
        "num_envs": 1, "device": "cpu", "auto_reset": False,
        "chrono_config": "configs/go2_chrono_eval_flat.json",
        "imported_policy_ckpt": str(ev.DEFAULT_IMPORTED_POLICY),
        "dynamics_checkpoint":
            "artifacts/training_runs/go2_transformer_v01_contact_mix25_onehot/checkpoints/best_val.pt",
        "reference_path": "artifacts/rl_references/go2_flat_crm_ref40.npz",
        "pre_roll_time_s": 0.0,
        "max_episode_steps": int(round(a.horizon_s / 0.05)),
        "initial_reference_ids": [0],
    })
    env = Go2ChronoTrackingEnv(cfg)
    print(f"tuning on {len(TUNE_ON)} held-out flat references: {TUNE_ON}")
    print(f"evaluating later on the eight: {EVAL_EIGHT}   (disjoint: "
          f"{not set(TUNE_ON) & set(EVAL_EIGHT)})\n")

    values = [float(v) for v in a.grid.split(",")]
    results = []
    t0 = time.time()
    for kx, ky, kyaw in itertools.product(values, values, values):
        ctrl = ev.ProportionalController([kx, ky, kyaw], env.action_low, env.action_high)
        errs = []
        for rid in TUNE_ON:
            r = ev.roll_one(env, rid, policy=None, p_controller=ctrl)
            errs.append(r["mean_position_error_m"])
        score = float(np.mean(errs))
        results.append({"gains": [kx, ky, kyaw], "mean_position_error_m": score,
                        "per_reference": errs})
        print(f"  k=({kx:>4}, {ky:>4}, {kyaw:>4})  mean err {score:.5f} m"
              f"   [{time.time()-t0:.0f}s]")

    results.sort(key=lambda r: r["mean_position_error_m"])
    best = results[0]
    print(f"\nBEST on the tuning set: k = {best['gains']}, mean {best['mean_position_error_m']:.5f} m")
    print(f"worst: k = {results[-1]['gains']}, mean {results[-1]['mean_position_error_m']:.5f} m")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"tuned_on": TUNE_ON, "eval_on": EVAL_EIGHT, "grid": values,
         "results": results, "best": best}, indent=2) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
