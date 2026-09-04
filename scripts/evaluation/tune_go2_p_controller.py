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

EVAL_EIGHT = {"flat": [16, 17, 10, 19, 12, 13, 6, 15],
              "crm": [28, 21, 38, 31, 24, 25, 26, 27]}
DOMAIN_IDS = {"flat": list(range(20)), "crm": list(range(20, 40))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=str, default="0.5,1.0,2.0,4.0")
    ap.add_argument("--domain", choices=["flat", "crm"], default="flat")
    ap.add_argument("--chrono-config", type=str, default=None)
    ap.add_argument("--max-tune-refs", type=int, default=None,
                    help="Cap the tuning set. CRM is ~1.6 min per rollout, so a full "
                         "12-reference sweep over a 3-axis grid is not affordable; the cap "
                         "is stated in the output rather than hidden.")
    ap.add_argument("--isotropic", action="store_true",
                    help="Search a single shared gain instead of the 3-axis product. The "
                         "rigid optimum was isotropic and its top three all shared kx=ky, "
                         "so this is a restriction the rigid sweep justifies -- and it is "
                         "the only way a soil sweep fits in the time available.")
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
    eval_eight = EVAL_EIGHT[a.domain]
    tune_on = [i for i in DOMAIN_IDS[a.domain] if i not in eval_eight]
    if a.max_tune_refs:
        tune_on = tune_on[:a.max_tune_refs]
    chrono_config = a.chrono_config or f"configs/go2_chrono_eval_{'crm' if a.domain=='crm' else 'flat'}.json"
    cfg.update({
        "num_envs": 1, "device": "cpu", "auto_reset": False,
        "chrono_config": chrono_config,
        "imported_policy_ckpt": str(ev.DEFAULT_IMPORTED_POLICY),
        "dynamics_checkpoint":
            "artifacts/training_runs/go2_transformer_v01_contact_mix25_onehot/checkpoints/best_val.pt",
        "reference_path": "artifacts/rl_references/go2_flat_crm_ref40.npz",
        "pre_roll_time_s": 0.0,
        "max_episode_steps": int(round(a.horizon_s / 0.05)),
        "initial_reference_ids": [0],
    })
    from nedm.rl.go2_chrono_tracking_env import Go2ChronoCRMTrackingEnv
    env_cls = Go2ChronoCRMTrackingEnv if a.domain == "crm" else Go2ChronoTrackingEnv
    env = env_cls(cfg)
    print(f"domain {a.domain}, config {chrono_config}")
    print(f"tuning on {len(tune_on)} held-out {a.domain} references: {tune_on}")
    print(f"evaluating later on the eight: {eval_eight}   (disjoint: "
          f"{not set(tune_on) & set(eval_eight)})")
    print(f"{len(combos)} gain combinations{' (isotropic)' if a.isotropic else ''}\n")

    values = [float(v) for v in a.grid.split(",")]
    combos = ([(v, v, v) for v in values] if a.isotropic
              else list(itertools.product(values, values, values)))
    results = []
    t0 = time.time()
    for kx, ky, kyaw in combos:
        ctrl = ev.ProportionalController([kx, ky, kyaw], env.action_low, env.action_high)
        errs = []
        for rid in tune_on:
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
        {"tuned_on": tune_on, "eval_on": eval_eight, "domain": a.domain, "grid": values,
         "isotropic": bool(a.isotropic),
         "results": results, "best": best}, indent=2) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
