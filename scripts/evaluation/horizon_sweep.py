#!/usr/bin/env python3
"""Measure err/dist against rollout horizon on a fixed grid.

Why this exists. The question "how far can we trust a rollout" has been answered in
conversation with numbers that are not recorded anywhere, and training only ever logs
rollout at 5 s and 10 s, so there has never been a grid to read an answer off. This
produces one, under the SAME code path training uses (`evaluate_rollouts`), so the
numbers are comparable with `rollout_crm_5.0s` and `rollout_crm_10.0s` already on record.

Two things are pinned, because both have silently varied across arms before:
`rollout_eval.num_episodes` (12 on older configs, 32 on newer ones) and the sampling
seed. A median over 12 draws and a median over 32 are different quantities.

Reading the result: `errdist` is planar position error divided by distance travelled, so
a model that predicts no motion at all scores 1.0 by construction. That is the floor the
curve has to stay under to be worth anything, and where it crosses 1.0 is where the model
stops carrying information about where the robot goes.
"""
import argparse, json, os, sys
from pathlib import Path

REPO = "/srv/home/kasha2/nedm/NeDM"
ART  = Path("/srv/home/kasha2/nedm")
sys.path.insert(0, REPO + "/src")
os.environ.setdefault("NEDM_REPO", REPO)
import torch
from nedm.training.trainer import HMMWVTrainer as Trainer

GRID = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="best_val")
    ap.add_argument("--horizons", default=",".join(str(g) for g in GRID))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    hs = [float(x) for x in a.horizons.split(",")]

    out = {"episodes": a.episodes, "seed": a.seed, "ckpt": a.ckpt,
           "horizons_s": hs, "no_motion_floor": 1.0, "arms": {}}
    for arm in a.arms:
        cfg_p = Path(REPO) / "configs" / f"go2_crm_{arm}.json"
        ck_p  = ART / "training_runs" / f"go2_crm_{arm}" / "checkpoints" / f"{a.ckpt}.pt"
        if not cfg_p.exists() or not ck_p.exists():
            print(f"  {arm:<16s} SKIP (config={cfg_p.exists()} ckpt={ck_p.exists()})", flush=True)
            continue
        cfg = json.loads(cfg_p.read_text())
        re_cfg = cfg.setdefault("rollout_eval", {})
        native_eps, native_h = re_cfg.get("num_episodes"), re_cfg.get("horizons_s")
        re_cfg["num_episodes"] = a.episodes
        re_cfg["horizons_s"] = hs
        re_cfg["selection_horizon_s"] = hs[-1]
        cfg["training"]["seed"] = a.seed
        cfg["training"]["num_epochs"] = 0
        try:
            tr = Trainer(cfg)
            sd = torch.load(ck_p, map_location=tr.device, weights_only=False)
            tr.model.load_state_dict(sd.get("model_state_dict", sd))
            tr.model.eval()
            with torch.no_grad():
                m = tr.evaluate_rollouts()
        except Exception as e:
            print(f"  {arm:<16s} FAIL {type(e).__name__}: {e}", flush=True); continue

        row = {}
        for h in hs:
            hit = [v for k, v in m.items()
                   if k.startswith("rollout_") and k.endswith(f"_{h:.1f}s")
                   and isinstance(v, dict) and "errdist" in v]
            if hit: row[f"{h:.1f}"] = float(hit[0]["errdist"])
        out["arms"][arm] = {"errdist_by_horizon": row,
                            "native_num_episodes": native_eps, "native_horizons_s": native_h}
        cells = "  ".join(f"{h:.1f}s={row.get(f'{h:.1f}', float('nan')):.3f}" for h in hs)
        print(f"  {arm:<16s} {cells}", flush=True)

    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
        print("  wrote", a.out)

if __name__ == "__main__":
    main()
