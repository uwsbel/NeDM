#!/usr/bin/env python3
"""Rescore rollout_sel for a set of arms under ONE common measurement.

Why this exists. rollout_sel is computed during training from
rollout_eval.num_episodes open-loop rollouts, and that count is NOT constant across
our configs: the older arms (baseline_s1, abl_w512, dq25) use 12, the newer ones
(qbase, mw25*, w768, w1024) use 32. A median taken from a 12-episode metric and a
median taken from a 32-episode metric are different quantities, so any table that
puts them in the same column is comparing measurements, not models. Several of our
"less data is better" and "capacity" numbers did exactly that.

This reloads each arm's last.pt -- last.pt and not best_val.pt, because best_val is
the minimum of eighty noisy draws and so selects the luckiest evaluation rather than
the best model -- and re-runs evaluate_rollouts() with num_episodes and the episode
sampling seed pinned to the same values for every arm.

Usage:
  rescore_rollout_sel.py --episodes 32 --seed 0 arm1 arm2 ...
"""
import argparse, json, sys, os
from pathlib import Path

REPO = next(p for p in ("/home/kyle/Documents/sbel/NeDM", "/home/kyle/sbel/NeDM") if os.path.isdir(p))
sys.path.insert(0, REPO + "/src")
os.environ.setdefault("NEDM_REPO", REPO)
import torch
from nedm.training.trainer import HMMWVTrainer as Trainer

ART = Path("/home/kyle/sbel-artifacts")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    results = {}
    for arm in a.arms:
        cfg_path = Path(REPO) / "configs" / f"go2_crm_{arm}.json"
        ck = ART / "training_runs" / f"go2_crm_{arm}" / "checkpoints" / f"{a.ckpt}.pt"
        if not cfg_path.exists():
            print(f"  {arm:<14s} SKIP no config on this box", flush=True); continue
        if not ck.exists():
            print(f"  {arm:<14s} SKIP no {a.ckpt}.pt", flush=True); continue

        cfg = json.loads(cfg_path.read_text())
        native = cfg.get("rollout_eval", {}).get("num_episodes")
        # Pin the measurement. This is the whole point of the script.
        cfg.setdefault("rollout_eval", {})["num_episodes"] = a.episodes
        cfg["training"]["seed"] = a.seed          # trainer reads training.seed, not top-level
        cfg["training"]["num_epochs"] = 0         # construct only; do not train

        try:
            tr = Trainer(cfg)
            sd = torch.load(ck, map_location=tr.device, weights_only=False)
            state = sd.get("model_state_dict", sd.get("model", sd.get("state_dict", sd)))
            tr.model.load_state_dict(state)
            tr.model.eval()
            with torch.no_grad():
                m = tr.evaluate_rollouts()
        except Exception as e:
            print(f"  {arm:<14s} FAIL {type(e).__name__}: {e}", flush=True); continue

        sel = m.get("rollout_sel")
        results[arm] = {"rollout_sel": sel, "native_num_episodes": native,
                        "rescored_at_episodes": a.episodes, "ckpt": a.ckpt}
        print(f"  {arm:<14s} rollout_sel={sel:.4f}   (native metric used {native} episodes)", flush=True)

    if a.out:
        Path(a.out).write_text(json.dumps(
            {"episodes": a.episodes, "seed": a.seed, "ckpt": a.ckpt, "arms": results}, indent=1))
        print(f"  wrote {a.out}")

if __name__ == "__main__":
    sys.exit(main())
