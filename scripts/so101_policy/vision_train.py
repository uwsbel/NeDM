"""Train the ACT camera student on expert demos (states; images drawn on the fly by Madrona).

    source ~/so101_policy/runs/render/madrona_luffy/env_luffy.sh
    PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -P scripts/so101_policy/vision_train.py \
        --demos ~/so101_policy/runs/vision/demos/demos_mujoco_1280.npz --out RUN \
        --config configs/so101_policy/vision/act_v1.json --max-minutes 35 [--max-episodes 1000] [--seed 0]

Writes RUN/{meta.json, learning_curve.csv, train_summary.json, last.pt, ckpt/step_*.pt} (``vision.student.train``).
Madrona's engine start may abort the process when the GPU is full (build2_render.md): rerun.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from nedm.so101_policy.vision import dataset as V
from nedm.so101_policy.vision import student as S


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demos", nargs="+", required=True), ap.add_argument("--out", required=True)
    ap.add_argument("--config", help="StudentConfig JSON"), ap.add_argument("--set", help="JSON overrides")
    ap.add_argument("--max-minutes", type=float, default=30.0), ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-episodes", type=int, default=0), ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.05), ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--ckpt-minutes", type=float, default=10.0), ap.add_argument("--backend", default="madrona")
    ap.add_argument("--all-episodes", action="store_true", help="keep failed episodes too (default: successes only)")
    ap.add_argument("--init", help="student checkpoint to fine-tune (weights and normalisation; DAgger rounds)")
    a = ap.parse_args()
    d = json.loads(Path(a.config).read_text()) if a.config else {}
    d.update(json.loads(a.set) if a.set else {})
    d["seed"] = a.seed
    cfg = S.StudentConfig.from_dict(d)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = V.DemoSet(a.demos, "cuda", cfg.chunk, success_only=not a.all_episodes, max_episodes=a.max_episodes,
                     val_frac=a.val_frac, seed=a.seed, sustain_place=cfg.sustain_place, privileged=cfg.privileged_cube)
    print(f"demos {a.demos}: train episodes {data.n_train_eps} ({len(data.train_rows)} decisions), val episodes "
          f"{data.n_val_eps} ({len(data.val_rows)}); DAgger episodes {data.dagger_eps}; PLACE labels held "
          f"{data.relabelled}; decoder {data.decoder_cfg}", flush=True)
    views = V.Views(cfg.render_batch, cfg.res, cfg.cameras, backend=a.backend)
    S.train(data, views, cfg, out, max_minutes=a.max_minutes, max_steps=a.max_steps, val_every=a.val_every,
            ckpt_minutes=a.ckpt_minutes, log=lambda s: print(s, flush=True),
            extra_meta=dict(argv=sys.argv, demos=a.demos, max_episodes=a.max_episodes, torch=torch.__version__),
            init_from=a.init)


if __name__ == "__main__":
    main()
