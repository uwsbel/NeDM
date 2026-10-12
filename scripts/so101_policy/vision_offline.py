"""Offline check of a camera student on demo decisions (validation episodes of its own split, or another demo file):
first-action L1 per action dim, and the trigger a[4] as a classifier (student a[4] > 0 vs expert a[4] > 0) split by
decoder mode (free = GRASP trigger, hold = PLACE trigger, GRASP macro = hold the trigger).

    source ~/so101_policy/runs/render/madrona_luffy/env_luffy.sh
    PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -P scripts/so101_policy/vision_offline.py \
        --student RUN/last.pt --demos DEMOS.npz [--split val|train|all] [--n 4096] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nedm.so101_policy.decoder import M
from nedm.so101_policy.vision import dataset as V
from nedm.so101_policy.vision import student as S


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", required=True), ap.add_argument("--demos", nargs="+", required=True)
    ap.add_argument("--split", default="val", choices=("val", "train", "all"))
    ap.add_argument("--n", type=int, default=4096), ap.add_argument("--json"), ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-episodes", type=int, default=1000)
    a = ap.parse_args()
    st, ck = S.load_student(a.student)
    ds = V.DemoSet(a.demos, "cuda", st.cfg.chunk, max_episodes=a.max_episodes, val_frac=0.05 if a.split != "all" else 0.0,
                   seed=a.seed)
    rows = ds.val_rows if a.split == "val" else ds.train_rows
    rows = rows[: a.n] if len(rows) > a.n else rows
    views = V.Views(256, st.cfg.res, st.cfg.cameras)
    first, labels, modes = [], [], []
    for s in range(0, len(rows), 256):
        b = ds.batch(rows[s:s + 256])
        img = V.to_input(views.render(b["q"], b["cube_pos"], b["cube_quat"], b["goal_pose"]))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ch = st.predict_chunk(img, b["state"], b["goal"]).float()
        first.append(ch[:, 0].cpu())
        labels.append(b["action"][:, 0].cpu())
        modes.append(b["state"][:, 6 + 11:].argmax(-1).cpu())
    f, l, md = torch.cat(first), torch.cat(labels), torch.cat(modes)
    out = dict(student=a.student, step=int(ck.get("step", -1)), demos=a.demos, split=a.split, n=len(f),
               l1_first=[float(x) for x in (f - l).abs().mean(0)])
    groups = dict(free=md == M["free"], hold=md == M["hold"], grasp_pre=(md >= M["g_brake"]) & (md <= M["g_settle"]))
    for name, g in groups.items():
        p, y = f[g, 4] > 0, l[g, 4] > 0
        out[f"trigger_{name}"] = dict(n=int(g.sum()), expert_pos=int(y.sum()), tp=int((p & y).sum()), fp=int((p & ~y).sum()),
                                      fn=int((~p & y).sum()), student_a4_at_expert_pos_p50=float(f[g][y, 4].median()) if y.any() else None)
    hold = groups["hold"]
    out["hold_l1_xy_yaw"] = [float(x) for x in (f[hold][:, [0, 1, 3]] - l[hold][:, [0, 1, 3]]).abs().mean(0)] if hold.any() else None
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
