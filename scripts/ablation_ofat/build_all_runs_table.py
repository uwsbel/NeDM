#!/usr/bin/env python3
"""One table over every trained HMMWV dynamics run: one-step loss + open-loop err/dist.

Both metrics are read at each run's SELECTED epoch (min `rollout_sel`, i.e. the
checkpoint kept as best_val.pt), so every row describes the model you would
actually ship rather than a per-metric best epoch.

The two metric families:
  one-step   `val_*_loss` -- next-step state-delta prediction loss on the val
             split, in the model's normalized space (weighted Huber). This is the
             1-step teacher-forced error.
  open-loop  `rollout_*.errdist` -- the NN rolled out open-loop for 10 s with the
             RECORDED action sequence, compared against Chrono's RECORDED val-split
             trajectory, distance-normalized (xy_rmse / mean_gt_distance). NOTE:
             this is Chrono data offline, NOT policy-in-the-loop live Chrono
             (that is `chrono_eval_*`, a separate RL-side artifact).
  S          = rollout_sel = 0.5*flat + 0.5*crm open-loop errdist. Lower better.

Comparability caveats are printed as flags, not buried:
  [7ch] one-step loss is over 7 channels, not 15 -> NOT comparable to other rows.
  [spec] single-domain specialist -> its off-domain column is a collapse, by design.

Usage:
    python scripts/ablation_ofat/build_all_runs_table.py [--csv out.csv] [--markdown]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO_ROOT / "artifacts/training_runs"
OFAT_ROOT = RUN_ROOT / "ablation_ofat"

ANCHOR_SLUG = "hmmwv_transformer_v07_tire_normal_force_omega_300g_crm2000_mix25_rebal_rollout_onehot"


def load_metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def selected_epoch(metrics: list[dict]) -> dict | None:
    valid = [m for m in metrics if isinstance(m.get("rollout_sel"), (int, float))
             and math.isfinite(m["rollout_sel"])]
    return min(valid, key=lambda m: m["rollout_sel"]) if valid else None


def errdist(entry: dict, key: str) -> float:
    value = entry.get(key, {})
    return float(value.get("errdist", float("nan"))) if isinstance(value, dict) else float("nan")


def describe(config: dict, slug: str) -> tuple[str, str, list[str]]:
    """-> (arch string, arm, flags)"""
    model = config.get("model", {})
    arch = f"L{model.get('n_layer')}/H{model.get('n_head')}/E{model.get('n_embd')}/c{model.get('block_size')}"
    arm = config.get("sweep_recipe", {}).get("arm", "anchor")

    flags = []
    state_dim = 15
    for source in config.get("train_mix", {}).get("datasets", []):
        if "body7" in source.get("processed_dataset_dir", ""):
            state_dim = 7
    if state_dim == 7:
        flags.append("7ch")
    if not config.get("terrain_conditioning", {}).get("enabled", False):
        flags.append("no1hot")

    fractions = {s.get("name"): s.get("batch_fraction", 0.0)
                 for s in config.get("train_mix", {}).get("datasets", [])}
    if fractions.get("crm", 0) == 0 or fractions.get("flat", 0) == 0:
        flags.append("spec")
    episode_fraction = None
    for source in config.get("train_mix", {}).get("datasets", []):
        if "train_episode_fraction" in source:
            episode_fraction = source["train_episode_fraction"]
    if episode_fraction is not None:
        flags.append(f"data{int(episode_fraction*100)}%")
    return arch, arm, flags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--markdown", action="store_true", help="emit a markdown table")
    args = ap.parse_args()

    entries: list[tuple[str, Path, Path]] = [
        ("L6_H8_E256_ctx128 (anchor)", RUN_ROOT / ANCHOR_SLUG, REPO_ROOT / f"configs/{ANCHOR_SLUG}.json"),
    ]
    for run_dir in sorted(OFAT_ROOT.iterdir()):
        if run_dir.is_dir() and (run_dir / "metrics.jsonl").exists():
            entries.append((run_dir.name, run_dir, REPO_ROOT / f"configs/ablation_ofat/{run_dir.name}.json"))

    rows = []
    for name, run_dir, config_path in entries:
        metrics = load_metrics(run_dir)
        best = selected_epoch(metrics)
        if not best:
            continue
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        arch, arm, flags = describe(config, name)
        rows.append({
            "run": name,
            "arch": arch,
            "arm": arm,
            "flags": ",".join(flags) or "-",
            "best_ep": int(best["epoch"]),
            "step1_flat": best.get("val_flat_loss", best.get("val_loss", float("nan"))),
            "step1_crm": best.get("val_crm_loss", float("nan")),
            "step1_mixed": best.get("val_mixed_loss", float("nan")),
            "ol_flat10": errdist(best, "rollout_flat_10.0s"),
            "ol_crm10": errdist(best, "rollout_crm_10.0s"),
            "S": float(best["rollout_sel"]),
        })

    rows.sort(key=lambda r: r["S"])

    if args.markdown:
        print("| # | run | arch | arm | flags | ep | 1-step flat | 1-step CRM | 1-step mixed | "
              "open-loop flat | open-loop CRM | **S** |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(rows, 1):
            print(f"| {i} | `{r['run']}` | {r['arch']} | {r['arm']} | {r['flags']} | {r['best_ep']} | "
                  f"{r['step1_flat']:.5f} | {r['step1_crm']:.5f} | {r['step1_mixed']:.5f} | "
                  f"{r['ol_flat10']:.4f} | {r['ol_crm10']:.4f} | **{r['S']:.4f}** |")
    else:
        print(f"{'#':>2s} {'run':38s} {'arch':20s} {'arm':16s} {'flags':12s} {'ep':>3s} "
              f"{'1step_flat':>10s} {'1step_crm':>10s} {'1step_mix':>10s} "
              f"{'ol_flat':>8s} {'ol_crm':>8s} {'S':>8s}")
        print("-" * 150)
        for i, r in enumerate(rows, 1):
            print(f"{i:2d} {r['run']:38s} {r['arch']:20s} {r['arm']:16s} {r['flags']:12s} {r['best_ep']:3d} "
                  f"{r['step1_flat']:10.5f} {r['step1_crm']:10.5f} {r['step1_mixed']:10.5f} "
                  f"{r['ol_flat10']:8.4f} {r['ol_crm10']:8.4f} {r['S']:8.4f}")

    print(f"\n{len(rows)} runs, sorted by S (open-loop err/dist, 0.5 flat + 0.5 CRM; lower better).")
    print("Both metric families are read at the SELECTED epoch (min rollout_sel = best_val.pt).")
    print("open-loop = NN rolled out 10 s on recorded actions vs Chrono's RECORDED trajectory,")
    print("distance-normalized. Offline Chrono data, NOT policy-in-the-loop live Chrono.")
    print("flags: [7ch] 1-step loss over 7 channels -> NOT comparable to 15-ch rows;")
    print("       [spec] single-domain specialist -> off-domain column is a designed collapse;")
    print("       [no1hot] no terrain key. dataNN% = trained on NN% of episodes.")

    if args.csv:
        out = Path(args.csv)
        with out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
