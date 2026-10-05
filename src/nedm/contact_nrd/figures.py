"""Example fresh-cohort shots: Chrono truth against the unified model and one earlier model.

Panels: each model's worst shot (by its own primary error) and two typical
shots (median error of the unified model). Ball: side view (x, z). Pool:
table top view with A and B. Also one shot's collision on/off over time for
every pair (unified model, rolled out from the first state) against Chrono's
contact flags.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from nedm.contact_nrd.adapters import BallView, PoolView
from nedm.contact_nrd.model_v2 import load_any


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("pool", "ball"), required=True)
    parser.add_argument("--data", type=Path, required=True, help="reference-format cohort (ball: merged; pool: packed campaign)")
    parser.add_argument("--certification", type=Path, required=True)
    parser.add_argument("--unified", type=Path, required=True)
    parser.add_argument("--reference", required=True, help="name=checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("SLURM_JOB_ID")
    args.output.mkdir(parents=True, exist_ok=True)
    cert = json.loads(args.certification.read_text())
    ref_name, ref_path = args.reference.split("=", 1)
    uni_name = args.unified.parent.name
    model, _ = load_any(args.unified, "cuda")
    if args.system == "pool":
        from nedm.pool_ball.evaluate import load_data
        from nedm.pool_ball.model import load_pool
        data = load_data(args.data, "cuda")
        ref = load_pool(Path(ref_path), "cuda")[0]
        view = PoolView(model)
        key = "b_at_target_m"
    else:
        from nedm.bouncing_ball.model import load_model
        from nedm.bouncing_ball.transformer_contact_eval import data_packet
        data = data_packet(args.data, "cuda")
        ref = load_model(Path(ref_path), "cuda")[0].double()
        view = BallView(model)
        key = "endpoint_error_m"
    ids = torch.nonzero(data["splits"] == 2).flatten()
    per_u = cert["per_episode"][uni_name]
    per_r = cert["per_episode"][ref_name]
    err_u = np.array([e.get(key, np.inf) for e in per_u])
    err_r = np.array([e.get(key, np.inf) for e in per_r])
    order = np.argsort(err_u)
    picks = [("worst shot of the earlier model", int(np.argmax(err_r))), ("worst shot of the unified model", int(np.argmax(err_u))),
             ("typical shot", int(order[len(order) // 2])), ("typical shot", int(order[len(order) // 3]))]
    rows = ids[[p[1] for p in picks]]
    with torch.no_grad():
        truth = data["states"][rows]
        if args.system == "ball":
            steps = int(data["lengths"][rows].max()) - 1
            pu, pr = view.rollout(truth[:, 0], steps), ref.rollout(truth[:, 0], steps)
        else:
            stride = data["stride"]
            truth = truth[:, ::stride]
            steps = truth.shape[1] - 1
            pu, pr = view.rollout(truth[:, 0], steps), ref.rollout(truth[:, 0], steps)
    truth, pu, pr = truth.cpu().numpy(), pu.cpu().numpy(), pr.cpu().numpy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5) if args.system == "pool" else (14, 7.5))
    for ax, (label, k), t, u, r in zip(axes.ravel(), picks, truth, pu, pr):
        if args.system == "ball":
            n = int(data["lengths"][ids[k]])
            ax.plot(t[:n, 0], t[:n, 1], color="k", lw=2.2, label="Chrono")
            ax.plot(u[:n, 0], u[:n, 1], color="C0", lw=1.3, ls="--", label=f"unified ({1e3 * err_u[k]:.1f} mm at the end)")
            ax.plot(r[:n, 0], r[:n, 1], color="C3", lw=1.1, ls=":", label=f"{ref_name} ({1e3 * err_r[k]:.1f} mm)")
            scene = data["index"]["config"]["scene"]
            ax.axhline(scene["ground_z_m"], color="#795548", lw=3)
            ax.axvline(scene["wall_front_x_m"], color="#795548", lw=3)
            ax.set(xlabel="x (m)", ylabel="z (m)")
        else:
            from nedm.pool_ball.video import table
            scene = data["scene"]
            table(ax, scene)
            T = data["target_step"] + 1
            ax.plot(t[:T, 0], t[:T, 1], color="white", lw=2.2, label="Chrono A")
            ax.plot(t[:T, 7], t[:T, 8], color="#ffcc00", lw=2.2, label="Chrono B")
            ax.plot(u[:T, 7], u[:T, 8], color="C0", lw=1.4, ls="--", label=f"unified B ({1e3 * err_u[k]:.1f} mm at 2 s)")
            ax.plot(r[:T, 7], r[:T, 8], color="C3", lw=1.2, ls=":", label=f"{ref_name} B ({1e3 * err_r[k]:.1f} mm)")
            ax.plot(*t[T - 1, 7:9], "o", color="#ffcc00", ms=7)
        ax.set_title(f"{label}: fresh shot {int(ids[k])}")
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"{args.system}: fresh shots, rolled out from the first state only (10 ms steps)")
    fig.tight_layout()
    fig.savefig(args.output / f"{args.system}_example_shots.png", dpi=130)
    plt.close(fig)

    # collision on/off of a typical shot, unified model
    k = picks[2][1]
    with torch.no_grad():
        init = data["states"][ids[k:k + 1]]
        if args.system == "pool":
            init = init[:, ::data["stride"]]
        u0 = view.to_unified(init[:, 0]) if args.system == "pool" else BallView.to_unified(init[:, 0])
        n = steps
        _, gates = model.rollout(u0, n, return_gates=True)
    gates = (gates[0].cpu().numpy() > 0.5)
    labels = data["contacts"][ids[k]].cpu().numpy()
    if args.system == "pool":
        stride = data["stride"]
        m = labels.shape[0] // stride
        labels = labels[: m * stride].reshape(m, stride, -1).max(1)
    labels = labels[: gates.shape[0]] > 0.5
    names = [f"{model.system['bodies'][i]['name']}-{model.system['bodies'][j]['name']}" for i, j in model.system["pairs"]]
    if labels.shape[1] != len(names):  # ball packet: ground, wall
        labels = labels[:, : len(names)]
    fig, ax = plt.subplots(figsize=(13, 1.2 + 0.45 * len(names)))
    time = np.arange(gates.shape[0]) * model.dt
    for p, name in enumerate(names):
        on_t = time[labels[:, p]]
        on_g = time[gates[:, p]]
        ax.scatter(on_t, np.full(len(on_t), p + 0.15), marker="|", s=200, color="k", label="Chrono contact" if p == 0 else None)
        ax.scatter(on_g, np.full(len(on_g), p - 0.15), marker="|", s=200, color="C1", label="collision network on" if p == 0 else None)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("time (s)")
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    ax.set_title(f"{args.system}: on/off per body pair, unified model rolled out on its own (shot {int(ids[k])})")
    fig.tight_layout()
    fig.savefig(args.output / f"{args.system}_collision_onoff.png", dpi=130)
    print(json.dumps({"picks": [(label, int(ids[k]), float(err_u[k]), float(err_r[k])) for label, k in picks]}))


if __name__ == "__main__":
    main()
