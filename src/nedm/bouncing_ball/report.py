"""Make measured coverage and full-rollout comparison figures on AMD."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from nedm.bouncing_ball.collection import STATE_FIELDS, atomic_json
from nedm.bouncing_ball.model import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("generate rollout comparisons in an AMD GPU allocation")
    torch.set_num_threads(4)
    model, checkpoint = load_model(args.run / "best.pt", "cuda")
    evaluation = json.loads((args.run / "evaluation.json").read_text())
    index = json.loads((args.data / "campaign_index.json").read_text())
    packed = np.load(args.data / "model_data.npz")
    timing = {"prediction_timestamp": "completed-impact state at a 100 Hz frame boundary", "truth_timestamp": "Chrono event at 0.125 ms physics resolution"}
    for split in ["validation","test"]:
        if evaluation[split] is None:
            continue
        errors = []
        for entry in evaluation[split]["per_episode"]:
            if not entry["finite"] or not entry["contact_order"]:
                continue
            true = {event["kind"]:event["time_s"] for event in index["episodes"][entry["index"]]["events"]}
            errors.append(max(abs(entry["ground_event_times_s"][0]-true["first_ground_contact"]),
                              abs(entry["wall_event_times_s"][0]-true["wall_contact"])))
        timing[split] = {"matched_episodes":len(errors),"median_s":float(np.median(errors)),"p95_s":float(np.quantile(errors,.95)),"max_s":float(max(errors))}
    atomic_json(args.run / "impact_timing.json",timing)
    folder = args.run / "figures"
    folder.mkdir(exist_ok=True)
    nx, nz = index["campaign"]["grid"]
    counts = np.zeros((3, nx, nz), dtype=int)
    for split, cell in zip(packed["splits"], packed["cells"], strict=True):
        counts[split, cell[0], cell[1]] += 1
    extent = [*index["config"]["launch"]["vx_range_mps"], *index["config"]["launch"]["vz_range_mps"]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for k, (ax, name) in enumerate(zip(axes, ["Train", "Validation", "Test"], strict=True)):
        im = ax.imshow(counts[k].T, extent=extent, origin="lower", aspect="auto", vmin=0, vmax=12)
        ax.set(title=f"{name}: {int(counts[k].sum()):,} episodes", xlabel="Initial vx (m/s)", ylabel="Initial vz (m/s)")
    fig.colorbar(im, ax=axes, label="Episodes per velocity cell")
    fig.savefig(folder / "launch_coverage.png", dpi=170)
    plt.close(fig)

    test = evaluation["test"]
    if test is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
        for ax, metric, title in zip(axes, ["position_rmse_m", "endpoint_error_m"], ["Full-rollout position RMSE", "Endpoint error"], strict=True):
            sums, number = np.zeros((nx, nz)), np.zeros((nx, nz))
            for entry in test["per_episode"]:
                cell = tuple(packed["cells"][entry["index"]])
                sums[cell] += entry.get(metric, np.nan)
                number[cell] += 1
            im = ax.imshow((sums/number).T*100, extent=extent, origin="lower", aspect="auto", cmap="viridis")
            ax.set(title=title, xlabel="Initial vx (m/s)", ylabel="Initial vz (m/s)")
            fig.colorbar(im, ax=ax, label="Mean per cell (cm)")
        fig.savefig(folder / "heldout_errors.png", dpi=170)
        plt.close(fig)

    reference_index = json.loads((args.reference / "dataset_index.json").read_text())
    stride = round(model.dt / reference_index["config"]["simulation"]["record_step_s"])
    fig, axes = plt.subplots(4, len(reference_index["episodes"]), figsize=(16, 10), constrained_layout=True)
    comparisons = []
    for col, episode in enumerate(reference_index["episodes"]):
        with (args.reference / episode["csv_path"]).open(newline="") as stream:
            rows = list(csv.DictReader(stream))[::stride]
        end = next(event["time_s"] for event in episode["events"] if event["kind"]=="next_ground_contact")-index.get("terminal_margin_s",0)
        rows = [row for row in rows if float(row["time_s"])<=end+1e-10]
        truth = np.array([[float(row[field]) for field in STATE_FIELDS] for row in rows], dtype=np.float32)
        with torch.no_grad():
            predicted = model.rollout(torch.as_tensor(truth[None, 0], device="cuda"), len(truth)-1)[0].cpu().numpy()
        time = np.arange(len(truth))*model.dt
        error = np.linalg.norm(predicted[:, :2]-truth[:, :2], axis=-1)
        comparisons.append({"launch": episode["launch"], "label": episode["launch_label"],
                            "position_rmse_m": float(np.sqrt(np.mean(error**2))), "endpoint_error_m": float(error[-1])})
        np.savez_compressed(folder / f"{episode['launch_label']}_rollout.npz", time_s=time, chrono=truth, nrd=predicted)
        ax = axes[0, col]
        ax.plot(truth[:,0], truth[:,1], color="black", linewidth=2, label="Chrono")
        ax.plot(predicted[:,0], predicted[:,1], color="tab:orange", linestyle="--", linewidth=1.8, label="NRD")
        ax.axhline(model.floor_center_z, color="gray", linewidth=.7)
        ax.axvline(model.wall_center_x, color="gray", linewidth=.7)
        ax.set(title=f"vx={episode['launch']['vx_mps']:g}, vz={episode['launch']['vz_mps']:g}\nRMSE {comparisons[-1]['position_rmse_m']*100:.2f} cm", xlabel="x (m)", ylabel="z (m)")
        ax.legend(fontsize=8)
        for row, channel, label in [(1,2,"vx (m/s)"), (2,3,"vz (m/s)"), (3,4,"omega y (rad/s)")]:
            ax = axes[row, col]
            ax.plot(time, truth[:,channel], color="black", linewidth=1.5)
            ax.plot(time, predicted[:,channel], color="tab:orange", linestyle="--", linewidth=1.4)
            for event in episode["events"]:
                if event["kind"] in ("first_ground_contact", "wall_contact"):
                    ax.axvline(event["time_s"], color="gray", alpha=.4, linewidth=.7)
            ax.set(xlabel="Time (s)", ylabel=label)
            ax.grid(alpha=.15)
    fig.savefig(folder / "approved_five_rollouts.png", dpi=170)
    plt.close(fig)
    atomic_json(args.run / "approved_five_comparison.json", {"free_rollout": True, "model_dt_s": model.dt, "episodes": comparisons})
    logs = [json.loads(line) for line in (args.run / "train_log.jsonl").read_text().splitlines()]
    logs = [row for row in logs if row["stage"]=="one_step"]
    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    ax.plot([row["elapsed_s"]/60 for row in logs], [row["position_p95_m"]*100 for row in logs], "o-")
    ax.axhline(5, color="gray", linestyle="--", label="5 cm gate")
    ax.set(xlabel="Training time (minutes)", ylabel="Validation p95 position RMSE (cm)", title="Initial-state full rollouts")
    ax.legend()
    fig.savefig(folder / "validation_curve.png", dpi=170)
    plt.close(fig)
    validation_indices = np.flatnonzero(packed["splits"]==1)
    chosen = validation_indices[np.linspace(0,len(validation_indices)-1,16,dtype=int)]
    initial = torch.as_tensor(packed["states"][chosen,0],device="cuda").clone().requires_grad_(True)
    endpoint = model.rollout(initial,170)[:,-1,:2]
    jacobian = torch.stack([torch.autograd.grad(endpoint[:,k].sum(),initial,retain_graph=True)[0][:,2:4]
                            for k in range(2)],dim=1)
    eps = 1e-3
    finite_difference = []
    with torch.no_grad():
        for axis in [2,3]:
            plus, minus = initial.detach().clone(), initial.detach().clone()
            plus[:,axis] += eps
            minus[:,axis] -= eps
            finite_difference.append((model.rollout(plus,170)[:,-1,:2]-model.rollout(minus,170)[:,-1,:2])/(2*eps))
    numerical = torch.stack(finite_difference,dim=-1)
    relative = (jacobian-numerical).abs()/(jacobian.abs()+numerical.abs()).clamp_min(.1)
    gradient_report = {"validation_episodes": len(chosen), "horizon_s": 1.7, "perturbation_mps": eps,
                       "finite": bool(torch.isfinite(jacobian).all() and torch.isfinite(numerical).all()),
                       "relative_difference_p95": float(torch.quantile(relative.flatten(),.95)),
                       "max_absolute_difference": float((jacobian-numerical).abs().max()),
                       "initial_velocities": packed["launches"][chosen].tolist(),
                       "autograd_dxz_dvxvz": jacobian.detach().cpu().tolist(),
                       "finite_difference_dxz_dvxvz": numerical.cpu().tolist()}
    # A 1e-3 m/s perturbation can cross a discrete contact frame. Also check
    # the local derivative at a smaller perturbation with float64 arithmetic.
    precise = copy.deepcopy(model).double()
    point = initial.detach().double().requires_grad_(True)
    value = precise.rollout(point,170)[:,-1,:2]
    local_jacobian = torch.stack([torch.autograd.grad(value[:,k].sum(),point,retain_graph=True)[0][:,2:4]
                                 for k in range(2)],dim=1)
    small_eps = 1e-5
    local_fd = []
    with torch.no_grad():
        for axis in [2,3]:
            plus,minus = point.detach().clone(),point.detach().clone()
            plus[:,axis] += small_eps;minus[:,axis] -= small_eps
            local_fd.append((precise.rollout(plus,170)[:,-1,:2]-precise.rollout(minus,170)[:,-1,:2])/(2*small_eps))
    local_numerical = torch.stack(local_fd,dim=-1)
    local_error = (local_jacobian-local_numerical).abs()
    local_relative = local_error/(local_jacobian.abs()+local_numerical.abs()).clamp_min(.1)
    gradient_report["local_derivative_check"] = {"dtype":"float64","perturbation_mps":small_eps,
        "finite":bool(torch.isfinite(local_jacobian).all() and torch.isfinite(local_numerical).all()),
        "relative_difference_p95":float(torch.quantile(local_relative.flatten(),.95)),
        "max_absolute_difference":float(local_error.max())}
    gradient_report["scope"] = "Autograd through the learned model; larger launch changes can cross contact frames. Chrono verification remains necessary for launch optimization."
    atomic_json(args.run / "launch_gradient_check.json",gradient_report)
    print(json.dumps({"figures": str(folder), "approved_five": comparisons}), flush=True)


if __name__ == "__main__":
    main()
