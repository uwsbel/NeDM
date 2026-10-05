"""Minimize endpoint distance by differentiating NRD w.r.t. launch velocity.

Model parameters are frozen. Targets alone enter the optimizer; the launch
that generated a benchmark target is retained only as provenance. Projected
gradient descent uses a batched backtracking line search on the exact loaded
model, with a separate two-impact validity check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from nedm.bouncing_ball.collection import atomic_json, load_config
from nedm.bouncing_ball.model import load_model


def initial_state(velocity, position=(0.0, 1.0), spin=0.0):
    """Preserve the launch's autograd connection when assembling the 5D state."""
    leading = velocity.shape[:-1]
    position = velocity.new_tensor(position).expand(*leading, 2)
    angular = velocity.new_full((*leading, 1), spin)
    return torch.cat((position, velocity, angular), dim=-1)


def two_impact_mask(trajectory, model):
    previous, following = trajectory[..., :-1, :], trajectory[..., 1:, :]
    free_z = previous[..., 1] + previous[..., 3]*model.dt - .5*model.gravity*model.dt*(model.dt+model.physics_dt)
    free_x = previous[..., 0] + previous[..., 2]*model.dt
    ground = (previous[..., 3] < 0) & (following[..., 3] >= 0) & (free_z <= model.floor_center_z+.005)
    wall = (previous[..., 2] > 0) & (following[..., 2] <= 0) & (free_x >= model.wall_center_x-.005)
    ground_step = ground.to(torch.int64).argmax(-1)
    wall_step = wall.to(torch.int64).argmax(-1)
    finite = torch.isfinite(trajectory).all(dim=-1).all(dim=-1)
    penetration = ((trajectory[..., 1].amin(-1) >= model.floor_center_z-.005)
                   & (trajectory[..., 0].amax(-1) <= model.wall_center_x+.005))
    return finite & penetration & (ground.sum(-1) == 1) & (wall.sum(-1) == 1) & (ground_step < wall_step)


def optimize(model, targets, starts, bounds, *, steps, iterations=200,
             learning_rate=.08, backtracks=16, max_step=.25, tolerance=.005,
             position=(0.0, 1.0), spin=0.0, require_two_impacts=True,
             stop_when_any=False, progress=None):
    """Batched, monotone projected gradient descent, without simulator feedback."""
    if iterations < 1 or backtracks < 1 or learning_rate <= 0 or max_step <= 0 or tolerance <= 0:
        raise ValueError("iterations, backtracks, learning rate, step cap, and tolerance must be positive")
    if starts.shape != targets.shape or starts.shape[-1] != 2:
        raise ValueError("targets and launches must have shape [cases, 2]")
    lower, upper = bounds
    if torch.any(lower >= upper) or torch.any(starts < lower) or torch.any(starts > upper):
        raise ValueError("initial launch must be within increasing velocity bounds")
    if not torch.isfinite(targets).all() or not torch.isfinite(starts).all():
        raise ValueError("targets and initial launches must be finite")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    velocity = starts.detach().clone()
    rates = velocity.new_full((len(velocity),), learning_rate)
    factors = 2.0 ** -torch.arange(backtracks, dtype=velocity.dtype, device=velocity.device)
    history, snapshots = [], []
    stalled = torch.zeros(len(velocity), dtype=torch.int64, device=velocity.device)

    def forward(v):
        trajectory = model.rollout(initial_state(v, position, spin), steps)
        distance_squared = (trajectory[..., -1, :2]-targets).square().sum(-1)
        return trajectory, distance_squared

    with torch.no_grad():
        before, _ = forward(velocity)
        if require_two_impacts and not two_impact_mask(before, model).all():
            raise ValueError("initial NRD trajectories must contain ground then wall contact before endpoint")

    for iteration in range(iterations+1):
        velocity = velocity.detach().requires_grad_(True)
        trajectory, loss = forward(velocity)
        if not torch.isfinite(loss).all():
            raise RuntimeError("nonfinite endpoint loss")
        gradient, = torch.autograd.grad(loss.sum(), velocity)
        if not torch.isfinite(gradient).all():
            raise RuntimeError("nonfinite launch gradient")
        entry = {
            "iteration": iteration, "loss_m2": loss.detach().cpu().tolist(),
            "distance_m": loss.detach().sqrt().cpu().tolist(),
            "velocity_mps": velocity.detach().cpu().tolist(),
            "gradient": gradient.detach().cpu().tolist(),
            "learning_rate": rates.cpu().tolist(),
        }
        history.append(entry)
        if iteration % 10 == 0 or iteration == iterations or bool((loss <= tolerance**2).all()):
            snapshots.append((iteration, trajectory.detach().cpu().numpy()))
        if progress:
            progress(entry)
        active = (loss.detach() > tolerance**2) & (stalled < 5) & (rates >= 1e-9)
        if iteration == iterations or not active.any() or (stop_when_any and bool((loss <= tolerance**2).any())):
            break
        with torch.no_grad():
            # Cap each proposed launch displacement before backtracking.
            full_step = -rates[:, None]*gradient
            full_step *= (max_step/full_step.norm(dim=-1).clamp_min(max_step))[:, None]
            candidate = (velocity[:, None, :]+full_step[:, None, :]*factors[None, :, None]).clamp(lower, upper)
            candidate_trajectory = model.rollout(initial_state(candidate.flatten(0, 1), position, spin), steps)
            candidate_loss = (candidate_trajectory[:, -1, :2]-targets.repeat_interleave(backtracks, 0)).square().sum(-1)
            candidate_loss = candidate_loss.reshape(len(velocity), backtracks)
            valid = torch.isfinite(candidate_loss)
            if require_two_impacts:
                valid &= two_impact_mask(candidate_trajectory, model).reshape_as(candidate_loss)
            predicted_decrease = (gradient[:, None, :]*(candidate-velocity[:, None, :])).sum(-1)
            acceptable = valid & (candidate_loss < loss.detach()[:, None]) & (candidate_loss <= loss.detach()[:, None]+1e-4*predicted_decrease)
            accepted = acceptable.any(-1) & active
            chosen = acceptable.to(torch.int64).argmax(-1)
            new_velocity = candidate[torch.arange(len(velocity), device=velocity.device), chosen]
            velocity = torch.where(accepted[:, None], new_velocity, velocity)
            rates = torch.where(accepted, (rates*factors[chosen]*1.5).clamp(max=learning_rate), rates*.25)
            selected_loss = candidate_loss[torch.arange(len(velocity), device=velocity.device), chosen]
            meaningful = accepted & ((loss.detach().sqrt()-selected_loss.clamp_min(0).sqrt()) > 1e-6)
            stalled = torch.where(meaningful, torch.zeros_like(stalled), stalled+active.to(stalled.dtype))
            entry["accepted_step"] = accepted.cpu().tolist()
            entry["backtrack_index"] = chosen.cpu().tolist()

    with torch.no_grad():
        after, final_loss = forward(velocity)
    if snapshots[-1][0] != history[-1]["iteration"]:
        snapshots.append((history[-1]["iteration"], after.cpu().numpy()))
    return {"velocity": velocity.detach(), "before": before, "after": after,
            "loss": final_loss, "history": history, "snapshots": snapshots}


def benchmark_targets(data_root, config, steps):
    index = json.loads((data_root/"campaign_index.json").read_text())
    data_steps = round(config["terminal_time_s"]/index["model_dt_s"])
    if not math.isclose(data_steps*index["model_dt_s"], config["terminal_time_s"], abs_tol=1e-9):
        raise ValueError("benchmark endpoint must be on the physical data grid")
    with np.load(data_root/"model_data.npz") as packet:
        states = packet["states"]
    eligible = []
    for i, episode in enumerate(index["episodes"]):
        events = {e["kind"]: e["time_s"] for e in episode["events"]}
        if (episode["split"] == config["benchmark_split"]
                and events["wall_release"] < config["terminal_time_s"] < events["next_ground_contact"]-.02):
            eligible.append(i)
    chosen, targets = set(), []
    for neighborhood in config["benchmark_launch_neighborhoods_mps"]:
        candidates = [i for i in eligible if i not in chosen]
        i = min(candidates, key=lambda j: sum((index["episodes"][j]["launch"][key]-v)**2
                                            for key, v in zip(("vx_mps", "vz_mps"), neighborhood, strict=True)))
        chosen.add(i)
        episode = index["episodes"][i]
        targets.append({"name": f"target_{len(targets)+1}", "xz_m": states[i, data_steps, :2].tolist(),
                        "source_episode": episode["episode_id"], "source_split": episode["split"],
                        "source_launch_mps": [episode["launch"]["vx_mps"], episode["launch"]["vz_mps"]],
                        "target_source": "Chrono position at terminal time; source launch is never passed to optimizer"})
    return targets


def check_loss_gradients(model, velocity, targets, steps, position, spin):
    """Compare loss autograd with local and larger central perturbations."""
    velocity = velocity.detach().requires_grad_(True)
    loss = (model.rollout(initial_state(velocity, position, spin), steps)[:, -1, :2]-targets).square().sum(-1)
    gradient, = torch.autograd.grad(loss.sum(), velocity)
    reports = []
    with torch.no_grad():
        for epsilon in (1e-5, 1e-3):
            offsets = torch.eye(2, dtype=velocity.dtype, device=velocity.device)*epsilon
            trials = torch.cat((velocity[:, None, :]+offsets, velocity[:, None, :]-offsets), dim=1)
            endpoint = model.rollout(initial_state(trials.flatten(0, 1), position, spin), steps)[:, -1, :2]
            trial_loss = (endpoint-targets.repeat_interleave(4, 0)).square().sum(-1).reshape(len(velocity), 4)
            difference = (trial_loss[:, :2]-trial_loss[:, 2:])/(2*epsilon)
            error = (gradient-difference).abs()
            relative = 2*error/(gradient.abs()+difference.abs()+1e-9)
            reports.append({"perturbation_mps": epsilon, "autograd": gradient.cpu().tolist(),
                            "finite_difference": difference.cpu().tolist(),
                            "finite": bool(torch.isfinite(difference).all() and torch.isfinite(gradient).all()),
                            "max_absolute_error": error.max().item(),
                            "symmetric_relative_error_p95": torch.quantile(relative.flatten(), .95).item()})
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/bouncing_ball/launch_optimization_v1.json"))
    parser.add_argument("--physics-config", type=Path, default=Path("configs/bouncing_ball/chrono_v1.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", nargs=2, type=float, metavar=("X", "Z"))
    parser.add_argument("--initial-velocity", nargs=2, type=float, metavar=("VX", "VZ"))
    parser.add_argument("--iterations", type=int)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("run launch optimization in an AMD GPU allocation")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    config = json.loads(args.config.read_text())
    physics = load_config(args.physics_config)
    if args.iterations is not None:
        config["iterations"] = args.iterations
    if args.initial_velocity:
        config["initial_velocity_mps"] = args.initial_velocity
    model, metadata = load_model(args.checkpoint, "cuda")
    model = model.double()
    steps = round(config["terminal_time_s"]/model.dt)
    if not math.isclose(steps*model.dt, config["terminal_time_s"], abs_tol=1e-9):
        raise ValueError("terminal time must be an integer multiple of the NRD step")
    if args.target:
        specifications = [{"name": "custom_target", "xz_m": args.target, "target_source": "user"}]
    elif args.data:
        specifications = benchmark_targets(args.data, config, steps)
    else:
        raise ValueError("provide --target X Z or --data for the five-target benchmark")
    targets = torch.tensor([s["xz_m"] for s in specifications], dtype=torch.float64, device="cuda")
    starts = targets.new_tensor(config["initial_velocity_mps"]).expand_as(targets).clone()
    bounds = tuple(targets.new_tensor([physics["launch"]["vx_range_mps"][k], physics["launch"]["vz_range_mps"][k]]) for k in range(2))
    with (args.output_dir/"history.jsonl").open("w") as stream:
        def progress(entry):
            stream.write(json.dumps(entry, allow_nan=False)+"\n")
            stream.flush()
            if entry["iteration"] % 10 == 0:
                print(json.dumps({"iteration": entry["iteration"], "distance_m": entry["distance_m"]}), flush=True)
        begin = time.perf_counter()
        result = optimize(model, targets, starts, bounds, steps=steps, iterations=config["iterations"],
                          learning_rate=config["learning_rate"], backtracks=config["backtracks"],
                          max_step=config["max_step_mps"], tolerance=config["nrd_tolerance_m"],
                          position=(physics["scene"]["launch_x_m"], physics["scene"]["launch_z_m"]),
                          spin=physics["launch"]["omega_y_radps"], progress=progress)
    restarts = []
    # Fixed seeds are the same for every target, independent of its source
    # launch. Only stalled cases use them, and final selection uses NRD loss.
    for i in range(len(targets)):
        if result["loss"][i] <= config["nrd_tolerance_m"]**2:
            continue
        restart_starts = targets.new_tensor(config.get("restart_velocities_mps", []))
        if restart_starts.numel() == 0:
            continue
        def restart_progress(entry):
            if entry["iteration"] % 10 == 0:
                print(json.dumps({"restart_case": specifications[i]["name"],
                                  "iteration": entry["iteration"], "distance_m": entry["distance_m"]}), flush=True)
        alternate = optimize(model, targets[i:i+1].expand_as(restart_starts), restart_starts,
                             bounds, steps=steps, iterations=config["iterations"],
                             learning_rate=config["learning_rate"], backtracks=config["backtracks"],
                             max_step=config["max_step_mps"], tolerance=config["nrd_tolerance_m"],
                             position=(physics["scene"]["launch_x_m"], physics["scene"]["launch_z_m"]),
                             spin=physics["launch"]["omega_y_radps"], stop_when_any=True, progress=restart_progress)
        chosen = int(alternate["loss"].argmin())
        selected = bool(alternate["loss"][chosen] < result["loss"][i])
        restarts.append({"case": specifications[i]["name"], "fixed_starts_mps": restart_starts.cpu().tolist(),
                         "final_distances_m": alternate["loss"].sqrt().cpu().tolist(), "chosen": chosen,
                         "improved": selected, "history": alternate["history"]})
        if not selected:
            continue
        offset = result["history"][-1]["iteration"]+1
        for row in alternate["history"]:
            merged = {"iteration": offset+row["iteration"], "restart_case": specifications[i]["name"]}
            for key in ("loss_m2", "distance_m", "velocity_mps", "gradient", "learning_rate"):
                merged[key] = list(result["history"][-1][key])
                merged[key][i] = row[key][chosen]
            result["history"].append(merged)
        for iteration, snapshot in alternate["snapshots"]:
            merged_snapshot = result["after"].cpu().numpy().copy()
            merged_snapshot[i] = snapshot[chosen]
            result["snapshots"].append((offset+iteration, merged_snapshot))
        result["velocity"][i] = alternate["velocity"][chosen]
        result["after"][i] = alternate["after"][chosen]
        result["loss"][i] = alternate["loss"][chosen]
    atomic_json(args.output_dir/"restarts.json", restarts)
    position = (physics["scene"]["launch_x_m"], physics["scene"]["launch_z_m"])
    atomic_json(args.output_dir/"gradient_checks.json", {
        "scope": "Frozen NRD derivatives; larger steps can cross contact frames, not a Chrono derivative check",
        "initial": check_loss_gradients(model, starts, targets, steps, position, physics["launch"]["omega_y_radps"]),
        "optimized": check_loss_gradients(model, result["velocity"], targets, steps, position, physics["launch"]["omega_y_radps"])})
    cases = []
    valid = two_impact_mask(result["after"], model).cpu().tolist()
    for i, spec in enumerate(specifications):
        cases.append({**spec, "initial_velocity_mps": starts[i].cpu().tolist(),
                      "optimized_velocity_mps": result["velocity"][i].cpu().tolist(),
                      "initial_nrd_endpoint_m": result["before"][i, -1, :2].cpu().tolist(),
                      "optimized_nrd_endpoint_m": result["after"][i, -1, :2].cpu().tolist(),
                      "initial_nrd_distance_m": result["history"][0]["distance_m"][i],
                      "optimized_nrd_distance_m": math.sqrt(result["loss"][i].item()),
                      "nrd_two_impacts": valid[i]})
    report = {"config": config, "physics_config": physics, "cases": cases,
              "objective": "L = (x(T)-target_x)^2 + (z(T)-target_z)^2; T fixed",
              "optimizer": "projected gradient descent with exact-NRD backtracking and fixed restarts for stalled cases",
              "dtype": "float64", "checkpoint": str(args.checkpoint),
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "model_source_sha256": hashlib.sha256(Path(__file__).with_name("model.py").read_bytes()).hexdigest(),
              "optimizer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "weights_frozen": all(not p.requires_grad and p.grad is None for p in model.parameters()),
              "model_config": model.config, "host": platform.node(), "job_id": os.environ.get("SLURM_JOB_ID"),
              "target_data_sha256": (hashlib.sha256((args.data/"model_data.npz").read_bytes()).hexdigest()
                                     if args.data and not args.target else None),
              "torch_version": torch.__version__, "iterations": result["history"][-1]["iteration"],
              "elapsed_s": time.perf_counter()-begin,
              "nrd_passed": all(c["nrd_two_impacts"] and c["optimized_nrd_distance_m"] <= config["nrd_tolerance_m"] for c in cases)}
    atomic_json(args.output_dir/"optimization.json", report)
    np.savez_compressed(args.output_dir/"trajectories.npz", time_s=np.arange(steps+1)*model.dt,
                        before=result["before"].cpu().numpy(), after=result["after"].cpu().numpy(),
                        snapshot_iterations=[s[0] for s in result["snapshots"]],
                        snapshots=np.stack([s[1] for s in result["snapshots"]]))
    atomic_json(args.output_dir/"history.json", result["history"])
    with (args.output_dir/"history.jsonl").open("w") as stream:
        for row in result["history"]:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
    print(json.dumps({"nrd_passed": report["nrd_passed"], "cases": cases}, indent=2), flush=True)


if __name__ == "__main__":
    main()
