"""Choose ball A's launch (vx, vy) so that ball B is at a target after t seconds,
by differentiating the frozen pool NRD rollout.

Search variables are normalised speed and cut angle z = (u_n, phi_n) in
[-1, 1]^2; (vx, vy) is computed from them inside the autograd graph
(aim theta = asin((2R/D) sin phi) from the A->B line). Loss
L = |p_B(t) - target|^2. Two arms from the same fixed start grid:
  lm  Levenberg-Marquardt on the 2x2 Jacobian (primary since the validation
      rehearsals of 2026-10-02, where it converged on 26/30 and 29/30
      targets against 15/30 and 17/30 for gd);
  gd  projected gradient descent with batched backtracking, a step cap,
      an adaptive learning rate and a stall rule (the originally planned arm,
      as in the bouncing-ball study and Newton's diffsim example).
Selection among converged starts is predeclared and model-only (fewest B
cushion switch firings, then least sensitive); if no start converges, the
smallest model residual is taken. Targets come from the sealed cohort; their
source launches are never given to the optimiser. Data-only baselines: the
nearest training shot, and a local linear fit over the 12 training launches
nearest to it.
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

from nedm.pool_ball.campaign import atomic_json
from nedm.pool_ball.evaluate import cut_angle_deg, load_data
from nedm.pool_ball.model import BALL, initial_state, load_pool


class Launch:
    """z in [-1,1]^2 -> (vx, vy), differentiable."""

    def __init__(self, scene, box):
        a, b = np.asarray(scene["ball_a_xy_m"], float), np.asarray(scene["ball_b_xy_m"], float)
        d = b - a
        self.base = math.atan2(d[1], d[0])
        self.k = 2 * scene["radius_m"] / np.linalg.norm(d)
        self.u_lo, self.u_hi = box["speed_mps"]
        self.phi_max = math.radians(box["max_cut_deg"])

    def speed_cut(self, z):
        return self.u_lo + (z[..., 0] + 1) / 2 * (self.u_hi - self.u_lo), z[..., 1] * self.phi_max

    def velocity(self, z):
        u, phi = self.speed_cut(z)
        direction = self.base + torch.asin(self.k * torch.sin(phi))
        return torch.stack((u * torch.cos(direction), u * torch.sin(direction)), -1)

    def normalise(self, speed, cut_rad):
        return np.stack((2 * (np.asarray(speed) - self.u_lo) / (self.u_hi - self.u_lo) - 1, np.asarray(cut_rad) / self.phi_max), -1)


def rollout_b(model, scene, launch, z, steps, details=False):
    velocity = launch.velocity(z)
    if not details:
        return model.rollout(initial_state(scene, velocity), steps)[..., -1, BALL:BALL + 2]
    trajectory, gates = model.rollout(initial_state(scene, velocity), steps, return_gates=True)
    return trajectory, gates


def model_checks(model, trajectory, gates, scene):
    """Model-only validity: B moved, and (structured) exactly one A-B gate firing."""
    moved = (trajectory[..., -1, BALL:BALL + 2] - trajectory.new_tensor(scene["ball_b_xy_m"])).norm(dim=-1) > 0.01
    if gates is not None and model.contact in ("structured", "structured4"):
        ab = gates[..., 0].sum(-1)
        b_cushion = gates[..., 2].sum(-1) if model.contact == "structured" else gates[..., 5:9].sum((-1, -2))
        return moved & (ab >= 1) & (ab <= 2), b_cushion
    return moved, torch.zeros_like(moved, dtype=trajectory.dtype)


def endpoint_and_jacobian(model, scene, launch, z, steps, chunk):
    """B's position at t and its 2x2 Jacobian w.r.t. z, in independent chunks
    (autograd memory grows with chunk x steps)."""
    ends, jacs = [], []
    for part in z.split(chunk):
        zz = part.detach().requires_grad_(True)
        e = rollout_b(model, scene, launch, zz, steps)
        rows = [torch.autograd.grad(e[:, k].sum(), zz, retain_graph=k == 0)[0] for k in (0, 1)]
        ends.append(e.detach())
        jacs.append(torch.stack(rows, 1).detach())
    return torch.cat(ends), torch.cat(jacs)


def optimise(model, scene, launch, targets, starts, steps, *, method, iterations, tolerance, max_step=0.15,
             learning_rate=0.3, backtracks=16, record=None, chunk=256):
    """Batched over M = len(targets) problems. Returns final z, loss, history."""
    z = starts.clone()
    rate = z.new_full((len(z),), learning_rate)
    lam = z.new_full((len(z),), 1e-2)
    factors = 2.0 ** -torch.arange(backtracks, dtype=z.dtype, device=z.device)
    stalled = torch.zeros(len(z), dtype=torch.int64, device=z.device)
    history = []
    for iteration in range(iterations + 1):
        endpoint, jac = endpoint_and_jacobian(model, scene, launch, z, steps, chunk)
        residual = endpoint - targets
        loss = residual.square().sum(-1)
        gradient = 2 * torch.einsum("mi,mij->mj", residual, jac)
        history.append({"iteration": iteration, "distance_m": loss.sqrt().cpu().tolist()})
        if iteration % 10 == 0:
            d = loss.sqrt()
            print(json.dumps({"method": method, "iteration": iteration, "median_mm": float(d.median() * 1e3),
                              "within_tol": int((d <= tolerance).sum()), "of": len(d)}), flush=True)
        active = (loss > tolerance ** 2) & (stalled < 6)
        final = iteration == iterations or not active.any()
        if record is not None:
            record(iteration, z.detach(), loss, final)
        if final:
            break
        with torch.no_grad():
            if method == "lm":
                jtj = jac.transpose(1, 2) @ jac
                damp = jtj + lam[:, None, None] * torch.diag_embed(torch.diagonal(jtj, dim1=1, dim2=2).clamp_min(1e-12))
                step = -torch.linalg.solve(damp, (jac.transpose(1, 2) @ residual[..., None]))[..., 0]
            else:
                step = -rate[:, None] * gradient
            step = step * (max_step / step.norm(dim=-1).clamp_min(max_step))[:, None]
            candidate = (z[:, None, :] + step[:, None, :] * factors[None, :, None]).clamp(-1, 1)
            flat = candidate.reshape(-1, 2)
            c_end = torch.cat([rollout_b(model, scene, launch, part, steps) for part in flat.split(32768)])
            c_loss = (c_end - targets.repeat_interleave(backtracks, 0)).square().sum(-1).reshape(len(z), backtracks)
            decrease = (gradient[:, None, :] * (candidate - z[:, None, :])).sum(-1)
            ok = torch.isfinite(c_loss) & (c_loss < loss[:, None]) & (c_loss <= loss[:, None] + 1e-4 * decrease)
            accepted = ok.any(-1) & active
            chosen = ok.to(torch.int64).argmax(-1)
            pick = torch.arange(len(z), device=z.device)
            new_loss = c_loss[pick, chosen]
            z = torch.where(accepted[:, None], candidate[pick, chosen], z)
            if method == "lm":
                lam = torch.where(accepted, (lam * 0.3).clamp_min(1e-6), (lam * 10).clamp_max(1e6))
            else:
                rate = torch.where(accepted, (rate * factors[chosen] * 1.5).clamp(max=learning_rate), rate * 0.25)
            meaningful = accepted & ((loss.sqrt() - new_loss.clamp_min(0).sqrt()) > 1e-6)
            stalled = torch.where(meaningful, torch.zeros_like(stalled), stalled + active.to(stalled.dtype))
    with torch.no_grad():
        endpoint = rollout_b(model, scene, launch, z, steps)
    return z.detach(), (endpoint - targets).square().sum(-1), history


def choose_targets(data, config, count, seed, split=2):
    """Seeded draw from eligible sealed-cohort episodes, half with no B cushion
    hit before t and half with one."""
    t = data["index"]["campaign"]["target_time_s"]
    box = config["optimiser_box"]
    speed = data["launches"].norm(dim=-1).cpu().numpy()
    cut = cut_angle_deg(data["index"]["config"], [e["aim_rad"] for e in data["episodes"]])
    # Only targets whose source launch lies inside the optimiser's search box.
    inside = (speed >= box["speed_mps"][0]) & (speed <= box["speed_mps"][1]) & (np.abs(cut) <= box["max_cut_deg"])
    eligible = np.flatnonzero(data["eligible"].cpu().numpy() & (data["splits"].cpu().numpy() == split) & inside)
    def cushions(i):
        return sum(1 for e in data["episodes"][i]["events"] if e["kind"] == "start" and e["pair"].startswith("B_") and e["time_s"] < t)
    groups = {0: [i for i in eligible if cushions(i) == 0], 1: [i for i in eligible if cushions(i) == 1]}
    rng = np.random.default_rng(seed)
    picks = [rng.permutation(groups[g]) for g in (0, 1)]
    order = [int(p[j]) for j in range(max(len(p) for p in picks)) for p in picks if j < len(p)]
    k = round(t / data["record_dt"])
    chosen = order[:count]
    return [{"episode": i, "episode_id": data["episodes"][i]["episode_id"],
             "xy_m": data["states"][i, k, BALL:BALL + 2].cpu().tolist(), "b_cushions_before_t": cushions(i),
             "source_launch_mps": data["launches"][i].cpu().tolist(),
             "source_cut_deg": float(cut_angle_deg(data["index"]["config"], data["episodes"][i]["aim_rad"]))} for i in chosen]


def baselines(train_data, targets_xy, k_neighbours=12):
    """Data-only inverse maps from training shots. Nearest: the launch whose
    B(t) is closest to the target. Local linear: around that launch, fit
    B(t) ~ A [vx, vy] + c on the k nearest launches (in launch space, so the
    fit stays on one cushion branch) and solve for the target; fall back to
    the nearest launch if the solution leaves the neighbourhood."""
    t = train_data["index"]["campaign"]["target_time_s"]
    k = round(t / train_data["record_dt"])
    train = (train_data["splits"] == 0).cpu().numpy()
    ends = train_data["states"][:, k, BALL:BALL + 2].cpu().numpy()[train]
    launches = train_data["launches"].cpu().numpy()[train]
    scale = launches.std(0)
    out = []
    for target in np.asarray(targets_xy):
        d = np.linalg.norm(ends - target, axis=1)
        i0 = int(np.argmin(d))
        near = np.argsort(np.linalg.norm((launches - launches[i0]) / scale, axis=1))[:k_neighbours]
        design = np.c_[launches[near], np.ones(len(near))]
        coef, *_ = np.linalg.lstsq(design, ends[near], rcond=None)
        a, c = coef[:2].T, coef[2]
        radius = np.linalg.norm((launches[near] - launches[i0]) / scale, axis=1).max()
        try:
            solved = np.linalg.solve(a, target - c)
            if np.linalg.norm((solved - launches[i0]) / scale) > 2 * radius:
                solved = launches[i0]
        except np.linalg.LinAlgError:
            solved = launches[i0]
        out.append({"nearest_launch_mps": launches[i0].tolist(), "nearest_endpoint_miss_m": float(d[i0]),
                    "local_linear_launch_mps": np.asarray(solved).tolist()})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True, help="campaign used for training (baselines)")
    parser.add_argument("--target-data", type=Path, required=True, help="sealed cohort (targets)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-max-episodes", type=int, help="restrict the baselines to the model's training subset")
    parser.add_argument("--target-split", type=int, default=2, help="2 = sealed/test (default); 1 = validation (rehearsal)")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("AMD allocation required")
    cfg = json.loads(args.config.read_text())
    torch.set_num_threads(1)
    model, packet = load_pool(args.checkpoint, "cuda")
    for p in model.parameters():
        p.requires_grad_(False)
    target_data = load_data(args.target_data, "cuda")
    scene = target_data["scene"]
    steps = target_data["target_step"]
    launch = Launch(scene, cfg["optimiser_box"])
    specs = choose_targets(target_data, cfg, cfg["targets"], cfg["target_seed"], args.target_split)
    targets = torch.tensor([s["xy_m"] for s in specs], dtype=torch.float64, device="cuda")
    grid = [(u, math.radians(c)) for u in cfg["start_speeds_mps"] for c in cfg["start_cuts_deg"]]
    start_z = torch.tensor(launch.normalise([g[0] for g in grid], [g[1] for g in grid]), dtype=torch.float64, device="cuda")
    n_t, n_s = len(targets), len(start_z)
    flat_targets = targets.repeat_interleave(n_s, 0)
    flat_starts = start_z.repeat(n_t, 1)
    args.output.mkdir(parents=True, exist_ok=False)
    results = {}
    for method in cfg["methods"]:
        began = time.perf_counter()
        trace = []

        def record(iteration, z, loss, final):
            if iteration % cfg.get("snapshot_every", 5) == 0 or final:
                trace.append((iteration, z.cpu().numpy().copy(), loss.cpu().numpy().copy()))
        chunk = cfg.get("autograd_chunk", 1024 if model.context == 1 else 128)
        z, loss, history = optimise(model, scene, launch, flat_targets, flat_starts, steps, method=method,
                                    iterations=cfg["iterations"][method], tolerance=cfg["nrd_tolerance_m"],
                                    learning_rate=cfg.get("learning_rate", 0.3), record=record, chunk=chunk)
        elapsed = time.perf_counter() - began
        with torch.no_grad():
            outs = [rollout_b(model, scene, launch, part, steps, details=True) for part in z.split(32768)]
            trajectory = torch.cat([o[0] for o in outs])
            gates = torch.cat([o[1] for o in outs]) if outs[0][1] is not None else None
            valid, b_cushion = model_checks(model, trajectory, gates, scene)
        _, jac = endpoint_and_jacobian(model, scene, launch, z, steps, chunk)
        sensitivity = torch.linalg.matrix_norm(jac, ord=2)
        distance = loss.sqrt()
        cases = []
        for i, spec in enumerate(specs):
            sl = slice(i * n_s, (i + 1) * n_s)
            d, ok, cush, sens = distance[sl], valid[sl], b_cushion[sl], sensitivity[sl]
            converged = (d <= cfg["nrd_tolerance_m"]) & ok
            # Predeclared, model-only pick: converged; fewest B cushion gate
            # firings; least sensitive. Otherwise the smallest residual.
            if converged.any():
                score = cush * 1e3 + sens / (sens.max() + 1e-12)
                score = torch.where(converged, score, torch.full_like(score, float("inf")))
                pick = int(score.argmin())
            else:
                pick = int(torch.where(ok, d, d + 1e3).argmin())
            vel = launch.velocity(z[sl][pick]).tolist()
            u, phi = launch.speed_cut(z[sl][pick])
            branches = []
            for j in torch.nonzero(converged).flatten().tolist():
                if all(float((z[sl][j] - z[sl][b]).norm()) > 0.02 for b in branches):
                    branches.append(j)
            cases.append({**spec, "launch_mps": vel, "speed_mps": float(u), "cut_deg": math.degrees(float(phi)),
                          "nrd_distance_m": float(d[pick]), "converged": bool(converged[pick]), "start_index": pick,
                          "converged_starts": int(converged.sum()),
                          "branches": [{"start_index": j, "launch_mps": launch.velocity(z[sl][j]).tolist(), "nrd_distance_m": float(d[j]),
                                        "b_cushion_gate_steps": float(cush[j])} for j in branches]})
        snapshots = [{"iteration": it, "launch_mps": launch.velocity(torch.as_tensor(zs, device="cuda")).cpu().numpy()
                      .reshape(n_t, n_s, 2)[np.arange(n_t), [c["start_index"] for c in cases]].tolist(),
                      "nrd_distance_m": np.sqrt(ls).reshape(n_t, n_s)[np.arange(n_t), [c["start_index"] for c in cases]].tolist()}
                     for it, zs, ls in trace]
        results[method] = {"cases": cases, "elapsed_s": elapsed, "iterations": history[-1]["iteration"], "snapshots": snapshots,
                           "nrd_converged": sum(c["converged"] for c in cases)}
        print(json.dumps({"method": method, "elapsed_s": round(elapsed, 1), "iterations": history[-1]["iteration"],
                          "converged": results[method]["nrd_converged"], "of": len(cases)}), flush=True)
    train_data = load_data(args.train_data, "cpu", max_episodes=args.train_max_episodes)
    base = baselines(train_data, [s["xy_m"] for s in specs])
    report = {"config": cfg, "checkpoint": str(args.checkpoint),
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "target_data_sha256": target_data["index"]["model_data_sha256"], "targets": specs, "results": results,
              "baselines": base, "host": platform.node(), "job_id": os.environ.get("SLURM_JOB_ID"),
              "objective": "L = |p_B(t) - target|^2, t fixed", "source_launch_used_by_optimiser": False,
              "physics_config": target_data["index"]["config"], "target_time_s": target_data["index"]["campaign"]["target_time_s"]}
    atomic_json(args.output / "optimization.json", report)


if __name__ == "__main__":
    main()
