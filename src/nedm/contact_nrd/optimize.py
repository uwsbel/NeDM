"""Gradient targeting through the frozen unified contact NRD, for any system.

Choose the launch so that the system's target body is at a target position
after t seconds. Search variables z in [-1, 1]^2 map to the launch inside the
autograd graph:
  pool  (speed, cut angle) -> (vx, vy) of ball A  (as in the pool study)
  ball  (vx, vz) of the bouncing ball, scaled to its box
Loss |p_target_body(t) - target|^2 over the plane of motion. Arms:
Levenberg-Marquardt on the 2x2 Jacobian (primary) and projected gradient
descent with backtracking. Fixed start grid; model-only pick among converged
starts (fewest target-body switch firings with fixed bodies, then least
sensitive; otherwise smallest residual). Targets come from the sealed cohort
and its source launches are never given to the optimiser. Data-only
baselines from the training split: nearest shot, and a local linear fit over
the 12 training launches nearest to it. Output follows the pool study's
optimization.json layout so the existing Chrono replay can read it.
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

from nedm.contact_nrd.evaluate import load_data
from nedm.contact_nrd.model_v2 import load_any as load


class Launch:
    def __init__(self, system, box):
        self.system = system
        self.box = box
        cfg = system["physics_config"]["scene"]
        if system["name"] == "pool":
            a, b = np.asarray(cfg["ball_a_xy_m"], float), np.asarray(cfg["ball_b_xy_m"], float)
            d = b - a
            self.base = math.atan2(d[1], d[0])
            self.k = 2 * cfg["radius_m"] / np.linalg.norm(d)
            self.u_lo, self.u_hi = box["speed_mps"]
            self.phi_max = math.radians(box["max_cut_deg"])
        else:
            self.lo = np.array([box["vx_mps"][0], box["vz_mps"][0]])
            self.hi = np.array([box["vx_mps"][1], box["vz_mps"][1]])

    def velocity(self, z):
        if self.system["name"] == "pool":
            u = self.u_lo + (z[..., 0] + 1) / 2 * (self.u_hi - self.u_lo)
            direction = self.base + torch.asin(self.k * torch.sin(z[..., 1] * self.phi_max))
            return torch.stack((u * torch.cos(direction), u * torch.sin(direction)), -1)
        lo, hi = z.new_tensor(self.lo), z.new_tensor(self.hi)
        return lo + (z + 1) / 2 * (hi - lo)

    def normalise(self, launch):
        launch = np.asarray(launch, float)
        if self.system["name"] == "pool":
            u = np.linalg.norm(launch, axis=-1)
            aim = np.arctan2(launch[..., 1], launch[..., 0]) - self.base
            phi = np.arcsin(np.clip(np.sin(aim) / self.k, -1, 1))
            return np.stack((2 * (u - self.u_lo) / (self.u_hi - self.u_lo) - 1, phi / self.phi_max), -1)
        return 2 * (launch - self.lo) / (self.hi - self.lo) - 1

    def initial_state(self, velocity):
        """[.., 2] launch -> [.., D, 9] initial moving-body states (autograd link kept)."""
        cfg = self.system["physics_config"]
        lead = velocity.shape[:-1]
        zeros3 = velocity.new_zeros(*lead, 3)
        if self.system["name"] == "pool":
            R = cfg["scene"]["radius_m"]
            a = velocity.new_tensor([*cfg["scene"]["ball_a_xy_m"], R]).expand(*lead, 3)
            b = velocity.new_tensor([*cfg["scene"]["ball_b_xy_m"], R]).expand(*lead, 3)
            va = torch.cat((velocity, velocity.new_zeros(*lead, 1)), -1)
            return torch.stack((torch.cat((a, va, zeros3), -1), torch.cat((b, zeros3, zeros3), -1)), -2)
        scene = cfg["scene"]
        p = velocity.new_tensor([scene["launch_x_m"], 0.0, scene["launch_z_m"]]).expand(*lead, 3)
        v = torch.stack((velocity[..., 0], velocity.new_zeros(lead), velocity[..., 1]), -1)
        return torch.cat((p, v, zeros3), -1).unsqueeze(-2)


def plane_axes(system):
    return [0, 1] if system["name"] == "pool" else [0, 2]


def endpoint(model, launch, z, steps, slot, axes):
    out = model.rollout(launch.initial_state(launch.velocity(z)), steps)
    return out[..., -1, slot, :][..., axes]


def endpoint_and_jacobian(model, launch, z, steps, slot, axes, chunk):
    ends, jacs = [], []
    for part in z.split(chunk):
        zz = part.detach().requires_grad_(True)
        e = endpoint(model, launch, zz, steps, slot, axes)
        rows = [torch.autograd.grad(e[:, k].sum(), zz, retain_graph=k == 0)[0] for k in (0, 1)]
        ends.append(e.detach())
        jacs.append(torch.stack(rows, 1).detach())
    return torch.cat(ends), torch.cat(jacs)


def optimise(model, launch, targets, starts, steps, slot, axes, *, method, iterations, tolerance, chunk,
             max_step=0.15, learning_rate=0.3, backtracks=16, record=None):
    z = starts.clone()
    rate = z.new_full((len(z),), learning_rate)
    lam = z.new_full((len(z),), 1e-2)
    factors = 2.0 ** -torch.arange(backtracks, dtype=z.dtype, device=z.device)
    stalled = torch.zeros(len(z), dtype=torch.int64, device=z.device)
    history = []
    for iteration in range(iterations + 1):
        end, jac = endpoint_and_jacobian(model, launch, z, steps, slot, axes, chunk)
        residual = end - targets
        loss = residual.square().sum(-1)
        gradient = 2 * torch.einsum("mi,mij->mj", residual, jac)
        history.append({"iteration": iteration, "distance_m": loss.sqrt().cpu().tolist()})
        active = (loss > tolerance ** 2) & (stalled < 6)
        final = iteration == iterations or not active.any()
        if record is not None:
            record(iteration, z.detach(), loss, final)
        if iteration % 10 == 0:
            print(json.dumps({"method": method, "iteration": iteration, "median_mm": float(loss.sqrt().median() * 1e3),
                              "within_tol": int((loss <= tolerance ** 2).sum()), "of": len(z)}), flush=True)
        if final:
            break
        with torch.no_grad():
            if method == "lm":
                jtj = jac.transpose(1, 2) @ jac
                damp = jtj + lam[:, None, None] * torch.diag_embed(torch.diagonal(jtj, dim1=1, dim2=2).clamp_min(1e-12))
                step = -torch.linalg.solve(damp, jac.transpose(1, 2) @ residual[..., None])[..., 0]
            else:
                step = -rate[:, None] * gradient
            step = step * (max_step / step.norm(dim=-1).clamp_min(max_step))[:, None]
            cand = (z[:, None, :] + step[:, None, :] * factors[None, :, None]).clamp(-1, 1)
            flat = cand.reshape(-1, 2)
            c_end = torch.cat([endpoint(model, launch, part, steps, slot, axes) for part in flat.split(32768)])
            c_loss = (c_end - targets.repeat_interleave(backtracks, 0)).square().sum(-1).reshape(len(z), backtracks)
            decrease = (gradient[:, None, :] * (cand - z[:, None, :])).sum(-1)
            ok = torch.isfinite(c_loss) & (c_loss < loss[:, None]) & (c_loss <= loss[:, None] + 1e-4 * decrease)
            accepted = ok.any(-1) & active
            chosen = ok.to(torch.int64).argmax(-1)
            pick = torch.arange(len(z), device=z.device)
            new_loss = c_loss[pick, chosen]
            z = torch.where(accepted[:, None], cand[pick, chosen], z)
            if method == "lm":
                lam = torch.where(accepted, (lam * 0.3).clamp_min(1e-6), (lam * 10).clamp_max(1e6))
            else:
                rate = torch.where(accepted, (rate * factors[chosen] * 1.5).clamp(max=learning_rate), rate * 0.25)
            meaningful = accepted & ((loss.sqrt() - new_loss.clamp_min(0).sqrt()) > 1e-6)
            stalled = torch.where(meaningful, torch.zeros_like(stalled), stalled + active.to(stalled.dtype))
    with torch.no_grad():
        final_end = torch.cat([endpoint(model, launch, part, steps, slot, axes) for part in z.split(32768)])
    return z.detach(), (final_end - targets).square().sum(-1), history


def choose_targets(data, launch, box, count, seed, split, axes):
    system = data["system"]
    t_index = round(system["target_time_s"] / data["record_dt"])
    body = system["target_body"]
    slot = [k for k, b in enumerate(system["bodies"]) if b["moving"]].index(body)
    launches = data["launches"].cpu().numpy()
    z = launch.normalise(launches)
    inside = (np.abs(z) <= 1).all(-1)
    eligible = np.flatnonzero(data["eligible"].cpu().numpy() & (data["splits"].cpu().numpy() == split) & inside
                              & (data["lengths"].cpu().numpy() > t_index))
    rng = np.random.default_rng(seed)
    if system["name"] == "pool":
        def cushions(i):
            return sum(1 for e in data["episodes"][i]["events"] if e["kind"] == "start" and e["pair"].startswith("B_")
                       and e["time_s"] < system["target_time_s"])
        groups = [rng.permutation([i for i in eligible if cushions(i) == c]) for c in (0, 1)]
        order = [int(g[j]) for j in range(max(len(g) for g in groups)) for g in groups if j < len(g)]
    else:
        order = [int(i) for i in rng.permutation(eligible)]
        cushions = lambda i: 0
    out = []
    for i in order[:count]:
        out.append({"episode": i, "episode_id": data["episodes"][i]["episode_id"],
                    "xy_m": data["states"][i, t_index, slot, axes].cpu().tolist(),
                    "b_cushions_before_t": cushions(i), "source_launch_mps": launches[i].tolist()})
    return out


def baselines(train, targets_xy, axes, k_neighbours=12):
    system = train["system"]
    t_index = round(system["target_time_s"] / train["record_dt"])
    slot = [k for k, b in enumerate(system["bodies"]) if b["moving"]].index(system["target_body"])
    keep = (train["splits"] == 0).cpu().numpy() & (train["lengths"].cpu().numpy() > t_index)
    ends = train["states"][:, t_index, slot][:, axes].cpu().numpy()[keep]
    launches = train["launches"].cpu().numpy()[keep]
    scale = launches.std(0)
    out = []
    for target in np.asarray(targets_xy):
        d = np.linalg.norm(ends - target, axis=1)
        i0 = int(np.argmin(d))
        near = np.argsort(np.linalg.norm((launches - launches[i0]) / scale, axis=1))[:k_neighbours]
        design = np.c_[launches[near], np.ones(len(near))]
        coef, *_ = np.linalg.lstsq(design, ends[near], rcond=None)
        radius = np.linalg.norm((launches[near] - launches[i0]) / scale, axis=1).max()
        try:
            solved = np.linalg.solve(coef[:2].T, target - coef[2])
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
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--target-data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-split", type=int, default=2)
    parser.add_argument("--train-max-episodes", type=int)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("AMD allocation required")
    cfg = json.loads(args.config.read_text())
    torch.set_num_threads(1)
    model, packet = load(args.checkpoint, "cuda")
    for p in model.parameters():
        p.requires_grad_(False)
    data = load_data(args.target_data, "cuda", model_step_s=model.dt)   # the model's own step (10 or 20 ms)
    system = data["system"]
    axes = plane_axes(system)
    launch = Launch(system, cfg["box"])
    steps = data["target_step"]
    slot = model.moving.index(system["target_body"])
    specs = choose_targets(data, launch, cfg["box"], cfg["targets"], cfg["target_seed"], args.target_split, axes)
    targets = torch.tensor([s["xy_m"] for s in specs], dtype=torch.float64, device="cuda")
    starts = torch.tensor(cfg["starts_normalised"], dtype=torch.float64, device="cuda")
    n_t, n_s = len(targets), len(starts)
    flat_targets, flat_starts = targets.repeat_interleave(n_s, 0), starts.repeat(n_t, 1)
    fixed_pairs = [p for p, (i, j) in enumerate(system["pairs"]) if i == system["target_body"] and not system["bodies"][j]["moving"]]
    mm_pairs = [p for p, (i, j) in enumerate(system["pairs"]) if system["bodies"][j]["moving"]]
    args.output.mkdir(parents=True, exist_ok=False)
    results = {}
    for method in cfg["methods"]:
        began, trace = time.perf_counter(), []

        def record(iteration, z, loss, final):
            if iteration % cfg.get("snapshot_every", 5) == 0 or final:
                trace.append((iteration, z.cpu().numpy().copy(), loss.cpu().numpy().copy()))
        z, loss, history = optimise(model, launch, flat_targets, flat_starts, steps, slot, axes, method=method,
                                    iterations=cfg["iterations"][method], tolerance=cfg["nrd_tolerance_m"],
                                    chunk=cfg.get("autograd_chunk", 1024), record=record)
        elapsed = time.perf_counter() - began
        with torch.no_grad():
            parts = [model.rollout(launch.initial_state(launch.velocity(part)), steps, return_gates=True) for part in z.split(32768)]
            gates = torch.cat([p[1] for p in parts])                                  # [M, steps, P]
            fixed_firings = gates[..., fixed_pairs].sum((-1, -2)) if fixed_pairs else torch.zeros(len(z), device=z.device)
            ok = torch.ones(len(z), dtype=torch.bool, device=z.device)
            if mm_pairs:
                firings = gates[..., mm_pairs].sum((-1, -2))
                ok = (firings >= 1) & (firings <= 2)
        _, jac = endpoint_and_jacobian(model, launch, z, steps, slot, axes, cfg.get("autograd_chunk", 1024))
        sens = torch.linalg.matrix_norm(jac, ord=2)
        dist = loss.sqrt()
        cases = []
        for i, spec in enumerate(specs):
            sl = slice(i * n_s, (i + 1) * n_s)
            d, good, fire, se = dist[sl], ok[sl], fixed_firings[sl], sens[sl]
            conv = (d <= cfg["nrd_tolerance_m"]) & good
            if conv.any():
                score = torch.where(conv, fire * 1e3 + se / (se.max() + 1e-12), torch.full_like(se, float("inf")))
                pick = int(score.argmin())
            else:
                pick = int(torch.where(good, d, d + 1e3).argmin())
            branches = []
            for j in torch.nonzero(conv).flatten().tolist():
                if all(float((z[sl][j] - z[sl][b]).norm()) > 0.02 for b in branches):
                    branches.append(j)
            cases.append({**spec, "launch_mps": launch.velocity(z[sl][pick]).tolist(), "nrd_distance_m": float(d[pick]),
                          "converged": bool(conv[pick]), "start_index": pick, "converged_starts": int(conv.sum()),
                          "branches": [{"start_index": j, "launch_mps": launch.velocity(z[sl][j]).tolist(),
                                        "nrd_distance_m": float(d[j])} for j in branches]})
        picks = [c["start_index"] for c in cases]
        snapshots = [{"iteration": it, "launch_mps": launch.velocity(torch.as_tensor(zs, device="cuda")).cpu().numpy()
                      .reshape(n_t, n_s, 2)[np.arange(n_t), picks].tolist(),
                      "nrd_distance_m": np.sqrt(ls).reshape(n_t, n_s)[np.arange(n_t), picks].tolist()} for it, zs, ls in trace]
        results[method] = {"cases": cases, "elapsed_s": elapsed, "iterations": history[-1]["iteration"], "snapshots": snapshots,
                           "nrd_converged": sum(c["converged"] for c in cases)}
        print(json.dumps({"method": method, "elapsed_s": round(elapsed, 1), "converged": results[method]["nrd_converged"], "of": n_t}), flush=True)
    train = load_data(args.train_data, "cpu", max_episodes=args.train_max_episodes)
    report = {"config": cfg, "checkpoint": str(args.checkpoint), "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "target_data_sha256": data["index"]["data_sha256"], "targets": specs, "results": results,
              "baselines": baselines(train, [s["xy_m"] for s in specs], axes), "host": platform.node(),
              "job_id": os.environ.get("SLURM_JOB_ID"), "source_launch_used_by_optimiser": False,
              "physics_config": system["physics_config"], "target_time_s": system["target_time_s"], "system": system["name"],
              "plane_axes": axes, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output / "optimization.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
