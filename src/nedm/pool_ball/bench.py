"""Latency of one targeting iteration for a single target (synchronised timers)."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from nedm.pool_ball.model import load_pool
from nedm.pool_ball.optimize import Launch, endpoint_and_jacobian, optimise, rollout_b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    model, packet = load_pool(args.checkpoint, "cuda")
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = json.loads(args.config.read_text())
    scene = {"ball_a_xy_m": [-0.635, 0.0], "ball_b_xy_m": [0.0, 0.0], "radius_m": 0.028575}
    launch = Launch(scene, cfg["optimiser_box"])
    steps = 200
    target = torch.tensor([[0.8, -0.3]], dtype=torch.float64, device="cuda")
    z = torch.tensor(launch.normalise([2.25], [math.radians(25)]), dtype=torch.float64, device="cuda")
    out = {"device": torch.cuda.get_device_name(0), "steps": steps, "model_step_s": model.dt}

    def timed(fn, repeats=10):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            t = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
        return sorted(times)[len(times) // 2]

    with torch.no_grad():
        out["rollout_2s_ms"] = 1e3 * timed(lambda: rollout_b(model, scene, launch, z, steps))
    out["rollout_plus_jacobian_ms"] = 1e3 * timed(lambda: endpoint_and_jacobian(model, scene, launch, z, steps, 1024))
    for method in ("lm", "gd"):
        out[f"{method}_one_iteration_ms"] = 1e3 * timed(lambda: optimise(model, scene, launch, target, z, steps, method=method,
                                                                         iterations=1, tolerance=1e-9), repeats=5)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
