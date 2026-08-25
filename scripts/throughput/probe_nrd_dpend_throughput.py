"""NRD double-pendulum throughput benchmark (study plan 11.5, gate G6).

Measures transitions/second for:
  1. Chrono rigid-body dynamics only (no rendering);
  2. Chrono dynamics + Chrono::Sensor camera at 50 Hz (the data-collection path);
  3. batched NRD transition, encoder off after context, decoder OFF;
  4. batched NRD transition with the decoder ON every step;
  5. the matched state-only NeDM model, batched.

The NRD/state models step a sliding (block_size) window exactly as the RL /
rollout consumers do. Batch sweep shows the scaling story: Chrono is serial per
instance; the learned models amortize across the batch.

Run AFTER training (GPU otherwise contended):
    PYTHONPATH=src conda run -n nedm python scripts/throughput/probe_nrd_dpend_throughput.py \
        --nrd-checkpoint artifacts/training_runs/dpend_nrd_v1/checkpoints/best_val.pt \
        --state-checkpoint artifacts/training_runs/dpend_state_v1/checkpoints/best_val.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from nedm.double_pendulum_data import (  # noqa: E402
    CONTROL_DT_S,
    FrameTap,
    _advance_to_next_boundary,
    build_scene,
    reset_state,
    run_episode,
    sample_action_sequence,
)


def bench_chrono(with_camera: bool, control_steps: int = 500) -> float:
    scene = build_scene(with_camera=with_camera)
    tap = FrameTap(scene.camera) if with_camera else None
    _advance_to_next_boundary(scene)
    rng = np.random.default_rng(3)
    actions = sample_action_sequence(rng, "smooth", control_steps)
    start = time.perf_counter()
    if with_camera:
        run_episode(scene, tap, "bench", "train", "smooth", actions, (0.5, 0.5, 1.0, 0.0), None)
    else:
        reset_state(scene, 0.5, 0.5, 1.0, 0.0)
        for action in actions:
            scene.elbow_torque.SetSetpoint(float(action) * 1.5, scene.system.GetChTime())
            _advance_to_next_boundary(scene)
    elapsed = time.perf_counter() - start
    return control_steps / elapsed  # transitions per second (one instance)


@torch.no_grad()
def bench_model(step_fn, batch: int, iters: int = 50, warmup: int = 5) -> float:
    for _ in range(warmup):
        step_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        step_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return batch * iters / elapsed  # transitions per second across the batch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nrd-checkpoint", type=Path, required=True)
    parser.add_argument("--state-checkpoint", type=Path, default=None)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 32, 256, 1024, 4096])
    parser.add_argument("--output", type=Path, default=Path("artifacts/nrd_eval/dpend_v1/throughput.json"))
    parser.add_argument("--skip-chrono", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict[str, object] = {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}

    from nedm.nrd.model import NRDDynamicsModel  # noqa: E402

    payload = torch.load(args.nrd_checkpoint.resolve(), map_location=device, weights_only=False)
    config, metadata = payload["config"], payload["metadata"]
    model = NRDDynamicsModel(
        state_dim=len(metadata["state_fields"]),
        action_dim=len(metadata["action_fields"]),
        transformer_cfg=config["model"],
        vision_cfg=config.get("vision", {}),
        normalization=metadata["normalization"],
        state_fields=list(metadata["state_fields"]),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    block = int(config["model"]["block_size"])
    state_dim = len(metadata["state_fields"])
    action_dim = len(metadata["action_fields"])
    z2_dim = model.z2_dim

    if not args.skip_chrono:
        chrono_only = bench_chrono(with_camera=False)
        chrono_camera = bench_chrono(with_camera=True)
        results["chrono_dynamics_only_tps"] = chrono_only
        results["chrono_with_camera_tps"] = chrono_camera
        results["chrono_with_camera_rtf"] = chrono_camera * CONTROL_DT_S
        print(f"chrono dynamics only : {chrono_only:10.0f} transitions/s (1 instance)")
        print(f"chrono + camera 50Hz : {chrono_camera:10.0f} transitions/s (1 instance)")

    for label, decode in (("nrd_decoder_off", False), ("nrd_decoder_on", True)):
        results[label] = {}
        for batch in args.batches:
            states = torch.randn(batch, block, state_dim, device=device)
            latents = torch.randn(batch, block, z2_dim, device=device) * 0.01
            actions = torch.randn(batch, block, action_dim, device=device)

            def step() -> None:
                nonlocal states, latents
                next_state, next_z2 = model.predict_next(states, latents, actions)
                if decode:
                    model.decode_latents(next_z2)
                states = torch.cat([states[:, 1:], next_state.unsqueeze(1)], dim=1)
                latents = torch.cat([latents[:, 1:], next_z2.unsqueeze(1)], dim=1)

            tps = bench_model(step, batch)
            results[label][str(batch)] = tps
            print(f"{label:20s} batch {batch:5d}: {tps:12.0f} transitions/s")

    if args.state_checkpoint is not None:
        from nedm.training.model import HMMWVDynamicsModel  # noqa: E402

        spayload = torch.load(args.state_checkpoint.resolve(), map_location=device, weights_only=False)
        sconfig, smeta = spayload["config"], spayload["metadata"]
        smodel = HMMWVDynamicsModel(
            state_dim=len(smeta["state_fields"]),
            action_dim=len(smeta["action_fields"]),
            target_dim=len(smeta["state_fields"]),
            transformer_cfg=sconfig["model"],
            normalization=smeta["normalization"],
            state_fields=list(smeta["state_fields"]),
            dt_s=float(smeta["dt_s"]),
        ).to(device)
        smodel.load_state_dict(spayload["model_state_dict"])
        smodel.eval()
        results["state_only"] = {}
        for batch in args.batches:
            states = torch.randn(batch, block, state_dim, device=device)
            actions = torch.randn(batch, block, action_dim, device=device)

            def sstep() -> None:
                nonlocal states
                delta = smodel.predict_next_delta(states, actions)
                states = torch.cat([states[:, 1:], (states[:, -1] + delta).unsqueeze(1)], dim=1)

            tps = bench_model(sstep, batch)
            results["state_only"][str(batch)] = tps
            print(f"{'state_only':20s} batch {batch:5d}: {tps:12.0f} transitions/s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
