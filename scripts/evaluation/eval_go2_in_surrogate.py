"""The same policy, the same references, the same horizon -- INSIDE the frozen model.

THIS IS THE LEVEL-3 COMPARISON. "Does a policy trained inside the frozen dynamics
model still work in the high-fidelity simulator" is answered by how much worse it
is OUTSIDE the model than INSIDE it. Everything else is context.

WHY THIS WAS NOT THE HEADLINE FROM THE START, RECORDED BECAUSE IT MATTERS. The
replay baseline -- Chrono driven by the reference's own recorded commands -- was
introduced as a CONTROL: the references ARE recordings, so replaying them should
reproduce them, and the residual is what the eval harness costs relative to the
collector. It was then reframed as a baseline the policy ought to beat, and that
reframe became the primary verdict criterion. Beating it means being better at
reproducing a recording than the process that produced the recording. That is a
real measurement and it is not the transfer question. Caught by Kyle.

The BEAT result stands as what it is: the policy corrects harness disturbances
that an open-loop replay cannot. It is no longer the headline.

MATCHED POPULATIONS OR THE NUMBER IS MEANINGLESS. Same checkpoint, same eight
reference indices in the same file, same 6 s horizon, same start offset. The
~0.0158 m in the PPO log is NOT comparable -- different reference set, different
horizon, and a mean over all forty during training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from nedm.rl.go2_tracking_env import Go2NeuralTrackingEnv, go2_default_env_cfg  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--policy-checkpoint", type=Path, required=True)
    ap.add_argument("--reference-path", type=Path, required=True)
    ap.add_argument("--reference-indices", type=str, required=True)
    ap.add_argument("--horizon-s", type=float, default=6.0)
    ap.add_argument("--terrain", type=str, default="flat")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    ids = [int(v) for v in a.reference_indices.split(",")]
    run_cfg = json.loads((a.run_dir / "env_cfg.json").read_text())
    cfg = go2_default_env_cfg()
    # Take the reward and the action space from the RUN, so the policy is scored
    # in the environment it optimised rather than in today's module defaults.
    for key in ("reward", "termination", "action_low", "action_high", "action_scale",
                "action_repeat", "obs_history_steps", "reference_preview_steps"):
        if key in run_cfg:
            cfg[key] = run_cfg[key]
    cfg.update({
        "num_envs": len(ids),
        "device": a.device,
        "dynamics_checkpoint": run_cfg["dynamics_checkpoint"],
        "reference_path": str(a.reference_path),
        "terrain": a.terrain,
        "auto_reset": False,
        "max_episode_steps": int(round(a.horizon_s / (0.01 * int(cfg["action_repeat"])))),
        "initial_reference_ids": ids,
    })
    env = Go2NeuralTrackingEnv(cfg)

    # THE NN ENV IGNORES cfg["initial_reference_ids"] -- that key is honoured only
    # by the Chrono env's reset(). Its reset_idx DOES take reference_ids, so the
    # assignment has to be made explicitly. Without this the references are
    # sampled at random WITH REPLACEMENT and the per-family labels below are
    # attached to whatever each env happened to draw.
    #
    # Caught because three families reported identical errors to four decimals,
    # which is what duplicate references look like: envs 0, 5 and 6 had all drawn
    # reference 18. A silently wrong assignment that produces plausible numbers is
    # the failure this project has spent a night on; here the tell was that they
    # were TOO plausible -- identical.
    want = torch.tensor(ids, dtype=torch.long, device=env.device)
    env.reset_idx(torch.arange(len(ids), device=env.device), reference_ids=want)
    env._compute_observations()
    assert env.ref_ids.tolist() == ids, f"reference assignment failed: {env.ref_ids.tolist()}"
    print(f"reference assignment verified: {env.ref_ids.tolist()}")

    from rsl_rl.runners import OnPolicyRunner
    train_cfg = json.loads((a.run_dir / "train_cfg.json").read_text())
    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=a.device)
    runner.load(str(a.policy_checkpoint))
    policy = runner.get_inference_policy(device=a.device)

    obs = env.obs_buf
    assert env.ref_ids.tolist() == ids, "reference assignment was lost after policy load"
    per_step = []
    for _ in range(env.max_episode_length):
        with torch.no_grad():
            action = policy(obs)
        obs, _, dones, extras = env.step(action)
        ref_pose = env._reference_state_pose()[1]
        err = torch.linalg.norm(env.pose[:, :2] - ref_pose[:, :2], dim=-1)
        per_step.append(err.detach().cpu().numpy().copy())
        if bool(dones.any()):
            print(f"note: {int(dones.sum())} env(s) terminated early at step {len(per_step)}")

    errs = np.stack(per_step)                       # (steps, num_envs)
    rows = []
    for k, rid in enumerate(ids):
        rows.append({
            "reference_id": rid,
            "episode_id": env.reference_set.episode_ids[rid],
            "family": env.reference_set.scenario_families[rid].removesuffix("_command").split("_", 2)[2],
            "mean_position_error_m": float(errs[:, k].mean()),
            "final_position_error_m": float(errs[-1, k]),
            "steps": int(errs.shape[0]),
        })
    pooled = float(np.mean([r["mean_position_error_m"] for r in rows]))

    print(f"\nIN-SURROGATE, {a.terrain}, {errs.shape[0]} steps = "
          f"{errs.shape[0] * 0.01 * int(cfg['action_repeat']):.2f} s, checkpoint "
          f"{a.policy_checkpoint.name}")
    print(f"  {'family':<14}{'mean err':>10}{'final':>10}")
    for r in sorted(rows, key=lambda r: -r["mean_position_error_m"]):
        print(f"  {r['family']:<14}{r['mean_position_error_m']:>10.4f}{r['final_position_error_m']:>10.4f}")
    print(f"  {'POOLED':<14}{pooled:>10.4f}")

    out = {"checkpoint": str(a.policy_checkpoint), "terrain": a.terrain,
           "reference_path": str(a.reference_path), "reference_ids": ids,
           "horizon_steps": int(errs.shape[0]),
           "per_reference": rows, "pooled_mean_position_error_m": pooled}
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
