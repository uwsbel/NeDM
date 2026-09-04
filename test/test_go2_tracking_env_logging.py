"""The throttle_brake strip must survive an EPISODE ENDING.

WHY THIS FILE EXISTS. Go2NeuralTrackingEnv removes the HMMWV's throttle_brake
term, whose factors are cmd_vy and cmd_wz for us -- turning while strafing, which
we do not penalise. The first strip popped the key from extras["log"] only, and I
verified it on a 3-iteration smoke at 24 steps/env against max_episode_steps 120.
No episode ever ENDED in that smoke, so the parent never built extras["episode"],
so the second copy the parent makes with episode_log.update(tracking_log)
(hmmwv_tracking_env.py:796) was never created and the leak could not appear. The
check passed without its subject ever having happened, and the key showed up in
the real training log ten iterations in.

max_episode_steps is 8 here for exactly that reason: terminations are
GUARANTEED inside the loop, and the test asserts it actually saw the "episode"
dict rather than trusting that it did.

The negative control re-runs the same loop against the OLD log-only strip and
requires it to LEAK. A test that cannot fail on the bug it was written for is
the same defect one level up.

Needs the trained Go2 checkpoint and the 40-reference set; skipped when absent.

RUNS BOTH WAYS. pytest is not installed in the nedm-src env that has torch and
rsl_rl (it is in the chrono env, which has neither), so a pytest-only test here
would be a test nobody can run. `python test/test_go2_tracking_env_logging.py`
executes the same two checks directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import pytest
except ModuleNotFoundError:  # nedm-src has torch but not pytest
    pytest = None

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

CHECKPOINT = REPO_ROOT / (
    "artifacts/training_runs/go2_transformer_v01_contact_mix25_onehot/checkpoints/best_val.pt"
)
REFERENCES = REPO_ROOT / "artifacts/rl_references/go2_flat_crm_ref40.npz"

if pytest is not None:
    pytestmark = pytest.mark.skipif(
        not (CHECKPOINT.is_file() and REFERENCES.is_file()),
        reason="needs the Go2 dynamics checkpoint and reference set",
    )

LEAKED_KEY = "/tracking/throttle_brake"


def _cfg():
    import torch  # noqa: F401  -- import cost is why this is not module level

    from nedm.rl.go2_tracking_env import go2_default_env_cfg

    cfg = go2_default_env_cfg()
    cfg.update(
        {
            "num_envs": 16,
            "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
            "dynamics_context_steps": 16,
            # Short enough that episodes END inside the loop below.
            "max_episode_steps": 8,
            "dynamics_checkpoint": str(CHECKPOINT),
            "reference_path": str(REFERENCES),
            "terrain_mix": "flat:1,crm:1",
        }
    )
    return cfg


def _roll(env, steps=30):
    """Step with zero commands and collect every extras dict produced."""
    import torch

    env.reset()
    torch.manual_seed(0)
    seen = []
    for _ in range(steps):
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        seen.append(env.step(actions)[3])
    return seen


def test_throttle_brake_absent_from_both_log_dicts():
    from nedm.rl.go2_tracking_env import Go2NeuralTrackingEnv

    cfg = _cfg()
    env = Go2NeuralTrackingEnv(cfg)
    saw_episode = False
    for step, extras in enumerate(_roll(env)):
        for key in ("log", "episode"):
            entry = extras.get(key)
            if isinstance(entry, dict):
                saw_episode |= key == "episode"
                assert LEAKED_KEY not in entry, f"leaked in extras[{key!r}] at step {step}"
    # Without this the test passes on a rollout where nothing terminated, which
    # is precisely how the bug shipped.
    assert saw_episode, "no episode ended: the branch under test never ran"


def test_the_old_log_only_strip_would_have_leaked():
    """Negative control: the test must be able to fail on the original defect."""
    import torch

    import nedm.rl.go2_tracking_env as module
    from nedm.rl.go2_tracking_env import Go2NeuralTrackingEnv

    class LogOnlyStrip(Go2NeuralTrackingEnv):
        def _make_extras(self, reward_terms, dones, time_outs):
            if "throttle_brake" not in reward_terms:
                reward_terms = {
                    **reward_terms,
                    "throttle_brake": torch.zeros_like(reward_terms["track_reward"]),
                }
            extras = module.HMMWVNeuralTrackingEnv._make_extras(
                self, reward_terms, dones, time_outs
            )
            log = extras.get("log")
            if isinstance(log, dict):
                log.pop(LEAKED_KEY, None)
            return extras

    env = LogOnlyStrip(_cfg())
    leaked = any(
        isinstance(extras.get("episode"), dict) and LEAKED_KEY in extras["episode"]
        for extras in _roll(env)
    )
    assert leaked, "the negative control did not reproduce the leak; the test is not sensitive"


if __name__ == "__main__":
    if not (CHECKPOINT.is_file() and REFERENCES.is_file()):
        raise SystemExit(f"missing {CHECKPOINT} or {REFERENCES}")
    test_throttle_brake_absent_from_both_log_dicts()
    print("PASS  throttle_brake absent from both log dicts, with a real episode end")
    test_the_old_log_only_strip_would_have_leaked()
    print("PASS  negative control: the old log-only strip does leak")
