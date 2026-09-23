#!/usr/bin/env python3
"""Where does a policy drive the model, and is that the model's fault or the corpus's?

Seed 6's surrogate collapsed under PPO (out-of-distribution spikes from the first hundred
iterations, value loss 480,000 by iteration 1150) while three others trained on the SAME
corpus with the SAME settings ran clean past dw 6.8. Two explanations fit that, and they
call for opposite fixes:

  a coverage gap  -- the corpus does not cover the region, every surrogate is unconstrained
                     there, and which one PPO can exploit is luck. Collect more data.
  a model lottery -- the corpus covers it well enough, and seed 6's surrogate alone is
                     wrong there. Ensemble, penalise, or stop early; more data changes
                     nothing.

This rolls one policy inside several surrogates from identical held-out starts and reports,
for each, how far outside the corpus the rollout goes (the same whitened kNN reference PPO
prices into its reward and Gate 4 uses) and which channels leave their recorded range. A
policy that goes far outside only in its home surrogate is that model's fiction. One that
goes far outside in every surrogate is standing on a hole in the data.

It measures, it does not decide: both stories predict a large number in the home model.
The discriminating comparison is the OTHER models' columns.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="the fine-tuned policy to probe")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--branches", type=int, default=256)
    ap.add_argument("--steps", type=int, default=100, help="control steps; 100 is 2 s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import torch
    import finetune as FT
    import train as T

    dev = torch.device(a.device)
    loaded = [FT.load_nnrom(torch, Path(m), dev) for m in a.models]
    ck = loaded[0][1]
    exc = yaml.safe_load((HERE / "params" / "excitation.yaml").read_text())
    ctrl_dt = 1.0 / float(exc["episode"]["control_hz"])
    dt_s = float(ck["config"].get("dt_s") or 1.0 / float(exc["episode"]["record_hz"]))
    hold = int(round(ctrl_dt / dt_s))
    ctx = ck["config"]["block_size"]
    state_fields = ck["state_fields"]

    pol_cfg = yaml.safe_load((HERE / "params" / "policy.yaml").read_text())
    obs = FT.ObsBuilder(torch, state_fields, pol_cfg).to(dev)
    policy, _params = FT.load_policy_torch(torch, Path(a.policy), dev)
    corpus = T.Corpus(Path(a.corpus), ck["config"]["preset"], ctx,
                      extra_fields=FT.EXTRA_FIELDS)
    pool = FT.build_pool(corpus, ctx, ctrl_dt)
    ood = FT.OODCost(corpus, seed=a.seed).to_torch(torch, dev)

    import random
    names = corpus.state_fields
    print(f"policy {a.policy}")
    print(f"{a.branches} branches x {a.steps} control steps ({a.steps * ctrl_dt:.1f} s) from "
          f"the same held-out starts in each model; corpus self-distance {ood.thresh:.3f}")
    print(f"{'surrogate':28s} {'mean ood':>9s} {'p99 ood':>9s} {'steps>0':>8s} "
          f"{'worst channel (max |z|)':>34s}")
    for mp, (model, _c) in zip(a.models, loaded):
        rng = random.Random(a.seed)          # identical starts for every model
        b = FT.start_batch(torch, corpus, pool, a.branches, rng, ctx, obs.p2c, dev)
        cmd = b["cmd"]
        with torch.no_grad():
            S = FT.closed_loop(torch, model, obs, policy, b, cmd, a.steps, hold)
            # The OOD reference is built on (state, action) pairs, so the action half has
            # to be the action actually taken there, not zeros: a zero action is itself far
            # from the corpus and would make every model look equally bad.
            AA = FT.closed_loop_actions(torch, model, obs, policy, b, cmd, a.steps, hold)
            flat_s = S.reshape(-1, S.shape[-1])
            flat_a = AA.reshape(-1, AA.shape[-1])
            c = ood.cost(torch, flat_s, flat_a)
            z = ((flat_s - ood.t_mu[:flat_s.shape[-1]]) / ood.t_sd[:flat_s.shape[-1]]).abs()
            worst = int(z.max(0).values.argmax())
            print(f"{Path(mp).parent.name:28s} {float(c.mean()):9.4f} "
                  f"{float(torch.quantile(c, 0.99)):9.4f} "
                  f"{100 * float((c > 0).float().mean()):7.1f}% "
                  f"{names[worst] + ' ' + format(float(z[:, worst].max()), '.1f'):>34s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
