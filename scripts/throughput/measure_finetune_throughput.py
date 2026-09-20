"""Measure, rather than assert, what the surrogate buys in wall-clock.

The claim "fine-tuning inside the model is cheaper than fine-tuning against the simulator"
has been carried on an amortisation argument -- corpus collected once, reused often -- and
never on a measured per-step ratio. That argument is weak on its own, because it says
nothing about whether the same policy improvement could have been reached directly in
Chrono for less total time.

Three quantities settle it:

  1. surrogate throughput   transitions per second through the frozen model, at the exact
                            batch and branch length the fine-tune uses
  2. simulator throughput   transitions per second of CRM, taken from real collection logs
                            rather than a synthetic benchmark
  3. optimisation budget    transitions actually consumed to reach the stopping rule

Multiplying (3) by the reciprocal of (1) and (2) gives the two wall-clock figures the
comparison needs, and their ratio is the honest speedup for ONE fine-tune. The corpus
collection cost is then a fixed charge amortised over however many fine-tunes are run.
"""
import json
import os
import statistics as st
import sys
import time

sys.path.insert(0, "/home/kyle/Documents/sbel/NeDM/src")
import torch

S = "/home/kyle/sbel-artifacts"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---- 1. surrogate throughput ------------------------------------------------
ck = torch.load(f"{S}/training_runs/go2_crm_baseline_s1/checkpoints/best_val.pt",
                map_location=DEV, weights_only=False)
cfg = ck["config"]
from nedm.training.model import HMMWVDynamicsModel  # noqa: E402

# Build the network directly from the checkpoint's own config and metadata rather than
# standing up a trainer, which would load the whole processed dataset into memory just to
# time a forward pass.
md = ck["metadata"]
sdim = len(md["state_fields"])
adim = len(md["action_fields"])
sd = ck["model_state_dict"]
# target_dim and the normalisation buffers are recoverable from the saved tensors, so the
# measurement cannot drift from the weights it is timing.
tgt = sd["head.net.%d.weight" % max(int(k.split(".")[2]) for k in sd if k.startswith("head.net.") and k.endswith(".weight"))].shape[0] \
    if any(k.startswith("head.net.") for k in sd) else sdim
norm = {k: sd[k].tolist() for k in sd if k in ("state_mean", "state_std", "action_mean",
                                               "action_std", "target_mean", "target_std")}
model = HMMWVDynamicsModel(
    state_dim=sdim, action_dim=adim, target_dim=tgt,
    transformer_cfg=cfg["model"], normalization=norm,
    state_fields=md["state_fields"], dt_s=md.get("dt_s"))
model.load_state_dict(sd, strict=False)
model.to(DEV).eval()

L = cfg["model"]["block_size"]
BATCH, BRANCH = 64, 15

states = torch.randn(BATCH, L, sdim, device=DEV)
actions = torch.randn(BATCH, L, adim, device=DEV)

with torch.no_grad():
    for _ in range(5):
        model(states, actions)
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    REPS = 40
    for _ in range(REPS):
        for _ in range(BRANCH):
            model(states, actions)
    if DEV == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

surr_steps = REPS * BRANCH * BATCH
surr_rate = surr_steps / dt
print(f"  surrogate: {surr_rate:,.0f} transitions/s "
      f"({BATCH} branches x {BRANCH} steps, {sdim}-D state, {DEV})")

# ---- 2. simulator throughput, measured from collection logs ------------------
# The episode sidecars carry no timing field, but the collector prints the wall time it
# spent on each episode, so this is measured under real collection conditions on the same
# hardware rather than from a synthetic benchmark.
import glob  # noqa: E402
import re  # noqa: E402

pat = re.compile(r"wrote (\d+) rows to \S+ \(([0-9.]+) s wall")
rows, walls = [], []
for lg in glob.glob(f"{S}/datasets/go2_crm_*/logs/*.log") + glob.glob("/tmp/*collect*.log"):
    try:
        txt = open(lg, errors="ignore").read()
    except OSError:
        continue
    for n, w in pat.findall(txt):
        rows.append(float(n)); walls.append(float(w))

if rows:
    per_ep = [n / w for n, w in zip(rows, walls)]
    chrono_rate = st.mean(per_ep)
    print(f"  chrono CRM: {chrono_rate:,.2f} transitions/s per worker "
          f"(mean over {len(per_ep)} episodes, range {min(per_ep):.2f}-{max(per_ep):.2f})")
else:
    chrono_rate = None
    print("  chrono CRM: no per-episode wall times found in any collection log")

# ---- 3. optimisation budget --------------------------------------------------
# Transitions consumed by one fine-tune to reach the stopping rule.
UPDATES = {"analytic dW 1.0": 94, "analytic dW 2.0": 205, "analytic dW 4.0": 1147}
print()
for label, upd in UPDATES.items():
    consumed = upd * BATCH * BRANCH
    t_surr = consumed / surr_rate
    line = f"  {label:<18} {upd:>5} updates  {consumed:>9,} transitions  surrogate {t_surr:7.1f} s"
    if chrono_rate:
        t_chrono = consumed / chrono_rate
        line += f"  chrono {t_chrono / 3600:7.1f} h  speedup {t_chrono / t_surr:,.0f}x"
    print(line)

if chrono_rate:
    print()
    print(f"  per-transition speedup: {surr_rate / chrono_rate:,.0f}x")
    print(f"  corpus collection to amortise: 24.1 h of Chrono wall-clock, paid once")
