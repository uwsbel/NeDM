"""Closed-loop gain margin k*: the factor by which loop gain can be multiplied
before the plant-policy loop goes unstable.

WHY SCALE THE WEIGHTS RATHER THAN PATCH THE COLLECTOR. sbel-pc added an
NEDM_ACTION_MULT env var for the same purpose. Scaling the actor's final layer is
exactly equivalent -- the output is affine in that layer, so multiplying its weight
and bias by k multiplies every action by k -- and it needs no change to the
collector, the policy loader or the screen. Two boxes patching one instrument from
different bases is the failure this project has already recorded twice.

k* is read off the divergence rate from standing_screen: the smallest k at which
the rate crosses 0.5.
"""
import argparse, sys, tempfile, torch
from pathlib import Path
sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))
from nedm.quadruped.policy_batched import BatchedGo2Policy
from standing_screen import screen


def scaled_copy(ckpt, k, out):
    ts = torch.jit.load(str(ckpt), map_location="cpu")
    w = BatchedGo2Policy(ts)
    # LAST LAYER ONLY. The actor is a plain MLP; its output is affine in the final
    # weight and bias, so this scales the action exactly. Scaling any earlier layer
    # would pass through a nonlinearity and would NOT be a uniform output gain.
    last = max(int(n.split(".")[0]) for n, _ in w.actor.named_parameters())
    with torch.no_grad():
        for name, p in w.actor.named_parameters():
            if name.startswith(f"{last}."):
                p.mul_(k)
    torch.jit.save(ts, str(out))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ks", nargs="*", type=float, default=[0.7, 0.85, 1.0, 1.2, 1.5])
    ap.add_argument("--duration-s", type=float, default=20.0)
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="gain_"))
    print(f"  {'k':>6s} {'divergence rate':>16s}")
    kstar, prev_k, prev_r = None, None, None
    for k in sorted(a.ks):
        p = scaled_copy(a.ckpt, k, tmp / f"k{k:.2f}.pt")
        v, info = screen(str(p), duration=a.duration_s)
        r = info["rate"]
        print(f"  {k:6.2f} {v:>10s} {r:5.2f}")
        if kstar is None and r >= 0.5:
            # linear interpolation between the last sub-0.5 rung and this one
            kstar = k if prev_k is None else prev_k + (0.5 - prev_r) * (k - prev_k) / (r - prev_r)
        prev_k, prev_r = k, r
    if kstar is None:
        print(f"\n  k* > {max(a.ks)}  (never crossed 0.5 in the swept range)")
    else:
        print(f"\n  k* = {kstar:.3f}   {'STABLE as deployed (k* > 1)' if kstar > 1 else 'UNSTABLE as deployed (k* < 1)'}")
