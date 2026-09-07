"""Export a short-branch fine-tune checkpoint to TorchScript.

The fine-tune scripts save only `{"state_dict": BatchedGo2Policy.state_dict(), ...}`,
but every consumer -- collect_go2_smoke.py's --imported-ckpt, the gate, the verdict
harness -- calls torch.jit.load. A state_dict is not loadable by any of them, so a
fine-tune is unusable until it passes through here.

BatchedGo2Policy aliases the TorchScript submodules rather than copying them
(policy_batched.py:22-23), so loading the state_dict into a wrapper built from a
freshly jit-loaded base mutates the base in place and it can be saved directly.
That aliasing is the whole mechanism; if it ever stops holding, the check below
fails loudly rather than writing a file that is silently still the base policy.
"""
import argparse, sys, torch
sys.path.insert(0, "src")
from nedm.quadruped.policy_batched import BatchedGo2Policy

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="TorchScript policy the fine-tune started from")
ap.add_argument("--ckpt", required=True, help="fine-tune best.pt (state_dict form)")
ap.add_argument("--out", required=True, help="TorchScript file to write")
a = ap.parse_args()

ts = torch.jit.load(a.base, map_location="cpu")
before = {k: v.clone() for k, v in BatchedGo2Policy(ts).state_dict().items()}

wrapper = BatchedGo2Policy(ts)
sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)["state_dict"]
missing, unexpected = wrapper.load_state_dict(sd, strict=True), None
after = wrapper.state_dict()

# The aliasing claim, tested rather than assumed: the state_dict must have actually
# reached `ts`, and it must differ from the base or we exported the wrong thing.
changed = sum(1 for k in after if not torch.equal(before[k], after[k]))
if changed == 0:
    raise SystemExit("ABORT: checkpoint is identical to the base policy in every tensor")
torch.jit.save(ts, a.out)

reloaded = BatchedGo2Policy(torch.jit.load(a.out, map_location="cpu")).state_dict()
bad = [k for k in sd if not torch.equal(sd[k].cpu(), reloaded[k].cpu())]
if bad:
    raise SystemExit(f"ABORT: {len(bad)} tensor(s) did not survive the round trip: {bad[:3]}")
print(f"  {changed}/{len(after)} tensors differ from the base policy")
print(f"  round trip exact for all {len(sd)} tensors")
print(f"  wrote {a.out}")
