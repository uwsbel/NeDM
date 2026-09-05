"""Make a surrogate 'treatment' policy by perturbing the baseline's weights.

WHY THIS EXISTS. The paired noise floor is the variance of the per-episode
DIFFERENCE between two arms, and that cannot be derived from one arm's data: in a
paired design the command structure and the realisation are shared and both cancel,
so what survives is the treatment-by-realisation interaction. Measuring it needs a
second policy. It does not need the FINE-TUNED policy -- only a different one whose
tracking differs by about the effect size we hope to detect.

The checkpoint is a TorchScript archive, not a state_dict, so the perturbation is
applied to the scripted module's parameters in place and re-saved with jit.save.
"""
import argparse, torch

ap = argparse.ArgumentParser()
ap.add_argument("--in-ckpt", required=True)
ap.add_argument("--out-ckpt", required=True)
ap.add_argument("--rel-sigma", type=float, default=0.01,
                help="Gaussian noise sd as a fraction of each tensor's own std.")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

torch.manual_seed(a.seed)
m = torch.jit.load(a.in_ckpt, map_location="cpu")
n_t = n_p = 0
with torch.no_grad():
    for p in m.parameters():
        if p.dim() < 2:          # leave biases alone; weights carry the behaviour
            continue
        sd = p.std()
        if not torch.isfinite(sd) or sd == 0:
            continue
        p.add_(torch.randn_like(p) * (a.rel_sigma * sd))
        n_t += 1; n_p += p.numel()
if n_t == 0:
    raise SystemExit("perturbed nothing -- refusing to write a checkpoint identical "
                     "to its input, which would make the two arms trivially equal")
torch.jit.save(m, a.out_ckpt)
print(f"perturbed {n_t} weight tensors ({n_p} params) at rel-sigma {a.rel_sigma} -> {a.out_ckpt}")
