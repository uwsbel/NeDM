"""Does checkpointing change the gradient? Measure it, do not assume it.

The patch claims the backward pass is unchanged. That claim is the whole reason the
w1024 arm stays comparable to the w256 arm, so it gets checked against the real
surrogate rather than argued from the documentation.

Same input, same frozen weights, gradient taken with respect to the INPUT, because that
is the path the fine-tune actually differentiates: actions -> state -> reward.
"""
import sys, torch
sys.path.insert(0, "/srv/home/kasha2/nedm/NeDM/src")
from nedm.training.trainer import HMMWVTrainer

CK = sys.argv[1] if len(sys.argv) > 1 else \
    "/srv/home/kasha2/nedm/training_runs/go2_crm_abl_w512/checkpoints/best_val.pt"
DEV = "cuda"

ck = torch.load(CK, map_location="cpu", weights_only=False)
ck["config"]["training"]["device"] = DEV
tr = HMMWVTrainer(ck["config"]); tr.model.load_state_dict(ck["model_state_dict"])
wrapper = tr.model.to(DEV).eval()
for p in wrapper.parameters(): p.requires_grad_(False)

# The flag lives on the transformer, not on the wrapper that holds it. Testing the
# wrapper is what hid the bug this script exists to catch, so resolve the backbone the
# same way the production setter does and test THAT.
from nedm.training.model_transformer import ContinuousTransformer
m = next(x for x in wrapper.modules() if isinstance(x, ContinuousTransformer))
print(f"surrogate {CK.split('/')[-3]}  n_embd={m.config.n_embd}  "
      f"layers={m.config.n_layer}  "
      f"params={sum(p.numel() for p in wrapper.parameters())/1e6:.1f}M")

torch.manual_seed(0)
B, T = 8, 24
D = m.config.input_dim
x0 = torch.randn(B, T, D, device=DEV)

def run(flag):
    m.grad_checkpointing = flag
    x = x0.clone().requires_grad_(True)
    y = m(x)
    # A scalar that depends on every output element, so every path is exercised.
    loss = (y * torch.linspace(0.3, 1.7, y.shape[-1], device=DEV)).pow(2).mean()
    loss.backward()
    torch.cuda.synchronize()
    return loss.item(), x.grad.detach().clone()

torch.cuda.reset_peak_memory_stats()
l_off, g_off = run(False)
mem_off = torch.cuda.max_memory_allocated() / 2**20

torch.cuda.reset_peak_memory_stats()
l_on, g_on = run(True)
mem_on = torch.cuda.max_memory_allocated() / 2**20

dmax = (g_on - g_off).abs().max().item()
scale = g_off.abs().max().item()
print(f"  loss      off={l_off:.12e}  on={l_on:.12e}  delta={abs(l_on-l_off):.3e}")
print(f"  grad max  |off|={scale:.6e}   max|on-off|={dmax:.3e}   rel={dmax/max(scale,1e-30):.3e}")
print(f"  peak mem  off={mem_off:.0f} MiB   on={mem_on:.0f} MiB   "
      f"saved={100*(1-mem_on/max(mem_off,1e-9)):.1f}%")
exact = torch.equal(g_on, g_off)
print(f"  bitwise identical gradient: {exact}")
print("VERDICT:", "EXACT" if exact else ("CLOSE" if dmax/max(scale,1e-30) < 1e-6 else "DIFFERENT"))
