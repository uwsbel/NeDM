"""Short-branch fine-tune of the imported Go2 policy inside the frozen surrogate.

WHY SHORT BRANCHES. The gate certified this surrogate over 0.1 s and only 0.1 s:
at that horizon body_vel gain 1.077 [1.007,1.152], corr 0.668 [0.583,0.738] and
cosine 0.962 [0.939,0.977] all PASS on intervals at n=200. At 0.5 s its apparatus
ratio is 1.12 and the measurement is swamped. So the policy is rolled 5 steps
(0.1 s at 50 Hz) from a RECORDED real state and then reset -- the model never runs
past the window it was measured to be trustworthy on.

CONFIG DECLARED BEFORE LAUNCH, and none of it is tuned afterwards:
  * fixed update budget -- not "until it converges"
  * checkpoint selected on a SURROGATE-INTERNAL metric only, never on anything
    from Chrono
  * the Chrono verdict harness runs EXACTLY ONCE, on the final selected
    checkpoint, and that number is the result whatever it says
  * the verdict's backward-low episode specs are EXCLUDED from the branch pool,
    so the policy is never trained on the test set

FIDELITY LIMIT, stated not engineered away: observations reconstructed from
surrogate state reproduce the policy's logged action to ~0.117 (3.6% of action
magnitude). Origin unexplained; confined to the training environment and absent
from the Chrono evaluation, where the policy builds its own state from a real spawn.
"""
import argparse, csv, glob, json, os, random, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src", "nedm", "quadruped"))
from policy_batched import BatchedGo2Policy
from nedm.quadruped.imported_policy import (CHRONO_TO_IMPORTED, SIGN, IMPORTED_DEFAULTS,
    ANG_VEL_SCALE, CMD_SCALE, DOF_POS_SCALE, DOF_VEL_SCALE, ACTION_SCALE)
from nedm.training.trainer import HMMWVTrainer

ap = argparse.ArgumentParser()
ap.add_argument("--surrogate", default="/home/kyle/sbel-artifacts/training_runs/go2_corrected_34d/checkpoints/best_val.pt")
ap.add_argument("--policy", default="/home/kyle/sbel-artifacts/checkpoints/go2_cts_150k.pt")
ap.add_argument("--root", default="/home/kyle/sbel-artifacts/datasets/go2_comprehensive_merged/flat")
ap.add_argument("--out", default="/home/kyle/sbel-artifacts/finetune_go2_shortbranch")
ap.add_argument("--updates", type=int, default=1500)      # FIXED BUDGET
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--branch-steps", type=int, default=5)    # 0.1 s, the certified window
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--val-every", type=int, default=50)
ap.add_argument("--val-branches", type=int, default=512)
ap.add_argument("--episodes", type=int, default=400)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
DEV = "cuda"

ck = torch.load(a.surrogate, map_location="cpu", weights_only=False)
ck["config"]["training"]["device"] = DEV
tr = HMMWVTrainer(ck["config"]); tr.model.load_state_dict(ck["model_state_dict"])
tr.model.to(DEV).eval()
for p in tr.model.parameters(): p.requires_grad_(False)          # surrogate FROZEN
md = json.load(open(ck["config"]["processed_dataset_dir"] + "/metadata.json"))
sf, af = md["state_fields"], md["action_fields"]; L = tr.sequence_length
ix = {n: i for i, n in enumerate(sf)}; aix = {n: i for i, n in enumerate(af)}
MOTOR = ["rr_hip","rr_thigh","rr_calf","rl_hip","rl_thigh","rl_calf",
         "fr_hip","fr_thigh","fr_calf","fl_hip","fl_thigh","fl_calf"]
JP = [ix[f"joint_{n}_pos_rad"] for n in MOTOR]; JV = [ix[f"joint_{n}_vel_radps"] for n in MOTOR]
# roll_rate_radps IS ang_vel_body_x_radps and yaw_rate_radps IS ang_vel_body_z_radps
# -- verified bit-identical over 500 rows. The 34-D state carries the body-frame
# vector under Euler-sounding names, so this is the policy's ang_vel block.
ANG = [ix["roll_rate_radps"], ix["ang_vel_body_y_radps"], ix["yaw_rate_radps"]]
GRV = [ix["grav_body_x"], ix["grav_body_y"], ix["grav_body_z"]]
ATG = torch.tensor([aix[f"joint_{n}_target_rad"] for n in MOTOR], device=DEV)
VX, VY, WZ = ix["vel_body_x_mps"], ix["vel_body_y_mps"], ix["yaw_rate_radps"]
C2I = torch.tensor(CHRONO_TO_IMPORTED, dtype=torch.long, device=DEV)
DEF = torch.tensor(np.asarray(IMPORTED_DEFAULTS, dtype=np.float32), device=DEV)
def _t(v): return torch.as_tensor(np.asarray(v, dtype=np.float32), device=DEV)
ANGS, CMDS, DPS, DVS, ACTS = _t(ANG_VEL_SCALE), _t(CMD_SCALE), _t(DOF_POS_SCALE), _t(DOF_VEL_SCALE), _t(ACTION_SCALE)

ts = torch.jit.load(a.policy, map_location=DEV)
policy = BatchedGo2Policy(ts).to(DEV)
for p in policy.parameters(): p.requires_grad_(True)
opt = torch.optim.Adam(policy.parameters(), lr=a.lr)

# ---- branch pool, with the verdict's episodes EXCLUDED -----------------------
idx = json.load(open(a.root + "/dataset_index.json"))["episodes"]
keep = {e["episode_id"] for e in idx}
def scored_cmd(rows):
    c = np.array([float(r["cmd_vx_mps"]) for r in rows])
    w = c[-1000:]
    return float(w[0]) if w.std() <= 1e-6 else None
paths = [p for p in sorted(glob.glob(a.root + "/episodes/*.json"))
         if not p.endswith(".config.json") and os.path.basename(p)[:-5] in keep
         and "_s2000000_" in p]
random.Random(a.seed).shuffle(paths)
S_all, A_all, C_all, excluded = [], [], [], 0
for p in paths:
    if len(S_all) >= a.episodes: break
    rows = list(csv.DictReader(open(p.replace(".json", ".csv"))))
    if len(rows) < L + 400: continue
    cmd = scored_cmd(rows)
    if cmd is not None and -0.18 < cmd <= -0.02:      # THE VERDICT'S CELL
        excluded += 1; continue
    S_all.append(np.array([[float(r[f]) for f in sf] for r in rows], dtype=np.float32))
    A_all.append(np.array([[float(r[f]) for f in af] for r in rows], dtype=np.float32))
    C_all.append(np.array([[float(r["cmd_vx_mps"]), float(r["cmd_vy_mps"]),
                            float(r["cmd_wz_radps"])] for r in rows], dtype=np.float32))
print(f"  branch pool: {len(S_all)} episodes; EXCLUDED {excluded} in the verdict's cell", flush=True)

WARM, BS = 5, a.branch_steps
def sample(rng, n):
    out = []
    for _ in range(n):
        e = rng.randrange(len(S_all))
        lo, hi = L + 2 * WARM + 1, len(S_all[e]) - 2 * BS - 2
        b = rng.randrange(lo, hi)
        out.append((e, b if b % 2 == 1 else b + 1))   # control acts on ODD rows
    return out

def obs_from(s, cmd, prev):
    ang = s[:, ANG] * ANGS; grav = s[:, GRV]
    q = SIGN * s[:, JP][:, C2I]; qd = SIGN * s[:, JV][:, C2I]
    return torch.cat([ang, grav, cmd * CMDS, (q - DEF) * DPS, qd * DVS, prev], dim=1)

def rollout(batch, grad=True):
    S = torch.tensor(np.stack([S_all[e][b - L:b] for e, b in batch]), device=DEV)
    A = torch.tensor(np.stack([A_all[e][b - L:b] for e, b in batch]), device=DEV)
    cmd = torch.tensor(np.stack([C_all[e][b] for e, b in batch]), device=DEV)
    hist = policy.initial_history(len(batch), DEV)
    prev = torch.zeros(len(batch), 12, device=DEV)
    with torch.no_grad():                                    # warm-up on RECORDED obs
        for k in range(WARM, 0, -1):
            s = torch.tensor(np.stack([S_all[e][b - 2 * k] for e, b in batch]), device=DEV)
            prev, hist = policy(obs_from(s, cmd, prev), hist)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    errs = []
    with ctx:
        hs, ha = S, A
        for _ in range(BS):
            act, hist = policy(obs_from(hs[:, -1], cmd, prev), hist)
            tgt = torch.zeros_like(act).index_copy(1, C2I, act * ACTS + DEF)
            newa = ha[:, -1].clone().index_copy(1, ATG, tgt)
            d = tr.model.predict_delta(hs[:, -L:], torch.cat([ha, newa.unsqueeze(1)], 1)[:, -L:],
                                       terrain=None)[:, -1, :]
            nxt = hs[:, -1] + d
            hs = torch.cat([hs, nxt.unsqueeze(1)], 1); ha = torch.cat([ha, newa.unsqueeze(1)], 1)
            prev = act
            errs.append(torch.stack([nxt[:, VX] - cmd[:, 0], nxt[:, VY] - cmd[:, 1],
                                     nxt[:, WZ] - cmd[:, 2]], dim=1))
    return torch.stack(errs, 1)                              # (B, BS, 3)

vrng = random.Random(a.seed + 991)
VAL = sample(vrng, a.val_branches)
rng = random.Random(a.seed)
best = (float("inf"), -1); hist_log = []
t0 = time.time()
for u in range(1, a.updates + 1):
    e = rollout(sample(rng, a.batch), grad=True)
    loss = (e ** 2).mean()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()
    if u % a.val_every == 0 or u == 1:
        with torch.no_grad():
            ve = torch.cat([rollout(VAL[i:i + 128], grad=False) for i in range(0, len(VAL), 128)])
        vm = float((ve ** 2).mean())               # SURROGATE-INTERNAL metric
        hist_log.append({"update": u, "train_loss": float(loss), "val_tracking_mse": vm})
        star = ""
        if vm < best[0]:
            best = (vm, u); star = "  <- best"
            torch.save({"state_dict": policy.state_dict(), "update": u,
                        "val_tracking_mse": vm}, f"{a.out}/best.pt")
        print(f"  update {u:5d}  train {float(loss):.6f}  val {vm:.6f}"
              f"  {time.time()-t0:6.0f}s{star}", flush=True)
json.dump({"config": vars(a), "history": hist_log,
           "best_update": best[1], "best_val_tracking_mse": best[0]},
          open(f"{a.out}/history.json", "w"), indent=2)
print(f"\n  DONE. best checkpoint from update {best[1]}, val {best[0]:.6f} -> {a.out}/best.pt")
print("  The Chrono verdict harness has NOT been run. It runs ONCE, on this checkpoint.")
