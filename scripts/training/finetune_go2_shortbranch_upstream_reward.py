"""Short-branch fine-tune, v3: the upstream objective PLUS the two effort terms.

v2 used 8 terms and STILL inflated actions 4.6x (mean |raw action| 4.725 against a
baseline 1.033), because none of those 8 penalises action MAGNITUDE -- action_rate and
action_smoothness penalise changes, dof_pos_limits penalises resulting positions.

v3 adds `torques` (-1e-4) and `dof_power` (-2e-5), which do. They were wrongly excluded:
tau = Kp(target - q) - Kd*qd is a FUNCTION of quantities the surrogate already predicts,
not a channel it must carry. 10 of 14 terms now; the 4 remaining omissions are stated in
go2_reward_terms.NOT_COMPUTABLE and each has a named missing quantity.

Predicted effect, recorded before running: at baseline effort `torques` contributes ~0.17
against a tracking maximum of 1.5; at v2's inflation it reaches ~3.6 and dominates. If
the mean |raw action| of v3 does NOT come down toward 1.033, this diagnosis is wrong.

v1 optimised tracking error alone and produced a policy that fell in 43/43 Chrono
episodes at a median of 1.52 s -- mean |raw action| 1.033 -> 5.244. That is what
optimising 1 of 14 reward terms buys. v2 uses the original CTS objective restricted to
the 8 terms the surrogate can compute, at the CONVERGED curriculum weights.

USING THE ORIGINAL OBJECTIVE RATHER THAN A PENALTY OF MY OWN removes a free parameter I
would otherwise pick after seeing the failure, and avoids a confound: a policy trained
under one reward and fine-tuned under another is being repurposed, not improved, and any
transfer failure would be unattributable between "the surrogate is inadequate" and "we
changed the objective".

SIX TERMS ARE OMITTED and are NOT silently dropped -- see NOT_COMPUTABLE. The largest,
correct_base_height at -10.0, needs a pos_z_m channel the 34-D state does not carry, so
THIS RUN OPTIMISES 8 OF 14 TERMS WITHOUT THE DOMINANT ONE. That limitation is recorded
before the run, and a 36-D state carrying pos_z_m is being built in parallel.

BUDGET RAISED 1500 -> 6000 on surrogate-side evidence predating any Chrono result: v1's
best checkpoint was update 1500 of 1500 and still improving.

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
import go2_reward_terms as RT

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
ap.add_argument("--require-substring", default="",
                help="restrict the branch pool to paths containing this. Was a "
                     "hardcoded '_s2000000_', which excludes every episode of any "
                     "corpus but the one it was written for.")
# DISPLACEMENT TEST: stop when the policy has moved a declared distance in weight space,
# rather than when a metric plateaus. Isolates ||dW|| from the objective, which the three
# previous runs confounded -- each used a different reward AND ended at a different ||dW||.
ap.add_argument("--height-target", type=float, default=None,
                help="Target pos_z_m for correct_base_height. TERRAIN-RELATIVE and has no "
                     "default: pos_z_m is absolute world z, so a rigid-calibrated target "
                     "would read a normally-standing robot on CRM soil as 0.20 m too high "
                     "and drive it into the ground.")
ap.add_argument("--ensemble", default="",
                help="Comma-separated ADDITIONAL surrogate checkpoints. With this set, the "
                     "rollout uses the ensemble MEAN prediction and the reward is penalised "
                     "by ensemble DISAGREEMENT. See --pessimism.")
ap.add_argument("--pessimism", type=float, default=1.0,
                help="Weight on the disagreement penalty. DECLARED, NOT TUNED: sweeping it "
                     "and reporting the best is the fitting-to-the-verdict pattern the "
                     "one-shot rule exists to prevent.")
ap.add_argument("--target-dw", type=float, default=None,
                help="Save and stop the first time ||W - W_baseline|| reaches this.")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
DEV = "cuda"

ck = torch.load(a.surrogate, map_location="cpu", weights_only=False)
ck["config"]["training"]["device"] = DEV
tr = HMMWVTrainer(ck["config"]); tr.model.load_state_dict(ck["model_state_dict"])
tr.model.to(DEV).eval()
for p in tr.model.parameters(): p.requires_grad_(False)          # surrogate FROZEN
# ENSEMBLE. The single-surrogate objective is exploitable: the optimiser finds regions
# where the model is confidently wrong and the real system disagrees. Measured on CRM --
# every arm drives the surrogate's own reward up while Chrono achieved velocity goes to
# zero, and a displacement-matched random control is flat, so it is the LEARNED GRADIENT
# at fault. Independently-seeded surrogates agree where the data constrained them and
# diverge where it did not, so their disagreement is a usable ignorance signal.
_MEMBERS = [tr.model]
for _extra in [x for x in a.ensemble.split(",") if x]:
    _c = torch.load(_extra, map_location="cpu", weights_only=False)
    _c["config"]["training"]["device"] = DEV
    _t = HMMWVTrainer(_c["config"]); _t.model.load_state_dict(_c["model_state_dict"])
    _t.model.to(DEV).eval()
    for _p in _t.model.parameters(): _p.requires_grad_(False)
    _MEMBERS.append(_t.model)
if len(_MEMBERS) > 1:
    print(f"  ENSEMBLE: {len(_MEMBERS)} surrogates, pessimism weight {a.pessimism}", flush=True)

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
PZ = ix.get("pos_z_m")
GRV = [ix["grav_body_x"], ix["grav_body_y"], ix["grav_body_z"]]
ATG = torch.tensor([aix[f"joint_{n}_target_rad"] for n in MOTOR], device=DEV)
VX, VY, WZ = ix["vel_body_x_mps"], ix["vel_body_y_mps"], ix["yaw_rate_radps"]
RR, PY_ = ix["roll_rate_radps"], ix["ang_vel_body_y_radps"]
C2I = torch.tensor(CHRONO_TO_IMPORTED, dtype=torch.long, device=DEV)
DEF = torch.tensor(np.asarray(IMPORTED_DEFAULTS, dtype=np.float32), device=DEV)
import xml.etree.ElementTree as _ET
# HARDCODED TO ONE BOX'S LAYOUT. This named dorm-pc's checkout, which has no
# "sbel/" segment, so every fine-tune on sbel died at the URDF load -- six of them,
# all reporting FINETUNE FAILED after the surrogates had already trained. The same
# split cost twenty minutes earlier tonight in the collector, and the collector's
# fix (NEDM_GO2_ASSETS with a per-box fallback) was never applied here.
#
# Search rather than assume, honour the env var the rest of the pipeline sets, and
# fail loudly naming what was tried.
import os as _os
_ASSET_ROOTS = [_os.environ.get("NEDM_GO2_ASSETS", ""),
                "/home/kyle/Documents/sbel/sbel-reproducibility/2025/multi-terrain-RL",
                "/home/kyle/Documents/sbel-reproducibility/2025/multi-terrain-RL"]
_REL = "data/robot/go2_irrvis/urdf/go2_description.urdf"
_U = next((_os.path.join(r, _REL) for r in _ASSET_ROOTS
           if r and _os.path.exists(_os.path.join(r, _REL))), None)
if _U is None:
    raise SystemExit("FATAL: go2_description.urdf not found under any of "
                     + repr([r for r in _ASSET_ROOTS if r]))
_lim = {}
for _j in _ET.parse(_U).getroot().iter("joint"):
    if _j.get("type") != "revolute" or _j.find("limit") is None: continue
    _lim[_j.get("name").removesuffix("_joint").lower()] = (
        float(_j.find("limit").get("lower")), float(_j.find("limit").get("upper")))
_lo = np.array([RT.shrunk_limits(*_lim[n])[0] for n in MOTOR], dtype=np.float32)
_hi = np.array([RT.shrunk_limits(*_lim[n])[1] for n in MOTOR], dtype=np.float32)
LO = torch.tensor(_lo, device=DEV)[C2I]      # chrono order -> policy order
HI = torch.tensor(_hi, device=DEV)[C2I]
HIPI = torch.tensor([0, 3, 6, 9], device=DEV)
HIPD = DEF[HIPI]                              # each hip paired with its OWN default
# WIRE IN THE GUARD THAT ALREADY EXISTED. go2_reward_terms.wrongly_omitted() was
# written precisely to stop a reward term being dropped for a reason that had stopped
# being true -- and was never called from anywhere, so it stopped nothing. Every CRM
# run so far dropped correct_base_height (-10.0, the LARGEST weight, 10x
# tracking_lin_vel) despite carrying pos_z_m, which is the term that tells the robot
# to hold its body up. On soil, where the feet sink ~4 cm, that is not a minor omission.
_wrong = RT.wrongly_omitted(sf)
_HEIGHT_TGT = None
if "correct_base_height" in _wrong:
    if a.height_target is None:
        raise SystemExit(
            "FATAL: this surrogate carries pos_z_m, so correct_base_height "
            f"({_wrong['correct_base_height']}) IS computable and would otherwise be "
            "silently dropped. It needs --height-target, which is TERRAIN-RELATIVE: "
            "pos_z_m is absolute world z, and the CRM soil surface sits ~0.20 m above "
            "the rigid datum. Measure the base policy's mean pos_z_m on this corpus "
            "and pass it.")
    _HEIGHT_TGT = a.height_target
    print(f"  reward: correct_base_height ENABLED, target pos_z_m = {_HEIGHT_TGT:.4f} m")
_still = {k: v for k, v in RT.NOT_COMPUTABLE.items() if k not in _wrong}
print(f"  reward: {len(RT.WEIGHTS) + (1 if _HEIGHT_TGT is not None else 0)} computable "
      f"terms, {len(_still)} omitted{'' if not _still else '; ' + ', '.join(_still)}")
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
# TAKE THE PATHS FROM THE INDEX, NOT FROM A GLOB. A merged index references CSVs
# that live under per-episode directories elsewhere, so `root/episodes/*.json`
# matches nothing and the pool comes out empty -- which it did, silently enough
# that the failure surfaced as "empty range for randrange()" three frames later.
#
# The "_s2000000_" filter this replaced was a hardcoded shard name from the
# previous corpus. On any other collection it excludes everything, and it excluded
# everything here. It is now --require-substring, defaulting to no filter, so
# restricting to a shard has to be asked for and is recorded in the invocation.
_want = getattr(a, "require_substring", "") or ""
paths = [e["csv_path"][:-4] + ".json" for e in idx
         if e["episode_id"] in keep and (not _want or _want in e["csv_path"])]
paths = [p for p in sorted(paths) if os.path.exists(p) or os.path.exists(p[:-5] + ".csv")]
print(f"  branch pool candidates from index: {len(paths)}"
      + (f"  (filtered on {_want!r})" if _want else "  (no shard filter)"))
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
    errs = []; rews = []; term_acc = {}
    dq_prev = None
    a_prev1 = prev.clone(); a_prev2 = prev.clone()
    with ctx:
        hs, ha = S, A
        for _ in range(BS):
            act, hist = policy(obs_from(hs[:, -1], cmd, prev), hist)
            tgt = torch.zeros_like(act).index_copy(1, C2I, act * ACTS + DEF)
            newa = ha[:, -1].clone().index_copy(1, ATG, tgt)
            _aw = torch.cat([ha, newa.unsqueeze(1)], 1)[:, -L:]
            if len(_MEMBERS) == 1:
                d = _MEMBERS[0].predict_delta(hs[:, -L:], _aw, terrain=None)[:, -1, :]
                disagree = None
            else:
                _ds = torch.stack([m.predict_delta(hs[:, -L:], _aw, terrain=None)[:, -1, :]
                                   for m in _MEMBERS], 0)
                d = _ds.mean(0)
                # Per-channel spread, self-normalised by that channel's spread ACROSS THE
                # BATCH so no channel dominates through its units alone. Detached scale:
                # the penalty should push the policy away from disputed states, not
                # reshape the normaliser.
                _sd = _ds.std(0)
                _scale = d.std(0, keepdim=True).detach().clamp_min(1e-6)
                disagree = (_sd / _scale).mean(dim=1)
            nxt = hs[:, -1] + d
            hs = torch.cat([hs, nxt.unsqueeze(1)], 1); ha = torch.cat([ha, newa.unsqueeze(1)], 1)
            prev = act
            qp = SIGN * nxt[:, JP][:, C2I]          # -> POLICY/URDF frame, applied ONCE
            dqp = SIGN * nxt[:, JV][:, C2I]
            t = RT.terms(qp, dqp, dq_prev if dq_prev is not None else dqp,
                         nxt[:, [VX, VY]], nxt[:, [RR, PY_]], nxt[:, WZ], cmd,
                         act, a_prev1, a_prev2, LO, HI, HIPD, HIPI,
                         act * ACTS + DEF,          # PD target, policy frame
                         pos_z=(nxt[:, PZ] if _HEIGHT_TGT is not None else None),
                         height_target=_HEIGHT_TGT)
            _r = RT.total(t)
            if disagree is not None:
                _r = _r - a.pessimism * disagree      # trust the model less where it argues
                term_acc["pessimism"] = term_acc.get("pessimism", 0.0) - float(
                    (a.pessimism * disagree).mean())
            rews.append(_r)
            for k, v in t.items(): term_acc[k] = term_acc.get(k, 0.0) + float(v.mean())
            dq_prev = dqp; a_prev2 = a_prev1; a_prev1 = act
            errs.append(torch.stack([nxt[:, VX] - cmd[:, 0], nxt[:, VY] - cmd[:, 1],
                                     nxt[:, WZ] - cmd[:, 2]], dim=1))
    return torch.stack(errs, 1), torch.stack(rews, 1), term_acc

W0 = {k: v.detach().clone() for k, v in policy.named_parameters()}
def dw():
    return float(torch.sqrt(sum(((p_ - W0[k]) ** 2).sum() for k, p_ in policy.named_parameters())))

vrng = random.Random(a.seed + 991)
VAL = sample(vrng, a.val_branches)
rng = random.Random(a.seed)
best = (float("inf"), -1); hist_log = []
t0 = time.time()
for u in range(1, a.updates + 1):
    e, r, ta = rollout(sample(rng, a.batch), grad=True)
    loss = -r.mean()                       # MAXIMISE the upstream reward
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()
    if a.target_dw is not None and dw() >= a.target_dw:
        d = dw()
        torch.save({"state_dict": policy.state_dict(), "update": u, "dw": d,
                    "val_neg_reward": float("nan")}, f"{a.out}/best.pt")
        print(f"  update {u:5d}  ||dW|| {d:.3f} >= target {a.target_dw} -- STOPPING", flush=True)
        hist_log.append({"update": u, "dw": d, "stopped_on": "target_dw"})
        best = (float("nan"), u)
        break
    if u % a.val_every == 0 or u == 1:
        with torch.no_grad():
            parts = [rollout(VAL[i:i + 128], grad=False) for i in range(0, len(VAL), 128)]
            vr = torch.cat([p[1] for p in parts])
        # NEGATED REWARD, NOT AN MSE. vr is the surrogate's rollout reward, so this
        # is -mean(reward): LOWER IS BETTER, which is what the `vm < best[0]`
        # selection below relies on. The old name was `val_tracking_mse`, which
        # both lies about the sign (an MSE cannot be negative) and invites a
        # comparison ACROSS runs. It is not comparable across runs: two fine-tunes
        # in different surrogates each score their own rollouts under their own
        # reward model, so the numbers share a name and not an instrument.
        vm = float(-vr.mean())                     # SURROGATE-INTERNAL metric
        hist_log.append({"update": u, "train_loss": float(loss), "val_neg_reward": vm})
        star = ""
        if vm < best[0]:
            best = (vm, u); star = "  <- best"
            torch.save({"state_dict": policy.state_dict(), "update": u,
                        "val_neg_reward": vm}, f"{a.out}/best.pt")
        print(f"  update {u:5d}  train {float(loss):.6f}  val {vm:.6f}  dW {dw():.3f}"
              f"  {time.time()-t0:6.0f}s{star}", flush=True)
json.dump({"config": vars(a), "history": hist_log,
           "best_update": best[1], "best_val_neg_reward": best[0]},
          open(f"{a.out}/history.json", "w"), indent=2)
print(f"\n  DONE. best checkpoint from update {best[1]}, val {best[0]:.6f} -> {a.out}/best.pt")
print("  The Chrono verdict harness has NOT been run. It runs ONCE, on this checkpoint.")
