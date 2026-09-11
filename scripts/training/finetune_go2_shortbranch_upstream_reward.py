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
ap.add_argument("--freeze-encoder", action="store_true",
                help="Optimise the ACTOR only. The student encoder is 55% of the\n                      parameters and absorbed 57% of the squared displacement in\n                      every run so far, yet it was trained by supervised regression\n                      onto a privileged teacher latent -- a mapping the branch\n                      objective contains no term to preserve. A global ||dW|| budget\n                      spends itself where the parameters are, so it was mostly an\n                      encoder-drift budget.")
ap.add_argument("--objective", choices=["analytic", "ppo", "rslrl"], default="analytic",
                help="analytic: backpropagate the reward through the frozen surrogate.\n                      ppo: optimise the SAME reward in the SAME surrogate with a\n                      score-function estimator, as the recipe that produced this\n                      policy does.\n\n                      They differ in a way that is measured, not theoretical. Under\n                      backprop a term's influence is its weight times the STIFFNESS of\n                      its path through the model. dof_acc carries weight -2.5e-7 and\n                      6.7%% of the reward, but it is a squared finite difference over\n                      dt, so its path is stiffer by (1/0.02)^2 -- and it takes 47%% of\n                      the gradient at a 5-step branch and 63%% at 15, against tracking's\n                      28%% and 18%%, while tracking is 89%% of the reward's VALUE.\n                      A score-function estimator multiplies grad-log-pi by a SCALAR\n                      reward, so no term can be over-weighted by its Jacobian.\n\n                      This is not a hypothesis about which optimiser is nicer. At\n                      matched ||dW|| = 4.0 a RANDOM direction costs nothing on CRM\n                      (0.1629 vs BASE 0.1632, 37/77 episodes won) while the analytic\n                      direction costs 0.41-0.74 m/s and wins 1-9 of ~50. The damage is\n                      in the direction, not the distance.")
ap.add_argument("--ppo-sigma", type=float, default=0.15,
                help="Initial exploration std in RAW action units (recorded raw action std\n                      is ~1.5, so 0.15 is ~10%%). Learned thereafter. The CTS policy's\n                      own training sigma is NOT recoverable -- no config ships with the\n                      checkpoint and it has no std head -- so this is a choice, not a\n                      reconstruction, and it is swept rather than assumed.")
ap.add_argument("--kl-base", type=float, default=0.0,
                help="Penalise squared deviation of the mean action from the FROZEN base\n                      policy, on the states actually visited. PPO's KL controller bounds\n                      each update; nothing bounds cumulative drift, and the first PPO\n                      arm reached ||dW|| 7.58 with its best surrogate reward at the most\n                      drifted iterate -- the shape of model exploitation, where the\n                      policy finds where the learned dynamics are wrong rather than\n                      where the robot walks better. The surrogate beats persistence to\n                      about 0.5 s and is not a licence to leave the data.")
ap.add_argument("--critic-lr", type=float, default=1e-3,
                help="The critic gets its OWN optimiser at its OWN fixed rate. Sharing one\n                      with the policy broke it twice over: the KL controller throttled\n                      the policy lr to 6e-6, far too slow for a value function starting\n                      from scratch, and a single clip_grad_norm over the union let the\n                      critic's huge early gradient (returns are ~25, initial prediction\n                      ~0) consume the whole norm budget and shrink the policy update.\n                      Measured consequence: explained variance sat at 0.00-0.12, so the\n                      advantages were noise and PPO was taking KL-bounded random walks.")
ap.add_argument("--det-every", type=int, default=50,
                help="Deterministic evaluation cadence, in updates. This is both the\n                      progress signal and the checkpoint-selection metric: the\n                      stochastic rew/step at batch 64 swings 1.08-1.19 update to\n                      update and no trend can be read from it.")
ap.add_argument("--ppo-epochs", type=int, default=5)
ap.add_argument("--ppo-minibatches", type=int, default=4)
ap.add_argument("--ppo-clip", type=float, default=0.2)
ap.add_argument("--ppo-kl", type=float, default=0.01,
                help="Target KL per update. The lr is divided by 1.5 above 2x this and\n                      multiplied by 1.5 below half, as the reference PPO does. This is\n                      a real trust region: it bounds movement in ACTION space on the\n                      states actually visited. ||dW|| does not -- measured, going from\n                      ||dW|| 0.52 to 4.0 is an 8x displacement and moves behaviour only\n                      15%% -> 39%%, and under Adam it mostly counts update steps.")
ap.add_argument("--gamma", type=float, default=0.99)
ap.add_argument("--lam", type=float, default=0.95)
ap.add_argument("--entropy", type=float, default=0.01)
ap.add_argument("--grad-balance", type=float, default=0.0,
                help="Rescale each reward term so its share of the GRADIENT matches its\n                      share of the RETURN. 0 off, 1 full balancing, values between\n                      partial. This is the principled alternative to --reg-scale 0.\n\n                      --reg-scale 0 WORKS -- it is the necessary ingredient in the only\n                      configuration that beats the base policy on CRM -- but it works by\n                      DELETING reward terms because our estimator mishandles them. It is\n                      a workaround for a defect in the method, not a fix to it, and the\n                      deleted terms do real work: the arms that drop them show small but\n                      consistent drift on the axes that are meant to stay near zero.\n\n                      Backprop through a surrogate weights a term by the STIFFNESS of its\n                      path, so dof_acc -- a squared finite difference over dt, stiffer by\n                      (1/0.02)^2 -- takes 47-63%% of the gradient on 6.7%% of the reward.\n                      A score-function estimator cannot do this because it multiplies\n                      grad-log-pi by a SCALAR. Rescaling by (value share / gradient share)\n                      reproduces that term balance analytically while keeping every term.")
ap.add_argument("--balance-every", type=int, default=50,
                help="Recompute the balancing coefficients every N updates. They are a\n                      property of the CURRENT policy and the surrogate's local Jacobian,\n                      not a constant, so they drift as the policy moves.")
ap.add_argument("--probe-grad-share", action="store_true",
                help="Measure each reward term's share of the GRADIENT norm and of the\n                      reward VALUE at the base policy, then exit without training.\n                      These two shares have no reason to match: backprop through a\n                      surrogate weights a term by the stiffness of its path, while a\n                      score-function estimator (PPO) sees reward as a scalar and\n                      cannot. When they diverge sharply, the branch is optimising\n                      something other than the return.")
ap.add_argument("--min-upright", type=float, default=None,
                help="Reject branch starts whose grav_body_z exceeds -THIS, i.e. keep\n                      only branches beginning upright. 9.9%% of starts on the flat\n                      corpus begin with grav_body_z > 0 -- the robot fully INVERTED,\n                      post-fall -- where every reward term is meaningless. Off by\n                      default because it changes the branch pool and that has to be\n                      asked for. 0.7 keeps tilt under ~45 deg.")
ap.add_argument("--reg-scale", type=float, default=1.0,
                help="Multiplier on the NEGATIVE-weight (penalty) reward terms.\n                      Not a tuning knob -- a diagnostic. Measured on this pipeline,\n                      penalties are 13.9% of reward VALUE but 48-57% of GRADIENT\n                      NORM, because backprop through the surrogate weights a term\n                      by the stiffness of its path, not by its contribution to\n                      return. PPO cannot do this: its score-function estimator sees\n                      reward as a scalar. --reg-scale 0 removes the penalty gradient\n                      entirely and asks whether the tracking gradient alone can move\n                      the policy -- which also tests whether d(vel)/d(action) through\n                      the surrogate is strong enough to carry any signal at all.")
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
ap.add_argument("--anchor", type=float, default=0.0,
                help="Behaviour-cloning anchor: penalise squared action distance from the "
                     "FROZEN base policy on the same branch states. ||dW|| is a GLOBAL "
                     "trust region -- it limits how far the weights move, not where the "
                     "BEHAVIOUR changes -- so it cannot keep the policy inside the region "
                     "the surrogate was trained on. This can. 0 disables (default), so the "
                     "unanchored path stays bit-identical.")
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
if a.freeze_encoder:
    _nf = 0
    for _n, _p in policy.named_parameters():
        if "student_encoder" in _n:
            _p.requires_grad_(False); _nf += _p.numel()
    print(f"  FROZEN: student_encoder, {_nf} parameters ({100*_nf/sum(q.numel() for q in policy.parameters()):.0f}%)",
          flush=True)
# A SECOND, FROZEN copy of the same checkpoint. The anchor needs the base policy's
# action on the SAME branch state, which cannot come from the policy being optimised.
_ref = None
if a.anchor > 0.0:
    _ref = BatchedGo2Policy(torch.jit.load(a.policy, map_location=DEV)).to(DEV).eval()
    for _p in _ref.parameters(): _p.requires_grad_(False)
    print(f"  ANCHOR: behaviour-cloning to base policy, weight {a.anchor}", flush=True)
opt = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=a.lr)

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
PRF = [f"policy_raw_{n}" for n in MOTOR]   # chrono order, like JP/JV
S_all, A_all, C_all, P_all, excluded = [], [], [], [], 0
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
    # THE PREVIOUS ACTION IS RECORDED. It was being invented.
    #
    # rollout() started every branch with prev = zeros and self-fed the policy's own
    # output through the warm-up, while the true previous raw action sat in these
    # columns the whole time. Measured against the logged action for the same control
    # step, the FIRST branch action came out at RMS 1.6455 -- larger than the raw-action
    # standard deviation itself (1.539). Seeding prev from the record: 0.2672.
    # Every gradient step taken so far was differentiated through a policy whose
    # opening action was essentially uncorrelated with what the base policy does there.
    if PRF[0] not in rows[0]:
        raise SystemExit(
            f"FATAL: {p} has no {PRF[0]} column. The previous-action observation channel "
            "is not reconstructible from this corpus, and defaulting it to zeros is the "
            "bug this check exists to prevent. Re-collect with policy_raw_* recorded.")
    P_all.append(np.array([[float(r[f]) for f in PRF] for r in rows], dtype=np.float32))
print(f"  branch pool: {len(S_all)} episodes; EXCLUDED {excluded} in the verdict's cell", flush=True)

_COLLECT_TERMS = False
_BAL = None          # term -> multiplier, set by _recompute_balance()
WARM, BS = 5, a.branch_steps

# CONTROL-ROW PARITY IS A PER-EPISODE PROPERTY, NOT A CONSTANT.
# sample() hardcoded "control acts on ODD rows". That is true of the corpus it was
# written against (go2_merged: 200/200 odd) and false for roughly half the episodes of
# the two newer ones -- go2_crm_merged splits 102 even / 98 odd -- because the parity is
# set by where warmup_s + prewalk_s lands on the 0.01 s record grid, and prewalk varies.
# Read it off the data instead: the raw action is held across the decimation, so the
# rows where it CHANGES are the control rows.
def _parity(Pe):
    ch = np.nonzero(np.abs(np.diff(Pe, axis=0)).max(axis=1) > 0)[0] + 1
    ch = ch[ch > 50]                                   # skip the pre-walk hold
    if len(ch) < 20:
        raise SystemExit("FATAL: cannot determine control-row parity -- fewer than 20 "
                         "action changes found. Refusing to fall back to a hardcode.")
    bc = np.bincount(ch % 2, minlength=2)
    if bc.max() < 0.9 * bc.sum():
        raise SystemExit(f"FATAL: control rows are not on a consistent parity "
                         f"(even={bc[0]}, odd={bc[1]}). The 50 Hz-on-100 Hz assumption "
                         "does not hold for this episode.")
    return int(bc.argmax())
PAR = [_parity(Pe) for Pe in P_all]
# Upright mask over candidate branch starts. Kept as a mask rather than applied by
# rejection sampling so the acceptance rate is REPORTED -- a filter that silently drops
# most of the pool is a different experiment, not a cleaner one.
UPR = None
if a.min_upright is not None:
    _gz = GRV[2]
    UPR = [Se[:, _gz] <= -a.min_upright for Se in S_all]
    _acc = float(np.mean([m.mean() for m in UPR]))
    print(f"  upright filter: grav_body_z <= {-a.min_upright:.2f}, "
          f"{100 * _acc:.1f}% of rows accepted", flush=True)
    if _acc < 0.25:
        raise SystemExit(f"FATAL: upright filter accepts only {100*_acc:.1f}% of rows. "
                         "That is a different corpus, not a cleaner one -- refusing.")
print(f"  control-row parity: {sum(PAR)} odd, {len(PAR) - sum(PAR)} even "
      f"(was hardcoded ODD for all)", flush=True)
def sample(rng, n):
    out = []
    for _ in range(n):
        e = rng.randrange(len(S_all))
        # 2*BS already reserved the decimated span; kept explicit now that the
        # branch consumes DECIM=2 rows per policy step rather than one.
        lo, hi = L + 2 * WARM + 1, len(S_all[e]) - 2 * BS - 2
        for _try in range(50):
            b = rng.randrange(lo, hi)
            b = b if b % 2 == PAR[e] else b + 1
            if UPR is None or UPR[e][b]: break
        out.append((e, b))
    return out

def obs_from(s, cmd, prev):
    ang = s[:, ANG] * ANGS; grav = s[:, GRV]
    q = SIGN * s[:, JP][:, C2I]; qd = SIGN * s[:, JV][:, C2I]
    return torch.cat([ang, grav, cmd * CMDS, (q - DEF) * DPS, qd * DVS, prev], dim=1)

def rollout(batch, grad=True):
    # WINDOW ENDS ON THE CONTROL ROW. It ended one row short, so the warm-up ran on
    # rows b-10, b-8, ... b-2 (parity of b) and then the first branch observation was
    # taken from hs[:, -1] = row b-1, the OPPOSITE parity: a single 0.01 s gap in an
    # otherwise 0.02 s sequence, at exactly the step whose gradient matters most.
    S = torch.tensor(np.stack([S_all[e][b - L + 1:b + 1] for e, b in batch]), device=DEV)
    A = torch.tensor(np.stack([A_all[e][b - L + 1:b + 1] for e, b in batch]), device=DEV)
    cmd = torch.tensor(np.stack([C_all[e][b] for e, b in batch]), device=DEV)
    hist = policy.initial_history(len(batch), DEV)
    rhist = _ref.initial_history(len(batch), DEV) if _ref is not None else None
    # Seed from the RECORDED raw action of the control step preceding the warm-up,
    # mapped chrono -> policy order. No SIGN: policy_raw_* is the network's own output.
    _p0 = torch.tensor(np.stack([P_all[e][b - 2 * WARM - 2] for e, b in batch]),
                       device=DEV)[:, C2I]
    rprev = _p0.clone()
    prev = _p0.clone()
    with torch.no_grad():                                    # warm-up on RECORDED obs
        for k in range(WARM, 0, -1):
            s = torch.tensor(np.stack([S_all[e][b - 2 * k] for e, b in batch]), device=DEV)
            prev, hist = policy(obs_from(s, cmd, prev), hist)
            if _ref is not None:
                rprev, rhist = _ref(obs_from(s, cmd, rprev), rhist)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    errs = []; rews = []; term_acc = {}; term_seq = []
    dq_prev = None
    a_prev1 = prev.clone(); a_prev2 = prev.clone()
    with ctx:
        hs, ha = S, A
        # CONTROL RUNS AT 50 Hz; THE RECORD STEP IS 100 Hz. The warm-up above already
        # honours this -- it indexes S_all[e][b - 2*k], stride 2 -- and sample() picks an
        # odd b because "control acts on ODD rows". The branch did NOT: it called the
        # policy once per SURROGATE step, i.e. at 100 Hz, so the actor's 5-slot
        # observation buffer was flushed with double-rate samples and the branch was
        # differentiated through a controller that does not exist at deployment.
        # A displacement-matched random control cannot expose this, because RAND never
        # enters this loop -- which is precisely the signature we measured on CRM.
        #
        # One policy action now spans DECIM surrogate steps, held constant, exactly as a
        # 50 Hz PD target is held across two 100 Hz steps. The branch becomes
        # BS x 0.02 = 0.10 s -- the horizon the docstring always claimed and the gate
        # certified -- instead of the 0.05 s that was actually being optimised.
        DECIM = 2
        for _ in range(BS):
            act, hist = policy(obs_from(hs[:, -1], cmd, prev), hist)
            anchor_pen = None
            if _ref is not None:
                with torch.no_grad():
                    # SAME state, SAME command, the reference's OWN action history -- the
                    # anchor asks "what would base do here", not "what did base do before".
                    ract, rhist = _ref(obs_from(hs[:, -1], cmd, rprev), rhist)
                anchor_pen = ((act - ract) ** 2).sum(dim=1)
                rprev = ract
            # SIGN ON THE WAY BACK. The read applies it -- obs_from() does
            # `q = SIGN * s[:, JP][:, C2I]` -- and this scatter is its inverse, so it
            # must apply SIGN too. It did not, so the joint target written into the
            # surrogate's action channel was NEGATED.
            #
            # Measured against the recorded targets of an episode this very policy
            # generated: as-written corr -0.8313 (RMS 1.797), with SIGN corr +0.8313
            # (RMS 0.596). The surrogate was being told the policy commanded the
            # opposite of what it did, at every branch step.
            #
            # This is invisible to everything that guarded this pipeline. The open-loop
            # audit fed RECORDED actions, so it certified a model that was never the
            # problem. The displacement-matched random control never enters this loop.
            # Both were sound and neither could see it.
            tgt = torch.zeros_like(act).index_copy(1, C2I, SIGN * (act * ACTS + DEF))
            for _sub in range(DECIM):          # hold the action across the decimation
                newa = ha[:, -1].clone().index_copy(1, ATG, tgt)
                # ALIGN THE WINDOWS. hs was not extended before slicing while the action
                # tensor was, so the action window dropped its oldest entry and every
                # context token but the last was paired with the action one step later.
                # ~50% of consecutive action rows are identical at 100 Hz, which is why
                # this corrupted the context quietly instead of breaking outright.
                # REPLACE THE LAST ACTION, DO NOT APPEND IT.
                #
                # `newa` is the last recorded action row with the joint-target channels
                # overwritten by the policy's target -- it is built to REPLACE ha[:, -1].
                # Appending it instead ran the action sequence one step ahead of the
                # state sequence, and the `torch.cat([hs, hs[:, -1:]])` above papered
                # over the length difference by DUPLICATING the current state. The last
                # two context tokens then carried the same state under two different
                # actions: a zero-delta step that occurs nowhere in the corpus, while
                # the oldest real token was pushed out.
                #
                # Training pairs states[i] with actions[i] and targets s[i+1]-s[i], and
                # the audited open-loop rollout preserves that. So the open-loop gate
                # certified a forward pass THIS LOOP WAS NOT USING. Measured one-step,
                # driving RECORDED actions so the policy is not even involved:
                #
                #     vel_body_x_mps   nRMSE 0.119 aligned -> 0.684 here   (5.75x)
                #     yaw_rate_radps         0.110 -> 0.153                (1.39x)
                #     joint pos (12)         0.034 -> 0.039                (1.15x)
                #     joint vel (12)         0.263 -> 0.280                (1.06x)
                #
                # The damage is almost entirely on vel_body_x_mps -- the channel
                # tracking_lin_vel is built from, which is 57% of the reward. Over the
                # branch, vx nRMSE 2.89 -> 4.77. The branch gradient is rotated ~59 deg
                # (cos +0.52) and points into the opposite half-space on 1 branch in 4,
                # against a null control of +0.92 for simply dropping a context token.
                # Worst on UPRIGHT, normally-walking branches, not on fallen ones.
                #
                # This is why the reward and the gradient disagreed: the penalty terms
                # read joint channels, which survived; the tracking terms read velocity,
                # which did not.
                _sw = hs[:, -L:]
                _aw = torch.cat([ha[:, :-1], newa.unsqueeze(1)], 1)[:, -L:]
                if len(_MEMBERS) == 1:
                    d = _MEMBERS[0].predict_delta(_sw, _aw, terrain=None)[:, -1, :]
                    disagree = None
                else:
                    _ds = torch.stack([m.predict_delta(_sw, _aw, terrain=None)[:, -1, :]
                                       for m in _MEMBERS], 0)
                    d = _ds.mean(0)
                    _sd = _ds.std(0)
                    _scale = d.std(0, keepdim=True).detach().clamp_min(1e-6)
                    disagree = (_sd / _scale).mean(dim=1)
                nxt = hs[:, -1] + d
                hs = torch.cat([hs, nxt.unsqueeze(1)], 1)
                ha = torch.cat([ha, newa.unsqueeze(1)], 1)
            prev = act
            qp = SIGN * nxt[:, JP][:, C2I]          # -> POLICY/URDF frame, applied ONCE
            dqp = SIGN * nxt[:, JV][:, C2I]
            t = RT.terms(qp, dqp, dq_prev if dq_prev is not None else dqp,
                         nxt[:, [VX, VY]], nxt[:, [RR, PY_]], nxt[:, WZ], cmd,
                         act, a_prev1, a_prev2, LO, HI, HIPD, HIPI,
                         act * ACTS + DEF,          # PD target, policy frame
                         pos_z=(nxt[:, PZ] if _HEIGHT_TGT is not None else None),
                         height_target=_HEIGHT_TGT)
            if _BAL is not None:
                _r = sum(_BALW[k] * _BAL[k] * v for k, v in t.items())
            else:
                _r = RT.total(t) if a.reg_scale == 1.0 else RT.total_scaled(t, a.reg_scale)
            if anchor_pen is not None:
                _r = _r - a.anchor * anchor_pen
                term_acc["anchor"] = term_acc.get("anchor", 0.0) - float(
                    (a.anchor * anchor_pen).mean())
            if disagree is not None:
                _r = _r - a.pessimism * disagree      # trust the model less where it argues
                term_acc["pessimism"] = term_acc.get("pessimism", 0.0) - float(
                    (a.pessimism * disagree).mean())
            rews.append(_r)
            if _COLLECT_TERMS: term_seq.append(t)
            for k, v in t.items(): term_acc[k] = term_acc.get(k, 0.0) + float(v.mean())
            dq_prev = dqp; a_prev2 = a_prev1; a_prev1 = act
            errs.append(torch.stack([nxt[:, VX] - cmd[:, 0], nxt[:, VY] - cmd[:, 1],
                                     nxt[:, WZ] - cmd[:, 2]], dim=1))
    if _COLLECT_TERMS:
        return torch.stack(errs, 1), torch.stack(rews, 1), term_acc, term_seq
    return torch.stack(errs, 1), torch.stack(rews, 1), term_acc

if a.probe_grad_share:
    _COLLECT_TERMS = True
    _prm = [p for p in policy.parameters() if p.requires_grad]
    _b = sample(random.Random(1234), a.batch)
    _e, _r, _acc, _seq = rollout(_b, grad=True)
    _w = dict(RT.WEIGHTS); _w["correct_base_height"] = RT.NOT_COMPUTABLE["correct_base_height"]
    _rows = []
    for _k in _seq[0]:
        _s = sum(_w[_k] * _t[_k].mean() for _t in _seq)
        _g = torch.autograd.grad(_s, _prm, retain_graph=True, allow_unused=True)
        _n = float(torch.sqrt(sum((gi ** 2).sum() for gi in _g if gi is not None)))
        _v = float(sum(_w[_k] * _t[_k].mean() for _t in _seq))
        _rows.append((_k, _v, _n))
    _gt = sum(r[2] for r in _rows); _vp = sum(abs(r[1]) for r in _rows)
    _trk = sum(r[2] for r in _rows if r[0].startswith("tracking"))
    print(f"\n  GRADIENT SHARE vs VALUE SHARE at the base policy"
          f"  (branch {BS} steps = {BS * 0.02:.2f} s, batch {a.batch})")
    print(f"  {'term':22s} {'weighted value':>15s} {'|grad|':>12s} {'grad share':>11s}")
    for _k, _v, _n in sorted(_rows, key=lambda r: -r[2]):
        print(f"  {_k:22s} {_v:15.4f} {_n:12.4f} {100 * _n / max(_gt, 1e-12):10.1f}%")
    print(f"  {'-' * 64}")
    print(f"  tracking terms: {100 * _trk / max(_gt, 1e-12):.1f}% of gradient norm, "
          f"{100 * sum(abs(r[1]) for r in _rows if r[0].startswith('tracking')) / max(_vp, 1e-12):.1f}% "
          f"of |value|")
    raise SystemExit(0)

# ============================ PPO VIA rsl_rl ================================
# The hand-written --objective ppo above had three bugs in its first hour: GAE zeroed
# the terminal non-terminal flag so the bootstrap was multiplied by zero, the critic
# shared an optimiser with the policy and was throttled to 6e-6 by the KL controller,
# and one grad-norm clip over both let the critic's early gradient eat the budget. All
# three were in the ALGORITHM, which is the part rsl_rl already gets right and which is
# installed in this very environment. Three bugs found in an afternoon means assume more.
#
# This keeps every piece that is genuinely ours -- corpus, parity, seeded prev, the
# corrected surrogate window, the upstream reward -- and hands the algorithm to rsl_rl.
#
# The imported policy fits rsl_rl's actor contract without an adapter, which is the part
# that makes this cheap. BatchedGo2Policy does:
#     history <- cat([history[:, 1:], obs.unsqueeze(1)]);  latent <- enc(history.flat)
#     out     <- actor(cat([latent, obs]))
# so if the ENV pushes the observation and hands back the POST-PUSH history flattened to
# 225, the action is a pure function of that tensor: obs is its last 45 entries. The env
# becomes Markovian from rsl_rl's point of view, which is exactly what VecEnv wants.
#
# One thing this gets for free that the hand-written version never had: rsl_rl
# distinguishes a TIME-OUT from a TERMINATION and bootstraps the former. Our branch ends
# after --branch-steps because the surrogate is only trustworthy to ~0.5 s, which is a
# time-out, not the robot failing. Treating it as terminal is what "everything past the
# branch is worth zero" meant, and it is handled properly here.
if a.objective == "rslrl":
    from rsl_rl.algorithms import PPO as RslPPO
    from rsl_rl.modules import ActorCritic as RslActorCritic

    OBS45, HIST = policy.obs_dim, policy.hist_len
    NOBS = OBS45 * HIST

    class _ImportedActor(torch.nn.Module):
        """rsl_rl calls actor(obs); obs here is the post-push history, flattened."""
        def __init__(self, src):
            super().__init__()
            self.student_encoder = src.student_encoder
            self.actor_net = src.actor
        def forward(self, obs):
            return self.actor_net(torch.cat([self.student_encoder(obs),
                                             obs[:, -OBS45:]], dim=1))

    _ac = RslActorCritic(num_actor_obs=NOBS, num_critic_obs=NOBS, num_actions=12,
                         init_noise_std=a.ppo_sigma).to(DEV)
    _ac.actor = _ImportedActor(policy).to(DEV)
    if a.freeze_encoder:
        for _p in _ac.actor.student_encoder.parameters(): _p.requires_grad_(False)
    _alg = RslPPO(_ac, num_learning_epochs=a.ppo_epochs, num_mini_batches=a.ppo_minibatches,
                  clip_param=a.ppo_clip, gamma=a.gamma, lam=a.lam, entropy_coef=a.entropy,
                  learning_rate=a.lr, desired_kl=a.ppo_kl, schedule="adaptive", device=DEV)
    _alg.init_storage(a.batch, BS, [NOBS], [NOBS], [12])

    _W0r = {k: v.detach().clone() for k, v in _ac.actor.named_parameters()}
    def _dwr():
        return float(torch.sqrt(sum(((p_ - _W0r[k]) ** 2).sum()
                                    for k, p_ in _ac.actor.named_parameters())))
    _ROLL = ix.get("roll_rad"); _PITCH = ix.get("pitch_rad")
    if _ROLL is None or _PITCH is None:
        raise SystemExit("FATAL: --objective rslrl needs roll_rad and pitch_rad for the "
                         "upstream |roll|,|pitch| > 0.2 termination.")

    class _SurrogateEnv:
        """Fixed-size sliding windows so a per-environment reset is trivial. The window
        construction is byte-identical to the analytic branch: replace the last action,
        never append it, and never duplicate the current state."""
        def __init__(self, n, rng):
            self.n, self.rng = n, rng
            self.hs = torch.zeros(n, L, len(sf), device=DEV)
            self.ha = torch.zeros(n, L, len(af), device=DEV)
            self.cmd = torch.zeros(n, 3, device=DEV)
            self.hist = torch.zeros(n, HIST, OBS45, device=DEV)
            self.prev = torch.zeros(n, 12, device=DEV)
            # dof_acc, action_rate and action_smoothness are FUNCTIONS OF HISTORY. A first
            # version passed the current joint velocity as its own previous value and the
            # current action as both previous actions, which sets dof_acc identically to
            # zero -- silently deleting the single term this whole investigation is about,
            # and turning the run into a partial --reg-scale 0 that did not say so.
            self.dqp = torch.zeros(n, 12, device=DEV)
            self.ap1 = torch.zeros(n, 12, device=DEV)
            self.ap2 = torch.zeros(n, 12, device=DEV)
            self.fresh = torch.ones(n, dtype=torch.bool, device=DEV)
            self.age = torch.zeros(n, dtype=torch.long, device=DEV)
            self.reset_idx(torch.arange(n, device=DEV))
        def reset_idx(self, ids):
            if ids.numel() == 0: return
            batch = sample(self.rng, int(ids.numel()))
            self.hs[ids] = torch.tensor(np.stack([S_all[e][b - L + 1:b + 1] for e, b in batch]), device=DEV)
            self.ha[ids] = torch.tensor(np.stack([A_all[e][b - L + 1:b + 1] for e, b in batch]), device=DEV)
            self.cmd[ids] = torch.tensor(np.stack([C_all[e][b] for e, b in batch]), device=DEV)
            p0 = torch.tensor(np.stack([P_all[e][b - 2 * WARM - 2] for e, b in batch]),
                              device=DEV)[:, C2I]
            h = torch.zeros(int(ids.numel()), HIST, OBS45, device=DEV)
            pv = p0
            with torch.no_grad():
                for k in range(WARM, 0, -1):
                    s = torch.tensor(np.stack([S_all[e][b - 2 * k] for e, b in batch]), device=DEV)
                    o = obs_from(s, self.cmd[ids], pv)
                    h = torch.cat([h[:, 1:], o.unsqueeze(1)], 1)
                    pv = _ac.actor(h.flatten(1))
            self.hist[ids] = h; self.prev[ids] = pv
            self.ap1[ids] = pv; self.ap2[ids] = pv
            self.dqp[ids] = (SIGN * self.hs[ids][:, -1][:, JV][:, C2I])
            self.fresh[ids] = True
            self.age[ids] = 0
        def observe(self):
            o = obs_from(self.hs[:, -1], self.cmd, self.prev)
            self.hist = torch.cat([self.hist[:, 1:], o.unsqueeze(1)], 1)
            return self.hist.flatten(1)
        def step(self, act):
            tgt = torch.zeros_like(act).index_copy(1, C2I, SIGN * (act * ACTS + DEF))
            for _ in range(2):
                newa = self.ha[:, -1].clone().index_copy(1, ATG, tgt)
                _aw = torch.cat([self.ha[:, :-1], newa.unsqueeze(1)], 1)
                d = _MEMBERS[0].predict_delta(self.hs, _aw, terrain=None)[:, -1, :]
                nxt = self.hs[:, -1] + d
                self.hs = torch.cat([self.hs[:, 1:], nxt.unsqueeze(1)], 1)
                self.ha = torch.cat([self.ha[:, 1:], newa.unsqueeze(1)], 1)
            qp = SIGN * nxt[:, JP][:, C2I]; dqp = SIGN * nxt[:, JV][:, C2I]
            t = RT.terms(qp, dqp, self.dqp, nxt[:, [VX, VY]], nxt[:, [RR, PY_]], nxt[:, WZ],
                         self.cmd, act, self.ap1, self.ap2, LO, HI, HIPD, HIPI,
                         act * ACTS + DEF,
                         pos_z=(nxt[:, PZ] if _HEIGHT_TGT is not None else None),
                         height_target=_HEIGHT_TGT)
            rew = RT.total(t) if a.reg_scale == 1.0 else RT.total_scaled(t, a.reg_scale)
            self.dqp = dqp; self.ap2 = self.ap1; self.ap1 = act
            self.prev = act
            self.fresh = torch.zeros_like(self.fresh)
            self.age += 1
            fell = (nxt[:, _ROLL].abs() > 0.2) | (nxt[:, _PITCH].abs() > 0.2)
            tout = self.age >= BS
            done = fell | tout
            return rew, done, tout

    _env = _SurrogateEnv(a.batch, random.Random(a.seed + 4242))
    print(f"  rsl_rl PPO: {a.batch} envs x {BS} steps, obs {NOBS}, sigma {a.ppo_sigma}, "
          f"gamma {a.gamma}, lam {a.lam}, clip {a.ppo_clip}, KL {a.ppo_kl} (adaptive)",
          flush=True)
    _obs = _env.observe()
    _dbest = (-float("inf"), -1); _hist_log = []; _t0 = time.time()
    for u in range(1, a.updates + 1):
        _rsum = torch.zeros((), device=DEV); _rn = 0
        with torch.inference_mode(False):
            for _ in range(BS):
                _act = _alg.act(_obs, _obs)
                _rew, _done, _tout = _env.step(_act)
                _infos = {"time_outs": _tout}
                _alg.process_env_step(_rew, _done, _infos)
                _rsum = _rsum + _rew.mean(); _rn += 1
                if _done.any(): _env.reset_idx(torch.nonzero(_done).squeeze(-1))
                _obs = _env.observe()
            _alg.compute_returns(_obs)
        _losses = _alg.update()
        if u % a.det_every == 0 or u == 1:
            with torch.no_grad():
                _sv = _ac.std.data.clone(); _ac.std.data.fill_(1e-8)
                # MASK THE FALLEN. Without this, an environment that terminates keeps
                # accumulating meaningless reward for the rest of the window and the
                # metric conflates "tracks worse" with "fell over at step 3", which are
                # different failures and want different responses. The training reward
                # above auto-resets and so never sees this, which is exactly why the two
                # numbers diverged: 0.986 deterministic against 1.131 on-policy.
                _e2 = _SurrogateEnv(256, random.Random(7777))
                _o2 = _e2.observe()
                _tot = torch.zeros((), device=DEV); _wsum = torch.zeros((), device=DEV)
                _alive2 = torch.ones(256, device=DEV)
                for _ in range(BS):
                    _a2 = _ac.act_inference(_o2)
                    _r2, _d2, _t2 = _e2.step(_a2)
                    _tot = _tot + (_r2 * _alive2).sum(); _wsum = _wsum + _alive2.sum()
                    _alive2 = _alive2 * (~(_d2 & ~_t2)).float()
                    _o2 = _e2.observe()
                _dr = float(_tot / _wsum.clamp_min(1.0))
                _surv2 = float(_alive2.mean())
                _ac.std.data.copy_(_sv)
            _star = ""
            if _dr > _dbest[0]:
                _dbest = (_dr, u); _star = "  <- best"
                # export_finetuned_policy.py does load_state_dict(strict=True) into a
                # BatchedGo2Policy, whose submodules are `student_encoder` and `actor`.
                # This wrapper names the head `actor_net` to avoid shadowing rsl_rl's own
                # `actor` attribute, so the key has to be renamed on the way out or the
                # export fails with a key mismatch AFTER the whole run has finished.
                _sd = {("actor." + k[len("actor_net."):] if k.startswith("actor_net.")
                        else k): v for k, v in _ac.actor.state_dict().items()}
                torch.save({"state_dict": _sd, "update": u, "dw": _dwr(),
                            "det_rew_per_step": _dr}, f"{a.out}/best.pt")
            print(f"    [deterministic] rew/step {_dr:+.4f}  survive {_surv2:5.1%}  "
                  f"dW {_dwr():.3f}{_star}", flush=True)
            _hist_log.append({"update": u, "det_rew_per_step": _dr,
                              "det_survive": _surv2, "dw": _dwr()})
        if u % 25 == 0 or u == 1:
            print(f"  update {u:5d}  rew/step {float(_rsum)/max(_rn,1):+.4f}  "
                  f"dW {_dwr():.3f}  lr {_alg.learning_rate:.2e}  "
                  f"{time.time()-_t0:5.0f}s", flush=True)
    json.dump(_hist_log, open(f"{a.out}/history.json", "w"), indent=1)
    if _dbest[1] < 0:
        raise SystemExit("FATAL: no deterministic evaluation ran; raise --updates above "
                         "--det-every.")
    print(f"\n  DONE (rsl_rl). best deterministic rew/step {_dbest[0]:+.4f} from update "
          f"{_dbest[1]}, final ||dW|| {_dwr():.3f} -> {a.out}/best.pt")
    raise SystemExit(0)

# =============================== PPO IN THE SURROGATE ========================
# Everything above -- corpus, parity, seeded prev, obs_from, the corrected surrogate
# window, the reward -- is shared. Only the estimator changes.
if a.objective == "ppo":
    _W0 = {k: v.detach().clone() for k, v in policy.named_parameters()}
    def _dwp():
        return float(torch.sqrt(sum(((p_ - _W0[k]) ** 2).sum()
                                    for k, p_ in policy.named_parameters())))
    LOGSTD = torch.nn.Parameter(torch.full((12,), float(np.log(a.ppo_sigma)), device=DEV))
    _base = None
    if a.kl_base > 0.0:
        _base = BatchedGo2Policy(torch.jit.load(a.policy, map_location=DEV)).to(DEV).eval()
        for _p in _base.parameters(): _p.requires_grad_(False)
        print(f"  ANCHOR: squared deviation from the frozen base action, weight {a.kl_base}",
              flush=True)
    critic = torch.nn.Sequential(
        torch.nn.Linear(policy.obs_dim, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 256), torch.nn.ELU(), torch.nn.Linear(256, 1)).to(DEV)
    _ROLL = ix.get("roll_rad"); _PITCH = ix.get("pitch_rad")
    if _ROLL is None or _PITCH is None:
        raise SystemExit("FATAL: --objective ppo needs roll_rad and pitch_rad to apply the "
                         "upstream termination (|roll| or |pitch| > 0.2). This surrogate "
                         "carries neither, and a rollout that cannot END is the failure the "
                         "analytic branch already has -- refusing to reproduce it.")
    _PPARAMS = [p for p in policy.parameters() if p.requires_grad] + [LOGSTD]
    _popt = torch.optim.Adam(_PPARAMS, lr=a.lr)
    _copt = torch.optim.Adam(critic.parameters(), lr=a.critic_lr)
    _lr = a.lr

    def ppo_rollout(batch):
        """Collect one on-policy batch inside the surrogate. No graph is retained: the
        gradient comes from grad-log-pi at update time, so the horizon costs memory
        linearly rather than quadratically. The analytic branch OOMed at 25 steps on a
        24 GB card; this does not."""
        S = torch.tensor(np.stack([S_all[e][b - L + 1:b + 1] for e, b in batch]), device=DEV)
        A = torch.tensor(np.stack([A_all[e][b - L + 1:b + 1] for e, b in batch]), device=DEV)
        cmd = torch.tensor(np.stack([C_all[e][b] for e, b in batch]), device=DEV)
        hist = policy.initial_history(len(batch), DEV)
        prev = torch.tensor(np.stack([P_all[e][b - 2 * WARM - 2] for e, b in batch]),
                            device=DEV)[:, C2I]
        with torch.no_grad():
            for k in range(WARM, 0, -1):
                s = torch.tensor(np.stack([S_all[e][b - 2 * k] for e, b in batch]), device=DEV)
                prev, hist = policy(obs_from(s, cmd, prev), hist)
        hs, ha = S, A
        alive = torch.ones(len(batch), device=DEV)
        dq_prev = None; a_prev1 = prev.clone(); a_prev2 = prev.clone()
        OBS, HIS, ACT, LGP, VAL, REW, ALV = [], [], [], [], [], [], []
        with torch.no_grad():
            for _ in range(BS):
                ob = obs_from(hs[:, -1], cmd, prev)
                mu, nhist = policy(ob, hist)
                std = LOGSTD.exp()
                act = mu + std * torch.randn_like(mu)
                lgp = (-0.5 * ((act - mu) / std) ** 2 - LOGSTD
                       - 0.5 * float(np.log(2 * np.pi))).sum(1)
                OBS.append(ob); HIS.append(hist); ACT.append(act); LGP.append(lgp)
                VAL.append(critic(ob).squeeze(1)); ALV.append(alive.clone())
                hist = nhist
                tgt = torch.zeros_like(act).index_copy(1, C2I, SIGN * (act * ACTS + DEF))
                for _sub in range(2):
                    newa = ha[:, -1].clone().index_copy(1, ATG, tgt)
                    _sw = hs[:, -L:]
                    _aw = torch.cat([ha[:, :-1], newa.unsqueeze(1)], 1)[:, -L:]
                    d = _MEMBERS[0].predict_delta(_sw, _aw, terrain=None)[:, -1, :]
                    nxt = hs[:, -1] + d
                    hs = torch.cat([hs, nxt.unsqueeze(1)], 1)
                    ha = torch.cat([ha, newa.unsqueeze(1)], 1)
                prev = act
                qp = SIGN * nxt[:, JP][:, C2I]; dqp = SIGN * nxt[:, JV][:, C2I]
                t = RT.terms(qp, dqp, dq_prev if dq_prev is not None else dqp,
                             nxt[:, [VX, VY]], nxt[:, [RR, PY_]], nxt[:, WZ], cmd,
                             act, a_prev1, a_prev2, LO, HI, HIPD, HIPI,
                             act * ACTS + DEF,
                             pos_z=(nxt[:, PZ] if _HEIGHT_TGT is not None else None),
                             height_target=_HEIGHT_TGT)
                r = RT.total(t) if a.reg_scale == 1.0 else RT.total_scaled(t, a.reg_scale)
                REW.append(r * alive)
                # UPSTREAM TERMINATION. Falling must cost the future, or a policy that
                # sacrifices it pays nothing. The analytic branch has no way to express
                # this: it is a fixed-length window that always completes.
                fell = ((nxt[:, _ROLL].abs() > 0.2) | (nxt[:, _PITCH].abs() > 0.2)).float()
                alive = alive * (1.0 - fell)
                dq_prev = dqp; a_prev2 = a_prev1; a_prev1 = act
            lastv = critic(obs_from(hs[:, -1], cmd, prev)).squeeze(1)
        # `alive` here is the mask AFTER the final step. It is returned separately rather
        # than folded into lastv, because GAE needs it as the non-terminal flag for the
        # last transition, not just as a scale on the bootstrap.
        return (torch.stack(OBS), torch.stack(HIS), torch.stack(ACT), torch.stack(LGP),
                torch.stack(VAL), torch.stack(REW), torch.stack(ALV), lastv, alive)

    print(f"  PPO in surrogate: {BS} steps x {a.batch} branches = {BS * a.batch} transitions"
          f"/update, sigma init {a.ppo_sigma}, gamma {a.gamma}, lam {a.lam}, "
          f"clip {a.ppo_clip}, KL target {a.ppo_kl}", flush=True)
    rng_ppo = random.Random(a.seed + 4242)
    _dbest = (-float('inf'), -1)
    _hist_log = []; _t0 = time.time()
    for u in range(1, a.updates + 1):
        OBS, HIS, ACT, LGP, VAL, REW, ALV, lastv, endalive = ppo_rollout(sample(rng_ppo, a.batch))
        # GAE. ALV[t] is alive at the START of step t, so "did not terminate during
        # step t" is ALV[t+1], and for the last step it is the mask returned by the
        # rollout. The previous version used zeros there, which multiplied the bootstrap
        # by zero and valued everything past the branch at NOTHING -- reintroducing, inside
        # PPO, the exact defect PPO was added to remove. It made the terminal value a
        # decoration and left the horizon truncated at 0.5 s with no tail.
        ALVN = torch.cat([ALV[1:], endalive.unsqueeze(0)], 0)
        adv = torch.zeros_like(REW); nextadv = torch.zeros_like(lastv)
        for t_ in range(BS - 1, -1, -1):
            nextv = lastv if t_ == BS - 1 else VAL[t_ + 1]
            nonterm = ALVN[t_]
            delta = REW[t_] + a.gamma * nextv * nonterm - VAL[t_]
            nextadv = delta + a.gamma * a.lam * nonterm * nextadv
            adv[t_] = nextadv
        ret = adv + VAL
        fo = OBS.reshape(-1, OBS.shape[-1]); fh = HIS.reshape(-1, *HIS.shape[-2:])
        fa = ACT.reshape(-1, 12); fl = LGP.reshape(-1); fr = ret.reshape(-1)
        fv = ALV.reshape(-1)
        fadv = adv.reshape(-1)
        fadv = (fadv - fadv.mean()) / fadv.std().clamp_min(1e-8)
        n = fo.shape[0]; mb = max(1, n // a.ppo_minibatches); kls = []
        for _ep in range(a.ppo_epochs):
            for i0 in range(0, n, mb):
                sl = slice(i0, i0 + mb)
                mu, _ = policy(fo[sl], fh[sl])
                std = LOGSTD.exp()
                lgp = (-0.5 * ((fa[sl] - mu) / std) ** 2 - LOGSTD
                       - 0.5 * float(np.log(2 * np.pi))).sum(1)
                ratio = (lgp - fl[sl]).exp()
                w = fv[sl]                     # dead steps contribute nothing
                s1 = ratio * fadv[sl]
                s2 = ratio.clamp(1 - a.ppo_clip, 1 + a.ppo_clip) * fadv[sl]
                pol_loss = -(torch.min(s1, s2) * w).sum() / w.sum().clamp_min(1.0)
                vpred = critic(fo[sl]).squeeze(1)
                val_loss = (((vpred - fr[sl]) ** 2) * w).sum() / w.sum().clamp_min(1.0)
                ent = (LOGSTD.sum() + 0.5 * 12 * float(np.log(2 * np.pi * np.e)))
                # Separate graphs (vpred depends only on the critic, pol_loss only on the
                # policy), so no retain_graph is needed and each gets its own norm budget.
                _anch = 0.0
                if _base is not None:
                    with torch.no_grad():
                        _bmu, _ = _base(fo[sl], fh[sl])
                    _anch = ((((mu - _bmu) ** 2).sum(1)) * w).sum() / w.sum().clamp_min(1.0)
                _popt.zero_grad(); (pol_loss - a.entropy * ent + a.kl_base * _anch).backward()
                torch.nn.utils.clip_grad_norm_(_PPARAMS, 1.0); _popt.step()
                _copt.zero_grad(); val_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0); _copt.step()
                kls.append(float((fl[sl] - lgp).mean().abs()))
        kl = float(np.mean(kls))
        if kl > 2.0 * a.ppo_kl: _lr = max(_lr / 1.5, 1e-7)
        elif kl < 0.5 * a.ppo_kl: _lr = min(_lr * 1.5, 1e-2)
        for g in _popt.param_groups: g["lr"] = _lr
        if u % a.det_every == 0 or u == 1:
            with torch.no_grad():
                _sv = float(LOGSTD.data.exp().mean()); LOGSTD.data.fill_(-20.0)
                _o = ppo_rollout(sample(random.Random(7777), 256))
                LOGSTD.data.fill_(float(np.log(max(_sv, 1e-8))))
                _dr = float((_o[5] * _o[6]).sum() / _o[6].sum().clamp_min(1.0))
                _ds = float(_o[8].mean())
            _star = ""
            # SELECT ON THE DETERMINISTIC REWARD, NOT ON ||dW||.
            #
            # PPO takes ppo_epochs x ppo_minibatches = 20 Adam steps per update, so ||dW||
            # races ahead of learning: the first --target-dw 1.0 arm stopped after ~40
            # updates, before the critic had even warmed up. ||dW|| was already a poor
            # trust region for the analytic path (0.52 -> 4.0 is an 8x displacement and
            # moves behaviour only 15% -> 39%); for PPO it is not a trust region at all,
            # because the KL controller is. So run a fixed budget, let KL do the bounding,
            # and keep the checkpoint that is actually best under the deterministic policy.
            #
            # This is still a SURROGATE-INTERNAL metric and selecting on it cannot tell us
            # anything about Chrono. It is only honest about which of ITS OWN iterates is
            # best, which is more than --target-dw did.
            if _dr > _dbest[0]:
                _dbest = (_dr, u); _star = "  <- best"
                torch.save({"state_dict": policy.state_dict(), "update": u, "dw": _dwp(),
                            "det_rew_per_step": _dr,
                            "log_std": LOGSTD.detach().cpu()}, f"{a.out}/best.pt")
            print(f"    [deterministic] rew/step {_dr:+.4f}  survive {_ds:5.1%}  "
                  f"dW {_dwp():.3f}{_star}", flush=True)
            _hist_log.append({"update": u, "det_rew_per_step": _dr, "det_survive": _ds,
                              "dw": _dwp()})
        if u % 10 == 0 or u == 1:
            mr = float((REW * ALV).sum() / ALV.sum().clamp_min(1.0))
            surv = float(ALV[-1].mean())
            # Explained variance of the critic. If this is near zero the value function
            # is not predicting returns, the advantages are noise, and every number above
            # it is meaningless -- the one diagnostic that says whether PPO is doing
            # anything at all, as opposed to taking KL-bounded random walks.
            _ev = float(1.0 - (ret - VAL).var() / ret.var().clamp_min(1e-8))
            print(f"  update {u:5d}  rew/step {mr:+.4f}  survive {surv:5.1%}  EV {_ev:+.3f}  "
                  f"KL {kl:.5f}  lr {_lr:.2e}  sigma {float(LOGSTD.exp().mean()):.3f}  "
                  f"dW {_dwp():.3f}  {time.time() - _t0:5.0f}s", flush=True)
            _hist_log.append({"update": u, "rew_per_step": mr, "survive": surv,
                              "kl": kl, "lr": _lr, "dw": _dwp()})
        if a.target_dw is not None and _dwp() >= a.target_dw:
            d = _dwp()
            torch.save({"state_dict": policy.state_dict(), "update": u, "dw": d,
                        "log_std": LOGSTD.detach().cpu()}, f"{a.out}/best.pt")
            print(f"  update {u:5d}  ||dW|| {d:.3f} >= target {a.target_dw} -- STOPPING",
                  flush=True)
            _hist_log.append({"update": u, "dw": d, "stopped_on": "target_dw"})
            break
    json.dump(_hist_log, open(f"{a.out}/history.json", "w"), indent=1)
    if _dbest[1] < 0:
        raise SystemExit("FATAL: no deterministic evaluation ever ran, so nothing was "
                         "selected. Raise --updates above --det-every.")
    print(f"\n  DONE (ppo). best deterministic rew/step {_dbest[0]:+.4f} from update "
          f"{_dbest[1]}, final ||dW|| {_dwp():.3f} -> {a.out}/best.pt")
    raise SystemExit(0)

_BALW = dict(RT.WEIGHTS)
_BALW["correct_base_height"] = RT.NOT_COMPUTABLE["correct_base_height"]

def _recompute_balance():
    """Set each term's multiplier to (its share of |value|) / (its share of |grad|).

    Measured at the CURRENT policy, because the ratio is a property of the surrogate's
    local Jacobian and drifts as the policy moves. Anchored on the largest-value term so
    tracking keeps a multiplier near 1 and the stiff terms scale DOWN, rather than the
    whole objective being rescaled (which Adam would ignore anyway)."""
    global _BAL, _COLLECT_TERMS
    _was, _COLLECT_TERMS = _COLLECT_TERMS, True
    _bsave, _BAL = _BAL, None                 # measure the UNBALANCED gradient
    try:
        _b = sample(random.Random(a.seed + 31337), min(a.batch, 64))
        _e, _r, _acc, _seq = rollout(_b, grad=True)
        _prm = [p for p in policy.parameters() if p.requires_grad]
        _gn, _vv = {}, {}
        for _k in _seq[0]:
            _s = sum(_BALW[_k] * _t[_k].mean() for _t in _seq)
            _g = torch.autograd.grad(_s, _prm, retain_graph=True, allow_unused=True)
            _gn[_k] = float(torch.sqrt(sum((gi ** 2).sum() for gi in _g if gi is not None)))
            _vv[_k] = abs(float(_s))
    finally:
        _COLLECT_TERMS = _was
    _gt = max(sum(_gn.values()), 1e-12); _vt = max(sum(_vv.values()), 1e-12)
    _raw = {k: (_vv[k] / _vt) / max(_gn[k] / _gt, 1e-12) for k in _gn}
    _anchor = max(_vv, key=_vv.get)
    _BAL = {k: float(np.clip((_raw[k] / max(_raw[_anchor], 1e-12)) ** a.grad_balance,
                             1e-3, 1e3)) for k in _raw}
    return _anchor, _gn, _vv

if a.grad_balance > 0.0:
    _an, _g0, _v0 = _recompute_balance()
    print(f"  GRAD-BALANCE {a.grad_balance}: anchored on {_an}; multipliers "
          + ", ".join(f"{k}={_BAL[k]:.3g}" for k in sorted(_BAL, key=lambda z: -_v0[z])[:6]),
          flush=True)


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
        # MEASURE THE STOP POINT. This wrote val_neg_reward = NaN into best.pt and set
        # best = (NaN, u), overwriting whatever validation had already selected -- so in
        # --target-dw mode no checkpoint selection happened at all, contradicting the
        # protocol declared at the top of this file, and history.json got a bare NaN,
        # which is not valid JSON. The displacement stop IS the intended deliverable
        # here, so keep it; just record an honest number for it.
        with torch.no_grad():
            _parts = [rollout(VAL[i:i + 128], grad=False) for i in range(0, len(VAL), 128)]
            _vm = float(-torch.cat([q[1] for q in _parts]).mean())
        torch.save({"state_dict": policy.state_dict(), "update": u, "dw": d,
                    "val_neg_reward": _vm}, f"{a.out}/best.pt")
        print(f"  update {u:5d}  ||dW|| {d:.3f} >= target {a.target_dw}  val {_vm:.6f}"
              f"  -- STOPPING", flush=True)
        hist_log.append({"update": u, "dw": d, "val_neg_reward": _vm,
                         "stopped_on": "target_dw"})
        best = (_vm, u)
        break
    if a.grad_balance > 0.0 and u % a.balance_every == 0:
        _recompute_balance()
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
