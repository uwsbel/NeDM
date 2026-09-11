"""Attribute the residual tracking error: model error, or fine-tuning shortfall?

The question, posed by the NRD author: if the fine-tuned policy reaches the commanded
velocity INSIDE the surrogate but not in Chrono, the gap is the surrogate failing to
represent Chrono. If it falls short in BOTH, the fine-tune simply has not finished the job.
Those need opposite responses -- a better model versus a better optimiser -- so the
decomposition decides where effort goes next.

Method: replay the SAME episodes the Chrono verdict used. For each, seed the surrogate from
the recorded start, close the loop with the fine-tuned policy for the full scored window,
and record the velocity the SURROGATE believes results. Compare against (a) the command and
(b) what Chrono actually produced for that same policy on that same episode.
"""
import sys, json, csv, glob, os, argparse, numpy as np, torch
# Derive the repo from THIS FILE, not from one box's layout. The hardcoded
# /home/kyle/Documents/sbel/NeDM meant the script ran only on sbel; a3 and sliger
# check the repo out at /home/kyle/sbel/NeDM and it died on cd before importing.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "src", "nedm", "quadruped"))
from policy_batched import BatchedGo2Policy
from nedm.quadruped.imported_policy import (CHRONO_TO_IMPORTED, SIGN, IMPORTED_DEFAULTS,
    ANG_VEL_SCALE, CMD_SCALE, DOF_POS_SCALE, DOF_VEL_SCALE, ACTION_SCALE)
from nedm.training.trainer import HMMWVTrainer

ap = argparse.ArgumentParser()
ap.add_argument("--policy", required=True)
ap.add_argument("--chrono", required=True, help="crmtrack json for the SAME policy")
ap.add_argument("--surrogate", default="/home/kyle/sbel-artifacts/training_runs/go2_crm_baseline_s1/checkpoints/best_val.pt")
ap.add_argument("--steps", type=int, default=150, help="control steps to roll (150 = 3 s)")
ap.add_argument("--episodes", type=int, default=30)
a = ap.parse_args()
DEV = "cuda"

ck = torch.load(a.surrogate, map_location=DEV, weights_only=False)
tr = HMMWVTrainer(ck["config"]); tr.model.load_state_dict(ck["model_state_dict"])
m = tr.model.to(DEV).eval()
for p in m.parameters(): p.requires_grad_(False)
L = tr.sequence_length
md = json.load(open(ck["config"]["processed_dataset_dir"] + "/metadata.json"))
sf, af = md["state_fields"], md["action_fields"]
ix = {n: i for i, n in enumerate(sf)}; aix = {n: i for i, n in enumerate(af)}
MOTOR = ["rr_hip","rr_thigh","rr_calf","rl_hip","rl_thigh","rl_calf",
         "fr_hip","fr_thigh","fr_calf","fl_hip","fl_thigh","fl_calf"]
JP=[ix[f"joint_{n}_pos_rad"] for n in MOTOR]; JV=[ix[f"joint_{n}_vel_radps"] for n in MOTOR]
ANG=[ix["roll_rate_radps"], ix["ang_vel_body_y_radps"], ix["yaw_rate_radps"]]
GRV=[ix["grav_body_x"], ix["grav_body_y"], ix["grav_body_z"]]
VX, VY, WZ = ix["vel_body_x_mps"], ix["vel_body_y_mps"], ix["yaw_rate_radps"]
ATG = torch.tensor([aix[f"joint_{n}_target_rad"] for n in MOTOR], device=DEV)
C2I = torch.tensor(CHRONO_TO_IMPORTED, dtype=torch.long, device=DEV)
DEF = torch.tensor(np.asarray(IMPORTED_DEFAULTS, dtype=np.float32), device=DEV)
T = lambda v: torch.as_tensor(np.asarray(v, dtype=np.float32), device=DEV)
ANGS, CMDS, DPS, DVS, ACTS = T(ANG_VEL_SCALE), T(CMD_SCALE), T(DOF_POS_SCALE), T(DOF_VEL_SCALE), T(ACTION_SCALE)
policy = BatchedGo2Policy(torch.jit.load(a.policy, map_location=DEV)).to(DEV).eval()
for p in policy.parameters(): p.requires_grad_(False)
PRF = [f"policy_raw_{n}" for n in MOTOR]

def obs_from(s, cmd, prev):
    return torch.cat([s[:, ANG]*ANGS, s[:, GRV], cmd*CMDS,
                      (SIGN*s[:, JP][:, C2I]-DEF)*DPS, SIGN*s[:, JV][:, C2I]*DVS, prev], 1)

chrono = {e["episode_id"]: e for e in json.load(open(a.chrono))
          if isinstance(e, dict) and e.get("mae_vx") is not None}
idx = json.load(open("/home/kyle/sbel-artifacts/datasets/go2_crm_merged/score_subset_index.json"))["episodes"]
rows = []
def uid(cp):
    # MATCH THE SCORER'S KEY, not the index's own episode_id field. The scorer builds ids
    # from the last four path components, so the index field and the result key differ:
    # "go2_crm_s9300000_arc_019" vs "go2_crm_9300000_c_crm_go2_crm_s9300000_arc_019".
    q = os.path.normpath(cp).split(os.sep)
    stem = os.path.splitext(q[-1])[0]
    return "_".join(q[-4:-2] + [stem]) if len(q) >= 4 else stem

for e in idx:
    eid = uid(e["csv_path"])
    if eid not in chrono: continue
    try: R = list(csv.DictReader(open(e["csv_path"])))
    except Exception: continue
    if len(R) < L + 2*a.steps + 40: continue
    P = np.array([[float(r[f]) for f in PRF] for r in R], dtype=np.float32)
    ch = np.nonzero(np.abs(np.diff(P, axis=0)).max(axis=1) > 0)[0] + 1
    ch = ch[ch > 50]
    if len(ch) < 20: continue
    par = int(np.bincount(ch % 2, minlength=2).argmax())
    b = L + 20; b = b if b % 2 == par else b + 1
    S = np.array([[float(r[f]) for f in sf] for r in R], dtype=np.float32)
    A = np.array([[float(r[f]) for f in af] for r in R], dtype=np.float32)
    cmd = torch.tensor([[float(R[b]["cmd_vx_mps"]), float(R[b]["cmd_vy_mps"]),
                         float(R[b]["cmd_wz_radps"])]], device=DEV)
    hs = torch.tensor(S[b-L+1:b+1], device=DEV).unsqueeze(0)
    ha = torch.tensor(A[b-L+1:b+1], device=DEV).unsqueeze(0)
    prev = torch.tensor(P[b-12], device=DEV).unsqueeze(0)[:, C2I]
    hist = policy.initial_history(1, DEV)
    with torch.no_grad():
        for k in range(5, 0, -1):
            s = torch.tensor(S[b-2*k], device=DEV).unsqueeze(0)
            prev, hist = policy(obs_from(s, cmd, prev), hist)
        vx = []
        for _ in range(a.steps):
            act, hist = policy(obs_from(hs[:, -1], cmd, prev), hist)
            tgt = torch.zeros_like(act).index_copy(1, C2I, SIGN*(act*ACTS+DEF))
            for _sub in range(2):
                newa = ha[:, -1].clone().index_copy(1, ATG, tgt)
                _aw = torch.cat([ha[:, :-1], newa.unsqueeze(1)], 1)
                d = m.predict_delta(hs, _aw, terrain=None)[:, -1, :]
                nxt = hs[:, -1] + d
                hs = torch.cat([hs[:, 1:], nxt.unsqueeze(1)], 1)
                ha = torch.cat([ha[:, 1:], newa.unsqueeze(1)], 1)
            prev = act
            vx.append(float(nxt[0, VX]))
    rows.append(dict(eid=eid, cmd=float(cmd[0,0]),
                     surro=float(np.mean(np.abs(vx))),
                     chrono=abs(float(chrono[eid]["ach_vx"]))))
    if len(rows) >= a.episodes: break

import statistics as st
c  = st.mean(abs(r["cmd"]) for r in rows)
su = st.mean(r["surro"] for r in rows)
crn= st.mean(r["chrono"] for r in rows)
print(f"\n  episodes: {len(rows)}   rollout {a.steps} control steps = {a.steps*0.02:.1f} s")
print(f"  |commanded vx|                    {c:.4f}")
print(f"  |achieved vx| the SURROGATE says  {su:.4f}   ({100*su/c:5.1f}% of command)")
print(f"  |achieved vx| CHRONO actually got {crn:.4f}   ({100*crn/c:5.1f}% of command)")
print(f"\n  shortfall inside the model  (fine-tuning left on the table) {c-su:+.4f}")
print(f"  extra shortfall in Chrono   (model over-promises)           {su-crn:+.4f}")
tot = c - crn
if abs(tot) > 1e-9:
    print(f"\n  attribution of the {tot:.4f} total gap:")
    print(f"    fine-tuning shortfall : {100*(c-su)/tot:5.1f}%")
    print(f"    model / sim mismatch  : {100*(su-crn)/tot:5.1f}%")
