"""STEP 3 GATE: does the surrogate transmit action influence the way Chrono does?

THE QUESTION, AND WHY IT COMES BEFORE FINE-TUNING. Fine-tuning a policy inside a
frozen surrogate only means anything if a change in the policy's ACTIONS produces
the same change in the resulting trajectory that it would in Chrono. A surrogate
can have excellent one-step accuracy and still be nearly action-blind: it predicts
the next state well because the state history already determines it, and the action
contributes little. Optimising a policy against such a model improves a number that
does not exist in the plant. This gate measures that directly and is reported BEFORE
step 4, pass or fail.

THE INTERVENTION IS A POLICY-WEIGHT PERTURBATION, not an injected action offset.
That is deliberate: fine-tuning changes weights, so this measures the class of change
fine-tuning will actually make, with the same closed-loop correlation structure.

DESIGN. For each episode, two Chrono arms that are BIT-IDENTICAL until a branch
instant and differ only after it:
  arm A  the already-collected episode (baseline policy throughout)
  arm B  same spec, baseline policy until the branch, perturbed weights after

THE BRANCH IS THE CORRECTION THAT MAKES THIS MEASURABLE, and it was added after a
control falsified the first design. Originally arm B ran the perturbed policy from
step 0. Measured consequence: at the first row of the rollout window the two Chrono
arms already differed by 0.068 m/s in body velocity, while the surrogate's two arms
start at exactly 0 by construction. The gain would then have read near zero at short
horizons for a purely bookkeeping reason -- the arms had been diverging for the whole
1.28 s history window before the comparison began -- and that artifact points the same
way as the finding it would be used to support, so it could not have been argued away
afterwards. The branch is set per episode at arm A's own recorded time at row 128, so
both Chrono arms and both surrogate arms start from a difference of exactly zero at the
same instant. Row 0 sits at t ~ 2.37 s and prewalk varies per episode, so a single
fixed branch time would not align.

A control also ruled out the competing explanation: matched arms differ by only ~9% of
the difference between two UNRELATED episodes, so they are far from decorrelated and
the flat profile was not saturation.
Both action sequences are then replayed OPEN-LOOP through the surrogate from arm A's
own initial history, so the only difference between the two surrogate rollouts is the
action sequence. Then

  d_chrono(h) = state_B(h) - state_A(h)      Chrono's response to the action change
  d_model(h)  = model_B(h) - model_A(h)      the surrogate's response to the same change

reported per horizon as a GAIN (median ||d_model||/||d_chrono||), a CORRELATION of the
magnitudes across episodes, and the COSINE between the two response vectors within each
episode.

THE COSINE IS A PASS CONDITION, NOT A DIAGNOSTIC. Gain and correlation are both
computed on norms, and two vectors of equal length can point anywhere relative to each
other -- so a surrogate whose response has the right size and the wrong direction would
pass a magnitude-only gate. For trainability direction dominates magnitude: a policy
gradient needs d(outcome)/d(action) to point the right way, and a magnitude error only
rescales the step, which optimisation absorbs, while a direction error sends the policy
the wrong way and no learning rate repairs it. Cosine >= 0.5 keeps the gradient in the
correct half-space. It is close to independent of the correlation: corr asks whether
episodes with large Chrono responses have large model responses, cosine asks whether,
within one episode, the model moved the state the way Chrono did. Gain near 1 with high correlation means the surrogate carries action
influence at the right magnitude. Gain much below 1 means it under-responds, and a
policy tuned inside it is being rewarded for changes the plant will not make.

READ THE LONG HORIZONS WITH CARE. d_chrono saturates once the two arms decorrelate,
so a gain computed past the e-folding time is comparing two saturated numbers and
means little. Horizons are reported separately for that reason and never averaged.

=======================================================================
VERDICT RULE, DECLARED BEFORE THE GATE IS RUN AND BEFORE ANY NUMBER EXISTS
=======================================================================

PRIMARY FAMILY: body_vel. The fine-tune's objective is velocity tracking, so that
is the family whose action-response has to be right. The others are reported for
diagnosis and do not enter the verdict.

VERDICT HORIZONS: 0.5 s and 1.0 s. Both must be evaluable. 0.1 s is too short for
an action difference to have moved the body; 2.0 s and 6.0 s are past the measured
e-folding time in the low-command band (0.50 s at vx 0.4), where both arms have
decorrelated and the ratio compares two saturated numbers.

  PASS        at BOTH verdict horizons: gain in [0.5, 2.0] AND corr >= 0.5
              AND median cosine >= 0.5.

  VERDICT LOGIC CORRECTED FOR SUBSEQUENT RUNS, declared 2026-09-05 before the
  rel-sigma 0.05 re-run and after the first run exposed the flaw. The original form
  -- "both horizons evaluable, else INCOMPLETE" -- conflates CANNOT CONFIRM SUCCESS
  with CANNOT CONCLUDE ANYTHING. A clean FAIL at a measurable horizon is evidence,
  and an unmeasurable horizon elsewhere does not erase it. The correct form for a
  conjunctive criterion with a missing-data branch is:

      PASS        all declared horizons evaluable AND passing
      FAIL        any horizon evaluable AND failing
      INCOMPLETE  no horizon evaluable

  Success needs complete evidence; failure needs one clear instance. The first run is
  reported under the ORIGINAL rule and is not restated. Note that both rules return
  FAIL for it -- the original by a wrong route, via a pooled veto that read 0.96 where
  the primary family's true ratio was 4.61.

  The flaw was well specified and still had an interaction nobody could see until one
  horizon went unmeasurable for one family and not another.

  WHAT A rel-sigma 0.05 RESULT CAN AND CANNOT ESTABLISH, fixed before the number
  exists. rel-sigma 0.01 is THRESHOLD-SIZED: it produces about a 0.021 m/s tracking
  shift against the acceptance criterion's 0.020 m/s threshold. rel-sigma 0.05 is
  therefore a policy change roughly FIVE TIMES LARGER than the one the criterion cares
  about. A PASS at 0.05 separates "the action pathway is broken" from "the signal is
  too small to see", which is diagnostically valuable. It does NOT establish that a
  threshold-sized fine-tune is measurable at 1.0 s, and must not be reported as
  though it did. If 1.0 s comes back marginal a second time, the reading is that
  1.0 s is not measurable with this apparatus at any practical perturbation -- not
  that the run was unlucky.
  INCOMPLETE  if at either horizon the apparatus check has err_over_signal >= 1.0
              FOR THE PRIMARY FAMILY. Corrected 2026-09-05 AFTER the first run, and
              recorded rather than quietly changed: the first version pooled the
              apparatus ratio over all 34 channels, which is dominated by joint
              velocities (magnitude ~5-15) and says nothing about body velocity
              (~0.04). On the first run the pooled ratio at 1.0 s read 0.96 -- just
              inside the veto -- while the body_vel ratio was 4.61, meaning that
              horizon was never measurable for the family the verdict rests on. The
              veto could not protect the quantity it existed to protect. This run's
              verdict is reported under the ORIGINAL rule; the correction applies to
              subsequent runs.
              -- the surrogate's own open-loop error exceeds the between-arm signal
              it is being asked to reproduce, so the gain is a ratio of two noise
              terms -- or if fewer than 8 episodes reach the horizon.
  FAIL        otherwise.

NECESSARY, NOT SUFFICIENT. This gate is an OPEN-LOOP test: both action sequences are
fixed recordings. The fine-tune runs CLOSED-LOOP, where the policy reacts to the
surrogate's own predicted state and errors compound through the feedback path. Passing
here does not clear closed-loop training, and a later closed-loop failure does not
contradict a PASS here.

WHAT A FAIL MEANS, stated now so it is not argued about later. FAIL means action
influence is not transmitted within a factor of 2 in magnitude, is not correlated
with Chrono's, or points the wrong way. It does NOT mean fine-tuning is impossible; it means a policy tuned
inside THIS surrogate would be optimising a response the plant does not have, so the
step-4 result could not be attributed to the fine-tune.

IF INCOMPLETE, the declared next action is ONE re-run at --rel-sigma 0.05, a larger
but still physical policy perturbation, with both results reported. Declared here so
that re-running is not a second bite chosen after seeing the first.

The gain band [0.5, 2.0] is deliberately generous, for the same reason the trim
bounds were: the gate is meant to exclude a surrogate that is action-BLIND or wildly
over-reactive, not to certify calibration. A 2x error in action gain is survivable by
a fine-tune; a 20x one is not.
"""
from __future__ import annotations
import argparse, csv, glob, hashlib, json, math, os, random, subprocess, sys
from math import comb
import numpy as np, torch

sys.path.insert(0, "/home/kyle/sbel/NeDM/src")
from nedm.quadruped.imported_policy import family_seed
from nedm.training.trainer import HMMWVTrainer

R_DEFAULT = "/home/kyle/sbel-artifacts/datasets/go2_comprehensive_merged/flat"
PY_ = "/home/kyle/miniconda3/envs/nedm/bin/python"
CHRONO = "/home/kyle/chrono-build/bin"
ASSETS = "/home/kyle/Documents/sbel-reproducibility/2025/multi-terrain-RL"
BASE_CKPT = "/home/kyle/sbel-artifacts/checkpoints/go2_cts_150k.pt"
PERTURB_MAX_N, GROUND_M = 120.0, 200.0
REPO = "/home/kyle/sbel/NeDM"


def spec_for(j):
    m = json.load(open(j))
    d = os.path.basename(json.load(open(j.replace(".json", ".config.json")))["output_subdir"])
    fam, idx = d[len("rigid_"):].rsplit("_", 1); idx = int(idx)
    off = int(m.get("seed_offset", 0))
    tr = random.Random(family_seed(fam, off) + 977 * idx)
    return dict(csv=j.replace(".json", ".csv"), eid=m["episode_id"], fam=fam, idx=idx, off=off,
                params=m["command_params"], duration=m["duration_s"], seed=m["seed"],
                spawn_x=m["spawn_m"][0], spawn_y=m["spawn_m"][1], heading=m["heading_deg"],
                peak=PERTURB_MAX_N * (idx % 6) / 5.0, prewalk=tr.uniform(0.0, 3.0),
                roll=tr.uniform(-3.0, 3.0), pitch=tr.uniform(-3.0, 3.0))


def arm_cmd(s, ckpt, outdir, switch_ckpt=None, switch_at=None):
    extra = ([] if switch_ckpt is None else
             ["--switch-ckpt", switch_ckpt, "--switch-ckpt-at-s", f"{switch_at:.6f}"])
    return [PY_, "scripts/collection/collect_go2_smoke.py", "--terrain", "rigid",
            "--duration-s", str(s["duration"]), "--imported-ckpt", ckpt, *extra,
            "--command-family", s["fam"], "--command-params", json.dumps(s["params"]),
            "--ground-size-m", str(GROUND_M), "--perturb-peak-n", f"{s['peak']:.1f}",
            "--prewalk-s", f"{s['prewalk']:.2f}",
            "--ground-tilt-roll-deg", f"{s['roll']:.2f}",
            "--ground-tilt-pitch-deg", f"{s['pitch']:.2f}",
            "--episode-index", str(s["idx"]), "--seed", str(s["seed"]),
            "--spawn-x-m", str(s["spawn_x"]), "--spawn-y-m", str(s["spawn_y"]),
            "--heading-deg", str(s["heading"]), "--patch-y", "4.0",
            "--output-dir", outdir, "--overwrite", "--progress-interval-s", "99"]


def env_for(s):
    return dict(os.environ, PYTHONPATH=f"{CHRONO}:{REPO}/src",
                NEDM_GO2_ASSETS=ASSETS, NEDM_SEED_OFFSET=str(s["off"]))


def run_arm(s, ckpt, outdir, switch_ckpt=None, switch_at=None):
    os.makedirs(outdir, exist_ok=True)
    p = subprocess.run(arm_cmd(s, ckpt, outdir, switch_ckpt, switch_at), env=env_for(s),
                       capture_output=True, text=True, cwd=REPO)
    got = glob.glob(f"{outdir}/episodes/*.csv")
    if p.returncode != 0 or not got:
        return None, (p.stderr or "")[-400:]
    return got[0], None


def read_arrays(csv_path, state_fields, action_fields):
    """State and action matrices, deriving the gravity channels when absent.

    Arm B comes straight from the collector and has no grav_body_* columns; the
    recorded arm A has them because add_gravity_channels.py ran afterwards. Deriving
    them here with the SAME quaternion formula keeps the two arms on one definition
    instead of one arm carrying a column the other lacks.
    """
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return None, None
    if "grav_body_x" not in rows[0]:
        for r in rows:
            w, x, y, z = (float(r["quat_e0"]), float(r["quat_e1"]),
                          float(r["quat_e2"]), float(r["quat_e3"]))
            r["grav_body_x"] = -2.0 * (x * z - w * y)
            r["grav_body_y"] = -2.0 * (y * z + w * x)
            r["grav_body_z"] = -(1.0 - 2.0 * (x * x + y * y))
    S = np.array([[float(r[f]) for f in state_fields] for r in rows], dtype=np.float64)
    A = np.array([[float(r[f]) for f in action_fields] for r in rows], dtype=np.float64)
    return S, A


def median_ci(x, alpha=0.05):
    """Exact order-statistic CI for a median. No distributional assumption, which
    matters because these ratios are heavy-tailed."""
    x = np.sort(np.asarray(x, dtype=float)); n = len(x)
    if n < 6:
        return float("nan"), float("nan")
    ks = [i for i in range(1, n // 2 + 1)
          if 2 * sum(comb(n, j) for j in range(i)) / 2 ** n <= alpha]
    if not ks:
        return float("nan"), float("nan")
    k = max(ks)
    return float(x[k - 1]), float(x[n - k])


def roll(trainer, states0, actions, steps, device):
    """Open-loop surrogate rollout: fixed initial history, supplied action sequence."""
    L = trainer.sequence_length
    hs = torch.tensor(states0[:L], dtype=torch.float32, device=device)
    ha = torch.tensor(actions[:L], dtype=torch.float32, device=device)
    out = []
    with torch.no_grad():
        for k in range(steps):
            d = trainer.model.predict_delta(hs[-L:].unsqueeze(0), ha[-L:].unsqueeze(0),
                                            terrain=None)[:, -1, :].squeeze(0)
            nxt = hs[-1] + d
            out.append(nxt)
            if L + k < actions.shape[0]:
                ha = torch.cat([ha, torch.tensor(actions[L + k], dtype=torch.float32,
                                                 device=device).unsqueeze(0)], 0)
            hs = torch.cat([hs, nxt.unsqueeze(0)], 0)
    return torch.stack(out).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--root", default=R_DEFAULT)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--rel-sigma", type=float, default=0.01)
    ap.add_argument("--horizons-s", type=str, default="0.1,0.5,1.0,2.0,6.0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--work", default="/home/kyle/sbel-artifacts/datasets/go2_gate")
    # ONLY EPISODES THIS MACHINE COLLECTED. The merged set holds two populations:
    # offset 2,000,000 from kyle-N7-B650E and 3,000,000 from kyle-sbel. Replaying a
    # kyle-sbel episode here is NOT bit-identical -- 147 columns differ from row 0 --
    # so its arms would differ by machine as well as by checkpoint. The branch
    # self-test rejects them anyway; this stops us paying for the Chrono run first.
    ap.add_argument("--require-substring", default="_s2000000_",
                    help="Only use episodes whose id contains this. Default keeps to "
                         "episodes collected on this machine, which are the only ones "
                         "that replay bit-identically here.")
    ap.add_argument("--out", default="/home/kyle/sbel-artifacts/go2_action_sensitivity.json")
    a = ap.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    ck["config"]["training"]["device"] = a.device
    tr = HMMWVTrainer(ck["config"])
    tr.model.load_state_dict(ck["model_state_dict"]); tr.model.to(tr.device).eval()
    md = json.load(open(ck["config"]["processed_dataset_dir"] + "/metadata.json"))
    sf, af, dt = md["state_fields"], md["action_fields"], md["dt_s"]
    hs_steps = [max(1, int(round(float(h) / dt))) for h in a.horizons_s.split(",")]
    print(f"surrogate {a.checkpoint}\n  state {len(sf)}D action {len(af)}D dt {dt}s "
          f"seq_len {tr.sequence_length}")

    # treated policy
    os.makedirs(a.work, exist_ok=True)
    tck = f"{a.work}/perturbed_{a.rel_sigma}.pt"
    if not os.path.exists(tck):
        m = torch.jit.load(BASE_CKPT, map_location="cpu")
        torch.manual_seed(0); n = 0
        with torch.no_grad():
            for p in m.parameters():
                if p.dim() < 2: continue
                sd = p.std()
                if not torch.isfinite(sd) or sd == 0: continue
                p.add_(torch.randn_like(p) * (a.rel_sigma * sd)); n += 1
        if n == 0:
            raise SystemExit("perturbed nothing: the two arms would be identical")
        torch.jit.save(m, tck)
        print(f"  perturbed {n} weight tensors at rel-sigma {a.rel_sigma}")

    keep = {e["episode_id"] for e in json.load(open(a.root + "/dataset_index.json"))["episodes"]
            if e.get("split") == "val"}
    js = [j for j in sorted(glob.glob(a.root + "/episodes/*.json"))
          if not j.endswith(".config.json") and os.path.basename(j)[:-5] in keep
          and (a.require_substring in j if a.require_substring else True)]
    rng = random.Random(0); rng.shuffle(js)

    rec, failed = [], 0
    for j in js:
        if len(rec) >= a.episodes: break
        s = spec_for(j)
        L = tr.sequence_length
        # Branch at arm A's OWN recorded time at row L, so the two Chrono arms are
        # identical through the history window and diverge exactly where the
        # surrogate rollout begins. Row 0 is not t=0 and prewalk varies per episode,
        # so this cannot be a constant.
        rowsA = list(csv.DictReader(open(s["csv"])))
        if len(rowsA) <= L + max(hs_steps): failed += 1; continue
        branch_at = float(rowsA[L]["time_s"])
        outdir = f"{a.work}/branch_{a.rel_sigma}/{s['eid']}"
        gotB = glob.glob(f"{outdir}/episodes/*.csv")
        if not gotB:
            got, err = run_arm(s, BASE_CKPT, outdir, switch_ckpt=tck, switch_at=branch_at)
            if got is None:
                failed += 1; print(f"  {s['eid']}: arm B failed {err[:120]}"); continue
            gotB = [got]
        SA, AA = read_arrays(s["csv"], sf, af)
        SB, AB = read_arrays(gotB[0], sf, af)
        if SA is None or SB is None: failed += 1; continue
        n = min(len(SA), len(SB))
        if n <= L + max(hs_steps): failed += 1; continue
        # ANCHORING SELF-TEST. If the branch worked, the arms are identical up to L.
        # If they are not, this episode's d_chrono starts from a nonzero offset the
        # surrogate cannot see, and the pair is unusable rather than merely noisy.
        # THRESHOLD 1e-6, NOT 0. Arm A's gravity channels were RECORDED (added
        # post-hoc from the quaternion); arm B's are DERIVED here by the same
        # formula, and the two agree to ~3e-8 in float, not exactly. A threshold of
        # 1e-9 rejected every pair and would have reported INCOMPLETE for a
        # float-representation reason. 1e-6 sits three orders above that residual and
        # four below the ~1e-2 divergence the branch actually produces.
        # ARM-B ADMISSIBILITY. Arm A is a collected episode that passed the dataset's
        # own predicate; arm B is generated here and passed nothing. Measured on 615
        # pairs: 0.8% of arm Bs DIVERGE, with action differences of 1e7-1e9 rad and
        # d_chrono 30-60x the median. Medians survive that; corr is Pearson and does
        # not, and corr is a pass condition. Hardening one arm and leaving the other
        # open is the asymmetry that produces a defect nobody looks for.
        if float(np.abs(AB).max()) > 5.0 or not np.isfinite(SB).all():
            failed += 1
            print(f"  {s['eid']}: arm B INADMISSIBLE, max|action| "
                  f"{float(np.abs(AB).max()):.4g} > 5.0 rad")
            continue
        pre = float(np.abs(SB[:L] - SA[:L]).max())
        if pre > 1e-6:
            failed += 1
            print(f"  {s['eid']}: BRANCH FAILED, arms differ by {pre:.3g} before the branch")
            continue
        steps = min(max(hs_steps), n - L)
        MA = roll(tr, SA, AA, steps, tr.device)
        MB = roll(tr, SA, AB, steps, tr.device)     # same (identical) history, arm B's actions
        rec.append(dict(eid=s["eid"], fam=s["fam"], n=n, steps=steps,
                        dc=(SB[L:L + steps] - SA[L:L + steps]),
                        dm=(MB - MA),
                        da=np.abs(AB[:steps] - AA[:steps]),
                        # APPARATUS CHECK, not part of the gate's verdict. Arm A's
                        # open-loop rollout against its OWN recording. If the surrogate
                        # is being driven wrongly -- misordered channels, wrong
                        # normalisation, a stale history -- this is large and the gain
                        # below is measuring nothing. It must be read before the gain.
                        err=np.abs(MA - SA[L:L + steps])))
        print(f"  {s['eid']:42s} {s['fam']:<12} rows {n} steps {steps}")

    if not rec:
        print("\nGATE: INCOMPLETE -- no usable episode pairs"); return 2

    report = {"checkpoint": a.checkpoint, "episodes": len(rec), "failed": failed,
              "rel_sigma": a.rel_sigma, "horizons_s": a.horizons_s, "families": {}}
    print(f"\n  usable pairs {len(rec)}, failed {failed}")
    print(f"  mean |action difference| between arms: "
          f"{np.mean([r['da'].mean() for r in rec]):.5f} rad")
    fams = {"body_vel": [i for i, f in enumerate(sf) if f.startswith("vel_body")],
            "body_rate": [i for i, f in enumerate(sf) if "rate_radps" in f or "ang_vel" in f],
            "joint_pos": [i for i, f in enumerate(sf) if f.endswith("_pos_rad")],
            "joint_vel": [i for i, f in enumerate(sf) if f.endswith("_vel_radps")],
            "gravity":   [i for i, f in enumerate(sf) if f.startswith("grav_body")]}
    print("\n=== APPARATUS CHECK: arm A open-loop error vs its own recording ===")
    print("  (read this first: if it is large the gain below is meaningless)")
    print(f"  {'horizon':>8} {'|err|':>12} {'|d_chrono|':>12} {'ratio':>8}")
    report["apparatus"] = {}
    # per-family, so the veto protects the family the verdict actually uses
    report["apparatus_by_family"] = {}
    for fname, idxs in fams.items():
        report["apparatus_by_family"][fname] = {}
        for h, hstep in zip(a.horizons_s.split(","), hs_steps):
            e = [float(np.linalg.norm(r["err"][hstep - 1, idxs])) for r in rec if r["steps"] >= hstep]
            d = [float(np.linalg.norm(r["dc"][hstep - 1, idxs])) for r in rec if r["steps"] >= hstep]
            if len(e) < 3:
                continue
            me, mdc = float(np.median(e)), float(np.median(d))
            report["apparatus_by_family"][fname][h] = dict(
                rollout_err=me, d_chrono=mdc, err_over_signal=me / max(mdc, 1e-12))
    for h, hstep in zip(a.horizons_s.split(","), hs_steps):
        e = [float(np.linalg.norm(r["err"][hstep - 1])) for r in rec if r["steps"] >= hstep]
        d = [float(np.linalg.norm(r["dc"][hstep - 1])) for r in rec if r["steps"] >= hstep]
        if len(e) < 3:
            continue
        me, mdc = float(np.median(e)), float(np.median(d))
        print(f"  {h:>8} {me:12.4g} {mdc:12.4g} {me / max(mdc, 1e-12):8.2f}")
        report["apparatus"][h] = dict(rollout_err=me, d_chrono=mdc,
                                      err_over_signal=me / max(mdc, 1e-12))
    for fname, idxs in fams.items():
        if not idxs: continue
        print(f"\n=== {fname} ({len(idxs)} channels) ===")
        print(f"  {'horizon':>8} {'d_chrono':>10} {'d_model':>10} {'gain':>7} "
              f"{'corr':>7} {'cosine':>7} {'n':>4}")
        report["families"][fname] = {}
        for h, hstep in zip(a.horizons_s.split(","), hs_steps):
            dc, dm, cs = [], [], []
            for r in rec:
                if r["steps"] < hstep: continue
                vc = r["dc"][hstep - 1, idxs]; vm = r["dm"][hstep - 1, idxs]
                nc = float(np.linalg.norm(vc)); nm = float(np.linalg.norm(vm))
                dc.append(nc); dm.append(nm)
                if nc > 1e-12 and nm > 1e-12:
                    cs.append(float(np.dot(vc, vm) / (nc * nm)))
            if len(dc) < 3:
                print(f"  {h:>8} -- too few episodes reach this horizon ({len(dc)})"); continue
            dc, dm = np.array(dc), np.array(dm)
            ratios = dm / np.maximum(dc, 1e-12)
            gain = float(np.median(ratios))
            corr = float(np.corrcoef(dc, dm)[0, 1]) if len(dc) > 2 else float("nan")
            cos = float(np.median(cs)) if cs else float("nan")
            g_lo, g_hi = median_ci(ratios)
            c_lo, c_hi = median_ci(cs) if cs else (float("nan"), float("nan"))
            _z = math.atanh(min(abs(corr), 0.999))
            _se = 1.959964 / math.sqrt(max(len(dc) - 3, 1))
            r_lo, r_hi = math.tanh(_z - _se), math.tanh(_z + _se)
            if corr < 0: r_lo, r_hi = -r_hi, -r_lo
            print(f"  {h:>8} {np.median(dc):10.4g} {np.median(dm):10.4g} "
                  f"{gain:7.3f} {corr:7.3f} {cos:7.3f} {len(dc):4d}")
            report["families"][fname][h] = dict(d_chrono=float(np.median(dc)),
                                                d_model=float(np.median(dm)),
                                                gain=gain, corr=corr, cosine=cos,
                                                gain_ci=[g_lo, g_hi], corr_ci=[r_lo, r_hi],
                                                cosine_ci=[c_lo, c_hi], n=len(dc))
    # --- verdict, against the rule declared in the docstring ------------------
    PRIMARY, VH, GLO, GHI, CMIN, NMIN = "body_vel", ["0.5", "1.0"], 0.5, 2.0, 0.5, 8
    reasons, ok = [], True
    evaluable, any_failing = [], False
    for h in VH:
        ap_ = report.get("apparatus_by_family", {}).get(PRIMARY, {}).get(h) \
            or report.get("apparatus", {}).get(h)
        fam = report["families"].get(PRIMARY, {}).get(h)
        if ap_ is None or fam is None:
            reasons.append(f"{h}s: not evaluable"); ok = None; continue
        if ap_["err_over_signal"] >= 1.0:
            reasons.append(f"{h}s: INCOMPLETE, surrogate open-loop error "
                           f"{ap_['err_over_signal']:.2f}x the between-arm signal")
            ok = None; continue
        if fam["n"] < NMIN:
            reasons.append(f"{h}s: INCOMPLETE, only {fam['n']} episodes reach it"); ok = None; continue
        # (The standalone corr power check that briefly lived here is subsumed by the
        # interval logic below, which applies the same discipline to ALL THREE
        # conditions rather than to corr alone. Keeping both would have let the corr
        # check `continue` past the gain and cosine intervals, hiding them.)
        evaluable.append(h)
        g_, c_, k_ = fam["gain"], fam["corr"], fam.get("cosine", float("nan"))
        gl, gh = fam["gain_ci"]; rl, rh = fam["corr_ci"]; kl, kh = fam["cosine_ci"]
        # EVERY condition is decided by its INTERVAL, not its point estimate. The first
        # version asked this of nothing: at n=16 the corr CI was [-0.380, +0.596] and
        # could not separate an observed 0.143 from a passing 0.5, yet it returned FAIL.
        # A point estimate that cannot exclude the threshold is not evidence either way.
        def _state(lo, hi, lo_ok, hi_ok):
            if lo != lo or hi != hi: return "INDETERMINATE"
            if lo >= lo_ok and hi <= hi_ok: return "PASS"
            if hi < lo_ok or lo > hi_ok: return "FAIL"
            return "INDETERMINATE"
        st_g = _state(gl, gh, GLO, GHI)
        st_c = _state(rl, rh, CMIN, 1.0)
        st_k = _state(kl, kh, CMIN, 1.0)
        reasons.append(f"{h}s n={fam['n']}: "
                       f"gain {g_:.3f} [{gl:.3f},{gh:.3f}] {st_g} | "
                       f"corr {c_:.3f} [{rl:.3f},{rh:.3f}] {st_c} | "
                       f"cosine {k_:.3f} [{kl:.3f},{kh:.3f}] {st_k}")
        if "FAIL" in (st_g, st_c, st_k):
            any_failing = True
            if ok is not None: ok = False
        elif "INDETERMINATE" in (st_g, st_c, st_k):
            ok = None
    # Corrected logic, declared in the docstring before the rel-sigma 0.05 run:
    #   PASS       all declared horizons evaluable AND passing
    #   FAIL       any horizon evaluable AND failing
    #   INCOMPLETE no horizon evaluable
    # Success needs complete evidence; failure needs one clear instance. The previous
    # form let an unmeasurable horizon erase a clean failure at a measurable one.
    if any_failing:
        verdict = "FAIL"
    elif not evaluable:
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS" if ok else "FAIL"
    report["verdict"] = verdict; report["verdict_reasons"] = reasons
    print(f"\n=== GATE VERDICT: {verdict} ===")
    for r in reasons:
        print(f"  {r}")
    print("  SCOPE, so a PASS is not over-read:")
    print("   - OPEN-LOOP. The fine-tune runs closed-loop, where the policy reacts to the")
    print("     surrogate's own predictions and errors compound through the feedback path.")
    print("     A PASS here is necessary but NOT sufficient for closed-loop training, and a")
    print("     later closed-loop failure does not contradict it.")
    print("   - ONE INSTANT, ONE KIND OF STATE. The weight change is applied at a single")
    print("     branch, post-128-row, mid-gait, after prewalk. A fine-tuned policy differs")
    print("     from step 0 everywhere. If action sensitivity varies with state, this")
    print("     samples a slice of it and does not establish that sensitivity holds")
    print("     everywhere.")
    print("   - RIGID TERRAIN ONLY, on val-split episodes of the merged flat set.")
    if verdict == "FAIL":
        print("  FAIL means action influence is not transmitted within 2x, is not\n"
              "  correlated, or points the wrong way. It does NOT mean fine-tuning is\n"
              "  impossible.")
    if verdict == "INCOMPLETE" and a.rel_sigma < 0.05:
        print("  Declared next action: ONE re-run at --rel-sigma 0.05, both reported.")
    elif verdict == "INCOMPLETE":
        print("  The rel-sigma 0.05 branch is already exhausted. A second unmeasurable\n"
              "  result means this horizon is not measurable with this apparatus at any\n"
              "  practical perturbation, not that the run was unlucky.")
    json.dump(report, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
