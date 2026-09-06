"""Does contact conditioning improve prediction SPECIFICALLY before a fall?

THE CLAIM UNDER TEST. The contact-coverage collection is justified by one measured
fact: lateral contact modes lie on the path into a fall, rising 1.33% -> 13.70% in
the final second. The mechanism proposed is that a quadruped is a contact-mode
SWITCHING system and a single continuous model must average across the
discontinuities. If that is why conditioning helps, the improvement should
concentrate where the switching happens.

WHY THIS IS SHARPER THAN THE AGGREGATE HORIZON NUMBER. An aggregate moves somewhat
whatever happens -- four extra input channels are informative in general. A targeted
prediction can FAIL, and this one does if the improvement turns out to be uniform.

  improvement concentrated pre-fall -> the discontinuity mechanism is confirmed and
                                       the collection rests on a measured mechanism
  improvement uniform               -> conditioning helped for some other reason;
                                       the justification for collecting weakens

THE BAR, DECLARED BEFORE EITHER MODEL EXISTS (see VERDICT RULE below). Written now
precisely so it cannot be fitted to the result.

WINDOWS ARE WITHIN-EPISODE, so command, spawn, terrain and tilt are held fixed:
  pre-fall   the final 1 s before fell_at_s
  steady     everything before fell_at_s - 2 s

POPULATION. The 10 genuine locomotion falls, NOT the 428 episodes flagged `fell`.
94% of those collapsed during the stand-up ramp at a median 1.39 s and lie still
afterwards, carrying no pre-fall trajectory. Ten is very nearly the complete
population of real falls in this half -- a much weaker limitation than "10 of 428"
sounds, but it is still ten, and the verdict rule below refuses to call an
underpowered result uniform.
"""
from __future__ import annotations
import argparse, csv, glob, json, sys
from math import comb
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from nedm.rl.dynamics import load_frozen_dynamics  # noqa: E402

# --- the declared bar -------------------------------------------------------------
MIN_GAP = 0.10          # pre-fall relative improvement must exceed steady's by this
RECORD_DT = 0.01

# THE ERROR IS SCORED ON ONE SHARED FAMILY, NOT ON EACH MODEL'S OWN FIELDS.
# An earlier version aggregated over whatever channels each model carried. That
# makes the ratio depend on how well the CONTACT INDICATORS happen to predict --
# near-binary targets on a different scale from the physical channels, in a delta
# model that can drift on them. If they predict easily the conditioned model's
# aggregate falls for a reason unrelated to the discontinuity claim; if they drift
# it rises and conditioning looks worse than it is. Neither is the question.
#
# The contact channels are the MECHANISM, not the quantity we care about predicting.
# Body velocity is what the horizon question has always been about, both models
# predict it, and they predict it on the same scale.
# THE SHARED PAIR, NOT THE TRIPLE. The 34-channel unconditioned model has no
# vel_body_z_mps; the 40-channel conditioned one does. Selecting by the `vel_body`
# prefix therefore scores 2 channels for one model and 3 for the other -- a real
# defect dorm-pc found in its own headline comparison. Named explicitly rather than
# derived from a prefix so this version cannot inherit it.
#
# NOT an amendment. The declared principle is "score one family both models predict
# on the same scale"; this list was written before anyone knew the 34ch model lacks
# the z channel, so it was wrong RELATIVE TO the principle it implements. Changing it
# implements the declaration rather than revising it.
SCORING_FIELDS = ("vel_body_x_mps", "vel_body_y_mps")
# A correlation over fewer than this many pooled windows is not worth quoting. Set so
# the guard can actually fire on a sparse resample rather than being decorative: with
# 21 episodes contributing ~20 pre-fall anchors each the pooled count is ~420, so a
# threshold of 20 would never trigger and would be a check whose failure path is
# unreachable.
MIN_POOLED_WINDOWS = 60


def bootstrap_gap_ci(per_ep, n_boot=20000, alpha=0.05, seed=0):
    """Percentile bootstrap over EPISODES for the correlation gap.

    A correlation is computed across a set of windows, so the scored quantity has no
    per-episode value and there is nothing to take a median of -- see AMENDMENT 2.
    Resampling episodes and recomputing all four correlations per draw is what
    supplies an interval.

    per_ep: list of dicts with arrays pred_c/true_c/pred_u/true_u for each window
    type. Returns (point, lo, hi, n_used, discard_reasons).
    """
    rng = np.random.default_rng(seed)
    ne = len(per_ep)

    def gap_from(idxs):
        out = {}
        for w in ("pre", "st"):
            for m in ("c", "u"):
                pr = np.concatenate([per_ep[i][f"{w}_{m}_pred"] for i in idxs])
                tr = np.concatenate([per_ep[i][f"{w}_{m}_true"] for i in idxs])
                if len(pr) < MIN_POOLED_WINDOWS:
                    return None, f"fewer than {MIN_POOLED_WINDOWS} pooled windows"
                if np.std(pr) < 1e-9 or np.std(tr) < 1e-9:
                    return None, "near-constant series"
                out[f"{w}_{m}"] = float(np.corrcoef(pr, tr)[0, 1])
        if any(not np.isfinite(v) for v in out.values()):
            return None, "non-finite correlation"
        return ((out["pre_c"] - out["pre_u"]) - (out["st_c"] - out["st_u"])), None

    point, why = gap_from(list(range(ne)))
    if point is None:
        raise SystemExit(f"cannot compute the point estimate: {why}")
    draws = []
    discards = {}
    for _ in range(n_boot):
        g, why = gap_from(rng.integers(0, ne, ne))
        if g is None:
            discards[why] = discards.get(why, 0) + 1
        else:
            draws.append(g)
    # A BOOTSTRAP THAT DISCARDS MOST OF ITS DRAWS IS NOT THE INTERVAL IT CLAIMS.
    # Abort rather than quoting a percentile over the surviving minority, which
    # would be an interval conditioned on the resamples that happened to work.
    if len(draws) < 0.9 * n_boot:
        raise SystemExit(
            f"bootstrap discarded {n_boot - len(draws)} of {n_boot} draws "
            f"({100 * (1 - len(draws) / n_boot):.0f}%): {discards}. Refusing to "
            f"report an interval conditioned on the resamples that survived.")
    draws = np.array(draws)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi), len(draws), discards


def exact_median_ci(x, alpha=0.05):
    x = np.sort(np.asarray(x)); n = len(x)
    ks = [i for i in range(1, n // 2 + 1)
          if 2 * sum(comb(n, j) for j in range(i)) / 2 ** n <= alpha]
    if not ks:
        return float("nan"), float("nan"), 0.0
    k = max(ks)
    return float(x[k - 1]), float(x[n - k]), 1 - 2 * sum(comb(n, j) for j in range(k)) / 2 ** n


def usable_falls(root, split=None):
    """Usable locomotion falls, optionally restricted to one train/val split.

    THE SPLIT MATTERS BECAUSE BOTH SURROGATES TRAINED ON THE TRAIN EPISODES. The
    contamination is symmetric and largely cancels in a ratio of two models' errors
    on the same episode -- but the conditioned model carries four extra channels and
    therefore more capacity to memorise, and if it memorises PRE-FALL dynamics
    preferentially the gap inflates by exactly the mechanism under test. So the
    subsets are run separately rather than pooled blind:

      both agree in direction              contamination does not matter
      val inconclusive, pooled supported   consistent; val simply cannot resolve it
      val contradicted, pooled supported   the memorisation signature, and caught
    """
    out = []
    for j in sorted(glob.glob(f"{root}/**/episodes/*.json", recursive=True)):
        try:
            m = json.load(open(j))
        except Exception:
            continue
        if not m.get("fell") or m.get("fell_at_s") is None:
            continue
        if split is not None and m.get("split") != split:
            continue
        csvp = j.replace(".json", ".csv")
        try:
            rows = list(csv.DictReader(open(csvp)))
        except Exception:
            continue
        if not rows:
            continue
        t = np.array([float(x["time_s"]) for x in rows])
        ft = float(m["fell_at_s"])
        if ((t >= ft - 1.0) & (t < ft)).sum() >= 50 and (t < ft - 2.0).sum() >= 200:
            out.append((csvp, rows, t, ft))
    return out


def scoring_indices(sfields):
    """Positions of SCORING_FIELDS within this model's state vector.

    Resolved BY NAME per model, because the two models order their fields
    differently and a positional assumption is the exact trap this project has hit
    four times. Missing fields abort rather than being skipped.
    """
    missing = [f for f in SCORING_FIELDS if f not in sfields]
    if missing:
        raise SystemExit(f"model lacks scoring fields {missing}; refusing to score "
                         f"on a different quantity than declared")
    return [sfields.index(f) for f in SCORING_FIELDS]


def rollout_error(dyn, rows, idx, horizon, sfields, afields, device, score_idx):
    """Open-loop error from anchor `idx`: recorded ACTIONS, predicted STATES.

    Teacher-forcing the actions and free-running the states is what isolates the
    dynamics model; feeding recorded states back would measure one-step error only.
    """
    k = dyn.context_steps
    stride = max(1, int(round(dyn.dt_s / RECORD_DT)))
    need = (k + horizon) * stride
    if idx - k * stride < 0 or idx + horizon * stride >= len(rows):
        return None
    sel = [idx - (k - 1 - i) * stride for i in range(k)]
    S = np.array([[float(rows[s][f]) for f in sfields] for s in sel], dtype=np.float32)
    A = np.array([[float(rows[s][f]) for f in afields] for s in sel], dtype=np.float32)
    if not (np.isfinite(S).all() and np.isfinite(A).all()):
        return None
    sh = torch.tensor(S, device=device).unsqueeze(0)
    ah = torch.tensor(A, device=device).unsqueeze(0)
    errs = []
    for h in range(1, horizon + 1):
        with torch.no_grad():
            delta = dyn.model.predict_next_delta(sh, ah, terrain=None)
        nxt = sh[:, -1, :] + delta
        j = idx + h * stride
        truth = np.array([float(rows[j][f]) for f in sfields], dtype=np.float32)
        if not np.isfinite(truth).all():
            return None
        # Restricted to the shared family. The model still PREDICTS every channel --
        # the full vector feeds the next step -- only the ERROR is restricted.
        pred_s = nxt.squeeze(0)[score_idx]
        true_s = torch.tensor(truth[score_idx], device=device)
        errs.append(float(torch.linalg.vector_norm(pred_s - true_s).item()))
        act = np.array([float(rows[j][f]) for f in afields], dtype=np.float32)
        sh = torch.roll(sh, -1, dims=1); ah = torch.roll(ah, -1, dims=1)
        sh[:, -1, :] = nxt.squeeze(0)
        ah[:, -1, :] = torch.tensor(act, device=device)
    return float(np.mean(errs))


def window_error(dyn, rows, t, ft, lo, hi, horizon, sfields, afields, device,
                 score_idx, stride_n=5):
    idxs = np.where((t >= lo) & (t < hi))[0][::stride_n]
    e = [rollout_error(dyn, rows, int(i), horizon, sfields, afields, device, score_idx)
         for i in idxs]
    e = [x for x in e if x is not None]
    return float(np.median(e)) if e else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditioned", required=True)
    ap.add_argument("--unconditioned", required=True)
    ap.add_argument("--baseline-root", required=True)
    ap.add_argument("--horizon", type=int, default=5, help="model steps; 5 = 0.1 s at 50 Hz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--split", choices=["train", "val"], default=None,
                    help="restrict to one split; omit to use all usable falls")
    a = ap.parse_args()

    falls = usable_falls(a.baseline_root, a.split)
    print(f"usable locomotion falls: {len(falls)}"
          + (f"   [split = {a.split}]" if a.split else "   [all splits]"))
    print("  (episodes flagged `fell` that actually walked first; 94% of the flagged\n"
          "   population collapsed during the stand-up ramp and are excluded)")
    if not falls:
        print("\nVERDICT: NOT MEASURABLE -- no episode carries a pre-fall trajectory")
        return 2

    dyn_c = load_frozen_dynamics(a.conditioned, device=a.device)
    dyn_u = load_frozen_dynamics(a.unconditioned, device=a.device)
    sf = list(dyn_c.metadata["state_fields"]); af = list(dyn_c.metadata["action_fields"])
    su = list(dyn_u.metadata["state_fields"])
    idx_c = scoring_indices(sf); idx_u = scoring_indices(su)
    print(f"\n  scoring on {len(SCORING_FIELDS)} shared channels: {', '.join(SCORING_FIELDS)}")
    print(f"    conditioned   {len(sf)} state fields, scoring at {idx_c}")
    print(f"    unconditioned {len(su)} state fields, scoring at {idx_u}")
    if sf != su:
        print("    the two differ in width, as expected -- the conditioned model adds\n"
              "    contact channels. Both models still predict every channel they carry;\n"
              "    only the ERROR is restricted, so the ratio compares one quantity on\n"
              "    one scale rather than two aggregates over different vectors.")
    rows_out = []
    for csvp, rows, t, ft in falls:
        rec = {}
        for tag, lo, hi, dyn, fields, idx in (
                ("pre_c", ft - 1.0, ft, dyn_c, sf, idx_c),
                ("pre_u", ft - 1.0, ft, dyn_u, su, idx_u),
                ("st_c", t.min(), ft - 2.0, dyn_c, sf, idx_c),
                ("st_u", t.min(), ft - 2.0, dyn_u, su, idx_u)):
            rec[tag] = window_error(dyn, rows, t, ft, lo, hi, a.horizon, fields,
                                    af, a.device, idx)
        if any(v is None for v in rec.values()):
            continue
        r_pre = rec["pre_c"] / rec["pre_u"]
        r_st = rec["st_c"] / rec["st_u"]
        rows_out.append(dict(episode=Path(csvp).stem, fell_at_s=ft,
                             r_pre=r_pre, r_steady=r_st, gap=r_st - r_pre))
    n = len(rows_out)
    print(f"\n  {'episode':<18}{'r_prefall':>11}{'r_steady':>11}{'gap':>9}")
    for r in rows_out:
        print(f"  {r['episode']:<18}{r['r_pre']:>11.3f}{r['r_steady']:>11.3f}{r['gap']:>9.3f}")
    if n < 5:
        print(f"\nVERDICT: NOT MEASURABLE -- {n} episodes")
        return 2
    gaps = np.array([r["gap"] for r in rows_out])
    med = float(np.median(gaps)); lo, hi, cov = exact_median_ci(gaps)
    half = (hi - lo) / 2
    print(f"\n  median gap (r_steady - r_prefall) {med:+.3f}")
    print(f"  exact 95% CI [{lo:+.3f}, {hi:+.3f}] (coverage {cov:.3f}), half-width {half:.3f}")
    print(f"  bar declared in advance: median gap > {MIN_GAP} with the CI excluding 0\n")

    # VERDICT RULE, declared before the models existed.
    if lo > 0 and med > MIN_GAP:
        print("VERDICT: MECHANISM SUPPORTED -- conditioning improves prediction")
        print("  materially more in the pre-fall window than in steady locomotion.")
    elif lo > 0:
        # A real directional effect below the declared bar is NOT a uniform
        # improvement, and an earlier version of this rule labelled it one. The
        # mechanism is present and smaller than we declared material; those are
        # different findings and only the first is evidence against the mechanism.
        print("VERDICT: MECHANISM PRESENT BUT BELOW THE BAR -- the improvement IS")
        print(f"  concentrated pre-fall (CI excludes 0) but the median gap {med:+.3f} is")
        print(f"  under the {MIN_GAP} declared material in advance. The mechanism is")
        print("  supported in direction and not in magnitude; the bar is not revised.")
    elif hi < 0:
        print("VERDICT: MECHANISM CONTRADICTED -- conditioning helps LESS before a fall")
        print("  than during steady locomotion, the opposite of the proposed mechanism.")
    elif half < MIN_GAP:
        print("VERDICT: MECHANISM UNSUPPORTED -- the improvement is uniform across the")
        print("  gait. Conditioning helped for some other reason, and the contact-")
        print("  discontinuity argument for collecting more lateral data is weakened.")
    else:
        print("VERDICT: INCONCLUSIVE -- the interval is wider than the effect the bar")
        print(f"  asks about (half-width {half:.3f} against a bar of {MIN_GAP}), so this")
        print("  CANNOT distinguish a uniform improvement from an underpowered test.")
        print("  Reported as inconclusive rather than as evidence of uniformity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
