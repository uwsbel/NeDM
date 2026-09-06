"""W0: is the step-to-step map a function of the pre-impact state at all?

THE QUESTION, AND WHY IT COMES FIRST. The joint-level plan's highest-ceiling item is an
event-indexed (Poincare) formulation: model x_{k+1} = f(x_k) between foot strikes instead
of a fixed-dt transition. That construction assumes the map IS a function of the robot
state. On granular terrain it may not be: soil carries its own state with memory, and a
trot re-loads ground it already compacted, so two identical pre-impact states landing on
loose vs packed soil evolve differently. This script measures that BEFORE any architecture
is built. See docs/state/decisions/quadruped-joint-level-plan.md, W0.

THE DECISION RULE, FIXED BEFORE THE NUMBERS EXIST:
  (all rules below read the INCREMENT R^2)
  R^2 high on rigid and low on crm  -> the deterministic event map is dead here; W4 does
                                        not run, and a terrain-state input is mandatory.
  R^2 comparable and non-trivial    -> the formulation is viable; W4 is licensed.
  R^2 low on BOTH                   -> suspect the apparatus, not the physics. The most
                                        likely cause is event-time quantisation, which
                                        this script measures and reports (see below).

SCORE THE INCREMENT, NOT THE LEVEL. R^2 for x_{k+1} given x_k is inflated by persistence:
a pure random walk scores near 1.0 on the level while carrying no learnable dynamics at
all, because x_k is already most of the answer. The decision metric is therefore R^2 on
the INCREMENT, dx = x_{k+1} - x_k, against the mean increment. The level-R^2 is printed
beside it as a persistence diagnostic and is NOT the verdict. This was caught by a
synthetic case that scored 0.97 on the level while the increment was unlearnable.

R^2 IS ALREADY THE COMPARISON AGAINST THE MEAN PREDICTOR. R^2 <= 0 on a held-out split
means the fit does no better than predicting the training mean, which is the null this
test is built around. Reported per state component, never pooled: a single averaged R^2
would let well-predicted joint angles hide an unpredictable body height.

EVENT DETECTION USES THE REPO'S HYSTERETIC DETECTOR, NOT A THRESHOLD. dataset.contact_mode
is a measured Schmitt trigger (5 N release / 60 N engage). Its justification is directly
load-bearing here: on CRM a plain threshold fires at 1.66x the rate implied by the gait's
own spectral peak, while the Schmitt trigger gives 0.96x. A naive rising edge would
manufacture events on soil and not on rigid -- i.e. it would fabricate exactly the
rigid-vs-soil difference this script exists to measure.

THAT REASONING IS ABOUT CRM AND THIS HARNESS ONLY EVER RAN ON RIGID, where
foot_*_in_contact records what Chrono's contact container actually resolved and the
proxy was never needed. The Schmitt constants were tuned where no ground truth exists;
on rigid they agree with it on 70.9% of samples, worst of every threshold in the band
(see contact_mode's docstring). The section foot is fl and the drop rule is keyed to
rr -- the two feet the proxy handles WORST, at 72.2% and 41.2%, because they carry the
least load and a 60 N engage rarely latches for them.

MEASURED CONSEQUENCE for rr touchdown detection, 60 rigid episodes:

    detector             median events   zero-event episodes
    contact_mode proxy        94                 10
    stored in_contact         78                  7

  On the 49 episodes where both detect events the proxy finds 1.08x as many -- mild
  chatter, not under-detection. But it finds NONE on 4 episodes where ground truth
  finds events, against 1 the other way. So the failure is bimodal: slight
  over-detection where it works, total failure on ~7% of episodes.

  Both modes handicap the EVENT-INDEXED arm specifically, since that arm's sampling
  depends on detecting the event; the fixed-dt arm is indifferent. W4 is recorded
  INCONCLUSIVE and this is not a reason to re-run it -- but if the event-indexed
  formulation is ever revisited, READ foot_*_in_contact ON RIGID rather than
  contact_mode, or it inherits this handicap and reaches the same answer for a reason
  nobody would look for twice.

R^2 IS UNDEFINED WHERE THE SECTION STATE DOES NOT VARY. If every episode converges to
the same limit cycle, the pre-impact state is nearly constant, ss_tot collapses, and R^2
sits near zero no matter how good the model is. That is exactly HALO's Figure 5, where
error falls with rollout index because the test distribution collapses onto the fixed
point. This script therefore reports each component's SPREAD alongside its R^2 and marks
components whose spread is too small to score. A near-zero R^2 on a degenerate component
is not evidence that the map is unpredictable.

THE ABSOLUTE R^2 CANNOT LICENSE W4, AND --fixed-dt IS WHY. "High licenses the event-
indexed formulation, low kills it" has no number attached, and picking the boundary after
seeing the value is the failure this whole harness exists to prevent. The decidable
question is not "is the map predictable" but "is EVENT indexing better than TIME
indexing", because that difference is W4's entire premise. --fixed-dt samples pairs at a
regular interval instead of at touchdowns, using an identical state vector and frame rule,
so the only thing that differs is when the samples are taken. HALO never ran this control;
its gait is clock-driven, so its Poincare map is approximately a fixed-dt flow map, and it
reports no baseline that would have shown that.

NOT EVERY DETECTED PAIR IS A POINCARE RETURN. A section crossing separated from the next
by many seconds is not a consecutive return -- the robot stood, fell, or the section was
missed. Such pairs are not hard cases, they are different objects, and they depress R^2
without carrying information about the map. --gait-band drops pairs whose interval falls
outside [lo, hi] times the median interval. It is OFF by default so the unfiltered number
is always visible, and when on the script also reports R^2 on the DROPPED pairs: if those
score like the kept ones, the contamination story is wrong and the filter is laundering
the result rather than cleaning it. Set the band before looking at the R^2.

THE SAMPLING CAVEAT IS REPORTED, NOT ASSUMED AWAY. Events are located to the nearest
logged row. At a 50 Hz log rate that is up to 20 ms of quantisation, sampled at the
instant of peak state derivative, which converts to millimetres of body-height error. This
script reports the log rate, the implied quantisation, and the resulting body-height error
scale next to the R^2 so the two can be compared. If the quantisation error is a large
fraction of the signal's own spread, the R^2 is bounded by the apparatus and the honest
verdict is "not measurable at this log rate" rather than a null.
"""
from __future__ import annotations
import argparse, csv, glob, hashlib, json, math, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, "src")
from nedm.quadruped.dataset import contact_mode, LEG_ORDER

SECTION_FOOT = "fl"            # diagonal partner is rr; see plan W0
STANCE_PARTNER = "rr"


def _yaw_frame(row, partner_xyz):
    """Base state re-expressed in a yaw-aligned frame at the stance foot.

    Mirrors NeRD's robot-centric reduction: translation and heading are quotiented out,
    everything else is expressed relative. Roll and pitch are already yaw-invariant and
    pass through. This is the ONLY part of the state that needs a frame; joint angles and
    joint velocities are reduced coordinates and are invariant by construction.
    """
    yaw = row["yaw_rad"]
    c, s = math.cos(-yaw), math.sin(-yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    p = np.array([row["pos_x_m"], row["pos_y_m"], row["pos_z_m"]]) - partner_xyz
    v = np.array([row["vel_world_x_mps"], row["vel_world_y_mps"], row["vel_world_z_mps"]])
    w = np.array([row["ang_vel_world_x_radps"], row["ang_vel_world_y_radps"],
                  row["ang_vel_world_z_radps"]])
    return np.concatenate([R @ p, R @ v, R @ w, [row["roll_rad"], row["pitch_rad"]]])


def build_state(row, joint_cols, partner_xyz):
    return np.concatenate([_yaw_frame(row, partner_xyz),
                           np.array([row[c] for c in joint_cols])])


def state_names(joint_cols):
    return (["px_rel", "py_rel", "pz_rel", "vx", "vy", "vz", "wx", "wy", "wz",
             "roll", "pitch"] + list(joint_cols))


def rising_edges(stance):
    """Indices of the LAST sample before each stance onset -- the pre-impact row."""
    st = np.asarray(stance, dtype=bool)
    if st.size < 2:
        return []
    return list(np.flatnonzero((~st[:-1]) & st[1:]))


def load_episode(path, joint_cols):
    cols = defaultdict(list)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            for k, v in row.items():
                cols[k].append(v)
    out = {}
    for k, v in cols.items():
        try:
            out[k] = np.array(v, dtype=float)
        except ValueError:
            out[k] = np.array(v, dtype=object)
    return out


def episode_is_diverged(ep, joint_cols, bound):
    """True if any numeric channel leaves a physically absurd bound.

    W0 HAS NO MAGNITUDE FILTER OF ITS OWN, and its only exclusion -- dropping
    episodes with no detectable stance events -- protects it by accident. Divergence
    usually destroys foot contact, so most diverged episodes are dropped anyway; of
    eight tested, six were and two were not. Two in eight is enough to matter, because
    R^2 is not robust and a single row carrying 1e34 sets the residual for the fit.

    Measured cost of not having this: on go2_joint_off3000000, 21% of the 400 episodes
    used carry at least one diverged row, and excluding them moves the median ridge
    increment R^2 from 0.182 to 0.378 -- on FEWER episodes, so it is not a volume
    effect. Every published W0 figure predating this flag was measured with the
    contamination in.

    The bound is deliberately far beyond anything physical, so a trip means numerical
    divergence and not an aggressive episode.
    """
    for k, v in ep.items():
        if v.dtype != float:
            continue
        if not np.isfinite(v).all():
            return True
        if np.abs(v).max() > bound:
            return True
    return False


def episode_events(ep, joint_cols, fixed_dt=None, log_dt=None):
    """States at section-foot touchdown, or -- as a control -- at a fixed interval.

    The frame anchor is the stance partner's foot position at the sampled instant, which
    is defined at any time and not only at touchdown, so both arms use an identical state
    construction and differ ONLY in when they sample.
    """
    # ONE call on the stacked 4-column force array: that is contact_mode's designed
    # input and the only form whose bit packing matches LEG_ORDER.
    try:
        fz = np.column_stack([ep[f"foot_{leg}_force_fz_n"] for leg in LEG_ORDER])
    except KeyError:
        return None
    st, _mode = contact_mode(fz)
    stance = {leg: st[:, k] for k, leg in enumerate(LEG_ORDER)}
    if fixed_dt is not None:
        step = max(1, int(round(fixed_dt / log_dt)))
        idx = list(range(0, len(ep["time_s"]), step))
    else:
        idx = rising_edges(stance[SECTION_FOOT])
    if len(idx) < 3:
        return None
    t = ep["time_s"]
    states, times = [], []
    for i in idx:
        partner = np.array([ep[f"foot_{STANCE_PARTNER}_pos_{a}_m"][i] for a in "xyz"])
        row = {k: (ep[k][i] if ep[k].dtype != object else 0.0) for k in ep}
        states.append(build_state(row, joint_cols, partner))
        times.append(t[i])
    return (np.array(states), np.array(times), stance, t)


def ridge_fit(X, Y, lam):
    Xa = np.hstack([X, np.ones((len(X), 1))])
    A = Xa.T @ Xa + lam * np.eye(Xa.shape[1])
    A[-1, -1] -= lam
    return np.linalg.solve(A, Xa.T @ Y)


def ridge_pred(W, X):
    return np.hstack([X, np.ones((len(X), 1))]) @ W


def knn_pred(Xtr, Ytr, Xte, k, chunk: int = 512):
    """Chunked over queries. The naive form builds an (n_te, n_tr, d) tensor, which is
    32.9 GiB on a 43k-event rigid set -- it ran on the synthetic test and died on real
    data. Chunking changes no number (verified identical to the unchunked form at
    k=1/5/16, max abs diff 0.00e+00)."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    an = (A ** 2).sum(1)
    out = np.empty((len(B), Ytr.shape[1]), dtype=float)
    for i in range(0, len(B), chunk):
        b = B[i:i + chunk]
        d = an[None, :] - 2.0 * (b @ A.T) + (b ** 2).sum(1)[:, None]
        nn = np.argpartition(d, min(k, d.shape[1] - 1), axis=1)[:, :k]
        out[i:i + chunk] = Ytr[nn].mean(1)
    return out


def r2_per_component(Yte, Pred, Ytr_mean, min_spread):
    """R^2 against the TRAINING mean -- the null this test is built around.

    Components whose held-out spread is below min_spread are returned as NaN rather than
    as a number near zero: with no variation to explain there is nothing for R^2 to mean,
    and reporting 0.00 there would read as "unpredictable" when it is "constant".
    """
    ss_res = ((Yte - Pred) ** 2).sum(0)
    ss_tot = ((Yte - Ytr_mean) ** 2).sum(0)
    ok = Yte.std(0) >= min_spread
    return np.where(ok & (ss_tot > 1e-12), 1.0 - ss_res / np.maximum(ss_tot, 1e-12), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="episode CSV glob")
    ap.add_argument("--label", required=True, help="terrain label, e.g. rigid or crm")
    ap.add_argument("--max-episodes", type=int, default=400)
    ap.add_argument("--knn-k", type=int, default=10)
    ap.add_argument("--ridge-lam", type=float, default=1.0)
    ap.add_argument("--split-seed", type=int, default=0,
                    help="episode-split seed; vary it to measure split-to-split spread, "
                         "which is the yardstick any arm-vs-arm gap must clear")
    ap.add_argument("--fixed-dt", type=float, default=None, metavar="SECONDS",
                    help="CONTROL ARM: pair states every SECONDS instead of at touchdowns. "
                         "Same state vector, same frame rule. W4 is licensed only if the "
                         "event-indexed arm BEATS this.")
    ap.add_argument("--gait-band", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="keep pairs whose inter-event interval is in [LO,HI]*median. "
                         "PRE-REGISTER THIS before reading any R^2.")
    ap.add_argument("--restrict-to", default=None, metavar="FILE",
                    help="newline-delimited episode paths; use ONLY these. Feed it the\n"
                         "intersection of two arms' used_episodes to make the arms paired.")
    ap.add_argument("--max-abs-state", type=float, default=None, metavar="BOUND",
                    help="drop episodes where any numeric channel exceeds BOUND in "
                         "absolute value, or is non-finite. Off by default so old runs "
                         "reproduce; 1e4 is a sane setting and 1e34 values have been "
                         "observed. See episode_is_diverged.")
    ap.add_argument("--min-spread", type=float, default=1e-4,
                    help="held-out sd below which a component is unscoreable, not unpredictable")
    ap.add_argument("--summary-json", default=None)
    a = ap.parse_args()

    paths = sorted(glob.glob(a.glob))[: a.max_episodes]
    if a.restrict_to:
        keep = {ln.strip() for ln in open(a.restrict_to) if ln.strip()}
        paths = [q for q in sorted(glob.glob(a.glob)) if q in keep]
        print(f"restricted to {len(paths)} episodes from {a.restrict_to}")
    if not paths:
        sys.exit(f"no episodes matched {a.glob}")

    probe = load_episode(paths[0], [])
    joint_cols = [c for c in probe
                  if c.startswith("joint_") and (c.endswith("_pos_rad") or c.endswith("_vel_radps"))]
    if not joint_cols:
        sys.exit("no joint pos/vel columns found; check the schema")
    joint_cols = sorted(joint_cols)

    dt = float(np.median(np.diff(probe["time_s"])))
    per_ep, per_ep_t, ev_dt, diag_off, n_drop, n_diverged = [], [], [], [], 0, 0
    used_paths = []
    for p in paths:
        ep = load_episode(p, joint_cols)
        if a.max_abs_state is not None and episode_is_diverged(ep, joint_cols, a.max_abs_state):
            n_diverged += 1
            continue
        got = episode_events(ep, joint_cols, a.fixed_dt, dt)
        if got is None:
            n_drop += 1
            continue
        S, T, stance, t = got
        used_paths.append(p)
        per_ep.append(S)
        per_ep_t.append(T)
        ev_dt.extend(np.diff(T).tolist())
        pe = rising_edges(stance[STANCE_PARTNER])
        for i in rising_edges(stance[SECTION_FOOT]):
            if pe:
                j = min(pe, key=lambda q: abs(t[q] - t[i]))
                diag_off.append(t[j] - t[i])

    if len(per_ep) < 10:
        sys.exit(f"only {len(per_ep)} usable episodes; need >= 10")

    # SPLIT BY EPISODE, never by event. Events within an episode share a soil realisation
    # and an initial condition, so a row-wise split leaks and inflates R^2.
    rng = np.random.default_rng(a.split_seed)
    order = rng.permutation(len(per_ep))
    n_tr = int(0.8 * len(per_ep))
    tr, te = order[:n_tr], order[n_tr:]
    med_iv = float(np.median(ev_dt))
    def pack(idxs, keep=True):
        Xs, Ys = [], []
        for i in idxs:
            S, T = per_ep[i], per_ep_t[i]
            iv = np.diff(T)
            if a.gait_band is None:
                m = np.ones(len(iv), dtype=bool)
            else:
                lo, hi = a.gait_band
                m = (iv >= lo * med_iv) & (iv <= hi * med_iv)
            m = m if keep else ~m
            if m.any():
                Xs.append(S[:-1][m]); Ys.append(S[1:][m])
        if not Xs:
            return np.zeros((0, per_ep[0].shape[1])), np.zeros((0, per_ep[0].shape[1]))
        return np.vstack(Xs), np.vstack(Ys)

    Xtr, Ytr = pack(tr)
    Xte, Yte = pack(te)
    n_all = sum(len(s) - 1 for s in per_ep)
    # THE DECISION TARGET IS THE INCREMENT. See the module docstring.
    Dtr, Dte = Ytr - Xtr, Yte - Xte

    names = state_names(joint_cols)
    R2d = lambda P: r2_per_component(Dte, P, Dtr.mean(0), a.min_spread)
    R2l = lambda P: r2_per_component(Yte, P, Ytr.mean(0), a.min_spread)
    Wd = ridge_fit(Xtr, Dtr, a.ridge_lam)
    res = {
        "ridge": R2d(ridge_pred(Wd, Xte)),
        f"knn{a.knn_k}": R2d(knn_pred(Xtr, Dtr, Xte, a.knn_k)),
    }
    level = {"ridge(level)": R2l(ridge_pred(ridge_fit(Xtr, Ytr, a.ridge_lam), Xte))}
    spread = Dte.std(0)
    # SPLIT THE HANDLERS. A missing sklearn is an acceptable skip; a broken MLP arm is
    # not, and a bare `except` made the two indistinguishable. This block referenced an
    # undefined `sy` and the LEVEL target from the moment the decision metric became the
    # increment, and every run since has printed "(MLP skipped: ...)" and carried on, so
    # any MLP column in an older table was never populated.
    try:
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        print(f"  (MLP skipped, sklearn unavailable: {e})")
    else:
        sx = StandardScaler().fit(Xtr)
        sy = StandardScaler().fit(Dtr)          # the INCREMENT, matching R2d
        # KEYWORD, NOT POSITIONAL. sklearn 1.8 made `loss` the first positional
        # parameter; 1.7 had `hidden_layer_sizes` there. The positional form meant
        # different things in the two analysis envs on this box -- it trained a net
        # under one and raised InvalidParameterError under the other.
        m = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=400, random_state=0)
        m.fit(sx.transform(Xtr), sy.transform(Dtr))
        res["mlp"] = R2d(sy.inverse_transform(m.predict(sx.transform(Xte))))

    # THE APPARATUS CHECK. Event-time quantisation converts directly into pre-impact state
    # error at the instant of peak derivative. If that error is a large fraction of the
    # component's own spread, R^2 on that component is bounded by the log rate.
    vz = np.abs(np.vstack([s for s in per_ep])[:, 5])
    quant_pz = float(np.median(vz) * dt)
    spread_pz = float(np.vstack([s for s in per_ep])[:, 2].std())

    arm = f"FIXED-dt {a.fixed_dt:.3f}s (CONTROL)" if a.fixed_dt else "EVENT-INDEXED"
    print(f"\n=== W0  {a.label}   [{arm}] ===")
    print(f"episodes {len(per_ep)} usable, {n_drop} dropped, {n_diverged} diverged   events {sum(len(s) for s in per_ep)}"
          f"   train {len(Xtr)} / test {len(Xte)}")
    print(f"log dt {dt*1000:.1f} ms  ({1/dt:.0f} Hz)")
    print(f"inter-event interval  mean {np.mean(ev_dt):.3f} s  sd {np.std(ev_dt):.3f} s"
          f"  ({100*np.std(ev_dt)/max(np.mean(ev_dt),1e-9):.1f}% of mean)")
    if diag_off:
        print(f"diagonal pair offset   mean {np.mean(diag_off)*1000:+.1f} ms"
              f"  sd {np.std(diag_off)*1000:.1f} ms  |max| {np.max(np.abs(diag_off))*1000:.1f} ms")
    print(f"APPARATUS: quantisation-implied pz error ~{quant_pz*1000:.1f} mm"
          f"  vs pz spread {spread_pz*1000:.1f} mm"
          f"   ratio {quant_pz/max(spread_pz,1e-9):.2f}"
          f"{'   <- BOUNDED BY LOG RATE' if quant_pz > 0.3*spread_pz else ''}")
    # THE FALSIFICATION CHECK ON THE FILTER ITSELF. If the pairs the band REMOVED are
    # about as predictable as the ones it kept, the band is not removing contamination.
    dropped_note = ""
    if a.gait_band is not None:
        Xd, Yd = pack(te, keep=False)
        if len(Xd) > 30:
            Dd = Yd - Xd
            r_drop = np.nanmedian(r2_per_component(Dd, ridge_pred(Wd, Xd), Dtr.mean(0), a.min_spread))
            r_keep = np.nanmedian(res["ridge"])
            dropped_note = (f"  band kept {len(Xte)}/{n_all} pairs "
                            f"(median interval {med_iv:.3f} s, band "
                            f"[{a.gait_band[0]*med_iv:.3f}, {a.gait_band[1]*med_iv:.3f}] s)\n"
                            f"  ridge increment R^2  KEPT {r_keep:.3f}   DROPPED {r_drop:.3f}")
            if r_drop > r_keep - 0.05:
                dropped_note += ("\n  WARNING: dropped pairs score like kept ones. The band is NOT\n"
                                 "  removing contamination; do not report the filtered number as cleaner.")
    # COUNT THE NaNs ACTUALLY RETURNED, not a proxy for them. r2_per_component returns
    # NaN through TWO paths -- held-out sd below min_spread, or ss_tot collapsing -- and
    # re-deriving the count from the TRAINING spread caught only one. On a 0.02 s run
    # every component came back NaN while this line reported "0 of 35 unscoreable" and
    # the withhold guard stayed silent, so a table of dashes was printed beside a claim
    # that nothing was wrong. A guard keyed off a different array from the one it guards
    # is not a guard.
    first = next(iter(res))
    deg_mask = np.isnan(res[first])
    n_deg = int(deg_mask.sum())
    allk = list(res) + list(level)
    if dropped_note:
        print("\n" + dropped_note)
    print(f"\n  R^2 on the INCREMENT is the verdict; (level) is a persistence diagnostic.")
    print(f"{'component':<28}{'sd(dx)':>10}" + "".join(f"{k:>14}" for k in allk))
    for i, nm in enumerate(names):
        cells = "".join("           ---" if np.isnan((res | level)[k][i]) else f"{(res | level)[k][i]:>14.3f}"
                        for k in allk)
        print(f"{nm:<28}{spread[i]:>10.4f}{cells}")
    def _med(v):
        return "         ---" if np.all(np.isnan(v)) else f"{np.nanmedian(v):>14.3f}"
    print(f"\n{'MEDIAN over SCOREABLE':<28}{'':>10}" + "".join(_med((res | level)[k]) for k in allk))
    print(f"{n_deg} of {len(names)} components unscoreable: held-out sd < {a.min_spread} "
          f"or no variance to explain. Constant at the section, NOT unpredictable.")
    if n_deg == len(names):
        print("\n  *** ALL UNSCOREABLE. NO VERDICT. ***\n"
              "  Every component is constant at this horizon: the increment is smaller than\n"
              "  the spread threshold, so this horizon is below the resolution of the metric.\n"
              "  That is a fact about the horizon, not about predictability.")
    elif n_deg > 0.7 * len(names):
        print("VERDICT WITHHELD: most of the section state does not vary. Widen the initial\n"
              "  conditions or the command distribution before reading anything into R^2.")

    if a.summary_json:
        # RECORD THE INPUTS, NOT ONLY THE RESULTS. A summary that omits which files it
        # read is reproducible in principle and not in fact: the 2026-09-06 corpus
        # divergence census could not determine whether an earlier W0 run had read a
        # contaminated dataset, because the glob was preserved nowhere. The resolved
        # file list is hashed rather than stored so the field stays small while still
        # identifying the exact input set.
        _flist = "\n".join(paths)
        json.dump({"label": a.label, "host": os.uname().nodename, "dt_s": dt,
                   "input_glob": a.glob,
                   "input_files_sha256": hashlib.sha256(_flist.encode()).hexdigest(),
                   "input_files_n": len(paths),
                   "input_files_first": paths[0], "input_files_last": paths[-1],
                   "max_abs_state": float(np.abs(np.vstack(per_ep)).max()),
                   "arm": "fixed_dt" if a.fixed_dt else "event", "fixed_dt": a.fixed_dt,
                   "split_seed": a.split_seed, "gait_band": a.gait_band,
                   "n_episodes": len(per_ep), "n_dropped": n_drop,
                   "n_diverged": n_diverged,
                   # THE PATHS, NOT ONLY THE COUNT. The event and fixed-dt arms drop
                   # different episodes -- 288/112 against 339/61 -- so they were never
                   # drawing from the same population, and a between-arm comparison over
                   # them is confounded by WHICH episodes each arm could use. Varying the
                   # split seed does not touch that, because the seed varies the split
                   # WITHIN an arm. Recording the used set makes the arms intersectable,
                   # which is what makes the comparison paired.
                   "used_episodes": used_paths,
                   "n_events": int(sum(len(s) for s in per_ep)),
                   "interevent_mean_s": float(np.mean(ev_dt)),
                   "interevent_sd_s": float(np.std(ev_dt)),
                   "diag_offset_sd_ms": float(np.std(diag_off) * 1000) if diag_off else None,
                   "quant_pz_m": quant_pz, "spread_pz_m": spread_pz,
                   "components": names,
                   "r2_increment": {k: [None if np.isnan(v) else float(v) for v in res[k]] for k in res},
                   "r2_increment_median": {k: float(np.nanmedian(res[k])) for k in res},
                   "r2_level_median": {k: float(np.nanmedian(level[k])) for k in level}},
                  open(a.summary_json, "w"), indent=2)
        print(f"\nwrote {a.summary_json}")


if __name__ == "__main__":
    main()
