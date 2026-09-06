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

R^2 IS UNDEFINED WHERE THE SECTION STATE DOES NOT VARY. If every episode converges to
the same limit cycle, the pre-impact state is nearly constant, ss_tot collapses, and R^2
sits near zero no matter how good the model is. That is exactly HALO's Figure 5, where
error falls with rollout index because the test distribution collapses onto the fixed
point. This script therefore reports each component's SPREAD alongside its R^2 and marks
components whose spread is too small to score. A near-zero R^2 on a degenerate component
is not evidence that the map is unpredictable.

THE SAMPLING CAVEAT IS REPORTED, NOT ASSUMED AWAY. Events are located to the nearest
logged row. At a 50 Hz log rate that is up to 20 ms of quantisation, sampled at the
instant of peak state derivative, which converts to millimetres of body-height error. This
script reports the log rate, the implied quantisation, and the resulting body-height error
scale next to the R^2 so the two can be compared. If the quantisation error is a large
fraction of the signal's own spread, the R^2 is bounded by the apparatus and the honest
verdict is "not measurable at this log rate" rather than a null.
"""
from __future__ import annotations
import argparse, csv, glob, json, math, os, sys
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


def episode_events(ep, joint_cols):
    """Pre-impact states at section-foot touchdown, plus event times."""
    # ONE call on the stacked 4-column force array: that is contact_mode's designed
    # input and the only form whose bit packing matches LEG_ORDER.
    try:
        fz = np.column_stack([ep[f"foot_{leg}_force_fz_n"] for leg in LEG_ORDER])
    except KeyError:
        return None
    st, _mode = contact_mode(fz)
    stance = {leg: st[:, k] for k, leg in enumerate(LEG_ORDER)}
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


def knn_pred(Xtr, Ytr, Xte, k):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    d = ((B[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    nn = np.argsort(d, axis=1)[:, :k]
    return Ytr[nn].mean(1)


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
    ap.add_argument("--min-spread", type=float, default=1e-4,
                    help="held-out sd below which a component is unscoreable, not unpredictable")
    ap.add_argument("--summary-json", default=None)
    a = ap.parse_args()

    paths = sorted(glob.glob(a.glob))[: a.max_episodes]
    if not paths:
        sys.exit(f"no episodes matched {a.glob}")

    probe = load_episode(paths[0], [])
    joint_cols = [c for c in probe
                  if c.startswith("joint_") and (c.endswith("_pos_rad") or c.endswith("_vel_radps"))]
    if not joint_cols:
        sys.exit("no joint pos/vel columns found; check the schema")
    joint_cols = sorted(joint_cols)

    dt = float(np.median(np.diff(probe["time_s"])))
    per_ep, ev_dt, diag_off, n_drop = [], [], [], 0
    for p in paths:
        ep = load_episode(p, joint_cols)
        got = episode_events(ep, joint_cols)
        if got is None:
            n_drop += 1
            continue
        S, T, stance, t = got
        per_ep.append(S)
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
    rng = np.random.default_rng(0)
    order = rng.permutation(len(per_ep))
    n_tr = int(0.8 * len(per_ep))
    tr, te = order[:n_tr], order[n_tr:]
    pack = lambda idxs: (np.vstack([per_ep[i][:-1] for i in idxs]),
                         np.vstack([per_ep[i][1:] for i in idxs]))
    Xtr, Ytr = pack(tr)
    Xte, Yte = pack(te)
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
    try:
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        sx = StandardScaler().fit(Xtr)
        m = MLPRegressor((256, 256), max_iter=400, random_state=0).fit(sx.transform(Xtr), sy.transform(Ytr))
        res["mlp"] = R2(sy.inverse_transform(m.predict(sx.transform(Xte))))
    except Exception as e:
        print(f"  (MLP skipped: {e})")

    # THE APPARATUS CHECK. Event-time quantisation converts directly into pre-impact state
    # error at the instant of peak derivative. If that error is a large fraction of the
    # component's own spread, R^2 on that component is bounded by the log rate.
    vz = np.abs(np.vstack([s for s in per_ep])[:, 5])
    quant_pz = float(np.median(vz) * dt)
    spread_pz = float(np.vstack([s for s in per_ep])[:, 2].std())

    print(f"\n=== W0  {a.label} ===")
    print(f"episodes {len(per_ep)} usable, {n_drop} dropped   events {sum(len(s) for s in per_ep)}"
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
    n_deg = int((spread < a.min_spread).sum())
    allk = list(res) + list(level)
    print(f"\n  R^2 on the INCREMENT is the verdict; (level) is a persistence diagnostic.")
    print(f"{'component':<28}{'sd(dx)':>10}" + "".join(f"{k:>14}" for k in allk))
    for i, nm in enumerate(names):
        cells = "".join("           ---" if np.isnan((res | level)[k][i]) else f"{(res | level)[k][i]:>14.3f}"
                        for k in allk)
        print(f"{nm:<28}{spread[i]:>10.4f}{cells}")
    print(f"\n{'MEDIAN over SCOREABLE':<28}{'':>10}"
          + "".join(f"{np.nanmedian((res | level)[k]):>14.3f}" for k in allk))
    print(f"{n_deg} of {len(names)} components unscoreable (sd < {a.min_spread}): "
          f"constant at the section, NOT unpredictable.")
    if n_deg > 0.7 * len(names):
        print("VERDICT WITHHELD: most of the section state does not vary. Widen the initial\n"
              "  conditions or the command distribution before reading anything into R^2.")

    if a.summary_json:
        json.dump({"label": a.label, "host": os.uname().nodename, "dt_s": dt,
                   "n_episodes": len(per_ep), "n_dropped": n_drop,
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
