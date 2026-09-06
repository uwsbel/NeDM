"""NeRD-style excitation windows: random PD targets, reset in process.

STATE INJECTION IS UNAVAILABLE. Joint angles and velocities cannot be written on a
ChLinkMotorRotation -- the angle is a consequence of body placement, and the motor
link exposes no setter. So "sample a recorded walking state and put the robot in it"
is not implementable, and the ONLY route onto the walking manifold is through the
dynamics: run the policy until the robot is walking, then branch. Everything about
the design below follows from that one constraint.

WHY. Every action in the existing dataset is what the policy would have done, so
~4% of action variance survives conditioning on state and the effective rank is ~2
of 12. A model trained on it never had to learn how actions affect the world. NeRD
avoids this because its data is not policy data.

WHAT IS COPIED, AND WHAT IS DERIVED. The structure is NeRD's ANYmal recipe --
random joint-target offset from a standing pose, gains redrawn every step, short
windows, discard only on numerical divergence. The NUMBERS are measured from our
own data rather than inherited, because their robot is not ours:

  initial q      NeRD uses per-robot constants. Ours are sampled around the stand
                 pose over 1.5x the per-joint range the policy actually visits,
                 measured on 250 non-fallen rigid episodes, 962,899 rows.
  action scale   NeRD uses 0.5 on ANYmal. Our policy commands |target - stand| up
                 to 1.53 rad at p99.9 (calf), so 0.5 would not even cover what the
                 policy itself does. Ours is 1.6.
  gains          NeRD uses Kp ~ U[30,200], Kd ~ U[0,1]. Our nominal is Kp 20 /
                 Kd 0.5, BELOW their whole range. Ours brackets the nominal
                 instead: Kp ~ U[8,60], Kd ~ U[0.1,1.5].
  window         NeRD uses 1.67 s at 1/60 s. Ours is 1.70 s at 100 Hz = 170 rows.
                 The measured gait period is 0.290 s, so a window spans ~5.9
                 cycles and contains contact transitions rather than one snapshot.

FALLS ARE KEPT. Discard is on NaN, Inf, or |state| > 1e5 only -- never on falling.
A surrogate that has never seen a fall cannot penalise one, and the optimiser then
walks into failures the model believes are fine; that is how the fine-tunes failed,
excellent inside the surrogate and 43 of 43 on the floor in Chrono. From a standing
pose 1.7 s rarely reaches collapse, so most falls here are INCIPIENT, which is the
part worth having.

RESET IS IN PROCESS. A fresh system, ground and robot cost ~0.11 s against ~0.96 s
for a process launch, so a window costs 0.8 s instead of 1.7 s. Joint angles cannot
be written directly on a ChLinkMotorRotation, so the initial configuration is
reached by driving the PD to a random target for a short unrecorded pre-roll --
physically consistent, unlike injecting a pose the dynamics never produced.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

# Per-joint deviation from STAND_ACTION that the policy actually visits, p1..p99,
# measured on 250 non-fallen rigid episodes. Chrono order RR, RL, FR, FL.
VISITED_DEV = np.array([0.49, 0.78, 0.54, 0.42, 0.65, 0.55,
                        0.50, 0.82, 0.50, 0.62, 0.79, 0.56])
INIT_Q_MARGIN = 1.5          # widen beyond what the policy visits
ACTION_SCALE = 1.6           # covers |target - stand| p99.9 = 1.53 with room
# ON HOLD. Gain randomisation is what would force torque as the action channel,
# and with gains FIXED torque is a bijection of the target given the state, so the
# two encodings carry identical information. NeRD randomises gains because its
# product is a general-purpose backend under controllers it never saw; ours is a
# surrogate for one plant with one PD law, and a fine-tune changes policy weights,
# not gains. Off by default; --randomise-gains re-enables it.
KP_RANGE, KD_RANGE = (8.0, 60.0), (0.1, 1.5)
WINDOW_ROWS, RECORD_DT = 170, 0.01
CONTROL_DT = 0.02            # 50 Hz, the policy rate the surrogate will be driven at
PREROLL_S = 0.5

# THE RATES, established from the resolved collector config rather than from a
# docstring: physics 5e-4 s = 2000 Hz, exchange 2e-3 s = 500 Hz (apply_pd runs
# here), logging 1e-2 s = 100 Hz, policy 50 Hz. robot.py previously claimed
# 2.5e-3 s / 400 Hz; that was wrong and had been read back out and reported as a
# measurement. Corrected there.
#
# GAINS AND TARGET ARE RESAMPLED AT THE CONTROL INSTANT, 50 Hz, matching the rate
# the surrogate will actually be driven at. The ACTION is defined at the same
# instant it is logged, 100 Hz, as tau = clip(Kp*(q*-q) - Kd*qd) evaluated from
# q and qd AT THAT ROW. So the logged torque is exactly what the interface would
# compute at test time -- same formula, same instant, same gains -- and the
# residual against a recomputation is 0.0 rather than merely small.
#
# The clip is INSIDE the definition because the plant applies a clipped torque; a
# definition without it would disagree with the plant on the ~1% of saturated
# samples, which is the one place torque is most informative.
DIVERGENCE_LIMIT = 1e5
# RUNAWAY IS DETECTED, NOT INFERRED FROM LENGTH. In the existing collection the
# diverged episodes are excluded only because they are too short to yield a
# training window -- a filter that works for the wrong reason, and one that stops
# working here: this spec keeps falls and randomises the initial pose, so
# divergences are more common and some will run long enough to produce a window.
#
#   PRIMARY   |q| outside the URDF joint limit by a margin -- physical, not tuned
#   SECONDARY torque pinned at the effort limit for N consecutive rows WITH |q|
#             growing; catches runaway before the magnitude explodes
#   BACKSTOP  NaN, Inf, |state| > 1e5, which is NeRD's own rule
#
# Each is COUNTED and reported by category rather than silently dropped: the
# divergence rate per action scale is the measurement that sizes the sweep, and
# dropping it quietly would hide the one number that tunes the perturbation.
JOINT_LIMIT_MARGIN = 1.5     # multiple of the URDF range, beyond which q is runaway
PIN_ROWS = 8                 # consecutive rows at the effort limit to call it pinned



def chrono_fingerprint():
    """Identify WHICH Chrono this process actually loaded, by hashing the binary.

    THERE IS NO ERROR TO NOTICE IF THIS IS WRONG. Two Chrono builds live on this
    box -- a conda pychrono inside the env, and a source build reached only via
    PYTHONPATH -- and the source build shadows the conda one when the path is
    set. Omit the path and the import still succeeds, the run completes, and the
    output looks entirely normal while the physics came from a different engine.
    Unlike the replay investigation, which had 145 differing columns as a
    symptom, this failure is silent.

    So the binary is fingerprinted rather than the version string: a version is
    what a build claims, a hash is what it is. Recorded per run in summary.json
    and as a short column on every row, because rows get pooled across
    collections and a summary does not travel with them.
    """
    import pychrono
    so = Path(pychrono.__file__).parent / "_core.so"
    h = hashlib.md5(so.read_bytes()).hexdigest() if so.exists() else "missing"
    return {"pychrono_path": str(so), "core_so_md5": h,
            "source_build": "chrono-build" in str(so)}


def checkpoint_fingerprint(path):
    """Hash the policy checkpoint, for the same reason the Chrono binary is hashed.

    THE PATH IS NOT THE IDENTITY. A checkpoint path is reused, moved and overwritten;
    two runs naming the same file can have used different weights. Every existing Go2
    dataset records a seed and no episode sidecar anywhere records which policy
    produced it, so a replay that differs cannot separate a build change from a policy
    change. This closes that forward.

    Provenance now has four fields and they answer four different questions:
    seed (can this be re-executed?), argv (under what arguments?), the Chrono hash
    (with which physics?) and this (with which policy?).
    """
    p = Path(path)
    if not p.exists():
        return {"checkpoint_path": str(p), "sha256": "missing"}
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return {"checkpoint_path": str(p), "sha256": h.hexdigest(), "bytes": p.stat().st_size}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ground-size-m", type=float, default=40.0)
    ap.add_argument("--action-scale", type=float, default=None,
                    help="override ACTION_SCALE; the default is derived from data")
    ap.add_argument("--window-rows", type=int, default=None)
    ap.add_argument("--randomise-gains", action="store_true",
                    help="ON HOLD -- forces torque as the action channel; see note above")
    ap.add_argument("--branch-from-policy", action="store_true",
                    help="reach the initial state by RUNNING the policy, then switch to "
                         "random targets. Joint q/qd cannot be written directly on a "
                         "ChLinkMotorRotation, so the only way onto the walking manifold "
                         "is through the dynamics.")
    ap.add_argument("--bursts", type=int, default=1,
                    help="alternate perturbation and policy recovery within ONE episode. "
                         "The policy pulls the robot back onto the walking manifold "
                         "between bursts, so one pre-run amortises over many windows "
                         "and the state cannot drift far.")
    ap.add_argument("--recover-rows", type=int, default=60,
                    help="policy-driven rows between bursts; logged and marked, they are "
                         "ordinary walking data and worth keeping")
    ap.add_argument("--branch-s", type=float, nargs=2, default=(1.5, 6.0),
                    help="uniform range for how long the policy runs before branching")
    ap.add_argument("--init-from-data", default=None,
                    help="glob of existing episode CSVs to draw initial joint "
                         "configurations from, instead of stand + random offset")
    a = ap.parse_args()

    import pychrono as chrono
    from nedm.quadruped.robot import Go2Robot
    from nedm.quadruped.constants import STAND_ACTION, PD_KP, PD_KD, JOINT_EFFORT_NM
    from nedm.quadruped.dataset import csv_field_names, capture_row
    from nedm.quadruped.terrain import build_rigid_ground
    from nedm.quadruped.imported_policy import ImportedGo2Policy
    CKPT = os.environ.get("NEDM_GO2_CKPT",
                          "/home/kyle/sbel-artifacts/checkpoints/go2_cts_150k.pt")

    assets = Path(os.environ.get("NEDM_GO2_ASSETS",
                  "/home/kyle/Documents/sbel/sbel-reproducibility/2025/multi-terrain-RL"))
    urdf = assets / "data/robot/go2_irrvis/urdf/go2_description.urdf"
    stand = np.asarray(STAND_ACTION, dtype=np.float64)
    global ACTION_SCALE, WINDOW_ROWS
    if a.action_scale is not None: ACTION_SCALE = a.action_scale
    if a.window_rows is not None: WINDOW_ROWS = a.window_rows
    rng = np.random.default_rng(a.seed)
    init_pool = None
    if a.init_from_data:
        import glob as _g
        ORD = [f"{l}_{k}" for l in ("rr", "rl", "fr", "fl") for k in ("hip", "thigh", "calf")]
        pool = []
        for f in sorted(_g.glob(a.init_from_data))[:120]:
            try: rr_ = list(csv.DictReader(open(f)))
            except Exception: continue
            for r in rr_[::200]:
                try: v = [float(r[f"joint_{c}_pos_rad"]) for c in ORD]
                except (KeyError, ValueError): break
                if all(abs(x) < 5 for x in v): pool.append(v)
        init_pool = np.array(pool) if pool else None
        print(f"  init pool: {0 if init_pool is None else len(init_pool)} recorded poses")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    # target and gains logged alongside so the torque is reproducible after the fact
    fp = chrono_fingerprint()
    print(f"  chrono: {fp['core_so_md5'][:8]}  {fp['pychrono_path']}"
          f"  {'SOURCE BUILD' if fp['source_build'] else 'CONDA -- is that intended?'}")
    ck = checkpoint_fingerprint(CKPT)
    print(f"  policy: {ck['sha256'][:8]}  {ck['checkpoint_path']}")
    fields = (["window", "phase", "burst", "kp", "kd", "chrono_build"]
              + [f"target_{i}" for i in range(12)] + list(csv_field_names()))
    fh = open(out / "windows.csv", "w", newline="")
    w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    w.writeheader()

    step = 5e-4
    exchange = 2.5e-3
    kept = fell = rows_written = 0
    reject = {"nan_inf": 0, "state_1e5": 0, "joint_limit": 0, "torque_pinned": 0}
    reject_events = []
    fell_rows = []
    import xml.etree.ElementTree as _ET
    _lim = {}
    for _j in _ET.parse(urdf).getroot().iter("joint"):
        _L = _j.find("limit")
        if _L is not None:
            _lim[_j.get("name")] = (float(_L.get("lower")), float(_L.get("upper")))
    _ORD = [f"{l}_{k}" for l in ("RR", "RL", "FR", "FL") for k in ("hip", "thigh", "calf")]
    LO = np.array([_lim[f"{n}_joint"][0] for n in _ORD])
    HI = np.array([_lim[f"{n}_joint"][1] for n in _ORD])
    MID, HALF = (LO + HI) / 2, (HI - LO) / 2
    t_start = time.perf_counter()
    for wi in range(a.windows):
        system = chrono.ChSystemSMC()
        system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
        system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
        build_rigid_ground(chrono, system, size_m=float(a.ground_size_m))
        frame = chrono.ChFramed(chrono.ChVector3d(0.0, 0.0, 0.5),
                                chrono.QuatFromAngleZ(float(rng.uniform(-math.pi, math.pi))))
        robot = Go2Robot(system, urdf, frame, actuation="torque")

        if a.branch_from_policy:
            # ONTO THE WALKING MANIFOLD, THE ONLY WAY AVAILABLE. The pilot started
            # from a randomised standing pose and landed in a different region --
            # forward velocity under-covered 3x, body roll rate over-covered 5x --
            # because random initial VELOCITIES are destabilising before any action
            # is applied. Running the policy first reaches a state the robot was
            # genuinely in, and the branch time is randomised so gait phase and
            # speed vary rather than every window starting at the same point in
            # the cycle.
            pol = ImportedGo2Policy(Path(CKPT), family="constant",  # noqa: F841 -- reused for recovery
                                    params={"vx": float(rng.uniform(-0.8, 0.8))})
            pol.reset()
            t_br = float(rng.uniform(*a.branch_s))
            tt = 0.0
            while tt < t_br:
                if int(tt / CONTROL_DT) != int((tt - exchange) / CONTROL_DT):
                    robot.actuate(pol.act(robot))
                robot.apply_pd(); system.DoStepDynamics(exchange); tt += exchange
            q_target0 = None
        # PRE-ROLL to a random configuration. Reaching the initial q through the
        # dynamics rather than writing it means the state is one the plant can
        # actually be in; an injected pose need not be.
        q_target0 = None
        if a.branch_from_policy:
            pass
        elif init_pool is not None:
            # DRAWN FROM A REAL RECORDED POSE. Sampling each joint independently
            # over its own visited range does NOT give a pose the robot can stand
            # in: the policy's configuration is strongly correlated across joints,
            # and the independent version tips the robot before the window starts
            # -- measured, 80-100% tipped at EVERY action scale including 0.3,
            # median first tip at row 13 of 170. Real poses are in-regime by
            # construction, which is also what pilot diagnostic 2 asks for.
            q_target0 = init_pool[rng.integers(len(init_pool))]
        else:
            q_target0 = stand + rng.uniform(-1, 1, 12) * VISITED_DEV * INIT_Q_MARGIN
        if q_target0 is not None:
            robot.actuate(q_target0)
            t = 0.0
            while t < PREROLL_S:
                robot.apply_pd(); system.DoStepDynamics(exchange); t += exchange

        rows = []
        bad = None
        pinned, qmag = [], []
        t = 0.0
        burst = 0
        phase = "perturb"
        phase_rows = 0
        kp = kd = None
        target = stand.copy()
        steps_per_row = int(round(RECORD_DT / exchange))
        rows_per_control = int(round(CONTROL_DT / RECORD_DT))
        total_rows = (WINDOW_ROWS * a.bursts
                      + a.recover_rows * max(0, a.bursts - 1))
        while len(rows) < total_rows:
            if phase == "perturb" and phase_rows >= WINDOW_ROWS and burst + 1 < a.bursts:
                phase, phase_rows, burst = "recover", 0, burst + 1
            elif phase == "recover" and phase_rows >= a.recover_rows:
                phase, phase_rows = "perturb", 0
            if phase == "recover":
                if phase_rows % rows_per_control == 0:
                    robot.actuate(pol.act(robot))
            elif phase_rows % rows_per_control == 0:
                # REDRAWN AT THE CONTROL INSTANT. Independence of the action from
                # the state is the whole point: under a policy, q* = pi(s), so the
                # action carries almost no information the state does not already
                # have -- measured at ~4% conditional variance, effective rank 2
                # of 12. Sampling it breaks that by construction.
                if a.randomise_gains:
                    kp = float(rng.uniform(*KP_RANGE)); kd = float(rng.uniform(*KD_RANGE))
                else:
                    kp, kd = PD_KP, PD_KD
                target = stand + ACTION_SCALE * rng.uniform(-1, 1, 12)
                robot.actuate(target)
                robot.kp, robot.kd = kp, kd
            # THE ACTION, defined at this row's instant from this row's q and qd.
            tau = np.clip(kp * (target - robot.joint_pos()) - kd * robot.joint_vel(),
                          -JOINT_EFFORT_NM, JOINT_EFFORT_NM)
            for _ in range(steps_per_row):
                robot.apply_pd()
                system.DoStepDynamics(exchange)
                t += exchange
            q = robot.joint_pos(); qd = robot.joint_vel()
            bp = robot.base().GetPos()
            if not (np.isfinite(q).all() and np.isfinite(qd).all()
                    and math.isfinite(bp.x) and math.isfinite(bp.y) and math.isfinite(bp.z)):
                bad = "nan_inf"; break
            if max(np.abs(q).max(), np.abs(qd).max(),
                   abs(bp.x), abs(bp.y), abs(bp.z)) > DIVERGENCE_LIMIT:
                bad = "state_1e5"; break
            # PRIMARY: physically impossible joint angle. Signs are the recorded
            # (negated) convention, so compare |q - mid| against the URDF half-range.
            if np.any(np.abs(-q - MID) > JOINT_LIMIT_MARGIN * HALF):
                bad = "joint_limit"; break
            # SECONDARY: pinned torque with |q| still growing.
            pinned.append(bool(np.all(np.abs(tau) >= JOINT_EFFORT_NM - 1e-9)))
            qmag.append(float(np.abs(q).max()))
            if (len(pinned) >= PIN_ROWS and all(pinned[-PIN_ROWS:])
                    and qmag[-1] > qmag[-PIN_ROWS] + 0.05):
                bad = "torque_pinned"; break
            # soil_z is NaN-per-foot on rigid, matching what collect_go2_smoke
            # passes; None is not accepted and would fail inside capture_row.
            row = capture_row(chrono, robot, None, 0.0, target,
                              (0.0, 0.0, 0.0), [float("nan")] * 4, float("nan"),
                              f"exc_{wi:06d}", "go2_excitation", f"exc_{wi:06d}",
                              "train", len(rows), t, tau=tau, policy_raw=None,
                              perturb=None, contacts=None, com=None,
                              gravity=[0.0, 0.0, -9.81])
            qz = robot.base().GetRot()
            row["_tilt"] = math.acos(max(-1.0, min(1.0, 1 - 2 * (qz.e1 ** 2 + qz.e2 ** 2))))
            row.update(window=wi, kp=kp, kd=kd, phase=phase, burst=burst)
            phase_rows += 1
            for i in range(12):
                row[f"target_{i}"] = float(target[i])
            rows.append(row)
        if bad:
            # WHERE, not just whether. A rejected episode writes no rows, so without
            # this the timing is not merely unlogged -- it never existed, and the
            # question "do rejections cluster at burst onsets" is unanswerable after
            # the fact. Row index, phase, and distance to the most recent onset.
            reject[bad] += 1
            reject_events.append({"window": wi, "reason": bad, "row": len(rows),
                                  "phase": phase, "burst": burst,
                                  "rows_since_onset": phase_rows})
            continue
        # WHEN it tipped, not only whether it ended tipped. A window that is
        # already over at row 20 is a different object from one that tips at row
        # 160: the first is mostly free-fall, the second is the incipient boundary
        # this collection exists to sample.
        tilts = np.array([r["_tilt"] for r in rows])
        over = np.where(tilts > math.radians(63.0))[0]
        if len(over):
            fell += 1
            fell_rows.append(int(over[0]))
        for r in rows:
            r["chrono_build"] = fp["core_so_md5"][:8]
            w.writerow(r)
        rows_written += len(rows)
        kept += 1
        if (wi + 1) % 100 == 0:
            el = time.perf_counter() - t_start
            print(f"  {wi+1}/{a.windows}  kept {kept} rejected {sum(reject.values())} fell {fell}"
                  f"  {el/(wi+1):.3f} s/window", flush=True)
    fh.close()
    el = time.perf_counter() - t_start
    json.dump({"windows_requested": a.windows, "kept": kept,
               "chrono": fp, "checkpoint": ck,
               # RECORDED SO THE RUN CAN BE REPRODUCED. Re-running one window at the
               # same seed and comparing physics columns settles a dataset's build
               # provenance by construction -- but only if the seed and arguments
               # survive. Two older diagnostics are permanently ungradeable because
               # they do not.
               "seed": a.seed, "argv": sys.argv[1:],
               "rejected_by_reason": reject, "discarded": sum(reject.values()),
               "reject_events": reject_events,
               "ended_fallen": fell, "rows": rows_written,
               "rows_per_episode": (rows_written // kept) if kept else 0,
               "seconds_per_window": el / max(a.windows, 1),
               "action_scale": ACTION_SCALE, "kp_range": KP_RANGE, "kd_range": KD_RANGE,
               "window_rows": WINDOW_ROWS, "preroll_s": PREROLL_S,
               "init_q_margin": INIT_Q_MARGIN},
              open(out / "summary.json", "w"), indent=1)
    fr = np.array(fell_rows) if fell_rows else np.array([])
    print(f"\nkept {kept}, rejected {sum(reject.values())} {reject}, tipped {fell}"
          + (f"; first tip at row median {int(np.median(fr))} of {rows_written // max(kept, 1)}"
             f" (p10 {int(np.percentile(fr,10))}, p90 {int(np.percentile(fr,90))})" if len(fr) else ""))
    print(f"{el/max(a.windows,1):.3f} s per window -> {1000*el/max(a.windows,1)/3600:.2f} h per 1000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
