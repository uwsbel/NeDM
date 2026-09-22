#!/usr/bin/env python3
"""Rigid (f104 family) collector with external-control and branch modes: the rigid launch path of PLAN B3 and A4.

Same CLI as scripts/gen_collect.py plus
  --mode native|branch|branch_auto|pid_perturbed|pid_held|policy      (default native)
  --branch-frame F --branch-route ROUTE.json   branch: the frozen follower drives the initial route up to frame F, then
                                               a new follower on ROUTE (gc_control.make_follower, no Initialize()) drives on
  --actor ACTOR.npz                            policy: a numpy tracking policy (gc_control.NumpyActor + PolicyObs)
  --substep-log                                write substep_actions.npz (N x substeps x 3 applied triples) + a summary
  --local                                      bypass the source-manifest and FDM_RUNTIME_FINGERPRINT gates (local smokes
                                               only; recorded in collection_request.json and outcome.json)
  branch_auto: --recorded RUNDIR --branch-frame F --n-cont 3 --cont-seed S   (in-job two-pass on one node, see branch_auto)

Everything physical is the frozen loop (traverse_fdm_rgbd_diverse_chrono.run_chrono, imported through
gen_collect.import_runner): settle, physics, recording, termination and file schemas.  gen_collect's three insertion
hooks (stop policy) are kept verbatim (the third is extended by the external-mode near-stop rule); this wrapper adds
three insertion hooks (bind, follower swap at the frame top, per-frame external command) and ONE replacement of the
per-substep follower-input block.  Every hook location is asserted by count in ``replacement_adapter``; the sha256 of
the original and of the adapted function source are recorded in collection_request.json.

Modes
  native         nothing external: output byte-identical to gen_collect.py on the same node (local checks S1/S2).
  pid_held       at the top of every frame k >= 0 the SHADOW follower's output (its last Advance of interval k-1, i.e.
                 exactly what the native loop would record as action[k] before its clamp) is clipped ONCE
                 (gc_control.hold_clip: steering within +-0.1 of the previously held steering, throttle/brake in [0, 1])
                 and the identical DriverInputs is written at all substeps of the frame.  The follower keeps
                 Synchronize/Advance-ing as a shadow, so settle and parking logic are unchanged.
  pid_perturbed  as pid_held with gc_control.OUPerturb around the shadow output before the clip.  A seed is REQUIRED
                 (--perturb-seed, else --episode-seed); without one the run is rejected before Chrono is loaded, so
                 no two episodes can silently share the same perturbation stream.
  policy         as pid_held with a numpy actor; the observation (gc_control.PolicyObs) is padded at frame 0 with the
                 rest state and the settle action (0, 0, 1).  The state the policy sees is captured at the top of the
                 frame BEFORE hmmwv.Synchronize; its 12 observable columns (vx, vy, roll, pitch, roll/pitch/yaw rate,
                 four spindle speeds, engine speed) equal the recorded state[k] EXACTLY (asserted after every policy
                 run).  Engine speed is read from the transmission's output motorshaft
                 (transmission.GetOutputMotorshaftSpeed(), the integrated shaft state that the SHAFTS engine's
                 Synchronize imposes on the engine shaft); engine.GetMotorSpeed() before Synchronize would return the
                 value imposed one substep earlier (0.5 rad/s off on average).  The simulator-only tyre-force and
                 torque columns are not observable and lag one substep.
  branch         follower swap at frame F: a new ChPathFollowerDriver on the branch route, built exactly like the
                 frozen make_driver but without Initialize() (the constructor already Reset() the controllers); the
                 loop's xy/speeds switch to the branch route and wp restarts at 0; steering continuity comes from the
                 frozen per-substep clamp on the first branch substep.  outcome.json / trajectory.npz gain branch_frame,
                 branch_pose, branch_route_sha256 and the prefix history window branch_hist (40 x 15: the 12 observable
                 state columns of state[F-39+t] and action[F-40+t]) + branch_hmask (40; action frame >= 0), in the
                 mixed dataset's convention (ga_build_mixed.py, the same block the CRM collector writes); rows [:F] of
                 the recorded arrays are the full prefix.
  branch_auto    orchestration on one node (PLAN A4 rigid): pass 1 replays the prefix natively with horizon F (in
                 <out>__replay_F<F>, keyed by the branch frame so a cached replay of another F is never reused), keeps
                 the anchor only if the replay ran the full F frames with status 'timeout' (a prefix that reached the
                 goal or terminated earlier is skipped with reason 'replay_ended_before_branch_frame') and the
                 replayed pose (within --replay-pose-tol-m) and stalled/moving class match the recording (else
                 skipped.json, exit 0); samples --n-cont continuations from the REPLAYED pose with
                 gc_control.sample_continuations(v0 = replayed vx at F, start-heading acceptance
                 --cont-heading-tol-deg, default 15; opt-in --cont-accel-cap also caps speeds at the acceleration
                 ramp from vx) so no continuation starts below the vehicle's speed (the follower would brake hard) or
                 with a heading kink; drives prefix+branch in subprocesses <out>__c0..;
                 <out>/branch_auto.json + episode_complete.json summarise.  A branch subprocess whose prefix ends
                 before F writes skipped.json + episode_complete.json {skipped: true} (exit 0), not a failure.

External-mode stop rule (added, never fires in native/branch; the CRM collector's rule): 40 s (800 consecutive
recorded frames) with |vx| < 0.3 m/s and not parked, REGARDLESS of throttle -> status 'prolonged_blockage_terminated'
(NEAR_STOP_STATUS, the same status the native rule uses) with an event {"rule": "near_stop_any_throttle"} in
outcome.json.  The native prolonged-blockage rule (which needs throttle > 0.3 in every interval, so it fires first for
a follower that is stalled and not parked) is checked before it; this rule catches a policy that stops with zero
throttle.

Recorded action[k] in every mode = the triple handed to hmmwv.Synchronize at substep 0 of interval k (frozen
convention); in external modes that is the held triple.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT / "src"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gen_collect  # noqa: E402  the unmodified rigid wrapper: gates, StopPolicy, observer, import_runner
import gc_control   # noqa: E402  hold_clip, OUPerturb, NumpyActor, PolicyObs, sample_continuations, make_follower

DT = gen_collect.DT
MODES = ("native", "branch", "branch_auto", "pid_perturbed", "pid_held", "policy")
EXTERNAL_MODES = ("pid_perturbed", "pid_held", "policy")
NEAR_STOP_STATUS = "prolonged_blockage_terminated"   # the CRM collector reuses the native status; the event carries the rule name
STALL_VX, STALL_THROTTLE = .3, .3
HIST_T = 40
HIST_STATE_COLS = tuple(int(c) for c in gc_control.OBSERVABLE_COLS)   # 0-6, 11-14, 15: the 12 deployable state columns
sha, read, dump, require = gen_collect.sha, gen_collect.read, gen_collect.dump, gen_collect.require


def perturb_seed(args):
    """--perturb-seed, else --episode-seed (the CRM collector's seed, forwarded per task row).  No silent default:
    a pid_perturbed row without a seed is rejected (validate_mode_args), so a task file built without seeds cannot
    give every episode the same OU realisation and brake-tap frames."""
    for name in ("perturb_seed", "episode_seed"):
        if getattr(args, name, None) is not None:
            return int(getattr(args, name))
    raise ValueError("pid_perturbed needs --perturb-seed or --episode-seed (no default seed)")


# =============================================================================================== adapter
def replacement_adapter(module):
    """gen_collect's three insertions + three gen_ext insertions + one replacement, every location asserted by count."""
    source = inspect.getsource(module.run_chrono)
    original = source
    insert_before = [
        # gen_collect hooks 1-3 (hook 3 extended: the external-mode near-stop rule runs when the native rule is silent)
        ("    hmmwv, system, terrain = scene.hmmwv, scene.system, scene.terrain\n",
         "    args.f104_policy.bind_scene(scene, tmap)\n", 1),
        ('                    if args.command == "observe":\n',
         "                    args.f104_policy.on_anchor(scene, state, pose, history)\n", 2),
        ('        frame += 1\n',
         '            f104_stop = args.f104_policy.check(frame, record_pose, record_action, record_parked, terminal_pose, wp, 0. if at_end else float(speed[wp]))\n'
         '            if f104_stop is None:\n'
         '                f104_stop = args.gen_ext.check_near_stop(frame, record_state, record_action, record_parked)\n'
         '            if f104_stop is not None:\n'
         '                status = f104_stop\n'
         '                break\n', 1),
        # gen_ext hook A: bind scene objects once (after driver, tire_radii, fields, xy/speed, goal are defined)
        ('    dt = float(config["simulation"]["step_size_s"])\n',
         '    args.gen_ext.bind(scene, chrono, veh, tmap, case, goal, tire_radii, fields, capture_row, engine)\n', 1),
        # gen_ext hook B: follower swap at the top of frame F (before the nearest-waypoint search)
        ('        ref = hmmwv.GetChassis().GetBody().GetFrameRefToAbs()\n        pos = np.array([ref.GetPos().x, ref.GetPos().y])\n',
         '        if frame == args.gen_ext.branch_frame:\n'
         '            driver, xy, speed = args.gen_ext.swap_follower(frame, driver, hmmwv, previous_steer)\n'
         '            wp = 0\n', 1),
    ]
    insert_after = [
        # gen_ext hook C: one external command per frame (None in native/branch and during the settle)
        ('        driver.SetDesiredSpeed(0. if frame < 0 or at_end else float(speed[wp]))\n',
         '        ext_cmd = args.gen_ext.command(frame, float(system.GetChTime()), driver, hmmwv, terrain, at_end, 0. if at_end else float(speed[wp]), previous_steer)\n', 1),
    ]
    replacement = (
        '            driver.Synchronize(ts)\n'
        '            inputs = driver.GetInputs()\n'
        '            previous_steer = 0. if frame < 0 else float(np.clip(inputs.m_steering, previous_steer-2.*dt, previous_steer+2.*dt))\n'
        '            inputs.m_steering = previous_steer\n',
        '            driver.Synchronize(ts)\n'
        '            inputs = driver.GetInputs()\n'
        '            if ext_cmd is not None:\n'
        '                inputs = args.gen_ext.driver_inputs(ext_cmd)\n'
        '                previous_steer = float(ext_cmd[0])\n'
        '            else:\n'
        '                previous_steer = 0. if frame < 0 else float(np.clip(inputs.m_steering, previous_steer-2.*dt, previous_steer+2.*dt))\n'
        '                inputs.m_steering = previous_steer\n')
    for old, _, count in insert_before + insert_after:
        require(source.count(old) == count, f"Frozen loop hook location changed: {old.strip()[:60]!r}")
    require(source.count(replacement[0]) == 1, "Frozen per-substep follower-input block changed")
    for old, insertion, _ in insert_before:
        source = source.replace(old, insertion+old, 1)
    for old, insertion, _ in insert_after:
        source = source.replace(old, old+insertion, 1)
    source = source.replace(replacement[0], replacement[1], 1)
    namespace = dict(module.__dict__)
    exec(compile(source, str(Path(__file__).resolve())+":gen_ext_adapted_run_chrono", "exec"), namespace)
    _, gen_adapter = gen_collect.adapted_function(module)
    return namespace["run_chrono"], {
        "adapter": "gen_collect_ext.replacement_adapter",
        "original_function_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "adapted_function_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "gen_collect_adapted_function_sha256": gen_adapter["adapted_function_sha256"],
        "hook_count": {"insertions": len(insert_before)+len(insert_after), "replacements": 1},
        "mutation_scope": ("gen_ext_v1: gen_collect's three additive hooks (stop check extended by the external-mode "
                           "near-stop rule) + three additive gen_ext hooks (bind; follower swap at the frame top; one "
                           "external command per frame) + ONE replacement of the per-substep follower-input block: when an "
                           "external command is set the identical DriverInputs is written at every substep and the held "
                           "steering becomes previous_steer, otherwise the original clamp statements run verbatim. Physics, "
                           "settle, recording, termination and file schemas retained verbatim.")}


# =============================================================================================== controller state
class GenExt:
    """Per-episode state of the added hooks; attached to ``args.gen_ext``."""

    def __init__(self, args, case, route, branch_route=None, actor=None):
        self.args, self.mode = args, args.mode
        self.external = self.mode in EXTERNAL_MODES
        self.branch_frame = int(args.branch_frame) if self.mode == "branch" else None   # None: the settle's negative frames never match
        self.branch_route = branch_route
        self.branch_start_tol = float(args.branch_start_tol_m)
        self.near_stop_frames = int(round(args.near_stop_s/DT))
        self.case, self.route, self.actor = case, route, actor
        self.events, self.branch_info, self.last_cmd, self.near_stop_fired = [], None, None, False
        self.near_stop_run, self.near_stop_longest = 0, 0
        self.ou = (gc_control.OUPerturb(seed=perturb_seed(args), tau_s=args.ou_tau_s, steer_sd=args.ou_steer_sd,
                                        throttle_sd=args.ou_throttle_sd, brake_p=args.brake_p)
                   if self.mode == "pid_perturbed" else None)
        self.pobs = gc_control.PolicyObs.from_meta(route, actor.meta) if self.mode == "policy" else None
        self.log = {"frame": [], "shadow": [], "held": [], "desired": [], "parked": [], "perturb": [], "obs": []}
        self.bound = False

    # ---- hook A
    def bind(self, scene, chrono, veh, tmap, case, goal, tire_radii, fields, capture_row, engine):
        self.scene, self.chrono, self.veh, self.tmap = scene, chrono, veh, tmap
        self.vehicle, self.engine = scene.hmmwv.GetVehicle(), engine
        self.transmission = self.vehicle.GetTransmission()     # its output motorshaft speed == state[k] col 15 (see _capture)
        self.goal = np.asarray(goal, float)
        self.tire_radii, self.fields, self.capture_row = tire_radii, list(fields), capture_row
        if self.branch_route is not None:
            bx = np.asarray(self.branch_route["waypoints"], float)
            require(np.linalg.norm(bx[-1]-self.goal) <= .25, "Branch route must end at the case goal (within 0.25 m)")
        if self.pobs is not None:
            require(self.pobs.num_obs == self.actor.num_obs, f"Actor expects {self.actor.num_obs} observations, PolicyObs builds {self.pobs.num_obs}")
        self.bound = True

    # ---- hook B
    def swap_follower(self, frame, old_driver, hmmwv, previous_steer):
        require(self.bound and self.branch_route is not None, "Branch swap without a bound branch route")
        ref = hmmwv.GetChassis().GetBody().GetFrameRefToAbs()
        pose = np.array([ref.GetPos().x, ref.GetPos().y, ref.GetRot().GetCardanAnglesZYX().z], np.float64)
        old = old_driver.GetInputs()
        vx_now = self.chrono.Vdot(ref.GetPosDt(), ref.GetRotMat().GetAxisX())
        xy, speed = np.asarray(self.branch_route["waypoints"], float), np.asarray(self.branch_route["speeds"], float)
        start_error = float(np.linalg.norm(xy[0]-pose[:2]))
        require(start_error <= self.branch_start_tol, f"Branch route starts {start_error:.3f} m from the vehicle at frame {frame} (tolerance {self.branch_start_tol} m)")
        driver = gc_control.make_follower(self.veh, self.chrono, self.vehicle, self.branch_route, self.tmap.height, initialize=False)
        self.branch_info = {"branch_frame": int(frame), "branch_time_s": frame*DT, "branch_pose": pose.tolist(),
            "branch_start_error_m": start_error, "branch_route_sha256": gc_control.route_sha256(self.branch_route),
            "branch_route_file_sha256": sha(self.args.branch_route), "branch_waypoints": int(len(xy)),
            "branch_follower_points": len(gc_control.follower_points(self.branch_route, self.tmap.height)),
            "follower_output_before_swap": [float(old.m_steering), float(old.m_throttle), float(old.m_braking)],
            "previous_steer_at_swap": float(previous_steer),
            "branch_start_speed_mps": float(speed[0]), "vehicle_vx_at_swap_mps": float(vx_now),
            "swap": "ChPathFollowerDriver rebuilt without Initialize() (constructor Reset()); new driver starts with steering 0/throttle 0 for one substep, steering continuity through the frozen per-substep clamp; the follower keeps braking while throttle <= 0.2 and the vehicle is faster than the branch route's first speed, so continuations are sampled with a speed floor from the vehicle's speed (gc_control.sample_continuations v0)"}
        return driver, xy, speed

    # ---- hook C
    def command(self, frame, ts, driver, hmmwv, terrain, at_end, desired_speed, previous_steer):
        if frame < 0 or not self.external:
            return None
        sh = driver.GetInputs()
        shadow = (float(sh.m_steering), float(sh.m_throttle), float(sh.m_braking))
        prev = float(previous_steer)
        if self.last_cmd is not None and abs(prev-self.last_cmd[0]) > 1e-12:
            raise RuntimeError("held steering diverged from the loop's previous_steer")
        perturb, obs = None, None
        if self.mode == "pid_held":
            cmd = gc_control.hold_clip(shadow, prev)
        elif self.mode == "pid_perturbed":
            cmd = gc_control.hold_clip(self.ou.step(shadow), prev)
            perturb = dict(self.ou.last)
        else:
            state, pose = self._capture(frame, ts, hmmwv, terrain)
            last = self.last_cmd if self.last_cmd is not None else gc_control.SETTLE_ACTION
            if frame == 0:
                self.pobs.reset(state)
            obs = self.pobs.observe(pose, state, last)
            cmd = gc_control.hold_clip(self.actor.act(obs), prev)
            self.pobs.push(state, cmd)
        self.last_cmd = cmd
        L = self.log
        L["frame"].append(frame); L["shadow"].append(shadow); L["held"].append(cmd)
        L["desired"].append(float(desired_speed)); L["parked"].append(bool(at_end))
        if perturb is not None:
            L["perturb"].append([perturb["d_steer"], perturb["d_throttle"], float(perturb["tap_active"]), perturb["tap_level"]])
        if obs is not None:
            L["obs"].append(np.asarray(obs, np.float32))
        return cmd

    def _capture(self, frame, ts, hmmwv, terrain):
        inputs = self.driver_inputs(self.last_cmd if self.last_cmd is not None else gc_control.SETTLE_ACTION)
        row = self.capture_row(hmmwv, terrain, "fdm_rgbd_eval", "follower", self.case["id"], "development",
                               frame, ts, inputs, include_tires=True, tire_radii=self.tire_radii)
        # Frame top, BEFORE this frame's hmmwv.Synchronize.  The frozen loop records engine.GetMotorSpeed() AFTER
        # Synchronize, where ChEngineShafts::Synchronize has just imposed the transmission's output motorshaft speed on
        # the engine shaft; reading the transmission's integrated shaft state here gives that same value, whereas
        # engine.GetMotorSpeed() here would still hold the previous substep's imposed value (VERIFY_gen_ext P1).
        row["engine_motor_speed_radps"] = float(self.transmission.GetOutputMotorshaftSpeed())
        row["engine_motorshaft_torque_nm"] = float(self.engine.GetOutputMotorshaftTorque())   # column 16, not observable
        state = np.array([float(row[field]) for field in self.fields], np.float32)
        pose = np.array([row["pos_x_m"], row["pos_y_m"], row["yaw_rad"]], np.float64)
        return state, pose

    def driver_inputs(self, cmd):
        inputs = self.veh.DriverInputs()
        inputs.m_steering, inputs.m_throttle, inputs.m_braking = float(cmd[0]), float(cmd[1]), float(cmd[2])
        return inputs

    # ---- stop rule (external modes only; the CRM collector's rule: same status, same counter)
    def check_near_stop(self, frame, record_state, record_action, record_parked):
        if not self.external or self.near_stop_fired:
            return None
        slow = (not record_parked[-1]) and abs(float(record_state[-1][0])) < STALL_VX
        self.near_stop_run = self.near_stop_run+1 if slow else 0
        self.near_stop_longest = max(self.near_stop_longest, self.near_stop_run)
        if self.near_stop_run >= self.near_stop_frames:
            self.near_stop_fired = True
            self.events.append({"kind": NEAR_STOP_STATUS, "rule": "near_stop_any_throttle", "interval_end_s": (frame+1)*DT,
                                "near_stop_s": self.near_stop_run*DT, "vx_threshold_mps": STALL_VX})
            return NEAR_STOP_STATUS
        return None

    # ---- after the run
    def finish(self, out):
        ext = {"mode": self.mode, "external_control": self.external,
               "near_stop_rule": {"active": self.external, "threshold_s": self.near_stop_frames*DT, "vx_threshold_mps": STALL_VX,
                                  "status": NEAR_STOP_STATUS, "rule": "near_stop_any_throttle", "fired": self.near_stop_fired,
                                  "longest_near_stop_run_s": self.near_stop_longest*DT,
                                  "definition": "consecutive recorded intervals with |vx| < 0.3 m/s and not parked, regardless of throttle; checked after the native rules of the frame (the CRM collector's rule)"},
               "events": self.events, "branch": self.branch_info}
        if self.external:
            L = self.log
            arrays = {"frame": np.asarray(L["frame"], np.int64), "shadow_action": np.asarray(L["shadow"], np.float64),
                      "held_action": np.asarray(L["held"], np.float64), "desired_speed_mps": np.asarray(L["desired"], np.float64),
                      "parked": np.asarray(L["parked"], bool)}
            if L["perturb"]:
                arrays["perturbation"] = np.asarray(L["perturb"], np.float64)
                arrays["perturbation_columns"] = np.asarray(["d_steer", "d_throttle", "tap_active", "tap_level"])
            if L["obs"]:
                arrays["policy_obs"] = np.stack(L["obs"])
            np.savez_compressed(out/"ext_control.npz", **arrays)
            ext["ext_control"] = {"frames": len(L["frame"]), "columns": "shadow_action = follower output read at the top of the frame (pre-clip); held_action = the triple written at every substep = recorded action"}
        if self.ou is not None:
            ext["perturbation"] = {**self.ou.summary(), "tau_s": self.args.ou_tau_s, "steer_sd": self.args.ou_steer_sd,
                                   "throttle_sd": self.args.ou_throttle_sd, "brake_p_per_frame": self.args.brake_p,
                                   "seed_source": "--perturb-seed" if self.args.perturb_seed is not None else "--episode-seed"}
        if self.pobs is not None:
            ext["policy"] = {"actor": str(Path(self.args.actor).resolve()), "actor_sha256": sha(self.args.actor),
                             "num_obs": self.pobs.num_obs, "nonfinite_obs_entries": self.pobs.n_nonfinite,
                             "obs_layout": self.pobs.layout(),
                             "state_capture": "top of frame, before hmmwv.Synchronize; engine speed = transmission.GetOutputMotorshaftSpeed(); the 12 observable columns equal the recorded state[k] (asserted, see current_state_row_check)"}
        return ext


# =============================================================================================== substep log
class SubstepLog:
    """Wraps the rich-telemetry observer; keeps the clamped triple handed to hmmwv.Synchronize at every substep."""

    def __init__(self, inner, out):
        self.inner, self.out, self.rows, self.summary = inner, Path(out), [], None

    def on_frame(self, *a, **k):
        return self.inner.on_frame(*a, **k)

    def on_substep(self, scene, frame, substep, dt_s, action3=None):
        self.inner.on_substep(scene, frame, substep, dt_s, action3)
        self.rows.append((frame, substep, float(action3[0]), float(action3[1]), float(action3[2])))

    def on_post_substep(self, *a, **k):
        return self.inner.on_post_substep(*a, **k)

    def finish(self, scene, N, terminal_state, terminal_pose, last_action):
        self.inner.finish(scene, N, terminal_state, terminal_pose, last_action)
        rows = np.asarray(self.rows, np.float64)
        frames, subs = rows[:, 0].astype(np.int64), rows[:, 1].astype(np.int64)
        nsub = int(subs.max())+1
        applied = np.full((N, nsub, 3), np.nan)
        applied[frames, subs] = rows[:, 2:5]
        require(np.isfinite(applied).all(), "Substep log has gaps")
        rng = applied.max(1)-applied.min(1)
        self.summary = {"frames": int(N), "substeps": nsub,
                        "max_intra_interval_range": rng.max(0).tolist(),
                        "intervals_with_range_gt_0": int((rng.max(1) > 0).sum()),
                        "mean_abs_dev_of_substep_mean_from_action_k": np.abs(applied.mean(1)-applied[:, 0]).mean(0).tolist()}
        np.savez_compressed(self.out/"substep_actions.npz", applied=applied, n_frames=np.int64(N), substeps=np.int64(nsub),
                            columns=np.asarray(["steer", "throttle", "brake"]))


# =============================================================================================== helpers
def stall_class(states, actions, run):
    """'stalled' when the last ``run`` rows all have |vx| < 0.3 and throttle > 0.3 (the cache contract), else 'moving'."""
    states, actions = np.asarray(states), np.asarray(actions)
    if len(states) < run:
        return "moving"
    if np.all(np.abs(states[-run:, 0]) < STALL_VX) and np.all(actions[-run:, 1] > STALL_THROTTLE):
        return "stalled"
    return "moving"


def prefix_history(states, actions, F, T=HIST_T, cols=HIST_STATE_COLS):
    """The causal T-frame window ending at the branch frame F in the mixed dataset's convention (ga_build_mixed.py,
    identical to crm_collect_ext.prefix_history): hist[t] = [state[F-(T-1)+t][cols], action[F-T+t]],
    hmask[t] = (F-T+t >= 0); masked rows are zero."""
    states, actions = np.asarray(states, np.float32), np.asarray(actions, np.float32)
    require(len(states) > F and len(actions) > F, "branch frame beyond the recorded prefix")
    t = np.arange(T)
    si, ai = F-(T-1)+t, F-T+t
    ok = ai >= 0
    hist, hmask = np.zeros((T, len(cols)+3), np.float32), np.zeros(T, bool)
    hmask[ok] = True
    hist[ok] = np.concatenate([states[si[ok]][:, list(cols)], actions[ai[ok]]], 1)
    return hist, hmask


def augment_trajectory(path, extra):
    """Re-save trajectory.npz with the original arrays plus ``extra`` (atomic replace); non-native modes only."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    for k in extra:
        require(k not in arrays, f"trajectory.npz already has {k}")
    arrays.update(extra)
    tmp = path.with_name(path.stem+".tmp.npz")   # keep the .npz suffix: savez would append one otherwise
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def parser():
    p = gen_collect.parser()
    p.description = __doc__
    p.add_argument("--mode", choices=MODES, default="native")
    p.add_argument("--branch-frame", type=int, help="branch / branch_auto: swap (or cut) frame F >= 1")
    p.add_argument("--branch-route", help="branch: route file (collector format) from the frame-F pose to the case goal")
    p.add_argument("--branch-start-tol-m", type=float, default=1., help="branch: max distance of the branch route's first waypoint from the vehicle at F")
    p.add_argument("--actor", help="policy: gc_control actor npz")
    p.add_argument("--substep-log", action="store_true")
    p.add_argument("--local", action="store_true", help="bypass source-manifest and FDM_RUNTIME_FINGERPRINT gates (local smokes only)")
    p.add_argument("--near-stop-s", type=float, default=40., help="external modes: any-throttle near-stop window (>= 40 s)")
    p.add_argument("--perturb-seed", type=int, default=None, help="pid_perturbed: OU seed (else --episode-seed; one of them is REQUIRED, no default)")
    p.add_argument("--episode-seed", type=int, default=None, help="provenance / perturbation seed (the CRM task-row convention)")
    p.add_argument("--brake-p", type=float, default=gc_control.OUPerturb.brake_p_for_rate(1.4/20.), help="pid_perturbed: per-frame tap-start probability")
    p.add_argument("--ou-tau-s", type=float, default=.5)
    p.add_argument("--ou-steer-sd", type=float, default=.15)
    p.add_argument("--ou-throttle-sd", type=float, default=.25)
    p.add_argument("--recorded", help="branch_auto: recorded run dir (trajectory.npz + outcome.json) of --case/--route")
    p.add_argument("--n-cont", type=int, default=3)
    p.add_argument("--cont-seed", type=int)
    p.add_argument("--cont-heading-tol-deg", type=float, default=15., help="branch_auto: max |first route heading - vehicle yaw| of a continuation")
    p.add_argument("--cont-accel-cap", action="store_true", help="branch_auto: also cap continuation speeds at the 1.5 m/s^2 acceleration ramp from the replayed vx (first speed == vx exactly)")
    p.add_argument("--cont-parallel", type=int, default=1, help="branch_auto: concurrent continuation subprocesses")
    p.add_argument("--replay-pose-tol-m", type=float, default=.5)
    p.add_argument("--stall-run-frames", type=int, default=20)
    p.add_argument("--sub-timeout-s", type=float, default=3600.)
    return p


def validate_mode_args(args):
    if args.mode == "branch":
        require(args.branch_frame is not None and args.branch_frame >= 1 and args.branch_route, "branch needs --branch-frame >= 1 and --branch-route")
        require(args.branch_frame*DT < args.horizon_s-1e-9, "branch frame must precede the horizon")
    if args.mode == "policy":
        require(bool(args.actor) and Path(args.actor).is_file(), "policy needs --actor npz")
    if args.mode in EXTERNAL_MODES:
        require(args.near_stop_s >= 40., "Do not shorten the 40 s near-stop safeguard")
    if args.mode == "pid_perturbed":
        require(args.perturb_seed is not None or args.episode_seed is not None,
                "pid_perturbed needs --perturb-seed or --episode-seed (no default seed: every episode must have its own)")
    if args.mode == "branch_auto":
        require(args.recorded and Path(args.recorded).is_dir(), "branch_auto needs --recorded run dir")
        require(args.branch_frame is not None and args.branch_frame >= 1 and args.n_cont >= 1 and args.cont_seed is not None,
                "branch_auto needs --branch-frame >= 1, --n-cont >= 1 and --cont-seed")
        require(args.branch_frame*DT < args.horizon_s-1e-9, "branch frame must precede the horizon")
        require(args.cont_heading_tol_deg is None or args.cont_heading_tol_deg > 0., "--cont-heading-tol-deg must be positive")


def policy_state_check(out, pobs):
    """policy mode: the newest row of the observation's past-state block (the state the actor saw at frame k) must
    equal the recorded state[k] on every observable column; returns the per-column maxima for outcome.json."""
    cols = list(pobs.cols)
    with np.load(out/"trajectory.npz", allow_pickle=False) as z:
        rec = np.asarray(z["state"], np.float64)[:, cols]
    with np.load(out/"ext_control.npz", allow_pickle=False) as z:
        obs = np.asarray(z["policy_obs"], np.float64)
    cur = obs[:, obs.shape[1]-len(cols):]
    if pobs.state_mean is not None:
        cur = cur*np.asarray(pobs.state_std, float)+np.asarray(pobs.state_mean, float)
    require(cur.shape == rec.shape, f"policy_obs rows {cur.shape} vs recorded states {rec.shape}")
    per_col = np.abs(cur-rec).max(axis=0)
    require(float(per_col.max()) <= 1e-6, f"policy's current-state row differs from recorded state[k]; per-column max {per_col.tolist()}")
    return {"frames": int(len(rec)), "state_columns": cols, "max_abs_diff_per_column": per_col.tolist(),
            "max_abs_diff": float(per_col.max()), "tolerance": 1e-6}


# =============================================================================================== one episode
def run_episode(args):
    source, out = Path(args.source_root).resolve(), Path(args.out).resolve()
    args.case, args.route, args.out = str(Path(args.case).resolve()), str(Path(args.route).resolve()), str(out)
    require(0. < args.horizon_s <= 120. and abs(args.horizon_s/DT-round(args.horizon_s/DT)) < 1e-8, "Horizon must be a positive 50 ms multiple, at most 120 s")
    require(args.minimum_elapsed_s >= 24. and args.confirm_s >= 2. and args.recovery_tail_s >= 8., "Do not shorten the declared failure/recovery safeguards")
    gates = {"source_manifest": "checked", "runtime_fingerprint": "checked"}
    manifest_path = source/"source_manifest.json"
    source_hashes = {name: sha(source/name) for name in gen_collect.SOURCE_FILES}
    if args.local:
        gates["source_manifest"] = "BYPASSED (--local)"
        manifest_sha = sha(manifest_path) if manifest_path.is_file() else None
    else:
        manifest = read(manifest_path)
        if args.source_manifest_sha256:
            require(sha(manifest_path) == args.source_manifest_sha256, "Source manifest mismatch")
        require(all(manifest["files"].get(name) == value for name, value in source_hashes.items()), "Frozen source file mismatch")
        manifest_sha = sha(manifest_path)
    case = read(args.case)
    require(case["split"] in ("train", "val", "test"), "Missing declared group split")
    arena = (source/case["arena"]).resolve()
    meta = read(arena/"arena_meta.json")
    allowed = read(HERE/"gen_arenas.json")
    require(float(meta["size_m"]) == 80., "Require the 80 m arena size of the f104 family")
    require(allowed.get(arena.name) == sha(arena/meta["bmp"]), f"Arena {arena.name} BMP not in gen_arenas.json allowlist")
    require(case["layout"]["assets"] == [], "This campaign is the exact F104 terrain without added assets")
    for field in ("case", "route"):
        if getattr(args, field+"_sha256"):
            require(sha(getattr(args, field)) == getattr(args, field+"_sha256"), f"{field} checksum mismatch")
    module = gen_collect.import_runner(source)
    route = module.read_route(args.route)
    branch_route = module.read_route(args.branch_route) if args.mode == "branch" else None
    actor = gc_control.NumpyActor.from_npz(args.actor) if args.mode == "policy" else None
    run, adapter = replacement_adapter(module)
    args.f104_policy = gen_collect.StopPolicy(args, case)
    args.gen_ext = GenExt(args, case, route, branch_route, actor)
    contract = {"schema": "f104_collection_request_v1", "source_root": str(source),
        "source_manifest_sha256": manifest_sha, "source_sha256": source_hashes,
        "wrapper_sha256": sha(__file__), "gen_collect_sha256": sha(gen_collect.__file__), "gc_control_sha256": sha(gc_control.__file__),
        "f104_n2_sampler_sha256": sha(HERE/"f104_n2_sampler.py"),   # sample_continuations' route family (branch_auto provenance)
        "adapter": adapter, "case": args.case, "case_sha256": sha(args.case),
        "route": args.route, "route_sha256": sha(args.route), "scene_id": case["id"], "split": case["split"],
        "arena_bmp_sha256": sha(arena/meta["bmp"]), "arena_meta_sha256": sha(arena/"arena_meta.json"),
        "horizon_s": args.horizon_s, "stop_policy": args.f104_policy.config(), "route_metadata": route.get("meta", {}),
        "observation": "One separately rendered terrain-only RGB-D map may be joined by BMP/camera/runtime hash. Per-episode measured anchor is distinct; no shared vehicle-anchor equality claimed.",
        "time_accounting": "Completed measured traversal intervals only; exclude 0.8 s settling, failed processes and any postprocessing window overlap",
        "ext": {"mode": args.mode, "gates": gates, "substep_log": bool(args.substep_log),
                "branch_frame": args.branch_frame if args.mode == "branch" else None,
                "branch_route": str(Path(args.branch_route).resolve()) if args.mode == "branch" else None,
                "branch_route_sha256": gc_control.route_sha256(branch_route) if branch_route is not None else None,
                "branch_route_file_sha256": sha(args.branch_route) if args.mode == "branch" else None,
                "actor_sha256": sha(args.actor) if actor is not None else None,
                "perturb_seed": perturb_seed(args) if args.mode == "pid_perturbed" else None,
                "episode_seed": args.episode_seed,
                "near_stop_s": args.near_stop_s if args.mode in EXTERNAL_MODES else None}}
    if args.check_only:
        print(json.dumps({"check_only": True, "scene_id": case["id"], "contract": contract}))
        return
    out.mkdir(parents=True, exist_ok=True)
    require(not any((out/name).exists() for name in ("outcome.json", "collection_request.json", "trajectory.npz")), "Preserve existing output; use a new directory")
    fingerprint = os.environ.get("FDM_RUNTIME_FINGERPRINT")
    if args.local and not (fingerprint and Path(fingerprint).is_file()):
        gates["runtime_fingerprint"] = "BYPASSED (--local, no FDM_RUNTIME_FINGERPRINT)"
        contract["runtime_fingerprint_sha256"] = contract["runtime_sha256"] = None
    else:
        require(fingerprint and Path(fingerprint).is_file(), "FDM_RUNTIME_FINGERPRINT must bind the full Chrono libraries and vehicle assets")
        runtime = read(fingerprint)["runtime_sha256"]
        require(runtime and any("_vehicle.so" in name for name in runtime) and any("/vehicle/hmmwv/" in name for name in runtime), "Runtime fingerprint lacks native vehicle library or HMMWV data")
        contract["runtime_fingerprint_sha256"] = sha(fingerprint)
        contract["runtime_sha256"] = runtime
    contract["thread_environment"] = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LP_NUM_THREADS")}
    dump(out/"collection_request.json", contract)
    shutil.copyfile(args.case, out/"case.json")
    shutil.copyfile(args.route, out/"reference.json")
    if branch_route is not None and Path(args.branch_route).resolve() != (out/"branch_route.json"):
        shutil.copyfile(args.branch_route, out/"branch_route.json")
    args.command, args.backend, args.depth_ray_scale = "collect", "Vulkan_RT_lavapipe", 1.
    args.record_rgbd_stride, args.path_height_source = 0, "truth"
    args.render_parity, args.rich_telemetry = False, True
    observer = gen_collect.make_observer(out, args.case)
    args.frame_observer = SubstepLog(observer, out) if args.substep_log else observer
    started = time.time()
    try:
        run(args)
        outcome = read(out/"outcome.json")
        ext = args.gen_ext.finish(out)
        ext["gates"] = gates
        if args.substep_log:
            ext["substep_log"] = args.frame_observer.summary
        if args.mode == "policy":
            ext["policy"]["current_state_row_check"] = policy_state_check(out, args.gen_ext.pobs)
        if args.mode == "branch" and args.gen_ext.branch_info is None:
            # the prefix drive ended (goal, rollover, bounds, stop rule) before frame F: nothing was branched.  A skip
            # with a reason, not a failure: the runner sees episode_complete.json {skipped: true} (0 h of new data).
            reason = f"episode_ended_before_branch_frame: status {outcome['status']} after {outcome['frames']} frames (branch frame {args.branch_frame})"
            outcome["ext"] = ext
            (out/"outcome.json").write_text(json.dumps(outcome, indent=2)+"\n")
            dump(out/"skipped.json", {"schema": "f104_episode_skipped_v1", "mode": "branch", "reason": reason, "status": outcome["status"],
                                      "frames": int(outcome["frames"]), "branch_frame": int(args.branch_frame)})
            dump(out/"episode_complete.json", {"schema": "f104_episode_complete_v1", "skipped": True, "reason": reason,
                                               "actual_elapsed_s": 0., "ext_mode": args.mode})
            print(json.dumps({"out": str(out), "mode": args.mode, "skipped": True, "reason": reason}))
            return
        if args.mode != "native":
            if args.mode == "branch":
                b = args.gen_ext.branch_info
                with np.load(out/"trajectory.npz", allow_pickle=False) as z:
                    hist, hmask = prefix_history(z["state"], z["action"], b["branch_frame"])
                b.update({"hist_T": HIST_T, "hist_cols": list(HIST_STATE_COLS),
                          "hist_layout": "hist[t] = [state[F-39+t][hist_cols], action[F-40+t]]; hmask[t] = action frame F-40+t >= 0 (ga_build_mixed convention)",
                          "hist": hist.tolist(), "hmask": hmask.tolist()})
                augment_trajectory(out/"trajectory.npz", {"ext_mode": np.asarray(args.mode), "branch_frame": np.int64(b["branch_frame"]),
                    "branch_pose": np.asarray(b["branch_pose"], np.float64), "branch_route_sha256": np.asarray(b["branch_route_sha256"]),
                    "branch_hist": hist, "branch_hmask": hmask, "branch_hist_cols": np.asarray(HIST_STATE_COLS, np.int16)})
            else:
                augment_trajectory(out/"trajectory.npz", {"ext_mode": np.asarray(args.mode)})
            outcome["ext"] = ext
            if args.mode == "branch":
                outcome.update({k: b[k] for k in ("branch_frame", "branch_pose", "branch_route_sha256")})
            if args.gen_ext.events:
                outcome["ext_stop_events"] = args.gen_ext.events
            (out/"outcome.json").write_text(json.dumps(outcome, indent=2)+"\n")
        with np.load(out/"trajectory.npz", allow_pickle=False) as data:
            n = len(data["state"])
            require(n > 0 and n == outcome["frames"] and abs(n*DT-outcome["elapsed_s"]) < 1e-8, "Actual interval count/duration disagree")
            require(data["terminal_state"].shape == (17,) and np.isfinite(data["terminal_pose"]).all(), "Missing finite actual terminal measurement")
            if args.mode in EXTERNAL_MODES:
                held = np.load(out/"ext_control.npz", allow_pickle=False)["held_action"]
                require(len(held) == n and np.abs(held-data["action"]).max() < 1e-6, "Recorded action is not the held triple")
        with np.load(out/"rich_telemetry.npz", allow_pickle=False) as rich:
            desired = rich["command_desired_speed_mps"][:n].copy()
        reference = dict(interval_start_s=np.arange(n)*DT, desired_speed_mps=desired,
            reference_waypoints=np.asarray(route["waypoints"]), reference_stations=np.asarray(route["stations"]),
            reference_speeds=np.asarray(route["speeds"]), reference_headings=np.asarray(route["headings"]))
        if branch_route is not None:
            reference.update(branch_frame=np.int64(args.branch_frame), branch_waypoints=np.asarray(branch_route["waypoints"]),
                branch_stations=np.asarray(branch_route["stations"]), branch_speeds=np.asarray(branch_route["speeds"]),
                branch_headings=np.asarray(branch_route["headings"]))
        np.savez_compressed(out/"command_reference.npz", **reference)
        require(all(sha(source/name) == value for name, value in source_hashes.items()), "Source changed during collection")
        require(sha(args.case) == contract["case_sha256"] and sha(args.route) == contract["route_sha256"], "Input changed during collection")
        summary = {"schema": "f104_collection_completed_v1", "scene_id": case["id"], "split": case["split"],
            "status": outcome["status"], "actual_elapsed_s": n*DT, "actual_elapsed_h": n*DT/3600.,
            "interval_count": n, "terminal_endpoint_recorded": True, "future_padding": False,
            "requested_horizon_s": args.horizon_s, "horizon_reached": abs(n*DT-args.horizon_s) < 1e-8,
            "early_stop_policy": args.f104_policy.config(), "stop_events": args.f104_policy.events+args.gen_ext.events,
            "native_height_valid": args.f104_policy.native_height_report["passed"],
            "initial_state_valid": args.f104_policy.initial_state_report["passed"],
            "outcome_contact_scope": "Original outcome asset_contact excludes chassis. Use rich_intervals max_chassis_contact_resultant_n OR max_asset_contact_max_resultant_n for physical contact labels.",
            "additional_normals_scope": "Geometric RigidTerrain.GetNormal under measured wheel hubs; projected tire force is a derived quantity, not an internal tire normal load/contact-pair measurement.",
            "mechanical_work_is_not_fuel": True, "wall_s_including_finalization": time.time()-started,
            "collection_request_sha256": sha(out/"collection_request.json"), "ext_mode": args.mode, "gates": gates}
        dump(out/"f104_episode.json", summary)
        artifacts = {str(path.relative_to(out)): sha(path) for path in sorted(out.iterdir()) if path.is_file() and path.suffix != ".log"}
        dump(out/"episode_complete.json", {"schema": "f104_episode_complete_v1", "actual_elapsed_s": n*DT, "ext_mode": args.mode,
            "artifacts_sha256": artifacts, "request_sha256": sha(out/"collection_request.json")})
        print(json.dumps({"out": str(out), "mode": args.mode, "status": outcome["status"], "actual_elapsed_s": n*DT, "complete": True}))
    except Exception as exc:
        dump(out/"collection_failure.json", {"type": type(exc).__name__, "error": str(exc), "mode": args.mode,
            "count_toward_completed_hours": False, "stop_events": args.f104_policy.events})
        raise


# =============================================================================================== branch_auto
def subprocess_cmd(args, mode, out, horizon_s, extra):
    cmd = [sys.executable, "-P", "-u", str(Path(__file__).resolve()), "--source-root", args.source_root,
           "--case", args.case, "--route", args.route, "--out", str(out), "--chrono-data", args.chrono_data,
           "--horizon-s", repr(float(horizon_s)), "--mode", mode,
           "--minimum-elapsed-s", repr(args.minimum_elapsed_s), "--confirm-s", repr(args.confirm_s), "--recovery-tail-s", repr(args.recovery_tail_s)]
    for flag, name in (("--local", "local"), ("--disable-early-stop", "disable_early_stop"), ("--substep-log", "substep_log")):
        if getattr(args, name):
            cmd.append(flag)
    for name in ("source_manifest_sha256", "case_sha256", "route_sha256"):
        if getattr(args, name):
            cmd += ["--"+name.replace("_", "-"), getattr(args, name)]
    return cmd+list(extra)


def run_subprocess(cmd, out, timeout_s):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out/"collector.log", "a") as log:
        log.write(" ".join(cmd)+"\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s)
    return {"rc": proc.returncode, "wall_s": time.time()-t0, "complete": (out/"episode_complete.json").is_file()}


def branch_auto(args):
    F, n = int(args.branch_frame), int(args.n_cont)
    out = Path(args.out).resolve()
    args.case, args.route, args.out = str(Path(args.case).resolve()), str(Path(args.route).resolve()), str(out)
    out.mkdir(parents=True, exist_ok=True)
    require(not (out/"branch_auto.json").exists() and not (out/"skipped.json").exists(), "Preserve existing output; use a new directory")
    rec_dir = Path(args.recorded).resolve()
    rec_outcome = read(rec_dir/"outcome.json")
    require(rec_outcome["route_sha256"] == sha(args.route) and rec_outcome["case_sha256"] == sha(args.case),
            "Recorded episode's route/case hashes differ from --route/--case")
    case = read(args.case)
    goal = np.asarray(case["goal_xy"], float)
    with np.load(rec_dir/"trajectory.npz", allow_pickle=False) as z:
        rs, ra, rp, rtp, rts = z["state"], z["action"], z["pose"], z["terminal_pose"], z["terminal_state"]
    T = int(len(rs))
    started = time.time()
    summary = {"schema": "gen_ext_branch_auto_v1", "mode": "branch_auto", "recorded": str(rec_dir), "recorded_status": rec_outcome["status"],
               "recorded_frames": T, "branch_frame": F, "branch_time_s": F*DT, "n_cont": n, "cont_seed": int(args.cont_seed),
               "replay_pose_tol_m": args.replay_pose_tol_m, "stall_run_frames": args.stall_run_frames, "local": bool(args.local),
               "cont_heading_tol_deg": args.cont_heading_tol_deg, "replay_dir": f"{out}__replay_F{F}"}

    def skip(reason, **extra):
        summary.update({"skipped": True, "reason": reason, "wall_s": time.time()-started, **extra})
        dump(out/"skipped.json", summary)
        dump(out/"episode_complete.json", {"schema": "gen_ext_branch_auto_v1", "skipped": True, "reason": reason, "actual_elapsed_s": 0.})
        print(json.dumps({"out": str(out), "mode": "branch_auto", "skipped": True, "reason": reason}))
        return 0

    if F > T:
        return skip(f"recording_shorter_than_branch_frame: {T} < {F}")
    rec_pose = rp[F] if F < T else rtp
    rec_vx = float(rs[F, 0]) if F < T else float(rts[0])
    rec_class = stall_class(rs[:F], ra[:F], args.stall_run_frames)
    # pass 1: native replay of the prefix with horizon F
    replay = Path(f"{out}__replay_F{F}")          # keyed by F: a cached replay of another branch frame is never reused
    if not (replay/"episode_complete.json").is_file():
        require(not (replay/"outcome.json").exists(), f"Incomplete replay dir {replay}; remove it")
        r = run_subprocess(subprocess_cmd(args, "native", replay, F/20., []), replay, args.sub_timeout_s)
        summary["replay"] = r
        if r["rc"] != 0 or not r["complete"]:
            raise RuntimeError(f"replay subprocess failed rc={r['rc']}; see {replay}/collector.log")
    else:
        summary["replay"] = {"cached": True}
        require(abs(float(read(replay/"f104_episode.json")["requested_horizon_s"])-F*DT) < 1e-9, f"Cached replay {replay} was not run with horizon F")
    ro = read(replay/"outcome.json")
    with np.load(replay/"trajectory.npz", allow_pickle=False) as z:
        ps, pa, pp, ptp, pts = z["state"], z["action"], z["pose"], z["terminal_pose"], z["terminal_state"]
    summary["replay"].update({"status": ro["status"], "frames": int(ro["frames"])})
    if int(ro["frames"]) != F or ro["status"] != "timeout":
        # fewer than F frames, or F frames but the last one reached the goal / a stop rule: the branch frame is never
        # driven, so a branch subprocess would end before F.  Skip with the reason instead.
        return skip(f"replay_ended_before_branch_frame: status {ro['status']} after {ro['frames']} frames (branch frame {F})")
    rep_pose, rep_vx = np.asarray(ptp, float), float(pts[0])
    rep_class = stall_class(ps, pa, args.stall_run_frames)
    d_xy = float(np.linalg.norm(rep_pose[:2]-rec_pose[:2]))
    d_yaw = float(abs(np.arctan2(np.sin(rep_pose[2]-rec_pose[2]), np.cos(rep_pose[2]-rec_pose[2]))))
    prefix_max_dev = float(np.abs(ps[:, :7]-rs[:F, :7]).max())
    prefix_pose_dev = float(np.linalg.norm(rp[:F, :2]-pp[:, :2], axis=1).max())
    compare = {"recorded_pose_F": np.asarray(rec_pose, float).tolist(), "replayed_pose_F": rep_pose.tolist(), "pose_xy_diff_m": d_xy,
               "yaw_diff_rad": d_yaw, "recorded_vx_F": rec_vx, "replayed_vx_F": rep_vx, "recorded_class": rec_class,
               "replayed_class": rep_class, "prefix_max_abs_state_dev_cols0_6": prefix_max_dev, "prefix_max_pose_xy_dev_m": prefix_pose_dev,
               "accepted": d_xy <= args.replay_pose_tol_m and rec_class == rep_class}
    summary["replay_comparison"] = compare
    if not compare["accepted"]:
        return skip("replay_mismatch: pose diff %.3f m (tol %.2f), class %s vs %s" % (d_xy, args.replay_pose_tol_m, rec_class, rep_class))
    # continuations from the REPLAYED pose, floored at the replayed speed (no brake burst at the branch) and within the
    # start-heading tolerance (no steering kink at the branch)
    summary["continuation_sampling"] = {"v0_mps": rep_vx, "max_start_heading_err_deg": args.cont_heading_tol_deg,
                                        "speed_floor": "min(max(speeds, sqrt(max(v0^2 - 2*2.0*(s - s0), 0))), 6.0)",
                                        "accel_cap": bool(args.cont_accel_cap)}
    try:
        routes = gc_control.sample_continuations(rep_pose, goal, k=n, seed=int(args.cont_seed), v0=rep_vx,
                                                 max_start_heading_err_deg=args.cont_heading_tol_deg, v0_accel_cap=bool(args.cont_accel_cap))
    except (RuntimeError, ValueError) as exc:
        return skip(f"no_valid_continuation: {exc}")
    dirs = []
    for i, route in enumerate(routes):
        d = Path(f"{out}__c{i}")
        d.mkdir(parents=True, exist_ok=True)
        rf = d/"branch_route.json"
        if not rf.exists():
            dump(rf, gc_control.route_to_json(route))
        dirs.append(d)
    summary["continuations"] = [{"dir": str(d), "route_sha256": r["meta"]["route_sha256"], "base_speed_mps": r["meta"]["base_speed_mps"],
                                 "draws": r["meta"]["draws"], "waypoints": int(len(r["waypoints"])),
                                 "start_speed_mps": r["meta"]["start_speed_mps"], "start_heading_err_deg": r["meta"]["start_heading_err_deg"],
                                 "speed_floor_raised_points": r["meta"]["speed_floor_raised_points"]} for d, r in zip(dirs, routes)]

    def drive(i):
        d = dirs[i]
        if (d/"episode_complete.json").is_file():
            return {"cached": True, "complete": True, "rc": 0}
        require(not (d/"outcome.json").exists(), f"Incomplete continuation dir {d}; remove it")
        cmd = subprocess_cmd(args, "branch", d, args.horizon_s, ["--branch-frame", str(F), "--branch-route", str(d/"branch_route.json"),
                                                                 "--branch-start-tol-m", repr(args.branch_start_tol_m)])
        return run_subprocess(cmd, d, args.sub_timeout_s)

    with ThreadPoolExecutor(max_workers=max(1, int(args.cont_parallel))) as ex:
        results = list(ex.map(drive, range(n)))
    ok = True
    for c, r in zip(summary["continuations"], results):
        c.update(r)
        if r["complete"] and (Path(c["dir"])/"skipped.json").is_file():
            sk = read(Path(c["dir"])/"skipped.json")
            c.update({"skipped": True, "reason": sk["reason"], "status": sk.get("status"), "frames": 0})
        elif r["complete"]:
            o = read(Path(c["dir"])/"outcome.json")
            c.update({"skipped": False, "status": o["status"], "frames": int(o["frames"]), "goal_reached": bool(o["goal_reached"]),
                      "branch_pose": o.get("branch_pose"), "branch_start_error_m": o["ext"]["branch"]["branch_start_error_m"],
                      "vehicle_vx_at_swap_mps": o["ext"]["branch"].get("vehicle_vx_at_swap_mps"),
                      "branch_start_speed_mps": o["ext"]["branch"].get("branch_start_speed_mps")})
        ok = ok and r["complete"]
    summary.update({"skipped": False, "all_complete": ok, "n_skipped_continuations": int(sum(bool(c.get("skipped")) for c in summary["continuations"])),
                    "wall_s": time.time()-started})
    dump(out/"branch_auto.json", summary)
    if ok:
        dump(out/"episode_complete.json", {"schema": "gen_ext_branch_auto_v1", "skipped": False, "branch_frame": F,
            "continuations": [c["dir"] for c in summary["continuations"]],
            "skipped_continuations": [c["dir"] for c in summary["continuations"] if c.get("skipped")],
            "actual_elapsed_s": sum(c["frames"]*DT for c in summary["continuations"] if not c.get("skipped"))})
    print(json.dumps({"out": str(out), "mode": "branch_auto", "skipped": False, "all_complete": ok,
                      "statuses": [c.get("status") for c in summary["continuations"]]}))
    return 0 if ok else 1


def main():
    args = parser().parse_args()
    validate_mode_args(args)
    if args.mode == "branch_auto":
        if args.check_only:
            print(json.dumps({"check_only": True, "mode": "branch_auto", "recorded": args.recorded, "branch_frame": args.branch_frame}))
            return
        sys.exit(branch_auto(args))
    run_episode(args)


if __name__ == "__main__":
    main()
