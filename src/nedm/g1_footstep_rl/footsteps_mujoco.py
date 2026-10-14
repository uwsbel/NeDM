"""The footstep-target task in CPU MuJoCo (ground truth): course scenes, the episode loop with the numpy scorer, the
FS policy runner, the H-FS baseline on the same courses, the same episodes inside the NRD from the hand-over
snapshots, statistics, and the visible evaluation (footprint markers, plank colours, top and chase cameras, HUD,
per-course step-length plots, side-by-side videos).

"spec N" and the decision ids M1-M7 refer to the internal design spec of the study (not released; see the package
docstring). Task, courses and scorer: nedm.g1_footstep_rl.footsteps_task (FT). Script:
scripts/g1_footstep_rl/eval_footsteps_mujoco.py; H-FS tuning: scripts/g1_footstep_rl/footstep_baselines.py. Needs
torch AND mujoco in one interpreter.

Scenes (spec 12.1). Flat courses: G1MujocoEnv(terrain=None, ...) = the collector scene, bit-identical
(make_course_env). Plank courses: make_planks_env(course.to_layout(), ...) (boxes, pit at -0.6 m, foot_in_pit). The
robot starts in the default standing pose at course.start_pose(); time = 1 s stand + 12 s policy (EPISODE_S 13 s).

Runner contract (FootstepRunner, spec 12.2; extends mujoco_policy (1)-(8)):
 (1)-(8) averaged rates only (env.avg_state()); 1 s stand at (0, 0, 0, 0.74) with the torch HOMIE copy; hand-over
     values (previous command = stand, previous u = tanh(action_of_command(0)), gate
     [1, 1]); label of a step = contact in this step OR the previous one; HOMIE preview without a memory update; the
     HOMIE copy steps once per control step; the planks env action map (ActionMap({"action": ec["action"]})).
 (9) the course is the placed numpy Course (float64). At the hand-over control step k_h = 50 the episode sets target 0
     from the hand-over state (course.with_target0(centre of foot f0)).
 (10) one FT.FootstepProgress per episode, made at the hand-over (start feet / yaws / spheres = the hand-over state,
     t0_step = k_h) and updated by run_course_episode after every control step from (avg state, gate). The runner
     only reads it (CourseEpisode.progress).
 (11) the observation is footsteps_env.footsteps_observation (the env's own pure function).
 (12) the leg rule and the support rule of the NRD env are evaluated as diagnostics on the same states as in the NRD
     (newest and previous averaged state; nrd_rule_diagnostics): MuJoCo itself ends an episode only by its own events.
 (13) snapshot at the hand-over (FootstepRunner.snapshot): the 4 last averaged states, HOMIE memory, previous command
     and action, gate, the course (to_dict, target 0 set) and the progress state, for
     G1FootstepsEnv.reset_from_snapshots.

End of a MuJoCo episode (run_course_episode): a collector end event or foot_in_pit (outcome = its EVENTS name), the
evaluation fall flag ("fall_flag"), the end rule of the controller, or the time limit ("timeout").
- end_rule "footsteps" (FS, H-FS): the scorer's first terminal event: "completed" or "fail_<reason>" (stall,
  wrong_foot, far_miss, restep_budget, overrun, no_stop; strict rules as the env config); "finished" = the last target
  scored as a non-strict far miss.
- end_rule "planks" (controllers without footstep input): completion = the planks success rule (planks_task: pelvis
  0.3 m into the final platform, all 8 sphere centres on it) -> outcome "success"; the scorer only logs (non strict, it
  never ends the episode), so drift and footprint numbers exist for these controllers too.
Clean (plank courses) = planks_task.PlanksProgress counters on the true planks over the whole episode (from step 0):
no touchdown over a gap and no unsupported contact step. Clean completion = completion and clean.

Usage
    course, _ = course_for("MN", 1000)                       # or FT.showcase(name), FT.eval_course(level, seed)
    pol = load_footstep_policy("RUN/best.pt")                # FS; load_spec / make_controller for every policy kind
    row, ep, env = run_course_episode(course, FootstepRunner(pol), keep_states=True)
    rows, steps, states = nrd_rollout_courses(pol, [runner.snapshot, ...], members=(0, 1))
    render_episode(env.model, row["states"], ep, row, "out.mp4", policy_name=pol.name); plot_course(ep, row, "x.png")
    Torch threads: workers use 1; the NRD part on a loaded host should use few threads (4 threads at load 47 made a
    6-env rollout 50x slower than 1 thread).

Decisions (the spec is silent; the simplest option was taken)
 M1  H-FS drives HOMIE through the onnxruntime teacher with instantaneous rates (HOMIE's native input, the same as the
     probe that measured the H-FS speed table); teacher="ort_avg" feeds averaged rates instead. FS uses the torch HOMIE
     copy with averaged rates (contract (1)-(8)).
 M2  H-FS (spec 13): lookahead 2 = the spec rule (L* = planned L of target p + 1 while the expected foot is in the air,
     else along(T_p - last placement of the other foot)); lookahead 1 = always along(T_p - other foot). Above the
     table (L* > 0.299 m) the speed is extrapolated with the last segment's slope (else steps of 0.30-0.36 m are out of
     reach at g_v = 1); below 0.051 m linear to 0 at 0.02 m. "Expected foot in the air" = its label of the last step is
     off. Stop: vx = 0 once the pointer reaches the first stop target (the approach target is scored). l and the
     heading are taken against the local line of target p (FootstepProgress.drift); wz uses psi of target p + 1.
     Speed floor v_min while targets remain (not in the spec): target 0 is only d0 = 0-6 cm ahead, so the literal rule
     commands <= 0.1 m/s and HOMIE never starts stepping (all 7 showcases: stall at 2.5 s). Scan on F1 tuning seeds
     0-7 (g_v 1, lookahead 2, k_y 1): v_min 0.15 -> 0 / 8 completed (stall), 0.2 -> 4 / 8, 0.3 -> 0 / 8 (HOMIE's first
     step lands 6-8 cm ahead: not scored, then the other foot = wrong_foot), 0.4 -> 1 / 8. So v_min is a tuned
     parameter: the grid of footstep_baselines.py is the spec's 18 settings x v_min {0.2, 0.3} = 36.
 M4  The NRD-rule diagnostics are computed after the episode in one batch from the stored averaged states (the
     per-step leg rule costs 5 ms at batch 1): first step where the swept leg rule (rules.legs_*) or the support rule
     (rules.support_min_off spheres off the planks shrunk by rules.edge_margin, gate on, two-state) would end the
     episode inside the NRD, and the smallest swept leg distance.
 M5  Toe gap (cliff courses): edge of the final platform minus (toe sphere centre s + 5 mm) of each foot in the last
     state; reported for every stop course with a cliff.
 M6  Per-step hit rates are reported over scored targets (spec) and over all planned targets (unreached = miss); their
     95 % intervals and those of the error percentiles are episode-cluster bootstraps (1,000 resamples, seed 0);
     episode rates get Wilson 95 % intervals.
 M7  Rendering never changes the physics model or data: each camera renders a deepcopy of the model (plank colours,
     dark pit floor, headlight) with its own MjData; markers are scene geoms added after update_scene.
"""
from __future__ import annotations

import contextlib
import copy
import importlib
import io
import math
import socket
from collections import deque
from pathlib import Path

import numpy as np
import torch

from nedm.g1_footstep_rl import footsteps_task as FT
from nedm.g1_footstep_rl import planks_task as P
from nedm.g1_footstep_rl.homie_policy import FULL_HEIGHT

F64 = torch.float64
CONTROL_S = 0.02
STAND_S = P.STAND_S                         # 1 s stand
EPISODE_S = 13.0                            # 1 s stand + 12 s policy (= the NRD episode, 600 steps)
HANDOVER_STEP = int(round(STAND_S / CONTROL_S))
POLICY_STEPS = int(round((EPISODE_S - STAND_S) / CONTROL_S))
STAND_CMD = (0.0, 0.0, 0.0, FULL_HEIGHT)
RULE_DEFAULTS = dict(support="spheres", support_min_off=1, two_state=True, edge_margin=0.02, legs_min_dist=0.004,
                     legs_pelvis_min_dist=-0.001, legs_sweep=[2, 1.0, 2])
HFS_TABLE = ((0.051, 0.132, 0.191, 0.238, 0.299), (0.3, 0.5, 0.7, 0.9, 1.1))   # probe.json step length -> vx
FS_DESCRIPTOR = "trained only in the learned model, 0 MuJoCo steps"


def footsteps_env():
    """The torch env module (imported on first use, so the MuJoCo parts load without it)."""
    return importlib.import_module("nedm.g1_footstep_rl.footsteps_env")


def _M():
    from nedm.g1_footstep_rl import mujoco_env as M
    return M


# ---- scenes ----------------------------------------------------------------------------------------------------
def make_course_env(course, teacher=None, avg_rates=True, episode_s=EPISODE_S, cfg=None):
    """The MuJoCo scene of a placed course: flat -> G1MujocoEnv(terrain=None) (the collector scene); planks ->
    make_planks_env(course.to_layout()). teacher: None (the policy brings its torch HOMIE copy), "ort", "numpy"."""
    M = _M()
    if course.is_planks:
        return M.make_planks_env(course.to_layout(), teacher=teacher, avg_rates=avg_rates, cfg=cfg, episode_s=episode_s)
    return M.G1MujocoEnv(cfg, None, teacher, avg_rates, episode_s)


def course_for(level, seed):
    """(course, variant) of an evaluation cell: showcase names (heading 0), new levels (FT.sample_course: random
    heading), old levels L1-L4 (planner; None if no variant has a plan)."""
    if level in FT.SHOWCASES:
        return FT.showcase(level), None
    return FT.eval_course(level, seed)


# ---- one episode -----------------------------------------------------------------------------------------------
class CourseEpisode:
    """Bookkeeping of one MuJoCo course episode, read by the controllers: course (placed, as built), course_h (target
    0 set at the hand-over), progress (FT.FootstepProgress from the hand-over on), planks (PlanksProgress on plank
    courses, from step 0), last_gate (label of the last control step), k (control steps done), log (confirmed
    touchdowns with placements), trace (per policy step: pointer, l, dpsi)."""

    def __init__(self, course, score_cfg, end_rule):
        self.course, self.cfg, self.end_rule = course, score_cfg, end_rule
        self.course_h, self.progress, self.planks = None, None, None
        self.last_gate = np.ones(2, bool)
        self.k = 0
        self.hand_over_step = None
        self.hand_over = None
        self.log, self.trace = [], []
        self.avg_states = []                     # averaged state at the end of each control step (index k)
        self.gates = []

    def start_policy(self, env, k):
        """Hand-over (contract (9), (10)): target 0 from the hand-over state, a new FootstepProgress."""
        x = env.avg_state()
        sph = env.spheres()
        fc, fy = FT.foot_pose(sph)
        c = self.course.with_target0(fc[int(self.course.foot_of(0)), :2])
        self.course_h = c
        self.progress = FT.FootstepProgress(c, self.cfg, start_feet=fc[:, :2], start_yaw=fy, t0_step=k,
                                            start_spheres=sph)
        self.hand_over_step = k
        self.hand_over = dict(state=x, spheres=sph, feet=fc[:, :2], yaws=fy)
        self.last_gate = np.ones(2, bool)

    def progress_snapshot(self):
        """The progress part of a hand-over snapshot (contract (13)): constructor values + state()."""
        p = self.progress
        st = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.state().items()}
        return dict(st, t0_step=int(p.t0), last_td_step=p.last_td_step.tolist(), start_feet=p.start_place[:, :2].tolist(),
                    start_yaw=p.start_place[:, 2].tolist(), start_spheres=np.asarray(self.hand_over["spheres"]).tolist(),
                    cfg=dict(p.cfg))


def run_course_episode(course, controller, score=None, end_rule=None, keep_states=False, episode_s=EPISODE_S,
                       rules=None, env=None):
    """One MuJoCo episode of `controller` on a placed course (module docstring). controller(env, ep) -> (target [27],
    raw [12] or None, command [4] or None) is called once per control step; it declares scene_teacher,
    scene_avg_rates and end_rule. score: score dict or env config for FT.score_cfg (default: the controller's
    score_source, else FT.SCORE_DEFAULTS). rules: NRD-rule diagnostics config (default RULE_DEFAULTS updated by the
    controller's rules). Returns (row, ep, env); row["states"] = [T, 67] 10 ms records if keep_states."""
    M = _M()
    end_rule = end_rule or getattr(controller, "end_rule", "footsteps")
    cfg = FT.score_cfg(score if score is not None else getattr(controller, "score_source", None))
    if end_rule == "planks":
        cfg = dict(cfg, strict=False)
    if env is None:
        env = make_course_env(course, getattr(controller, "scene_teacher", None),
                              getattr(controller, "scene_avg_rates", True), episode_s)
    env.reset(pose=course.start_pose())
    ep = CourseEpisode(course, cfg, end_rule)
    ep.planks = P.PlanksProgress(course.to_layout(), spheres0=env.spheres()) if course.is_planks else None
    outcome, last = None, np.ones(2, bool)
    planks_success = False
    k = 0
    while outcome is None:
        if ep.progress is None and env.time >= STAND_S - 1e-9:
            ep.start_policy(env, k)
        target, raw, cmd = controller(env, ep)
        done, event = env.step(target, raw, cmd)
        if event:
            outcome = M.EVENTS[event]
            break
        x = env.avg_state()
        sph = env.spheres()
        rc = env.step_contacts()
        gate = rc | last                                   # the NRD label: contact in step k or k - 1
        last = rc
        ep.last_gate = gate
        ep.avg_states.append(x)
        ep.gates.append(gate)
        if ep.planks is not None:
            planks_success = ep.planks.update(x[:2], sph, gate) or planks_success
        scorer_end = None
        if ep.progress is not None:
            pr = ep.progress
            was_done = pr.done
            ev = pr.update_state(k, x, sph, gate)
            for td in ev["touchdowns"]:
                f = td["foot"]
                ep.log.append(dict(td, proc=k, x=float(pr.last_place[f, 0]), y=float(pr.last_place[f, 1]),
                                   yaw=float(pr.last_place[f, 2])))
            l, dpsi = pr.drift(x[:2], FT.quat_yaw(x[3:7]))
            ep.trace.append((k, pr.ptr, l, dpsi))
            if end_rule == "footsteps" and pr.done and not was_done:
                scorer_end = ("completed" if pr.completed else f"fail_{pr.fail}" if pr.fail else "finished")
        k += 1
        ep.k = k
        if M.fallen(x):                                    # a fall wins over a completion in the same step
            outcome = "fall_flag"
        elif scorer_end is not None:
            outcome = scorer_end
        elif end_rule == "planks" and planks_success:
            outcome = "success"
        elif done:
            outcome = "timeout"
    row = episode_row(course, controller, ep, env, outcome, rules)
    if keep_states:
        s = env.slot
        row["states"] = s.buf["state"][:s.rec + 1].copy()
    return row, ep, env


def toe_gap(course, spheres):
    """[2] distance (m) from the front edge of the final platform to the front of each foot's toe spheres (centre s +
    5 mm), course frame."""
    sl = course.path_coords(np.asarray(spheres)[:, 2:4, :2])            # [2, 2, 2]
    return course.intervals[-1, 1] - (sl[..., 0].max(-1) + FT.FOOT_RADIUS)


def episode_row(course, controller, ep, env, outcome, rules=None):
    """Per-episode metrics (spec 12.4) of a finished episode."""
    M = _M()
    blame = env.slot.event_contact
    pr = ep.progress
    row = dict(level=course.level, seed=int(course.seed), variant=(course.program or {}).get("variant"),
               end=course.end, cliff=bool(course.cliff), n_targets=int(course.n), n_planks=int(course.n_planks),
               half_width=float(course.half_width), controller=getattr(controller, "name", type(controller).__name__),
               end_rule=ep.end_rule, outcome=outcome,
               fell=bool(outcome in M.EVENTS[1:] or outcome == "fall_flag"), self_contact=outcome == "self_contact",
               event_body=blame[0] if blame else None, event_other=blame[1] if blame else None,
               end_s=round(env.time, 2), policy_steps=max(0, ep.k - (ep.hand_over_step or ep.k)),
               hand_over_step=ep.hand_over_step, host=socket.gethostname())
    if pr is not None:
        s = pr.metrics()["summary"]
        keep = ("completed", "finished", "fail", "stop_success", "stop_step", "err_p50", "err_p95", "err_max",
                "along_abs_p95", "across_abs_p95", "max_abs_l", "scored", "resteps", "resteps_free", "wrong_foot",
                "far_miss", "flickers", "overruns", "hit3", "hit5", "hit8", "hits3", "hits5", "hits8", "success3",
                "success5", "success8")
        row.update({k: s[k] for k in keep})
        row.update(pointer=int(pr.ptr), max_abs_dpsi_deg=math.degrees(s["max_abs_dpsi"]),
                   mean_abs_dpsi_deg=math.degrees(s["mean_abs_dpsi"]) if s["mean_abs_dpsi"] == s["mean_abs_dpsi"] else None,
                   scorer_touchdowns=s["touchdowns"], scorer_touchdowns_over_gap=s["touchdowns_over_gap"],
                   scorer_unsupported_steps=s["unsupported_steps"])
        for x in (3, 5, 8):
            row[f"hit{x}_all"] = s[f"hits{x}"] / course.n
    else:
        row.update(completed=False, finished=False, fail=None, scored=0, pointer=0)
    if ep.planks is not None:
        m = ep.planks.metrics()
        row.update(touchdowns=m["touchdowns"], touchdowns_over_gap=m["touchdowns_over_gap"],
                   unsupported_steps=m["unsupported_steps"], planks_reached=m["reached"],
                   planks_crossed=m["planks_crossed"])
        clean = m["touchdowns_over_gap"] == 0 and m["unsupported_steps"] == 0
    else:
        clean = True
    if ep.end_rule == "planks":
        row["score_completed"] = row["completed"]
        row["completed"] = outcome == "success"
        for x in (3, 5, 8):
            row[f"success{x}"] = False          # no footstep input: course success is not defined
    elif row["fell"] and row["completed"]:
        # the scorer completed in the step of a fall: the fall wins (run_course_episode), so no completion either
        row["score_completed"] = True
        row["completed"] = False
        for x in (3, 5, 8):
            row[f"success{x}"] = False
    row["planks_success"] = outcome == "success" if ep.end_rule == "planks" else None
    row["clean"] = bool(clean)
    row["clean_completion"] = bool(clean and row["completed"])
    if course.is_planks and course.cliff and ep.avg_states:
        g = toe_gap(course, env.spheres())
        row["toe_gap_m"] = [round(float(v), 4) for v in g]
    tr = getattr(controller, "trace", None)
    if tr and tr.get("cmd"):
        cmd = np.asarray(tr["cmd"], np.float64)
        row.update(cmd_mean=cmd.mean(0).round(4).tolist(), cmd_abs_max=np.abs(cmd).max(0).round(4).tolist())
        if tr.get("a") and len(tr["a"][0]) > 3:
            a = np.asarray(tr["a"], np.float64)
            amap = controller.amap
            u = np.tanh(np.clip(a[:, 3:], -amap.clip, amap.clip))
            row.update(residual_abs_mean_rad=float(amap.alpha * np.abs(u).mean()),
                       residual_abs_max_rad=float(amap.alpha * np.abs(u).max()),
                       residual_saturation=float((np.abs(u) > 0.95).mean()))
            u3 = np.tanh(np.clip(a[:, :3], -amap.clip, amap.clip))
            row["command_saturation"] = float((np.abs(u3) > 0.95).mean())
    row.update(nrd_rule_diagnostics(course, ep, rules if rules is not None else getattr(controller, "rules", None)))
    return row


def nrd_rule_diagnostics(course, ep, rules=None):
    """M4: the NRD env's leg and support rules on the stored averaged states of the policy phase (newest and previous
    state of each step): first control step where each would end the NRD episode, and the smallest swept distances."""
    from nedm.g1_footstep_rl import fk
    rl = dict(RULE_DEFAULTS, **(rules or {}))
    k0 = ep.hand_over_step
    out = dict(rule_legs_step=None, rule_support_step=None, rule_legs_min=None, rule_pelvis_min=None)
    if k0 is None or len(ep.avg_states) <= k0:
        return out
    X = np.asarray(ep.avg_states[k0 - 1:] if k0 >= 1 else ep.avg_states, np.float64)
    if len(X) < 2:
        return out
    prev, cur = torch.as_tensor(X[:-1]), torch.as_tensor(X[1:])
    steps = np.arange(k0, k0 + len(cur))
    with torch.inference_mode():
        n_b, ahead, n_a = rl["legs_sweep"]
        d = fk.swept_leg_distances(prev, cur, int(n_b), float(ahead), int(n_a))
        legs, pel = d["legs"].numpy(), d["pelvis"].numpy()
        sph = fk.foot_points(torch.as_tensor(X))["spheres"].numpy()          # [T, 2, 4, 3]
    fire = np.zeros(len(legs), bool)                     # a threshold of None switches that part off (as the env)
    if rl.get("legs_min_dist") is not None:
        fire |= legs < float(rl["legs_min_dist"])
    if rl.get("legs_pelvis_min_dist") is not None:
        fire |= pel < float(rl["legs_pelvis_min_dist"])
    out.update(rule_legs_min=float(legs.min()), rule_pelvis_min=float(pel.min()),
               rule_legs_step=int(steps[np.argmax(fire)]) if fire.any() else None)
    if course.is_planks:
        n_off = (course.support_index(sph[..., :2], margin=float(rl["edge_margin"])) < 0).sum(-1)    # [T, 2]
        bad = n_off >= int(rl["support_min_off"])
        gates = np.asarray(ep.gates[k0:k0 + len(cur)], bool)
        b = bad[1:] & gates & (bad[:-1] if rl.get("two_state", True) else True)
        out["rule_support_step"] = int(steps[np.argmax(b.any(-1))]) if b.any() else None
    return out


# ---- policies --------------------------------------------------------------------------------------------------
class FootstepPolicy:
    """A footstep policy: deterministic actor (obs [B, n_obs] float32 -> mean action [B, n]), its env config (merged
    by footsteps_env.merge_cfg, local data paths), the planks env action map, the frozen HOMIE network."""

    def __init__(self, actor, env_cfg, name, meta=None):
        from nedm.g1_footstep_rl.homie import HomieNet
        from nedm.g1_footstep_rl.mujoco_policy import ActionMap
        self.actor, self.env_cfg, self.name, self.meta = actor, env_cfg, name, dict(meta or {})
        self.amap = ActionMap({"action": env_cfg["action"]})
        self.num_actions = self.amap.num_actions
        self.net = HomieNet.load(env_cfg["homie_weights"], "cpu")


def local_footsteps_cfg(env_cfg, manifest=None, map_paths=True):
    """footsteps_env.merge_cfg(env_cfg) with the NRD checkpoints and the HOMIE weights mapped to this machine
    (mujoco_policy.resolve; sha256 checked against the manifest when it lists them). Course banks are not needed."""
    from nedm.g1_footstep_rl.mujoco_policy import resolve
    ec = footsteps_env().merge_cfg(env_cfg)
    if not map_paths:
        return ec
    man = manifest or {}
    shas = man.get("checkpoints") or {}
    ec["checkpoints"] = [resolve(p, shas.get(p), "checkpoint") for p in ec["checkpoints"]]
    hw = man.get("homie_weights") or {}
    ec["homie_weights"] = resolve(ec["homie_weights"], hw.get("sha256") if isinstance(hw, dict) else None, "homie")
    return ec


def n_obs_of(env_cfg):
    return len(footsteps_env().obs_names(int(env_cfg["obs"]["n_targets"])))


def load_footstep_policy(ckpt, map_paths=True, zero_residual=False):
    """An RSL-RL checkpoint of train_footsteps_ppo.py (env_cfg.json, train_cfg.json next to it): ActorCritic rebuilt
    from the train config, actor mean = the action (observation normaliser applied if the run had one).
    zero_residual: the 12 leg-correction actions set to 0 (spec 13 "FS correction zeroed")."""
    from rsl_rl.modules import ActorCritic
    from nedm.g1_footstep_rl.mujoco_policy import run_files, sha256
    ec, tc, man = run_files(ckpt)
    ec = local_footsteps_cfg(ec, man, map_paths)
    n_act = 3 if float(ec["action"]["alpha"]) == 0 else 15
    n_obs = n_obs_of(ec)
    pc = {k: v for k, v in tc["policy"].items() if k != "class_name"}
    if tc["policy"].get("class_name", "ActorCritic") != "ActorCritic":
        raise ValueError(f"{ckpt}: policy class {tc['policy']['class_name']} (only ActorCritic)")
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = d["model_state_dict"]
    if tuple(sd["actor.0.weight"].shape)[1] != n_obs:
        raise ValueError(f"{ckpt}: actor input {tuple(sd['actor.0.weight'].shape)}, the env config gives {n_obs}")
    with contextlib.redirect_stdout(io.StringIO()):          # rsl_rl prints the network
        ac = ActorCritic(n_obs, n_obs, n_act, **pc)
    ac.load_state_dict(sd, strict=True)
    ac.eval()
    norm = None
    if tc.get("empirical_normalization"):
        from rsl_rl.modules import EmpiricalNormalization
        norm = EmpiricalNormalization(shape=[n_obs], until=1.0e8)
        norm.load_state_dict(d["obs_norm_state_dict"])
        norm.eval()

    @torch.inference_mode()
    def actor(obs):
        a = ac.act_inference(norm(obs) if norm is not None else obs)
        if zero_residual:
            a = a.clone()
            a[:, 3:] = 0.0
        return a

    name = Path(ckpt).parent.name + "/" + Path(ckpt).name + (" | correction zeroed" if zero_residual else "")
    meta = dict(kind="footsteps", path=str(Path(ckpt).resolve()), iteration=d.get("iter"), sha256=sha256(ckpt),
                num_actions=n_act, alpha=float(ec["action"]["alpha"]), policy=pc, zero_residual=bool(zero_residual),
                empirical_normalization=bool(tc.get("empirical_normalization")))
    return FootstepPolicy(actor, ec, name, meta)


# ---- controllers -----------------------------------------------------------------------------------------------
class FootstepRunner:
    """FS policy in MuJoCo (contract (1)-(13) of the module docstring). After the episode: .snapshot (hand-over state
    for G1FootstepsEnv.reset_from_snapshots), .trace (per policy step: action a, command, gate, step), .records
    (record=True: every input of each observation, for parity checks). noise = (command std, correction std) adds
    Gaussian noise to the pre-tanh actions (data-round rollouts, spec 16 step 1: noise at the std floor), drawn from
    np.random.default_rng(noise_seed); None (default) = the deterministic mean action."""
    scene_teacher, scene_avg_rates, end_rule = None, True, "footsteps"

    def __init__(self, policy, record=False, noise=None, noise_seed=0):
        from nedm.g1_footstep_rl.homie import HomieTeacher
        self.noise = None
        if noise is not None:
            n = int(policy.num_actions)
            self.noise = np.array([float(noise[0])] * 3 + [float(noise[1])] * (n - 3), np.float64)
            self.rng = np.random.default_rng(noise_seed)
        self.pol, self.record = policy, bool(record)
        self.name, self.descriptor = policy.name, FS_DESCRIPTOR
        ec = policy.env_cfg
        self.score_source = ec
        self.rules = {k: ec["rules"][k] for k in RULE_DEFAULTS if k in ec.get("rules", {})}
        self.obs_cfg = ec["obs"]
        self.amap = policy.amap
        self.teacher = HomieTeacher(1, "cpu", net=policy.net)
        self.stand_cmd = torch.tensor([STAND_CMD], dtype=F64)
        self.k = 0
        self.started = False
        self.hist = deque(maxlen=4)
        self.snapshot, self.records = None, []
        self.trace = dict(a=[], cmd=[], gate=[], step=[])

    def observe(self, st, preview, gate, ep):
        """footsteps_env.footsteps_observation (11) of the MuJoCo state with the views of the episode's course and
        progress (footsteps_env.views_from_progress: slack with rules.edge_margin and obs.slack_clip)."""
        FE = footsteps_env()
        ec = self.pol.env_cfg
        tv, cv = FE.views_from_progress(ep.course_h, ep.progress, float(ec["rules"]["edge_margin"]),
                                        tuple(ec["obs"]["slack_clip"]))
        return FE.footsteps_observation(st, preview, self.prev_u, gate, tv, cv, self.obs_cfg)

    @torch.inference_mode()
    def __call__(self, env, ep):
        x = env.avg_state()                                # (2) averaged rates, never env.state()
        st = torch.as_tensor(x, dtype=F64)[None]
        self.hist.append(st[0].clone())
        amap = self.amap
        k, self.k = self.k, self.k + 1
        if k == 0 and not np.array_equal(env.default_upper(), amap.upper.numpy()):
            raise ValueError("the MuJoCo plan's upper body targets are not the default the NRD env uses")
        if env.time < STAND_S - 1e-9:                      # (3) the standing second
            raw, legs = self.teacher.step(self.stand_cmd, st)
            target = torch.cat((legs, amap.upper[None]), 1)
            return target[0].numpy(), raw[0].numpy(), self.stand_cmd[0].numpy()
        if not self.started:                               # (4) hand-over
            self.started = True
            self.prev_cmd = self.stand_cmd.clone()
            self.prev_u = amap.squash(amap.action_of_command(torch.zeros(1, 3, dtype=F64)))
        gate = torch.as_tensor(np.asarray(ep.last_gate, bool), dtype=F64)[None]     # (5) label of the last step
        mem = (self.teacher.history.clone(), self.teacher.last_action.clone())
        from nedm.g1_footstep_rl.planks_env import homie_preview
        preview = homie_preview(self.teacher, self.pol.net, self.prev_cmd, st, self.obs_cfg["homie_preview"])   # (6)
        obs = self.observe(st, preview, gate, ep)          # (11)
        if self.snapshot is None:                          # (13)
            self.snapshot = dict(step=k, hist=torch.stack(list(self.hist)).numpy(), teacher_history=mem[0][0].numpy(),
                                 teacher_last_action=mem[1][0].numpy(), prev_cmd=self.prev_cmd[0].numpy(),
                                 prev_u=self.prev_u[0].numpy(), gate=gate[0].numpy(), obs=obs[0].numpy(),
                                 course=ep.course_h.to_dict(), progress=ep.progress_snapshot(),
                                 level=ep.course_h.level, seed=int(ep.course_h.seed))
        a = self.pol.actor(obs)
        if self.noise is not None:                         # data-round rollouts only (spec 16 step 1)
            a = a + torch.as_tensor(self.rng.standard_normal(self.noise.shape) * self.noise, dtype=a.dtype)[None]
        u = amap.squash(a)
        cmd = amap.command(u)
        raw, legs = self.teacher.step(cmd, st)             # (7) once per step, new command
        target = amap.target(legs, u)
        if self.record:
            self.records.append(dict(step=k, state=st[0].numpy(), teacher_history=mem[0][0].numpy(),
                                     teacher_last_action=mem[1][0].numpy(), prev_cmd=self.prev_cmd[0].numpy(),
                                     prev_u=self.prev_u[0].numpy(), gate=gate[0].numpy(), preview=preview[0].numpy(),
                                     obs=obs[0].numpy(), a=a[0].numpy(), target=target[0].numpy(),
                                     hist=torch.stack(list(self.hist)).numpy(),
                                     progress=ep.progress_snapshot(), pointer=int(ep.progress.ptr)))
        self.trace["a"].append(a[0].numpy().astype(np.float32))
        self.trace["cmd"].append(cmd[0, :3].numpy().astype(np.float32))
        self.trace["gate"].append(gate[0].numpy().astype(np.uint8))
        self.trace["step"].append(k)
        self.prev_u, self.prev_cmd = u, cmd
        return target[0].numpy(), raw[0].numpy(), cmd[0].numpy()


def hfs_speed(L, g_v=1.0, table=HFS_TABLE):
    """H-FS forward command for a desired step length L* (M2): table interpolation, linear to 0 at L* <= 0.02 m below
    the table, the last segment's slope above it; x g_v, clipped to [0, 1.2]."""
    xs, ys = (np.asarray(v, np.float64) for v in table)
    L = float(L)
    if L <= 0.02:
        v = 0.0
    elif L < xs[0]:
        v = ys[0] * (L - 0.02) / (xs[0] - 0.02)
    elif L > xs[-1]:
        v = ys[-1] + (L - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    else:
        v = float(np.interp(L, xs, ys))
    return float(np.clip(g_v * v, 0.0, 1.2))


class HomieController:
    """Base of the HOMIE baselines: HOMIE's legs + default upper body under command(env, ep) -> (vx, vy, wz); zero
    command during the stand. teacher "ort" (native, instantaneous rates; M1), "ort_avg" or "numpy"."""
    end_rule = "footsteps"

    def __init__(self, teacher="ort"):
        self.scene_teacher = "numpy" if teacher == "numpy" else "ort"
        self.scene_avg_rates = teacher == "ort_avg"
        self.teacher_kind = teacher
        self.trace = dict(cmd=[], step=[])
        self.k = 0
        self.descriptor = "HOMIE teacher, no learning"

    def command(self, env, ep):
        raise NotImplementedError

    def __call__(self, env, ep):
        k, self.k = self.k, self.k + 1
        if env.time < STAND_S - 1e-9:
            cmd = np.array(STAND_CMD)
        else:
            cmd = np.array(tuple(self.command(env, ep)) + (FULL_HEIGHT,), np.float64)
            self.trace["cmd"].append(cmd[:3].astype(np.float32))
            self.trace["step"].append(k)
        raw, legs = env.teacher_action(cmd)
        return np.concatenate((legs, env.default_upper())), raw, cmd


class HfsController(HomieController):
    """H-FS (spec 13, M2): HOMIE + a footstep-to-speed controller, every control step after the stand."""

    def __init__(self, g_v=1.0, lookahead=2, k_y=1.0, k_psi=1.0, v_min=0.2, teacher="ort", L_fixed=None):
        super().__init__(teacher)
        self.L_fixed = None if L_fixed is None else float(L_fixed)      # checks: a constant L* (spec 17)
        self.g_v, self.lookahead, self.k_y, self.k_psi = float(g_v), int(lookahead), float(k_y), float(k_psi)
        self.v_min = float(v_min)
        self.params = dict(g_v=self.g_v, lookahead=self.lookahead, k_y=self.k_y, k_psi=self.k_psi, v_min=self.v_min,
                           teacher=teacher)
        self.name = f"H-FS g_v {self.g_v:g} look {self.lookahead} k_y {self.k_y:g} v_min {self.v_min:g}"
        self.end_rule = "footsteps"
        self.L_star = []

    def desired_length(self, ep):
        if self.L_fixed is not None:
            return self.L_fixed
        pr = ep.progress
        p, N = pr.ptr, pr.N
        if p >= N:
            return 0.0
        fp = int(pr.foot[p])
        if self.lookahead >= 2 and not ep.last_gate[fp]:
            return float(pr.L_plan[p + 1]) if p + 1 < N else 0.0
        c = pr.C[p]
        d = pr.T[p, :2] - pr.last_place[1 - fp, :2]
        return float(d[0] * math.cos(c[2]) + d[1] * math.sin(c[2]))

    def command(self, env, ep):
        pr = ep.progress
        x = env.avg_state()
        yaw = FT.quat_yaw(x[3:7])
        p, N = pr.ptr, pr.N
        stop = p >= N or (ep.course_h.end == "stop" and pr.T[min(p, N - 1), 4] > 0)
        L = self.desired_length(ep)
        self.L_star.append(L)
        vx = 0.0 if stop else max(self.v_min, hfs_speed(L, self.g_v))
        l, _ = pr.drift(x[:2], yaw)
        vy = float(np.clip(-self.k_y * l, -0.5, 0.5))
        psi = pr.C[min(p + 1, N - 1), 2]
        wz = float(np.clip(self.k_psi * FT.wrap(psi - yaw), -0.8, 0.8))
        return vx, vy, wz


# ---- policy specs (picklable, for worker processes) -----------------------------------------------------------
def load_spec(spec):
    """spec -> a loaded object for make_controller: {"kind": "footsteps", "path", "zero_residual"} or {"kind": "hfs",
    "params"}."""
    kind = spec["kind"]
    if kind == "footsteps":
        return load_footstep_policy(spec["path"], spec.get("map_paths", True), bool(spec.get("zero_residual")))
    if kind == "hfs":
        return None
    raise ValueError(f"unknown policy spec {spec}")


def make_controller(spec, loaded, course, record=False):
    """spec["noise"] (footsteps only): (command std, correction std) pre-tanh action noise, seeded per course by
    (seed, crc32(level)), for data-round rollouts."""
    kind = spec["kind"]
    if kind == "footsteps":
        if spec.get("noise") is None:
            return FootstepRunner(loaded, record=record)
        import zlib
        return FootstepRunner(loaded, record=record, noise=spec["noise"],
                              noise_seed=[int(course.seed) % 2 ** 32, zlib.crc32(str(course.level).encode())])
    if kind == "hfs":
        return HfsController(**spec.get("params", {}))
    raise ValueError(f"unknown policy spec {spec}")


def spec_name(spec):
    k = spec["kind"]
    if k == "footsteps":
        return Path(spec["path"]).parent.name + ("_zero_residual" if spec.get("zero_residual") else "")
    if k == "hfs":
        p = spec.get("params", {})
        return f"HFS_g{p.get('g_v', 1.0):g}_la{p.get('lookahead', 2)}_ky{p.get('k_y', 1.0):g}_vmin{p.get('v_min', 0.2):g}"
    raise ValueError(f"unknown policy spec {spec}")


_W = {}


def _worker_init(spec, threads=1, gl=None):
    import os
    if gl:
        os.environ["MUJOCO_GL"] = gl
    torch.set_num_threads(int(threads))
    _W["spec"], _W["loaded"] = spec, load_spec(spec)


def run_spec_episode(spec, loaded, level, seed, keep_states=False, record=False):
    """One episode of a policy spec on the course of (level, seed): (row, ep, env, controller), or None if the old
    layout has no plan. spec["score"]: score overrides (e.g. strict false for S0-S1 checkpoints)."""
    course, _ = course_for(level, seed)
    if course is None:
        return None
    ctrl = make_controller(spec, loaded, course, record=record)
    score = None
    if spec.get("score"):
        score = dict(FT.score_cfg(getattr(ctrl, "score_source", None)), **spec["score"])
    row, ep, env = run_course_episode(course, ctrl, score=score, keep_states=keep_states)
    return row, ep, env, ctrl


def _worker_episode(job):
    """job: dict(level, seed, episode_id, keep_states, video (path) or None, plot (path) or None, video_opts) ->
    (row, snapshot, step_rows)."""
    spec, loaded = _W["spec"], _W["loaded"]
    keep = bool(job.get("video") or job.get("keep_states"))
    out = run_spec_episode(spec, loaded, job["level"], job["seed"], keep_states=keep)
    if out is None:
        return dict(level=job["level"], seed=int(job["seed"]), outcome="no_plan", completed=False, clean=False,
                    clean_completion=False, episode_id=job.get("episode_id")), None, []
    row, ep, env, ctrl = out
    row.update(episode_id=job.get("episode_id"), policy=spec_name(spec))
    steps = []
    if ep.progress is not None:
        for r in ep.progress.metrics()["rows"]:
            r = dict(r, level=row["level"], seed=row["seed"])
            r.pop("place", None)
            steps.append(r)
    if job.get("plot"):
        where = f"{row['level']} seed {row['seed']}" if row["seed"] >= 0 else f"showcase {row['level']}"
        plot_course(ep, row, job["plot"], title=f"{row['controller']} | {where}")
    if job.get("video"):
        render_episode(env.model, row["states"], ep, row, job["video"], policy_name=row["controller"],
                       descriptor=getattr(ctrl, "descriptor", ""), **job.get("video_opts", {}))
    row.pop("states", None)
    snap = getattr(ctrl, "snapshot", None)
    return row, snap, steps


def run_pool(spec, jobs, workers=8, threads=1, gl=None):
    """Run course episodes in `workers` spawned processes -> (rows, snapshots, step_rows) in job order."""
    import multiprocessing as mp
    if workers <= 1:
        _worker_init(spec, threads, gl)
        out = [_worker_episode(j) for j in jobs]
    else:
        with mp.get_context("spawn").Pool(workers, initializer=_worker_init, initargs=(spec, threads, gl)) as pool:
            out = pool.map(_worker_episode, jobs, chunksize=1)
    return [o[0] for o in out], [o[1] for o in out], [s for o in out for s in o[2]]


# ---- statistics ------------------------------------------------------------------------------------------------
def _wilson(k, n):
    from nedm.g1_footstep_rl.mujoco_policy import wilson
    return wilson(k, n)


def cluster_bootstrap(groups, fn, n_boot=1000, seed=0):
    """95 % interval of fn(concatenated values) under resampling of the groups (episodes) with replacement."""
    groups = [np.asarray(g, np.float64) for g in groups]
    if not groups or not sum(len(g) for g in groups):
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        cat = np.concatenate([groups[i] for i in pick])
        if len(cat):
            vals.append(fn(cat))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


RATE_KEYS = ("completed", "clean_completion", "success3", "success5", "success8", "fell", "self_contact")


def cell_summary(rows, step_rows=(), n_boot=1000):
    """Summary of one evaluation cell (spec 12.4, M6): episode rates with Wilson 95 %, per-step hit rates and error
    percentiles with episode-cluster bootstrap 95 %, outcomes, drift, commands."""
    rows = [r for r in rows if r.get("outcome") != "no_plan"]
    n = len(rows)
    out = dict(n=n, no_plan=None)
    if not n:
        return out
    for k in RATE_KEYS:
        c = int(sum(bool(r.get(k)) for r in rows))
        out[k] = dict(k=c, rate=c / n, ci=_wilson(c, n))
    keyed = {(r["level"], r["seed"]) for r in rows}
    by_ep = {}
    for s in step_rows:
        if (s["level"], s["seed"]) in keyed:
            by_ep.setdefault((s["level"], s["seed"]), []).append(s)
    groups = [[s["e"] for s in by_ep.get((r["level"], r["seed"]), [])] for r in rows]
    e_all = np.concatenate([np.asarray(g, np.float64) for g in groups]) if groups else np.zeros(0)
    out["scored_steps"] = int(len(e_all))
    out["planned_targets"] = int(sum(r["n_targets"] for r in rows))
    for x in (3, 5, 8):
        thr = x / 100
        hits = int((e_all <= thr).sum())
        out[f"hit{x}"] = dict(k=hits, n=len(e_all), rate=hits / len(e_all) if len(e_all) else math.nan,
                              ci=cluster_bootstrap(groups, lambda a: float((a <= thr).mean()), n_boot))
        nt = [r["n_targets"] for r in rows]
        hg = [[float((np.asarray(g) <= thr).sum()), t] for g, t in zip(groups, nt)]
        out[f"hit{x}_all"] = dict(k=hits, n=out["planned_targets"], rate=hits / max(1, out["planned_targets"]),
                                  ci=cluster_bootstrap([np.asarray([h, t]) for h, t in hg],
                                                       lambda a: float(a[0::2].sum() / a[1::2].sum()), n_boot))
    for q in (50, 95):
        out[f"err_p{q}"] = dict(value=float(np.percentile(e_all, q)) if len(e_all) else math.nan,
                                ci=cluster_bootstrap(groups, lambda a, q=q: float(np.percentile(a, q)), n_boot))
    out["err_max"] = float(e_all.max()) if len(e_all) else math.nan
    for key in ("along", "across"):
        a = np.abs([s[key] for g in by_ep.values() for s in g])
        out[f"{key}_abs_p95"] = float(np.percentile(a, 95)) if len(a) else math.nan
    oc = {}
    for r in rows:
        oc[r["outcome"]] = oc.get(r["outcome"], 0) + 1
    out["outcomes"] = dict(sorted(oc.items(), key=lambda kv: -kv[1]))
    bodies = {}
    for r in rows:
        if r.get("fell"):
            b = r.get("event_body") or r["outcome"]
            bodies[b] = bodies.get(b, 0) + 1
    out["event_bodies"] = bodies
    num = lambda k: [r[k] for r in rows if r.get(k) is not None]
    for k in ("max_abs_l", "max_abs_dpsi_deg", "mean_abs_dpsi_deg"):
        v = np.asarray(num(k), np.float64)
        out[k] = dict(p50=float(np.median(v)), p95=float(np.percentile(v, 95)), max=float(v.max())) if len(v) else None
    for k in ("resteps", "wrong_foot", "flickers", "far_miss", "touchdowns_over_gap", "unsupported_steps"):
        v = num(k)
        out[f"{k}_mean"] = float(np.mean(v)) if v else None
    for k in ("residual_abs_mean_rad", "residual_saturation", "command_saturation", "residual_abs_max_rad"):
        v = num(k)
        out[k] = float(np.mean(v)) if v and k != "residual_abs_max_rad" else (float(np.max(v)) if v else None)
    cm = [r["cmd_mean"] for r in rows if r.get("cmd_mean")]
    out["cmd_mean"] = np.mean(cm, 0).round(4).tolist() if cm else None
    tg = [min(r["toe_gap_m"]) for r in rows if r.get("toe_gap_m") and r.get("completed")]
    out["toe_gap_min_completed"] = dict(p50=float(np.median(tg)), min=float(np.min(tg)), max=float(np.max(tg)),
                                        n=len(tg)) if tg else None
    out["rule_legs_fired"] = int(sum(r.get("rule_legs_step") is not None for r in rows))
    out["rule_support_fired"] = int(sum(r.get("rule_support_step") is not None for r in rows))
    out["end_s_p50"] = float(np.median([r["end_s"] for r in rows]))
    var = {}
    for r in rows:
        if r.get("variant"):
            var[r["variant"]] = var.get(r["variant"], 0) + 1
    out["planner_variants"] = var or None
    return out


def paired_mcnemar(rows, base_rows, key="completed"):
    """Paired exact McNemar of `key` on the (level, seed) pairs present in both row sets."""
    from nedm.g1_footstep_rl.mujoco_policy import mcnemar_exact
    base = {(r["level"], r["seed"]): r for r in base_rows if r.get("outcome") != "no_plan"}
    pr = [(bool(r.get(key)), bool(base[(r["level"], r["seed"])].get(key))) for r in rows
          if (r["level"], r["seed"]) in base and r.get("outcome") != "no_plan"]
    if not pr:
        return None
    both = sum(a and b for a, b in pr)
    pol = sum(a and not b for a, b in pr)
    bas = sum(b and not a for a, b in pr)
    return dict(n=len(pr), policy=sum(a for a, _ in pr), baseline=sum(b for _, b in pr), both=both,
                policy_only=pol, baseline_only=bas, neither=len(pr) - both - pol - bas, mcnemar_p=mcnemar_exact(pol, bas))


# ---- visible evaluation (spec 12.6; render time only, M7) -------------------------------------------------------
SIDE_RGBA = {1: (0.45, 0.72, 1.00, 1.0), -1: (1.00, 0.45, 0.00, 1.0)}         # left light blue, right orange
UNDERLAY_RGBA = (0.02, 0.02, 0.03, 1.0)   # dark rim under the outlines (orange outlines on orange planks)
ERR_CLASSES = ((0.03, (0.15, 0.75, 0.25)), (0.05, (0.62, 0.85, 0.20)), (0.08, (1.00, 0.70, 0.10)),
               (math.inf, (0.90, 0.15, 0.15)))                                 # green, yellow-green, amber, red
PLANK_SHORT, PLANK_LONG, PLANK_RANGE = (0.90, 0.55, 0.20), (0.25, 0.45, 0.80), (0.2, 0.8)
PLATFORM_RGBA = (0.55, 0.56, 0.58, 1.0)
PIT_RGBA = (0.08, 0.08, 0.10, 1.0)
SOLE_HALF = (0.09, 0.035)                 # sole outline 0.18 x 0.07 m
BAR_HALF_W, BAR_HALF_H, MARK_Z = 0.003, 0.001, 0.001
CAMERAS = dict(chase=dict(distance=3.2, elevation=-18.0, az_offset=90.0, lookat_z=0.45),
               top=dict(distance=3.5, elevation=-89.0, az_offset=90.0, lookat_z=0.0))


def err_rgb(e):
    for thr, rgb in ERR_CLASSES:
        if e <= thr:
            return rgb
    return ERR_CLASSES[-1][1]


def plank_rgba(length):
    t = float(np.clip((length - PLANK_RANGE[0]) / (PLANK_RANGE[1] - PLANK_RANGE[0]), 0.0, 1.0))
    return tuple((1 - t) * a + t * b for a, b in zip(PLANK_SHORT, PLANK_LONG)) + (1.0,)


def render_model(model, course=None, top=False, size=(960, 544)):
    """A render-only deepcopy of a physics model (M7): planks coloured by length on one sequential scale (0.2 m orange
    -> 0.8 m blue), platforms grey, pit floor near black (plank courses); top=True dims the headlight (diffuse 0.45,
    ambient 0.35, specular 0); offscreen buffer >= size."""
    import mujoco
    m = copy.deepcopy(model)
    if course is not None and course.is_planks:
        n = len(course.intervals)
        for k in range(n):
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"plank_{k:02d}")
            if g < 0:
                continue
            a, b = course.intervals[k]
            m.geom_rgba[g] = PLATFORM_RGBA if k in (0, n - 1) else plank_rgba(b - a)
        f = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        m.geom_matid[f] = -1
        m.geom_rgba[f] = PIT_RGBA
    if top:
        m.vis.headlight.diffuse[:] = 0.45
        m.vis.headlight.ambient[:] = 0.35
        m.vis.headlight.specular[:] = 0.0
    vis = m.vis.global_
    vis.offwidth, vis.offheight = max(vis.offwidth, size[0]), max(vis.offheight, size[1])
    return m


def _add_geom(scn, kind, size, pos, yaw, rgba):
    import mujoco
    if scn.ngeom >= scn.maxgeom:
        return
    c, s = math.cos(yaw), math.sin(yaw)
    mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64).ravel()
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], kind, np.asarray(size, np.float64), np.asarray(pos, np.float64), mat,
                        np.asarray(rgba, np.float32))
    scn.ngeom += 1


def add_sole(scn, xy, yaw, rgba, filled=False):
    """Sole marker 0.18 x 0.07 m at foot yaw, 1 mm above z = 0: filled box or an outline of 4 bars 6 mm wide, 2 mm
    high."""
    import mujoco
    box = mujoco.mjtGeom.mjGEOM_BOX
    z = MARK_Z + BAR_HALF_H
    p = np.array([xy[0], xy[1], z])
    if filled:
        _add_geom(scn, box, (SOLE_HALF[0], SOLE_HALF[1], BAR_HALF_H), p, yaw, rgba)
        return
    c, s = math.cos(yaw), math.sin(yaw)
    fwd, left = np.array([c, s, 0.0]), np.array([-s, c, 0.0])
    hl, hw = SOLE_HALF
    for w, h, z, col in ((BAR_HALF_W + 0.0025, 0.5 * BAR_HALF_H, -0.5 * BAR_HALF_H, UNDERLAY_RGBA),
                         (BAR_HALF_W, BAR_HALF_H, 0.0, rgba)):
        for (a, b), half in (((hl - BAR_HALF_W, 0), (w, hw + w - BAR_HALF_W, h)),
                             ((-hl + BAR_HALF_W, 0), (w, hw + w - BAR_HALF_W, h)),
                             ((0, hw - BAR_HALF_W), (hl + w - BAR_HALF_W, w, h)),
                             ((0, -hw + BAR_HALF_W), (hl + w - BAR_HALF_W, w, h))):
            _add_geom(scn, box, half, p + a * fwd + b * left + np.array([0, 0, z]), yaw, col)


def add_stop_bar(scn, xy, yaw):
    """White bar across the toes of a stop target."""
    import mujoco
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.array([xy[0] + (SOLE_HALF[0] + 0.012) * c, xy[1] + (SOLE_HALF[0] + 0.012) * s, MARK_Z + 1.5 * BAR_HALF_H])
    _add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, (0.005, SOLE_HALF[1] + 0.01, 1.5 * BAR_HALF_H), p, yaw, (1, 1, 1, 1))


def draw_markers(scn, course, pointer, scored, touchdowns):
    """Footprint markers (spec 12.6): future targets outlined (left light blue, right orange), the next target filled
    at alpha 0.5, scored targets filled by error class, touchdowns as white spheres (r 0.012 m), stop targets with a
    white bar across the toes. scored {target: error m}; touchdowns [(x, y)]."""
    import mujoco
    T = course.targets
    for i in range(course.n):
        xy, yaw, side = T[i, :2], T[i, 2], int(np.sign(T[i, 3]))
        if i in scored:
            add_sole(scn, xy, yaw, err_rgb(scored[i]) + (0.9,), filled=True)
        elif i == pointer:
            add_sole(scn, xy, yaw, SIDE_RGBA[side][:3] + (0.5,), filled=True)
        else:
            add_sole(scn, xy, yaw, SIDE_RGBA[side])
        if T[i, 4] > 0:
            add_stop_bar(scn, xy, yaw)
    for x, y in touchdowns:
        _add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, (0.012, 0.012, 0.012), (x, y, 0.012), 0.0, (1, 1, 1, 1))


class EpisodeView:
    """What the overlay shows at each 10 ms record of an episode: pointer, scored targets with errors, touchdown
    placements, last error, hits within 5 cm, drift (l, heading error). Events processed at control step k are shown
    from the end of step k on (record 2 (k + 1))."""

    def __init__(self, ep, record_s=0.01):
        self.ep, self.course = ep, ep.course_h if ep.course_h is not None else ep.course
        self.per_ctrl = int(round(CONTROL_S / record_s))
        self.trace = {k: (p, l, d) for k, p, l, d in ep.trace}

    def at(self, r):
        kc = r // self.per_ctrl - 1
        scored, tds, last = {}, [], None
        for td in self.ep.log:
            if td["proc"] > kc:
                break
            tds.append((td["x"], td["y"]))
            if td["kind"] == "scored":
                scored[td["target"]] = td["err"]
                last = td["err"]
        ks = [k for k in self.trace if k <= kc]
        p, l, d = self.trace[max(ks)] if ks else (0, None, None)
        hits5 = sum(e <= 0.05 for e in scored.values())
        return dict(pointer=p, scored=scored, touchdowns=tds, last_err=last, hits5=hits5, n_scored=len(scored), l=l,
                    dpsi=d, t=r * CONTROL_S / self.per_ctrl)


class CourseRenderer:
    """Offscreen renderer of one course: per camera a render-only model copy (render_model) and its own MjData;
    frame(state, view) -> [H, W x n_cameras, 3] uint8 with markers; the physics model and data are never touched."""

    def __init__(self, model, course, cameras=("chase", "top"), size=(960, 544), max_geom=8000):
        import mujoco
        self.course, self.size, self.names = course, tuple(size), tuple(cameras)
        self.models, self.datas, self.rends, self.cams = [], [], [], []
        for name in self.names:
            m = render_model(model, course, top=name == "top", size=self.size)
            self.models.append(m)
            self.datas.append(mujoco.MjData(m))
            self.rends.append(mujoco.Renderer(m, height=self.size[1], width=self.size[0], max_geom=max_geom))
            spec = CAMERAS[name]
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance, cam.elevation = spec["distance"], spec["elevation"]
            cam.azimuth = math.degrees(course.heading) + spec["az_offset"]
            self.cams.append(cam)
        self.ghost_data = [mujoco.MjData(m) for m in self.models]

    def frame(self, state, view=None, ghost=None):
        import mujoco
        from nedm.g1_footstep_rl import g1_scene as G
        imgs = []
        for i, name in enumerate(self.names):
            m, d, r, cam = self.models[i], self.datas[i], self.rends[i], self.cams[i]
            d.qpos[:], d.qvel[:] = G.state_to_qpos_qvel(state)
            mujoco.mj_forward(m, d)
            cam.lookat[:] = [d.qpos[0], d.qpos[1], CAMERAS[name]["lookat_z"]]
            r.update_scene(d, cam)
            scn = r.scene
            if view is not None:
                draw_markers(scn, self.course, view["pointer"], view["scored"], view["touchdowns"])
            if ghost is not None:
                g = self.ghost_data[i]
                g.qpos[:], g.qvel[:] = G.state_to_qpos_qvel(ghost)
                mujoco.mj_forward(m, g)
                first = scn.ngeom
                opt = mujoco.MjvOption()
                opt.geomgroup[:] = 0
                opt.geomgroup[1] = opt.geomgroup[2] = 1
                mujoco.mjv_addGeoms(m, g, opt, mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
                for j in range(first, scn.ngeom):
                    scn.geoms[j].rgba[:] = (0.3, 0.6, 1.0, 0.35)
            if name == "top":
                for j in range(scn.ngeom):
                    scn.geoms[j].specular = 0.0
            imgs.append(r.render())
        return np.concatenate(imgs, 1)

    def close(self):
        for r in self.rends:
            r.close()


def _font(size=20):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_hud(img, lines, banner=None, banner_rgb=(255, 255, 255)):
    """HUD (spec 12.6): black band with text lines (each a list of (text, rgb) segments, PIL default font 20) and an
    optional end banner (outcome) at the bottom centre."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    f = _font(20)
    dr.rectangle((0, 0, im.width, 26 * len(lines) + 8), fill=(0, 0, 0))
    for i, segs in enumerate(lines):
        x = 10
        for text, rgb in segs:
            dr.text((x, 6 + 26 * i), text, fill=tuple(rgb), font=f)
            x += dr.textlength(text, font=f)
    if banner:
        fb = _font(34)
        w = dr.textlength(banner, font=fb)
        y0 = im.height - 60
        dr.rectangle(((im.width - w) / 2 - 16, y0 - 8, (im.width + w) / 2 + 16, y0 + 44), fill=(0, 0, 0))
        dr.text(((im.width - w) / 2, y0), banner, fill=tuple(banner_rgb), font=fb)
    return np.asarray(im)


def outcome_banner(row):
    o = row.get("outcome") or ""
    if row.get("completed"):
        return ("COMPLETED" + (" (clean)" if row.get("clean") and row.get("n_planks") else "")
                + (" - stopped" if row.get("stop_success") else "")), (60, 220, 90)
    if o.startswith("fail_"):
        return f"FAILED: {o[5:]}", (235, 70, 70)
    if row.get("fell"):
        return f"FELL: {o}", (235, 70, 70)
    return o.upper(), (240, 200, 60)


def hud_lines(view, ep, row, policy_name, descriptor):
    c = ep.course_h if ep.course_h is not None else ep.course
    white, grey = (255, 255, 255), (200, 200, 200)
    l1 = [(policy_name, white)] + ([(f"  |  {descriptor}", grey)] if descriptor else [])
    where = f"{c.level}  seed {c.seed}" if c.seed >= 0 else f"showcase {c.level}"
    l2 = [(f"{where}  |  t = {view['t']:5.2f} s  |  step {min(view['pointer'], c.n)} / {c.n}", white)]
    if view["last_err"] is None:
        l3 = [("last error  -", white)]
    else:
        rgb = tuple(int(255 * v) for v in err_rgb(view["last_err"]))
        l3 = [("last error ", white), (f"{100 * view['last_err']:.1f} cm", rgb)]
    l3.append((f"  |  hits within 5 cm {view['hits5']} / {view['n_scored']}", white))
    if view["l"] is not None:
        l3.append((f"  |  lateral {view['l']:+.2f} m  |  heading {math.degrees(view['dpsi']):+.0f} deg", white))
    return [l1, l2, l3]


def render_episode(model, states, ep, row, out, policy_name="", descriptor="", fps=25, size=(960, 544),
                   cameras=("chase", "top"), t0_s=0.0, t1_s=None, record_s=0.01, ghost=None, ghost_dt_s=0.5,
                   banner_s=1.0, hud=True, t_offset_s=0.0):
    """mp4 of a recorded episode (states [T, 67] at record_s) with markers, the HUD and the end banner (last banner_s
    seconds); cameras side by side (chase + top = 1920 x 544 at the default size). ghost: optional [T', 67] states at
    20 ms from the hand-over (an NRD rollout) drawn translucent ghost_dt_s ahead. t_offset_s is added to the HUD time
    (an NRD rollout starts at the hand-over, 1 s). Returns the number of frames."""
    import imageio
    view = EpisodeView(ep, record_s)
    rend = CourseRenderer(model, view.course, cameras, size)
    step = max(1, round(1 / fps / record_s))
    n = len(states)
    r0, r1 = int(round(t0_s / record_s)), n if t1_s is None else min(n, int(round(t1_s / record_s)) + 1)
    banner, rgb = outcome_banner(row)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    frames = 0
    with imageio.get_writer(out, fps=round(1 / (step * record_s)), codec="libx264", quality=7,
                            macro_block_size=8) as w:
        for r in range(r0, r1, step):
            v = view.at(r)
            v["t"] = r * record_s + t_offset_s
            gh = None
            if ghost is not None and ep.hand_over_step is not None:
                j = int(round((r * record_s - ep.hand_over_step * CONTROL_S + ghost_dt_s) / CONTROL_S))
                gh = ghost[min(max(j, 0), len(ghost) - 1)]
            img = rend.frame(states[r], v, gh)
            if hud:
                img = draw_hud(img, hud_lines(v, ep, row, policy_name, descriptor),
                               banner if (n - 1 - r) * record_s < banner_s else None, rgb)
            w.append_data(img)
            frames += 1
    rend.close()
    return frames


def compose_videos(inputs, out, cols=2, width=None, labels=None):
    """Side-by-side / grid composition with ffmpeg (xstack): inputs (mp4 paths), cols per row, each scaled to `width`
    (default: as is; all inputs must then have one size), shorter clips padded with their last frame. labels: one
    text per input drawn at its top-right corner (drawtext). Returns the ffmpeg command."""
    import subprocess
    inputs = [str(p) for p in inputs]
    dur = []
    for p in inputs:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                           capture_output=True, text=True, check=True)
        dur.append(float(r.stdout.strip()))
    tmax = max(dur)
    n = len(inputs)
    rows = math.ceil(n / cols)
    parts = []
    for i in range(n):
        f = f"[{i}:v]tpad=stop_mode=clone:stop_duration={tmax - dur[i] + 0.1:.3f}"
        if width:
            f += f",scale={int(width)}:-2"
        if labels and labels[i]:
            txt = str(labels[i]).replace(":", "\\:").replace("'", "")
            f += f",drawtext=text='{txt}':x=w-tw-12:y=h-th-12:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.6"
        parts.append(f + f"[v{i}]")
    if n == 1:
        graph = ";".join(parts) + f";[v0]null[out]"
    elif rows == 1 and n == cols:
        graph = ";".join(parts) + ";" + "".join(f"[v{i}]" for i in range(n)) + f"hstack=inputs={n}[out]"
    else:
        lay = "|".join(f"{'+'.join(['w0'] * (i % cols)) or '0'}_{'+'.join(['h0'] * (i // cols)) or '0'}" for i in range(n))
        graph = (";".join(parts) + ";" + "".join(f"[v{i}]" for i in range(n))
                 + f"xstack=inputs={n}:layout={lay}:fill=black[out]")
    cmd = ["ffmpeg", "-y", "-v", "error"] + sum((["-i", p] for p in inputs), []) + [
        "-filter_complex", graph, "-map", "[out]", "-t", f"{tmax:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "23", str(out)]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    return cmd


# ---- per-course plot (dataviz skill: reference palette, thin marks, recessive grid, legend for >= 2 series) ----
VIZ = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
           planned="#2a78d6", achieved="#eb6834", plank="#e9e8e2",
           status=(("within 3 cm", "#0ca30c"), ("3-5 cm", "#fab219"), ("5-8 cm", "#ec835a"), ("over 8 cm", "#d03b3b")))


def _status(e):
    for (name, hexc), thr in zip(VIZ["status"], (0.03, 0.05, 0.08, math.inf)):
        if e <= thr:
            return name, hexc
    return VIZ["status"][-1]


def plot_course(ep, row, out, title=""):
    """PNG (spec 12.6): planned vs achieved step length per target (bars) and a top view of the targets and the scored
    touchdowns (course frame; touchdown dots coloured by error class, with labels in the legend)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    c = ep.course_h if ep.course_h is not None else ep.course
    pr = ep.progress
    rows = pr.metrics()["rows"] if pr is not None else []
    L_plan = pr.L_plan if pr is not None else FT.step_vectors(c)[0]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7.2), gridspec_kw=dict(height_ratios=[1.1, 1.0]))
    fig.patch.set_facecolor(VIZ["surface"])
    for a in (a1, a2):
        a.set_facecolor(VIZ["surface"])
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(VIZ["axis"])
        a.tick_params(colors=VIZ["ink2"], labelsize=9)
        a.grid(True, color=VIZ["grid"], linewidth=0.8)
        a.set_axisbelow(True)
    idx = np.arange(c.n)
    wbar = 0.38
    a1.bar(idx - wbar / 2 - 0.01, L_plan, wbar, color=VIZ["planned"], label="planned")
    ach = {r["target"]: r["L_act"] for r in rows}
    if ach:
        k = np.array(sorted(ach))
        a1.bar(k + wbar / 2 + 0.01, [ach[i] for i in k], wbar, color=VIZ["achieved"], label="achieved")
    if pr is not None and pr.ptr < c.n:
        a1.axvspan(pr.ptr - 0.5, c.n - 0.5, color=VIZ["grid"], alpha=0.5, lw=0)
        a1.text(pr.ptr - 0.4, a1.get_ylim()[1] * 0.92, "not reached", color=VIZ["ink2"], fontsize=9)
    a1.axhline(0, color=VIZ["axis"], lw=1)
    a1.set_xlabel("target", color=VIZ["ink2"])
    a1.set_ylabel("step length L (m)", color=VIZ["ink2"])
    a1.set_xticks(idx)
    a1.legend(frameon=False, loc="upper left", ncols=2, labelcolor=VIZ["ink"])
    a1.set_title(title or f"{c.level} seed {c.seed}", color=VIZ["ink"], loc="left", fontsize=12)
    # top view, course frame
    t, _ = c.local()
    if c.is_planks:
        for k, (s0, s1) in enumerate(c.intervals):
            a2.add_patch(Rectangle((s0, -min(c.half_width, 0.45)), s1 - s0, 2 * min(c.half_width, 0.45),
                                   color=VIZ["plank"], lw=0))
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * np.asarray(SOLE_HALF)
    for i in range(c.n):
        cs, sn = math.cos(t[i, 2]), math.sin(t[i, 2])
        pts = t[i, :2] + corners @ np.array([[cs, sn], [-sn, cs]])
        a2.add_patch(Polygon(pts, closed=True, fill=False, ec=VIZ["ink2"], lw=1.0))
    seen = set()
    for td in ep.log:
        if td["kind"] != "scored":
            continue
        s, l = c.path_coords(np.array([td["x"], td["y"]]))
        name, hexc = _status(td["err"])
        a2.plot(s, l, "o", ms=8, mfc=hexc, mec=VIZ["surface"], mew=2, label=None if name in seen else name, zorder=3)
        seen.add(name)
    a2.set_aspect("equal")
    lo, hi = t[:, :2].min(0), t[:, :2].max(0)
    a2.set_xlim(min(lo[0], -0.2) - 0.15, hi[0] + 0.3)
    a2.set_ylim(lo[1] - 0.25, hi[1] + 0.25)
    a2.set_xlabel("along the course s (m)", color=VIZ["ink2"])
    a2.set_ylabel("l (m)", color=VIZ["ink2"])
    order = [n for n, _ in VIZ["status"] if n in seen]
    h, lab = a2.get_legend_handles_labels()
    if h:
        hl = dict(zip(lab, h))
        a2.legend([hl[n] for n in order], [f"touchdown error {n}" for n in order], frameon=False, loc="upper left",
                  bbox_to_anchor=(0, -0.18), ncols=4, labelcolor=VIZ["ink"], fontsize=9)
    summ = (f"outcome {row.get('outcome')}  |  scored {row.get('scored', 0)} / {c.n}  |  "
            f"error p50 {100 * (row.get('err_p50') or float('nan')):.1f} cm, p95 {100 * (row.get('err_p95') or float('nan')):.1f} cm")
    a2.set_title("top view: planned footprints (outlines) and scored touchdowns  |  " + summ, color=VIZ["ink"],
                 loc="left", fontsize=10)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, facecolor=VIZ["surface"])
    plt.close(fig)


# ---- the same episodes inside the NRD (spec 12.4 --nrd) ---------------------------------------------------------
def nrd_env(policy, snaps, max_steps=POLICY_STEPS, device="cpu", sim=None):
    """A G1FootstepsEnv with the policy's env config (rules, score, reward, observation and action unchanged;
    auto_reset off, episode_steps = max_steps, num_envs = len(snaps)); NRD = the policy's members, HOMIE = its network,
    start bank = the snapshots (mujoco_policy.SnapshotBank; only the constructor's initial reset uses it), course bank
    = the local validation bank (reset_from_snapshots brings the MuJoCo courses)."""
    from types import SimpleNamespace
    from nedm.g1_footstep_rl.mujoco_policy import SnapshotBank
    from nedm.g1_footstep_rl.sim import NrdSim
    FE = footsteps_env()
    ec = policy.env_cfg
    cfg = FE.merge_cfg(dict(ec, num_envs=len(snaps), auto_reset=False, episode_steps=int(max_steps), seed=0,
                            course_bank=str(FE.VAL_COURSE_BANK), course_bank_max=8, level_weights={"F0": 1.0},
                            members=sorted(set(int(m) for m in ec["members"]) | {0, 1})))
    share = SimpleNamespace(sim=sim or NrdSim(cfg["checkpoints"], device=device), net=policy.net,
                            bank=SnapshotBank(snaps, device))
    return FE.G1FootstepsEnv(cfg, device, share=share)


@torch.inference_mode()
def nrd_rollout_courses(policy, snapshots, members=(0, 1), max_steps=POLICY_STEPS, device="cpu", keep_states=False,
                        sim=None):
    """The policy inside the NRD from the MuJoCo hand-over snapshots (contract (13)), once per member. Returns (rows
    (one per snapshot and member, snapshot order then member: outcome, completed, clean_completion, success3/5/8,
    fail reasons, steps, scored, hits, errors, drift, first_obs_max_abs_diff), steps (per scored target: target, e,
    along, across, yaw_err, member), states {(level, seed, member): [T, 67] newest NRD state per step from the
    hand-over} if keep_states)."""
    FE = footsteps_env()
    members = [int(m) for m in members]
    snaps = [s for s in snapshots for _ in members]
    mem = [m for _ in snapshots for m in members]
    n = len(snaps)
    env = nrd_env(policy, snaps, max_steps, device, sim)
    obs = env.reset_from_snapshots(snaps, mem)
    first = (obs.cpu() - torch.as_tensor(np.stack([s["obs"] for s in snaps]))).abs().amax(1)
    tr = env.tracker
    done = torch.zeros(n, dtype=torch.bool)
    keep, step_rows = {}, []
    states = [[env.hist[i, -1].cpu().numpy().copy()] for i in range(n)] if keep_states else None
    gates = [[np.asarray(snaps[i]["gate"], np.float64) > 0.5] for i in range(n)] if keep_states else None
    for t in range(max_steps):
        p0 = tr.ptr.clone()
        obs, _, d, ex = env.step(policy.actor(obs))
        si = ex["step_info"]
        alive = ~done
        off = torch.zeros_like(p0)
        for ps in si["tracker"]["passes"]:
            sc = ps["scored"].cpu() & alive
            for i in torch.nonzero(sc).flatten().tolist():
                step_rows.append(dict(level=snaps[i]["level"], seed=int(snaps[i]["seed"]), member=mem[i],
                                      target=int(p0[i] + off[i]), e=float(ps["e"][i]), along=float(ps["along"][i]),
                                      across=float(ps["across"][i]), yaw_err=float(ps["yaw_err"][i]),
                                      foot=int(ps["foot"][i]), step=int(ps["step"][i])))
            off = off + ps["scored"].cpu().long()
        if keep_states:
            for i in torch.nonzero(alive).flatten().tolist():
                states[i].append(si["state"][i].cpu().numpy().copy())
                gates[i].append(si["gates"][i].cpu().numpy().copy())
        new = (d.cpu() > 0) & ~done
        if bool(new.any()):
            res = si["episode"]
            for i in torch.nonzero(new).flatten().tolist():
                reasons = [k for k, v in si["reasons"].items() if bool(v[i])]
                r = dict(level=snaps[i]["level"], seed=int(snaps[i]["seed"]), member=mem[i],
                         completed=bool(si["completed"][i]), clean_completion=bool(si["clean_completion"][i]),
                         finished=bool(si["finished"][i]), fail=bool(si["fail"][i]), time_out=bool(si["time_out"][i]),
                         reasons=reasons, steps=int(si["steps"][i]), pointer=int(res["pointer"][i]),
                         n_targets=int(res["n"][i]), first_obs_max_abs_diff=float(first[i]))
                for k in ("scored", "hits3", "hits5", "hits8", "resteps", "resteps_free", "wrong_foot", "far_miss",
                          "flickers", "overruns", "touchdowns", "touchdowns_over_gap", "unsupported_steps"):
                    r[k] = int(res[k][i])
                for x in (3, 5, 8):
                    r[f"success{x}"] = bool(si["completed"][i]) and bool(res[f"success{x}"][i])
                    r[f"hit{x}"] = r[f"hits{x}"] / r["scored"] if r["scored"] else float("nan")
                r.update(max_abs_l=float(res["max_abs_l"][i]), max_abs_dpsi_deg=math.degrees(float(res["max_abs_dpsi"][i])),
                         stop_success=bool(res["stop_success"][i]), end_s=round(STAND_S + r["steps"] * CONTROL_S, 2))
                r["outcome"] = ("completed" if r["completed"] else "finished" if r["finished"] else
                                "timeout" if r["time_out"] else "fail_" + "+".join(reasons))
                r["fell"] = "unstable" in reasons or "fall" in reasons
                keep[i] = r
        done |= d.cpu() > 0
        if bool(done.all()):
            break
    rows = []
    for i in range(n):
        r = keep.get(i) or dict(level=snaps[i]["level"], seed=int(snaps[i]["seed"]), member=mem[i], completed=False,
                                clean_completion=False, outcome="not_ended", first_obs_max_abs_diff=float(first[i]))
        ep_steps = [s for s in step_rows if s["level"] == r["level"] and s["seed"] == r["seed"] and s["member"] == mem[i]]
        e = np.array([s["e"] for s in ep_steps])
        r.update(err_p50=float(np.percentile(e, 50)) if len(e) else float("nan"),
                 err_p95=float(np.percentile(e, 95)) if len(e) else float("nan"))
        rows.append(r)
    out_states = None
    if keep_states:
        out_states = {(snaps[i]["level"], int(snaps[i]["seed"]), mem[i]): dict(states=np.asarray(states[i]),
                                                                              gates=np.asarray(gates[i], bool))
                      for i in range(n)}
    return rows, step_rows, out_states


def snapshot_from_record(rec, course):
    """A reset_from_snapshots snapshot (fresh progress) from a FootstepRunner record (record=True)."""
    keys = ("hist", "teacher_history", "teacher_last_action", "prev_cmd", "prev_u", "gate", "obs")
    return dict({k: rec[k] for k in keys}, course=course.to_dict(), level=course.level, seed=int(course.seed))


@torch.inference_mode()
def observation_parity(ckpt, level, seed, n_steps=24, policy=None):
    """Contract (11) check: one recorded MuJoCo episode of the policy; at >= n_steps steps (first steps, single-support
    steps, steps right after a pointer advance, the last step) G1FootstepsEnv.observe() after reset_from_snapshots of
    the runner's inputs (4 states, HOMIE memory, previous command / action, gate, course) and the tracker values of
    the record (pointer, last placements, time since touchdowns, last error, start window) must equal the runner's
    observation. Returns a result dict (passed = max abs diff <= 1e-6)."""
    pol = policy or load_footstep_policy(ckpt)
    course, _ = course_for(level, seed)
    ctrl = FootstepRunner(pol, record=True)
    row, ep, env = run_course_episode(course, ctrl)
    recs = ctrl.records
    single = [i for i, r in enumerate(recs) if r["gate"].min() < 0.5]
    adv = [i for i in range(1, len(recs)) if recs[i]["pointer"] != recs[i - 1]["pointer"]]
    pick = {0, 1, 2, 5, 10, len(recs) // 2, len(recs) - 1}
    pick |= set(single[::max(1, len(single) // 8)][:8]) | set(adv[:8])
    pick = sorted(i for i in pick if 0 <= i < len(recs))
    snaps = [snapshot_from_record(recs[i], ep.course_h) for i in pick]
    nenv = nrd_env(pol, snaps, 10)
    nenv.reset_from_snapshots(snaps, [0] * len(snaps))
    tr = nenv.tracker
    for j, i in enumerate(pick):                    # the record's tracker values (steps relative to the hand-over)
        pr = recs[i]["progress"]
        t0 = pr["t0_step"]
        tr.ptr[j] = int(pr["pointer"])
        tr.last_place[j] = torch.as_tensor(np.asarray(pr["last_place"], np.float64))
        tr.last_td_step[j] = torch.as_tensor(np.asarray(pr["last_td_step"]) - t0)
        tr.k[j] = int(pr["step"]) - t0 + 1
        tr.last_err[j] = torch.as_tensor(np.asarray(pr["last_err"], np.float64))
    obs = nenv.observe().numpy()
    ref = np.stack([recs[i]["obs"] for i in pick])
    diff = np.abs(obs - ref)
    names = footsteps_env().obs_names(int(pol.env_cfg["obs"]["n_targets"]))
    worst = diff.max(0)
    return dict(policy=pol.name, level=level, seed=int(seed), outcome=row["outcome"], policy_steps=len(recs),
                steps_checked=[int(recs[i]["step"]) for i in pick], pointers=[recs[i]["pointer"] for i in pick],
                single_support_checked=int(sum(recs[i]["gate"].min() < 0.5 for i in pick)),
                pointer_advances_checked=int(len(set(adv) & set(pick))), max_abs_diff=float(diff.max()),
                max_abs_diff_per_step=diff.max(1).tolist(),
                worst_entries={names[k]: float(worst[k]) for k in np.argsort(-worst)[:5]},
                passed=bool(diff.max() <= 1e-6))
