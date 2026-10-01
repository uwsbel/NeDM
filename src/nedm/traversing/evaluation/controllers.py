"""Controllers of a drive (FINAL_DESIGN 2.3; PARITY #7, #8 in episode.py): the stock PID, the held PID and the tracker.

A controller turns the path follower's output into the triple handed to vehicle.Synchronize. The episode loop owns
the follower, the settle, the desired speed and the stop rules (the 40 s near-stop rule: EvalConfig.near_stop_rule);
it calls, per frame k:
  frame_top(ep)       once, after SetDesiredSpeed (k >= 0), before the frame's first Synchronize
  inputs(ep, u)       every substep, u = follower.GetInputs() = the output of the PREVIOUS Advance; returns the inputs
  after_sync(ep)      every substep after the vehicle Synchronize (k >= 0)
  frame_end(ep)       after the frame's physics and record (k >= 0)
  switch(ep, route)   when a hook swaps the route
Held controllers (crm_collect_ext.py:217-296, gen_collect_ext.py:243-282) decide one command at the top of each frame,
hold_clip it against the previous held steering and write that triple at every substep: hold_clip REPLACES the
per-substep clamp. The follower keeps running as a shadow, so settle, parking and desired speed stay native. The
previous command is fed back as the recorded float32 action on soil and as the float64 command on rigid ground
(SPEC 0.10).
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from nedm.traversing.training.numpy_actor import NumpyActor
from nedm.traversing.training.state import OBSERVABLE_COLS, SETTLE_ACTION

COLS = list(OBSERVABLE_COLS)
# the observation the Tracker builds = the released actor's meta obs_layout (gc_control.PolicyObs.layout)
OBS_LAYOUT = dict(preview_points=10, preview_spacing_m=1.0, search_window=40, hist_steps=8, state_cols=COLS,
                  settle_action=list(SETTLE_ACTION), state_mean=None, state_std=None)


def hold_clip(cmd, prev_steer, rate=0.1):
    """gc_control.hold_clip (gc_control.py:75-99): steering to prev +- rate, then [-1, 1]; throttle and brake to [0, 1].
    One clip per frame; the result is held over every substep. A non-finite command raises."""
    s, t, b = (float(v) for v in cmd)
    p = float(prev_steer)
    if not all(map(math.isfinite, (s, t, b))):
        raise ValueError(f'hold_clip: non-finite command {cmd!r}')
    s = min(max(s, p - rate), p + rate)
    return min(max(s, -1.0), 1.0), min(max(t, 0.0), 1.0), min(max(b, 0.0), 1.0)


def driver_inputs(cmd):
    """A fresh pychrono DriverInputs holding the triple `cmd` (gen_collect_ext.GenExt.driver_inputs)."""
    import pychrono.vehicle as veh
    u = veh.DriverInputs()
    u.m_steering, u.m_throttle, u.m_braking = (float(v) for v in cmd)
    return u


def _noop(*_):
    pass


class StockPID:
    """The follower's own steering, throttle and brake (M2, M3 pid_native, M4): steering clamped every substep to
    ep.prev_steer +- 2 dt, in place on the DriverInputs as recorded, and forced to 0 while settling
    (traverse_fdm_rgbd_diverse_chrono.py:222-225)."""
    owns_speed, info = False, None      # owns_speed: no SetDesiredSpeed after the settle; info: record provenance
    reset = frame_top = after_sync = frame_end = switch = _noop       # the hooks (module docstring) it does not use

    def inputs(self, ep, u):
        p = ep.prev_steer
        ep.prev_steer = 0. if ep.k < 0 else float(np.clip(u.m_steering, p - 2. * ep.sim.dt, p + 2. * ep.sim.dt))
        u.m_steering = ep.prev_steer
        return u


class HeldPID(StockPID):
    """M3 pid_held_50ms: the shadow follower's output at the frame top (its last Advance of frame k-1), held."""

    def reset(self, ep):
        self.f32, self.last = ep.sim.ground == 'soil', np.asarray(SETTLE_ACTION, np.float64)    # action[k-1]

    def command(self, ep):
        u = ep.fol.GetInputs()
        return u.m_steering, u.m_throttle, u.m_braking

    def frame_top(self, ep):
        self.held = hold_clip(self.command(ep), self.last[0])
        self.u = driver_inputs(self.held)

    def inputs(self, ep, u):
        return super().inputs(ep, u) if ep.k < 0 else self.u

    def frame_end(self, ep):
        self.last = ep.action[-1].astype(np.float64) if self.f32 else np.asarray(self.held, np.float64)


class Tracker(HeldPID):
    """M3 nrd_policy_v2: main's NumpyActor on the 158-number observation of gc_control.PolicyObs + RouteTracker
    (gc_control.py:700-888, the WP3 tracker env's), squashed to (steering, throttle, brake), then held:
      [0:3]     e_along / 10, e_ct / 10, e_h / pi to the nearest waypoint (the first frame of a route searches all of
                it, later frames waypoints idx-2 .. idx+39)
      [3:33]    10 points 1..10 m ahead (index step round(j / mean waypoint spacing)): body x / 10, y / 10, v_ref / 5
      [33:38]   vx / 10, yaw rate, the previous command
      [38:62]   the previous 8 commands, oldest first (the settle action (0, 0, 1) before frame 0)
      [62:158]  the 12 observable state columns of frames k-7 .. k, oldest first (the frame-0 state before frame 0)
    The state is captured at the frame top BEFORE Synchronize; column 15 is read from the transmission's output shaft,
    whose value Synchronize then imposes on the engine (the engine's own value still lags one substep there), so the
    observable columns equal the recorded state[k] (info['pre_capture_max_abs_diff'], 0 in every recorded drive).
    History pushes follow the feedback: soil (state[k], action[k]) as recorded in float32, rigid (the captured state,
    the float64 command). A non-finite observation raises (none in the 846 released tracker drives)."""

    def __init__(self, actor_npz):
        self.actor = NumpyActor.from_npz(actor_npz)
        lay = self.actor.meta.get('obs_layout', {})
        if self.actor.num_obs != 158 or {k: lay.get(k) for k in OBS_LAYOUT} != OBS_LAYOUT:
            raise ValueError(f'{actor_npz}: observation layout {lay} is not the ported one {OBS_LAYOUT}')
        self.info = dict(actor=str(actor_npz), actor_sha256=hashlib.sha256(Path(actor_npz).read_bytes()).hexdigest(),
                         pre_capture_max_abs_diff=0.)

    def reset(self, ep):
        super().reset(ep)
        self.switch(ep, ep.route)
        self.S = None                   # state rows k-8 .. k-1 (float64); self.A: the commands k-8 .. k-1

    def switch(self, ep, route):        # a fresh route tracker; the history is kept (a branch is unvalidated)
        self.xy, self.v, self.h = (np.asarray(route[k], np.float64) for k in ('waypoints', 'speeds', 'headings'))
        n, self.idx = len(self.xy), None
        ds = np.asarray(route['stations'], np.float64)[-1] / max(n - 1, 1)
        self.step = np.round(np.arange(1, 11) * 1.0 / ds).astype(int)        # preview spacing 1 m

    def command(self, ep):
        sim = ep.sim
        state, _, pose = sim.measure(ep.k, sim.now(), driver_inputs(self.last))[:3]
        state[15] = sim.transmission.GetOutputMotorshaftSpeed()
        if self.S is None:              # frame 0: the rest state and the settle action fill the history
            self.S, self.A = np.tile(state.astype(np.float64), (8, 1)), np.tile(SETTLE_ACTION, (8, 1))
        self.pre = state
        return self.actor.act(self.observe(pose, state))

    def observe(self, pose, state):
        x, y, yaw = (float(v) for v in pose)
        st, n = np.asarray(state, np.float64), len(self.xy)
        cand = np.arange(n) if self.idx is None else np.clip(self.idx + np.arange(-2, 40), 0, n - 1)
        self.idx = int(cand[int(np.argmin(np.hypot(self.xy[cand, 0] - x, self.xy[cand, 1] - y)))])
        dx, dy, h = x - self.xy[self.idx, 0], y - self.xy[self.idx, 1], self.h[self.idx]
        i = np.minimum(self.idx + self.step, n - 1)
        px, py, c, s = self.xy[i, 0] - x, self.xy[i, 1] - y, math.cos(yaw), math.sin(yaw)
        obs = np.concatenate([
            [(dx * math.cos(h) + dy * math.sin(h)) / 10.0, (-dx * math.sin(h) + dy * math.cos(h)) / 10.0,
             math.atan2(math.sin(yaw - h), math.cos(yaw - h)) / math.pi],
            np.stack([(c * px + s * py) / 10.0, (-s * px + c * py) / 10.0, self.v[i] / 5.0], axis=-1).reshape(-1),
            [st[0] / 10.0, st[6]], self.last, self.A.reshape(-1), np.vstack([self.S[1:], st])[:, COLS].reshape(-1)])
        if not np.isfinite(obs).all():
            raise FloatingPointError(f'non-finite tracker observation at {pose}: {obs}')
        return obs

    def frame_end(self, ep):
        super().frame_end(ep)
        rec = ep.state[-1]
        self.info['pre_capture_max_abs_diff'] = max(self.info['pre_capture_max_abs_diff'],
                                                    float(np.abs(self.pre[COLS] - rec[COLS]).max()))
        self.S = np.vstack([self.S[1:], rec if self.f32 else self.pre])
        self.A = np.vstack([self.A[1:], self.last])


def make_controller(cfg, env=None):
    """The controller of an arm; the tracker's actor is a checked release file (env.file)."""
    if cfg.controller == 'tracker':
        return Tracker(env.file(cfg.actor))
    if cfg.controller == 'nav_pid':
        raise NotImplementedError('controller nav_pid: the M1 mission controller is not ported yet')
    return {'pid': StockPID, 'pid_held': HeldPID}[cfg.controller]()
