"""One drive: the episode loop, the stop rules and the approach-protocol branch (FINAL_DESIGN 2.2-2.3).

Frame k (k < 0: the 0.8 s settle, 16 frames):
  top  [k == 0: hook.start] -> waypoint search (window 60, monotone) -> parking -> desired speed (0 while settling or
       parked) -> follower.SetDesiredSpeed -> ctrl.frame_top
  sub  follower.Synchronize(t) -> ctrl.inputs(GetInputs()) -> terrain + vehicle Synchronize -> [k >= 0: record at
       substep 0 (+ launch check at frame 0) -> ctrl.after_sync -> positive work] -> follower.Advance -> sim.advance
  end  post-advance chassis reference -> ctrl.frame_end -> status = hook.end_frame(ep) or stops.check(ep)
A Branch hook swaps the route at the end of frame F-1, which is the frozen collectors' top of frame F: nothing runs in
between (gen_collect_ext.py:222-240).

PARITY (recorded behaviour, FINAL_DESIGN 5; each follows from ground, vehicle, controller and task kind, never a switch):
  #5   the first follower is Initialize()d once before the settle; a branch builds a fresh one without Initialize()
       (zero PID memory) and the old ones stay alive; steering continuity comes from ep.prev_steer via the clamp.
  #6   settle: 16 frames at desired speed 0 with steering forced to 0, the follower's PID still integrating.
  #7   per-substep steering clamp ep.prev_steer +- 2 dt on the follower's DriverInputs (controllers.StockPID); held
       controllers REPLACE it by one hold_clip per frame, the same triple at every substep (controllers.HeldPID).
  #8   held feedback float32 (soil) vs float64 (rigid); the tracker observes before Synchronize with column 15 from
       the transmission and pushes the recorded float32 rows (soil) or its capture + float64 command (rigid).
  #9   the 40 s near-stop rule only for the held PID and the tracker (EvalConfig.near_stop_rule), after the native rules.
  #10  stop order: [soil breakthrough ->] rollover -> goal -> bounds -> blockage -> [near-stop] -> timeout. Breakthrough:
       the deepest wheel (stock tyre radius) below the BMP by more than soil depth + margin on 5 frames in a row
       (crm_collect.py:285-293); a soil fall-through (sim.FellThrough) or a non-finite pose raises, never a label.
  #11  goal and rollover on the chassis reference after the frame's physics.
  #12  blockage: bounds first; < 40 recorded frames -> none; the last 40 poses + the post-frame pose within a 0.25 m
       diameter, all 40 throttles > 0.3 (float32 compare), no frame parked; nothing before 24 s; confirm 2 s, then an
       8 s tail; any failing window cancels; tolerance 1e-9 (gen_collect.py:120-154). Earliest stop 34 s.
  #13  branch: rigid keeps the case goal (route end within 0.25 m of it); soil re-targets to the route end (within
       0.5 m of the case goal, crm_collect_ext.py:157-158, 212); start within 1.0 m at F.
  #16  launch check and terminal Synchronize for single-goal tasks only.
Positive work is summed per substep into the frame, then frame by frame (Python float, sequential).
"""

from __future__ import annotations

import math

import numpy as np

from .config import DT
from .routes import ENDS_TOL_M, route_sha256

SETTLE_FRAMES, WP_WINDOW = 16, 60
RAD60 = math.radians(60.)


class Episode:
    """The state of one drive: the active follower and route, the recorded 50 ms rows and the post-frame state."""

    def __init__(self, sim, route, ctrl, stops):
        self.sim, self.ctrl, self.stops = sim, ctrl, stops
        self.k, self.prev_steer, self.after, self.branch, self.old_followers = -SETTLE_FRAMES, 0., None, None, []
        self.state, self.action, self.pose, self.power, self.parked, self.work, self.desired_log, self.extra = (
            [] for _ in range(8))
        self.total_work, self.status, self.terminal_state = 0., None, None
        self._follow(route, initialize=True)

    def _follow(self, route, initialize):
        self.fol = self.sim.follower(route, initialize=initialize)
        self.route, self.wp = route, 0
        self.xy, self.speeds = np.asarray(route['waypoints'], float), np.asarray(route['speeds'], float)

    def switch(self, route, *, initialize):
        self.old_followers.append(self.fol)
        self._follow(route, initialize)
        self.ctrl.switch(self, route)

    def record(self, m):                            # (state, action, pose, power[, soil crm_extra row])
        for rows, v in zip((self.state, self.action, self.pose, self.power, self.extra), m):
            rows.append(v)

    def end_frame(self, work, after):
        self.work.append(work)
        self.total_work += work
        self.parked.append(self.at_end)
        self.desired_log.append(self.desired)
        self.after = after

    def finish(self, status, terminal_state):
        self.status, self.terminal_state = status, terminal_state

    def arrays(self) -> dict:
        """trajectory.npz in the collectors' key names (row k = substep 0 of frame k; terminal = after the last)."""
        a = dict(state=np.asarray(self.state, np.float32).reshape(-1, 17),
                 action=np.asarray(self.action, np.float32).reshape(-1, 3),
                 pose=np.asarray(self.pose, np.float64).reshape(-1, 3), parked=np.asarray(self.parked, bool),
                 power_kw=np.asarray(self.power, np.float64), positive_work_kj_per_interval=np.asarray(self.work),
                 desired_speed_mps=np.asarray(self.desired_log, np.float64))
        if self.after is not None:
            a['terminal_pose'] = self.after.pose
        if self.terminal_state is not None:
            a['terminal_state'] = self.terminal_state
        return a


def drive_episode(sim, route, ctrl, stops, hook=None, *, launch_check=True) -> Episode:
    """Drive `route` until a stop rule (or the hook) returns a status; see the module docstring for the order."""
    ep = Episode(sim, route, ctrl, stops)
    ctrl.reset(ep)
    k, status, u = -SETTLE_FRAMES, None, None
    while status is None:
        if k == 0 and hook is not None and (status := hook.start(ep)):
            break
        ep.k = k
        xy = sim.ref_xy()
        ep.wp += int(np.argmin(np.linalg.norm(ep.xy[ep.wp:ep.wp + WP_WINDOW] - xy, axis=1)))
        ep.at_end = ep.wp >= len(ep.xy) - 2 and float(np.linalg.norm(xy - ep.xy[-1])) < 3.
        ep.desired = 0. if k < 0 or ep.at_end else float(ep.speeds[ep.wp])
        if k < 0 or not ctrl.owns_speed:
            ep.fol.SetDesiredSpeed(ep.desired)
        if k >= 0:
            ctrl.frame_top(ep)
        work = 0.
        for sub in range(sim.substeps):
            t = sim.now()
            ep.fol.Synchronize(t)
            u = ctrl.inputs(ep, ep.fol.GetInputs())
            sim.sync(t, u)
            if k >= 0:
                if sub == 0:
                    ep.record(sim.measure(k, t, u))
                    if k == 0 and launch_check:
                        sim.launch_check(ep)
                ctrl.after_sync(ep)
                work += max(sim.power_kw(), 0.) * sim.dt
            ep.fol.Advance(sim.dt)
            sim.advance()
        if k >= 0:
            ep.end_frame(work, sim.after_frame())
            ctrl.frame_end(ep)
            status = hook.end_frame(ep) if hook is not None else stops.check(ep)
        k += 1
    ep.finish(status, sim.terminal(u) if launch_check else None)      # u: set by the settle
    return ep


class Blockage:
    """Bounds exit and the prolonged-blockage rule (gen_collect.StopPolicy.check); `elapsed` = (frame + 1) x 0.05 s."""

    def __init__(self):
        self.first = None                                # when the current qualifying run of windows began

    def check(self, elapsed, pose, action, parked, post_pose):
        if np.max(np.abs(np.asarray(post_pose)[:2])) > 40.:
            return 'terrain_bounds_exit'
        if len(action) < 40:
            return None
        bounded = False
        if not any(parked[-40:]) and np.all(np.asarray(action[-40:])[:, 1] > .3):
            pts = np.concatenate((np.asarray(pose[-40:])[:, :2], np.asarray(post_pose)[None, :2]))
            bounded = bool(np.square(pts[:, None, :] - pts[None, :, :]).sum(-1).max() <= .25 ** 2)
        if not bounded:
            self.first = None
            return None
        if elapsed + 1e-9 < 24.:
            return None
        if self.first is None:
            self.first = elapsed
        return 'prolonged_blockage_terminated' if elapsed + 1e-9 >= self.first + 2. + 8. else None


class Stops:
    """The stop rules of a drive, first match wins (module docstring #10). Pure in the episode's recorded rows and its
    post-frame state `ep.after`, so stored drives can be replayed through it (test A3)."""

    def __init__(self, goal, radius, *, near_stop=False, horizon_s=120., breakthrough_m=None):
        self.goal, self.radius, self.near_stop = np.asarray(goal, float), float(radius), near_stop
        self.frames, self.breakthrough_m = round(horizon_s / DT), breakthrough_m       # breakthrough_m: soil only
        self.blk, self.near, self.near_stop_fired, self.deep, self.max_sinkage = Blockage(), 0, False, 0, 0.

    def check(self, ep):
        a, k = ep.after, ep.k
        if self.breakthrough_m is not None:              # 5 frames in a row deeper than soil depth + margin
            self.max_sinkage = max(self.max_sinkage, a.sinkage)
            self.deep = self.deep + 1 if a.sinkage > self.breakthrough_m else 0
            if self.deep >= 5:
                return 'soil_breakthrough_terminated'
        if abs(a.roll) > RAD60 or abs(a.pitch) > RAD60:
            return 'rollover'
        if np.linalg.norm(a.pose[:2] - self.goal) <= self.radius:
            return 'goal_reached'
        if s := self.blk.check((k + 1) * DT, ep.pose, ep.action, ep.parked, a.pose):
            return s
        if self.near_stop:                               # 800 frames |vx| < 0.3, not parked, any throttle
            self.near = self.near + 1 if not ep.parked[-1] and abs(float(ep.state[-1][0])) < .3 else 0
            if self.near >= 800:
                self.near_stop_fired = True
                return 'prolonged_blockage_terminated'
        return 'timeout' if k + 1 >= self.frames else None


class Branch:
    """Pass 2 of the approach protocol (SPEC 1.2): drive the approach, then at the top of frame F a fresh follower
    (no Initialize) on the picked route from wherever the vehicle is; rigid keeps the case goal, soil takes the route
    end (module docstring #13)."""

    def __init__(self, F, route, goal, soil=False):
        end, tol = float(np.linalg.norm(np.asarray(route['waypoints'][-1], float) - goal)), .5 if soil else ENDS_TOL_M
        if not (int(F) == F >= 1 and end <= tol):
            raise ValueError(f'branch at frame {F}: need F >= 1 and the route to end within {tol} m of the goal '
                             f'({end:.3f})')
        self.F, self.route, self.soil = int(F), route, soil

    def start(self, ep):
        return None

    def end_frame(self, ep):
        s = ep.stops.check(ep)
        if s is None and ep.k + 1 == self.F:
            d0 = float(np.linalg.norm(np.asarray(self.route['waypoints'][0], float) - ep.after.pose[:2]))
            if len(ep.state) != self.F or d0 > 1.:
                raise RuntimeError(f'branch at frame {self.F}: {len(ep.state)} recorded frames, route starts {d0:.3f} m '
                                   'from the vehicle (tolerance 1.0 m)')
            ep.switch(self.route, initialize=False)
            if self.soil:
                ep.stops.goal = np.asarray(self.route['waypoints'][-1], float).copy()
            ep.branch = dict(frame=self.F, pose=ep.after.pose.tolist(), start_error_m=d0,
                             route_sha256=route_sha256(self.route))
        return s
