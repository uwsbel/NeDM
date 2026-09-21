"""Excitation: what is deliberately added to a collection run, and why each piece has the
shape it has.

This module is PURE. No Chrono, no torch, no file I/O, no clock. Everything it produces is
a function of its arguments and of a `numpy.random.Generator` handed in by the caller.
That is not tidiness for its own sake: the excitation layer is the part of the collector
whose output cannot be checked by looking at the corpus afterwards -- an isotropic push
sampler and a shallow-disc one produce corpora that look alike until you plot the
elevation histogram -- so it has to be testable standalone, and it is.

The admissibility rule from `params/excitation.yaml` governs what lives here: an excitation
is admissible iff it enters through a quantity the model reads, or through the initial
condition. Action injection enters through the logged action. Pushes do not enter through
anything, which is why the push WINDOW is segmented out of the corpus rather than carried
as a channel, and why this module ships `PushSchedule.segments()` alongside the sampler.

RNG DRAW ORDER IS PART OF THE CONTRACT
--------------------------------------
`docs/LESSONS.md`: "multiplying a draw by zero still consumes it, and reordering draws
inside an existing corpus's seed changes every subsequent value. A force-only episode once
replayed with different forces because a torque draw moved."

So every sampler here obeys two rules:

1. **Fixed draw count.** Each sampler consumes the SAME number of values from the Generator
   no matter which branch it takes. A disabled OU process still draws its twelve normals
   and multiplies them by sigma=0. An explicitly supplied chirp joint index still draws the
   candidate index it then discards. This costs nothing and means that turning a feature
   off does not shift the stream for every feature after it.
2. **Fixed draw order**, written down per sampler in its docstring, in the order the values
   come off the Generator. Changing the order is a corpus-breaking change even when the
   marginal distributions are identical.

`SAMPLER_VERSION` is stamped into the run manifest. Per LESSONS.md the rule is "version the
sampler and never edit one in place": if a draw order or count has to change, bump this and
branch on it, so an old corpus stays replayable instead of quietly becoming un-replayable.
"""
from __future__ import annotations

import math

import numpy as np

# Twelve joint position targets at 50 Hz, Chrono order (RR, RL, FR, FL). See
# params/transforms.py for the remap to and from the policy's own order -- this module
# works in whatever order it is handed and never reorders anything itself, because a
# reorder buried in a noise source is exactly the kind of silent sign/index bug the
# policy gate exists to prevent.
N_JOINTS = 12

# Bump, never edit in place. See the module docstring.
SAMPLER_VERSION = 1


# --------------------------------------------------------------------------- scalars
def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    """One draw, log-uniform on [low, high].

    Log-uniform rather than uniform wherever a parameter is a SCALE: uniform over
    [8, 140] N puts 80% of its mass above 35 N, so a corpus sampled that way is a corpus of
    hard shoves with a rounding error of gentle ones. Log-uniform gives every decade equal
    weight, which is what "spans scales" means.
    """
    if not (low > 0.0 and high > 0.0):
        raise ValueError(f"log-uniform needs strictly positive bounds, got [{low}, {high}]")
    if high < low:
        raise ValueError(f"log-uniform bounds are inverted: [{low}, {high}]")
    return float(np.exp(rng.uniform(math.log(low), math.log(high))))


def sample_sigma(rng: np.random.Generator, low: float, high: float) -> float:
    """Per-episode action-injection sigma, in radians. Log-uniform. ONE draw.

    Draw order: [sigma].

    One scalar per EPISODE, not per corpus. The old collector had a single
    `--action-noise-sigma-rad` defaulting to zero, so the corpus sat at one excitation level
    or at none at all. Sampling it per episode makes one corpus span the ladder, which is
    what lets the calibration sweep read a truncation-rate-versus-sigma curve off a single
    collection instead of needing one collection per rung.

    The ceiling in the yaml (0.15 rad) is empirical and provisional: enough injection and
    the robot falls, and those episodes truncate. Gate 1 (action identifiability) fails from
    below and the truncation rate spikes from above, so the usable band is pinned from both
    sides by measurement rather than chosen.
    """
    return _log_uniform(rng, low, high)


# ------------------------------------------------------------------------- OU process
class OUActionNoise:
    """Ornstein-Uhlenbeck noise on the twelve joint position targets.

    WHY NOT i.i.d. PER-STEP NOISE
    -----------------------------
    The action is a joint POSITION TARGET at 50 Hz, tracked by a PD loop (kp 20, kd 0.5)
    against real leg inertia. A position target is never applied directly; it is low-passed
    by the controller and the plant before it becomes motion. Independent Gaussian noise on
    each control step puts most of its energy ABOVE that bandwidth, so the target moves and
    the body does not.

    That is strictly worse than injecting nothing. It manufactures action variance with no
    matching state response, and a model fit to it learns the one lesson we cannot afford:
    that changing the action barely changes the next state. Since d s'/d a is the single
    quantity fine-tuning consumes, white noise on a position target actively suppresses the
    thing the corpus exists to identify. A corpus can therefore look well-excited by action
    variance and be worthless, which is why the acceptance gate measures the action RESIDUAL
    variance and rank after conditioning on the state, not the raw action variance.

    OU is the bounded form of "inject a random rate": integrating a random rate is a random
    walk, and the mean reversion is what stops the walk drifting off. With tau = 0.15 s the
    correlation time is ~7 control steps, inside the band the robot actually follows, so the
    injected motion shows up in the state.

    PER-JOINT INDEPENDENT
    ---------------------
    Twelve independent processes, not one shared signal. The previous corpus's residual
    action covariance was effectively RANK 2 in twelve dimensions; a single shared noise
    signal would have left it rank-deficient by construction, and no amount of it would
    identify the columns of d s'/d a that the shared direction does not touch. `sigma` may
    also be given as a 12-vector to vary the per-joint scale, which the plan lists as
    optional -- the independence is what buys the rank, the per-joint scale only shapes the
    conditioning.

    DISCRETISATION
    --------------
    The exact-update form,

        x <- alpha * x + sigma * sqrt(1 - alpha^2) * eps,    alpha = exp(-dt/tau),

    not Euler-Maruyama. This matters at our step size: EM's stationary variance is inflated
    by 1/(1 - dt/(2 tau)), which at dt = 0.02, tau = 0.15 is +3.5% on the standard deviation.
    The exact update makes the stationary std equal the REQUESTED sigma for any dt and tau,
    so the calibration sweep's x-axis means what it says and is not a function of the
    control rate.

    BOUNDEDNESS
    -----------
    Stationary marginal is N(0, sigma), so the process is bounded in probability, not
    absolutely: |x| exceeds 4 sigma on about 6e-5 of joint-steps. Nothing is clipped here on
    purpose, because clipping inside the process would bias the stationary std away from the
    sigma the sweep is indexed by. The real bound is applied downstream, where the injected
    target is clipped to the joint limits -- the excitation must never command a
    configuration the actuator cannot hold.

    Draw order per call:
      __init__ / reset(): [12 normals]  (the stationary initial condition)
      step():             [12 normals]

    Both draw unconditionally. A sigma of exactly 0.0 still consumes its twelve normals per
    step and multiplies them by zero, so a no-injection episode and an injected one at the
    same seed stay aligned downstream. That is the LESSONS.md rule applied literally.
    """

    def __init__(
        self,
        sigma_rad: float | np.ndarray,
        tau_s: float,
        dt: float,
        rng: np.random.Generator,
    ):
        sigma = np.asarray(sigma_rad, dtype=np.float64)
        if sigma.ndim == 0:
            sigma = np.full(N_JOINTS, float(sigma))
        if sigma.shape != (N_JOINTS,):
            raise ValueError(
                f"sigma_rad must be a scalar or a {N_JOINTS}-vector, got shape {sigma.shape}"
            )
        if np.any(sigma < 0.0) or not np.all(np.isfinite(sigma)):
            raise ValueError("sigma_rad must be finite and non-negative")
        if not (tau_s > 0.0):
            raise ValueError(f"tau_s must be positive, got {tau_s}")
        if not (dt > 0.0):
            raise ValueError(f"dt must be positive, got {dt}")

        self.sigma = sigma
        self.tau_s = float(tau_s)
        self.dt = float(dt)
        self.rng = rng

        # alpha is the one-step correlation. dt/tau = 0.133 at the nominal settings, i.e.
        # alpha = 0.875: the target keeps most of its value from step to step, which is the
        # whole point.
        self.alpha = math.exp(-self.dt / self.tau_s)
        self._innovation = self.sigma * math.sqrt(max(0.0, 1.0 - self.alpha * self.alpha))

        self.x = np.zeros(N_JOINTS, dtype=np.float64)
        self.reset()

    @property
    def correlation_steps(self) -> float:
        """tau expressed in control steps. Sanity number for a log line: ~7.5 at 50 Hz."""
        return self.tau_s / self.dt

    def reset(self) -> np.ndarray:
        """Re-seed the process from its STATIONARY distribution, N(0, sigma). 12 draws.

        Starting from zero instead would spend the first ~3 tau (about 22 rows at the
        nominal settings) below the requested amplitude, so the opening of every episode
        would be systematically under-excited -- a small bias, but one that lands on exactly
        the rows following a randomised initial state, which are the most informative rows
        in the episode. Seeding from the stationary distribution makes row 0 as excited as
        row 1000.
        """
        self.x = self.sigma * self.rng.standard_normal(N_JOINTS)
        return self.x.copy()

    def step(self) -> np.ndarray:
        """Advance one control step and return the 12-vector of target offsets, in radians.

        Returns a COPY. The caller adds this to the policy's target, and the sum is what
        gets logged (`logged: applied` in the yaml) -- logging the clean target instead
        would put an unlogged input back into the corpus while making it look clean, which
        is the confound this whole layer is designed to remove.
        """
        self.x = self.alpha * self.x + self._innovation * self.rng.standard_normal(N_JOINTS)
        return self.x.copy()


# ------------------------------------------------------------------------------- push
def sample_push(rng: np.random.Generator, mag_low: float, mag_high: float) -> np.ndarray:
    """One push force vector, world frame, in newtons. THREE draws.

    Draw order: [z, theta, magnitude].

    DIRECTION is uniform over the sphere:

        z ~ U(-1, 1);  theta ~ U(0, 2 pi);  r = sqrt(1 - z^2);  d = (r cos t, r sin t, z)

    The old sampler drew a full azimuth but capped elevation at atan(0.3), about +/-16.7
    degrees from horizontal, so every push was effectively sideways: the measured maxima
    were 53.6 and 51.8 N in x and y against 17.0 N in z. A corpus built that way contains
    almost no vertical loading or unloading -- nothing that presses the robot into the soil
    or lifts it out of it -- and those are precisely the transitions that distinguish
    deformable terrain from rigid. On CRM terrain that is not a coverage gap at the margin,
    it is the coverage gap.

    MAGNITUDE is log-uniform over [mag_low, mag_high], rather than `uniform(0.25, 1.0) *
    peak`, so the decade from 8 N to 140 N is covered evenly instead of piling up near the
    peak.

    NORM IS EXACT. The direction is renormalised before scaling, so ||f|| equals the sampled
    magnitude to floating-point rounding. The old vector was never normalised: its length
    ran from `mag` to `1.044 * mag` depending on elevation, so the reported "peak N" was
    wrong by up to 4.4% and, worse, wrong in a way CORRELATED with elevation -- the shallow
    pushes it was already biased towards were also the ones whose magnitude was understated.

    Torque is not sampled at all. The old schema carried a torque channel that was never
    driven; keeping it would add three identically-zero inputs, and a channel that is zero
    at collection time is also zero at fine-tuning time.
    """
    if not (mag_high >= mag_low > 0.0):
        raise ValueError(f"magnitude bounds must be positive and ordered, got [{mag_low}, {mag_high}]")

    z = float(rng.uniform(-1.0, 1.0))
    theta = float(rng.uniform(0.0, 2.0 * math.pi))
    magnitude = _log_uniform(rng, mag_low, mag_high)

    r = math.sqrt(max(0.0, 1.0 - z * z))
    direction = np.array([r * math.cos(theta), r * math.sin(theta), z], dtype=np.float64)

    # Analytically a unit vector already; divided anyway so the guarantee holds at the bit
    # level rather than at the level of the algebra. ||direction|| >= |z| and z^2 + r^2 = 1,
    # so this can never divide by zero.
    direction /= np.linalg.norm(direction)
    return magnitude * direction


# ------------------------------------------------------------------------------ probe
class SingleJointChirp:
    """A deliberate single-joint chirp: finite-differencing the Jacobian inside the data.

    On a minority of episodes (15% in the yaml) one joint target carries a swept-frequency
    sinusoid and the other eleven carry nothing. This is the cleanest identifiability signal
    available, because perturbing ONE input at a time and recording the response is exactly
    what a column of d s'/d a is, and it is what Gate 2 compares against Chrono's own
    Jacobian.

    It is a minority mode and not the default for the symmetric reason: real policy updates
    move many joints together, so a corpus of single-joint probes would identify the columns
    while saying nothing about the coordinated directions a policy actually explores. OU
    covers the coordinated directions; probes pin down the columns. Neither alone does both,
    which is why the corpus carries both.

    A CHIRP rather than a step because a step excites one transient once, while a linear
    sweep from 0.5 to 6.0 Hz walks the response across the band the PD loop and the leg
    inertia actually pass, which is where the interesting frequency dependence of the
    terrain response lives.

    Draw order:
      __init__: [candidate joint index]
      sample(): [amplitude, f0, f1, candidate joint index]

    The candidate joint index is drawn EVEN WHEN `joint` is given explicitly and then
    discarded, so a scripted probe and a random one consume the same stream. `value()`
    consumes nothing at all -- all randomness is resolved at construction, so the stream
    does not depend on how many times the collector happens to evaluate the probe.
    """

    def __init__(
        self,
        amplitude_rad: float,
        f0_hz: float,
        f1_hz: float,
        duration_s: float,
        joint: int | None = None,
        rng: np.random.Generator | None = None,
        t0_s: float = 0.0,
    ):
        if not (duration_s > 0.0):
            raise ValueError(f"duration_s must be positive, got {duration_s}")
        if amplitude_rad < 0.0:
            raise ValueError(f"amplitude_rad must be non-negative, got {amplitude_rad}")
        if f0_hz < 0.0 or f1_hz < 0.0:
            raise ValueError(f"chirp frequencies must be non-negative, got [{f0_hz}, {f1_hz}]")

        # Unconditional draw, then overridden. See the class docstring.
        drawn = int(rng.integers(0, N_JOINTS)) if rng is not None else 0
        if rng is None and joint is None:
            raise ValueError("SingleJointChirp needs either an explicit joint or an rng to draw one")
        self.joint = drawn if joint is None else int(joint)
        if not (0 <= self.joint < N_JOINTS):
            raise ValueError(f"joint index out of range: {self.joint}")

        self.amplitude_rad = float(amplitude_rad)
        self.f0_hz = float(f0_hz)
        self.f1_hz = float(f1_hz)
        self.duration_s = float(duration_s)
        self.t0_s = float(t0_s)

    @classmethod
    def sample(
        cls,
        rng: np.random.Generator,
        amplitude_range: tuple[float, float],
        chirp_hz: tuple[float, float],
        duration_s: float,
        joint: int | None = None,
        t0_s: float = 0.0,
    ) -> "SingleJointChirp":
        """Draw a probe from the yaml's ranges. FOUR draws: [amplitude, f0, f1, joint].

        Two readings of the yaml are settled here and neither is stated there, so both are
        recorded rather than buried:

        * `amplitude_rad: [0.02, 0.12]` is a per-episode SAMPLING range, drawn log-uniform.
          The yaml marks `sigma_rad` and `magnitude_n` log-uniform explicitly and is silent
          here, but the stated reason -- one corpus spanning scales -- applies to a probe
          amplitude exactly as it does to an injection sigma, and this is the same kind of
          quantity.
        * `chirp_hz: [0.5, 6.0]` is the SWEEP, f0 -> f1, not a range to draw endpoints from.
          f0 and f1 are nevertheless drawn (degenerately, from a zero-width range) so that
          the draw count stays at four if that reading is ever revised.
        """
        amplitude = _log_uniform(rng, float(amplitude_range[0]), float(amplitude_range[1]))
        f0 = float(rng.uniform(float(chirp_hz[0]), float(chirp_hz[0])))
        f1 = float(rng.uniform(float(chirp_hz[1]), float(chirp_hz[1])))
        return cls(amplitude, f0, f1, duration_s, joint=joint, rng=rng, t0_s=t0_s)

    def value(self, t: float) -> np.ndarray:
        """Target offset at time `t`, radians, 12-vector. Zero outside the probe window.

        Linear-frequency chirp: the instantaneous frequency rises linearly from f0 to f1
        across the window, so the phase is quadratic in elapsed time.

        The phase starts at zero, so the probe opens continuously -- no step into the
        actuator at window open. It does NOT in general close continuously, since
        sin(phase(T)) is whatever the sweep lands on; the residual is at most `amplitude`,
        which at the yaml's ceiling is 0.12 rad, and the guard band applied around probe
        windows is what keeps that edge out of the corpus. Tapering is deliberately not
        applied, because it would make the delivered amplitude something other than the
        requested one and the probe's whole value is that its amplitude is known.
        """
        out = np.zeros(N_JOINTS, dtype=np.float64)
        u = float(t) - self.t0_s
        if 0.0 <= u < self.duration_s:
            rate = (self.f1_hz - self.f0_hz) / self.duration_s
            phase = 2.0 * math.pi * (self.f0_hz * u + 0.5 * rate * u * u)
            out[self.joint] = self.amplitude_rad * math.sin(phase)
        return out


# --------------------------------------------------------------------- push schedule
class PushSchedule:
    """When the pushes happen, and which rows survive them.

    The push window is the one part of an episode whose cause is not a model input, so it
    cannot be in the corpus; the RECOVERY after it is fully explained by the post-push
    velocity and pose, which are in the state, and is exactly the off-manifold coverage the
    push exists to buy. So the window comes out and the recovery stays.

    ROWS ARE NEVER DELETED IN PLACE. Training windows are built inside an episode as
    `length - sequence_length + 1`, so deleting rows from the middle silently produces
    windows that span a temporal jump -- a worse defect than the one being fixed, and an
    invisible one. The episode is SPLIT into independent segments instead, each contiguous
    by construction, each with its own id and length. `segments()` returns those row ranges.

    WHY THIS CLASS REFUSES
    ----------------------
    Splitting is expensive in a superlinear way. A 1,475-row episode yields 1,348 windows
    whole; split into seven ~190-row segments it yields 441, about a third. So the schedule
    has to be sized around the training sequence length rather than inherited from an
    interval, and `min_segment_rows` (512, >= 4x the sequence length) is a HARD constraint,
    not a preference.

    A scheduler that silently emitted short segments would be the worst outcome available:
    collection would succeed, the corpus would load, training would run, and the window
    count would be a third of what the run was planned around with nothing anywhere saying
    so. So an infeasible request raises at construction, before a single step is simulated,
    and the error says how many events would fit and how long the episode would have to be.

    With the yaml's own numbers -- 2 events, 0.10 s push, guard 2, min_segment_rows 512 at
    50 Hz -- an episode needs >= 1,556 rows, i.e. >= 31.1 s. A 30 s episode CANNOT host two
    recorded pushes. That is the constraint doing its job; the yaml's own preferred remedy
    is the warmup push, which is delivered before recording starts and therefore costs no
    segments at all.

    Draw order:
      __init__: [events_per_episode + 1 uniforms]  (the slack allocated to each gap)

    Always exactly that many, including when the slack is zero and the draws change nothing.
    """

    def __init__(
        self,
        duration_s: float,
        events_per_episode: int,
        rng: np.random.Generator,
        warmup_s: float = 0.0,
        push_duration_s: float = 0.10,
        guard_steps: int = 2,
        min_segment_rows: int = 512,
        dt: float = 0.02,
    ):
        if not (dt > 0.0):
            raise ValueError(f"dt must be positive, got {dt}")
        if not (duration_s > 0.0):
            raise ValueError(f"duration_s must be positive, got {duration_s}")
        if events_per_episode < 0:
            raise ValueError(f"events_per_episode must be >= 0, got {events_per_episode}")
        if guard_steps < 0:
            raise ValueError(f"guard_steps must be >= 0, got {guard_steps}")
        if min_segment_rows < 1:
            raise ValueError(f"min_segment_rows must be >= 1, got {min_segment_rows}")
        if push_duration_s <= 0.0:
            raise ValueError(f"push_duration_s must be positive, got {push_duration_s}")

        self.duration_s = float(duration_s)
        self.events_per_episode = int(events_per_episode)
        self.warmup_s = float(warmup_s)
        self.push_duration_s = float(push_duration_s)
        self.guard_steps = int(guard_steps)
        self.min_segment_rows = int(min_segment_rows)
        self.dt = float(dt)
        self.rng = rng

        self.n_rows = self._rows(self.duration_s)
        push_rows = int(math.ceil(self.push_duration_s / self.dt - 1e-9))

        # Rows lost to one event: the force window, the guard on both sides, and one spare
        # for the floor/ceil straddle when an event does not land exactly on the grid.
        # Events ARE snapped to the grid below, so the spare is never actually consumed --
        # it is budgeted anyway so that a caller who re-derives segments() off the grid
        # cannot turn a feasible schedule into an infeasible one.
        self.drop_rows = push_rows + 2 * self.guard_steps + 1

        warmup_rows = self._rows(self.warmup_s)
        # The first segment is bounded below by BOTH constraints: it must clear the warmup
        # exclusion and it must be a usable segment in its own right.
        self._first_min_rows = max(self.min_segment_rows, warmup_rows)
        self._gap_rows = self.drop_rows + self.min_segment_rows

        required = self._required_rows(self.events_per_episode)
        if self.n_rows < required:
            fits = max(0, (self.n_rows - self._first_min_rows) // self._gap_rows)
            raise ValueError(
                f"cannot fit {self.events_per_episode} push events in {self.duration_s:g} s "
                f"({self.n_rows} rows at dt={self.dt:g}) while keeping every segment "
                f">= {self.min_segment_rows} rows: needs {required} rows "
                f"(>= {required * self.dt:.2f} s). At this duration {fits} event(s) would "
                f"fit. Lengthen the episode, lower events_per_episode, or move the push into "
                f"the unrecorded warmup, where it costs no segments at all."
            )

        self.min_separation_s = self._gap_rows * self.dt
        self._event_rows = self._place(self.n_rows - required)
        # Written as (row + guard) * dt, one multiply, so the event time is bit-identical to
        # the sample time of the row it lands on. Computed as row*dt + guard*dt it would
        # differ by an ulp and `active()` would disagree with the row grid at the edges.
        self._event_times = [(row_lo + self.guard_steps) * self.dt
                             for row_lo in self._event_rows]

    # ----------------------------------------------------------------- construction
    def _rows(self, seconds: float) -> int:
        """Rows at times 0, dt, 2dt, ... strictly inside [0, seconds)."""
        return int(math.floor(max(0.0, seconds) / self.dt + 1e-9))

    def _required_rows(self, k: int) -> int:
        """Minimum episode length, in rows, for k events. k+1 segments and k drop windows."""
        return self._first_min_rows + k * self._gap_rows

    def _place(self, slack_rows: int) -> list[int]:
        """First dropped row of each event, jittered inside the available slack.

        The layout is built from the minimum feasible one and then loosened, so feasibility
        is structural rather than checked after the fact by a rejection loop -- a rejection
        loop would consume a random number of draws and break replay outright.

        Exactly `events + 1` uniforms are drawn regardless of slack, normalised into a
        partition of it. Drawing unconditionally keeps the count fixed; normalising means
        the jitter distribution does not change shape when the episode gets longer.
        """
        k = self.events_per_episode
        weights = self.rng.uniform(0.0, 1.0, size=k + 1)  # unconditional: see docstring
        if k == 0:
            return []

        total = float(weights.sum())
        if total <= 0.0:  # measure zero; keep it deterministic rather than dividing by zero
            share = np.full(k + 1, slack_rows / (k + 1))
        else:
            share = weights / total * slack_rows
        extra = np.floor(share).astype(int)

        rows: list[int] = []
        cursor = self._first_min_rows + int(extra[0])
        for i in range(k):
            rows.append(cursor)
            cursor += self.drop_rows + self.min_segment_rows + int(extra[i + 1])
        return rows

    # ------------------------------------------------------------------------ query
    def event_times(self) -> list[float]:
        """Start time of each push force window, seconds from the first recorded row.

        Snapped to the control grid, because the collector can only switch a force on at a
        control step anyway, and snapping makes the dropped-row arithmetic exact instead of
        conservative-by-one.
        """
        return list(self._event_times)

    def active(self, t: float) -> bool:
        """Is the force ON at time t? Half-open [t_event, t_event + push_duration_s).

        This is the question the collector asks each step in order to decide whether to
        apply the force. It deliberately EXCLUDES the guard band: the guard is a data-hygiene
        margin applied when carving segments, not part of the physical event. Conflating the
        two would either apply force during the guard or drop the guard from the drop set,
        and both failures are invisible in the resulting corpus.
        """
        t = float(t)
        for t0 in self._event_times:
            if t0 <= t < t0 + self.push_duration_s:
                return True
        return False

    def drop_ranges(
        self,
        guard_steps: int | None = None,
        dt: float | None = None,
    ) -> list[tuple[int, int]]:
        """Half-open row ranges removed around each event: force window plus guard."""
        g = self.guard_steps if guard_steps is None else int(guard_steps)
        step = self.dt if dt is None else float(dt)
        out = []
        for t0 in self._event_times:
            lo = int(math.floor(t0 / step + 1e-9)) - g
            hi = int(math.ceil((t0 + self.push_duration_s) / step - 1e-9)) + g
            out.append((lo, hi))
        return out

    def segments(
        self,
        duration_s: float | None = None,
        guard_steps: int | None = None,
        dt: float | None = None,
    ) -> list[tuple[int, int]]:
        """Half-open (start_row, end_row) ranges that survive after removing every push.

        Returns `events + 1` ranges for a feasible schedule. Each becomes an independent
        segment with its own id and length; no returned range spans a removed row, so
        training windows built inside one can never span a temporal jump.

        The three arguments default to the values the schedule was constructed with. They
        are re-passable because the caller that carves the corpus is not always the one that
        planned the episode -- a truncated episode is shorter than planned, and the honest
        thing is to carve against the length actually written rather than the length
        intended. Ranges are clamped to [0, rows) and empty ones dropped, so a truncation
        that lands inside a push window yields fewer segments rather than a negative one.
        """
        total = self.duration_s if duration_s is None else float(duration_s)
        step = self.dt if dt is None else float(dt)
        n_rows = int(math.floor(max(0.0, total) / step + 1e-9))

        out: list[tuple[int, int]] = []
        cursor = 0
        for lo, hi in self.drop_ranges(guard_steps=guard_steps, dt=dt):
            lo = max(0, min(lo, n_rows))
            hi = max(0, min(hi, n_rows))
            if lo > cursor:
                out.append((cursor, lo))
            cursor = max(cursor, hi)
        if n_rows > cursor:
            out.append((cursor, n_rows))
        return out


# ---------------------------------------------------------------------- initial state
# Names and widths of the randomised initial condition. The widths are NOT in the yaml,
# which gives only the bounds, so they are fixed here and recorded:
#   base_vel_mps   3  -- full linear velocity of the base, x, y and z.
#   base_tilt_rad  2  -- roll and pitch ONLY. Yaw is not tilt, and the state is yaw-free by
#                        construction (projected gravity from the quaternion, see
#                        params/transforms.py), so a randomised yaw would be an excitation
#                        the model cannot see -- inadmissible by the rule at the top of the
#                        yaml. Heading variation belongs in the command families.
_INITIAL_STATE_FIELDS: tuple[tuple[str, int], ...] = (
    ("joint_offset_rad", N_JOINTS),
    ("base_vel_mps", 3),
    ("base_tilt_rad", 2),
    ("base_height_offset_m", 1),
)


def sample_initial_state(rng: np.random.Generator, bounds: dict) -> dict:
    """Randomised initial condition. Uniform within each bound. 18 draws.

    Draw order: [12 joint offsets, 3 base velocities, 2 base tilts, 1 height offset].

    This is the cheapest admissible excitation there is. The initial condition is explicitly
    admissible under the rule at the top of the yaml -- there is no unexplained acceleration
    in starting somewhere off-nominal, because the state at the first recorded row simply IS
    that condition -- and unlike a push it costs no segments.

    Uniform rather than log-uniform: these are OFFSETS straddling zero, not scales, and a
    log-uniform draw is not even defined on an interval containing zero.

    `bounds` is the `initial_state` block of excitation.yaml, as a dict of
    `{name: [low, high]}`. Every one of the four fields must be present. A missing key is an
    error rather than a skipped draw, because a sampler whose draw COUNT depends on which
    keys happen to be in a config file is exactly the replay hazard LESSONS.md describes:
    an episode collected with one config would be un-replayable under another that differs
    only in a default.
    """
    missing = [name for name, _ in _INITIAL_STATE_FIELDS if name not in bounds]
    if missing:
        raise ValueError(
            f"initial_state bounds are missing {missing}. Every field must be present: "
            f"omitting one would change the draw count and silently break replay."
        )

    out: dict = {}
    for name, width in _INITIAL_STATE_FIELDS:
        lo, hi = (float(v) for v in bounds[name])
        if hi < lo:
            raise ValueError(f"{name} bounds are inverted: [{lo}, {hi}]")
        draw = rng.uniform(lo, hi, size=width)
        out[name] = float(draw[0]) if width == 1 else draw
    return out


# ------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    _FAILURES: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
        if not ok:
            _FAILURES.append(name)

    print(f"excitation.py self-test  (SAMPLER_VERSION={SAMPLER_VERSION}, "
          f"numpy {np.__version__})")
    print("-" * 78)

    # ------------------------------------------------------------------ OU process
    print("\nOUActionNoise")
    SIGMA, TAU, DT, N = 0.06, 0.15, 0.02, 200_000
    ou = OUActionNoise(SIGMA, TAU, DT, np.random.default_rng(20260920))
    traj = np.empty((N, N_JOINTS))
    for i in range(N):
        traj[i] = ou.step()

    mean = traj.mean(axis=0)
    std = traj.std(axis=0)
    # Effective sample size, allowing for the correlation: n / (2 * tau_steps).
    n_eff = N / (2.0 * TAU / DT)
    mean_tol = 5.0 * SIGMA / math.sqrt(n_eff)
    check("mean reverts to zero",
          bool(np.all(np.abs(mean) < mean_tol)),
          f"max|mean| = {np.abs(mean).max():.3e} rad  (5-sigma tol {mean_tol:.3e})")

    rel = np.abs(std - SIGMA) / SIGMA
    # Per-joint n_eff is only ~13k, so a per-joint spread of ~1% is sampling noise, not
    # bias. Pooling the twelve independent joints gives 12x the n_eff, which is tight
    # enough (~0.2%) to show that the exact discretisation is UNBIASED -- Euler-Maruyama
    # would sit 3.5% high here and would fail this line.
    pooled = float(traj.std())
    pooled_rel = abs(pooled - SIGMA) / SIGMA
    check("stationary std matches requested sigma",
          bool(rel.max() < 0.02) and pooled_rel < 0.01,
          f"sigma = {SIGMA:g}, per-joint {std.min():.5f}..{std.max():.5f} rad "
          f"(max rel err {rel.max() * 100:.2f}%, sampling noise at n_eff~{n_eff:.0f}); "
          f"pooled {pooled:.5f} = {pooled_rel * 100:.2f}% off "
          f"(Euler-Maruyama would be +3.5%)")

    # Mean reversion, stated as the property rather than assumed from the construction:
    # regressing the increment on the current value must give a negative slope, and that
    # slope must be -(1 - alpha).
    x0, dx = traj[:-1], traj[1:] - traj[:-1]
    slope = float((x0 * dx).sum() / (x0 * x0).sum())
    want_slope = -(1.0 - math.exp(-DT / TAU))
    check("increment regresses negatively on state (mean-reverting)",
          slope < 0.0 and abs(slope - want_slope) < 0.01,
          f"slope = {slope:.5f}, expected {want_slope:.5f}")

    lag1 = float(np.mean([np.corrcoef(traj[:-1, j], traj[1:, j])[0, 1] for j in range(N_JOINTS)]))
    check("lag-1 autocorrelation is exp(-dt/tau), i.e. NOT white noise",
          abs(lag1 - math.exp(-DT / TAU)) < 0.01,
          f"measured {lag1:.4f}, expected {math.exp(-DT / TAU):.4f}  "
          f"(i.i.d. noise would give 0.0)")

    check("bounded in practice",
          bool(np.abs(traj).max() < 6.0 * SIGMA),
          f"max|x| = {np.abs(traj).max() / SIGMA:.2f} sigma over {N * N_JOINTS:,} joint-steps")

    # Per-joint independence: the whole point is a full-rank residual, so an off-diagonal
    # correlation would defeat the class before any of the rest matters.
    corr = np.corrcoef(traj.T)
    off = np.abs(corr - np.eye(N_JOINTS)).max()
    check("joints are independent (full-rank excitation)",
          off < 0.02,
          f"max |off-diagonal correlation| = {off:.4f}")

    # Draw-count invariance: sigma = 0 must consume exactly as much stream as sigma > 0.
    def _stream_after_ou(sigma: float) -> float:
        g = np.random.default_rng(7)
        n = OUActionNoise(sigma, TAU, DT, g)
        for _ in range(50):
            n.step()
        return float(g.standard_normal())

    check("zero sigma consumes the same stream as non-zero (draw-count invariance)",
          _stream_after_ou(0.0) == _stream_after_ou(SIGMA),
          f"next value after 50 steps = {_stream_after_ou(0.0):.12f} either way")

    # --------------------------------------------------------------------- push
    print("\nsample_push")
    MAG_LO, MAG_HI, M = 8.0, 140.0, 200_000
    g = np.random.default_rng(31337)
    pushes = np.array([sample_push(g, MAG_LO, MAG_HI) for _ in range(M)])
    norms = np.linalg.norm(pushes, axis=1)
    dirs = pushes / norms[:, None]

    dmean = dirs.mean(axis=0)
    # Each component of a uniform direction has std 1/sqrt(3).
    iso_tol = 5.0 / math.sqrt(3.0 * M)
    check("push directions are isotropic (mean direction ~ 0)",
          bool(np.all(np.abs(dmean) < iso_tol)),
          f"mean = [{dmean[0]:+.5f}, {dmean[1]:+.5f}, {dmean[2]:+.5f}], "
          f"5-sigma tol {iso_tol:.5f}")

    # The old disc sampler is the thing this must not be: |z| <= 0.29 for every sample.
    zfrac = float(np.mean(np.abs(dirs[:, 2]) > 0.30))
    check("elevation is not a shallow disc",
          abs(zfrac - 0.70) < 0.01,
          f"{zfrac * 100:.1f}% of pushes have |cos(elevation)| > 0.30 "
          f"(uniform sphere: 70.0%, old +/-16.7 deg disc: 0.0%)")

    zstd = float(dirs[:, 2].std())
    check("z component is U(-1,1), so no preferred axis",
          abs(zstd - 1.0 / math.sqrt(3.0)) < 0.005,
          f"std(z) = {zstd:.5f}, U(-1,1) gives {1.0 / math.sqrt(3.0):.5f}")

    # Norm equals the sampled magnitude EXACTLY: re-derive the magnitude by replaying the
    # documented draw order on a fresh Generator. This tests the order as well as the norm.
    worst = 0.0
    for seed in range(2000):
        f = sample_push(np.random.default_rng(seed), MAG_LO, MAG_HI)
        h = np.random.default_rng(seed)
        _z = h.uniform(-1.0, 1.0)
        _t = h.uniform(0.0, 2.0 * math.pi)
        mag = float(np.exp(h.uniform(math.log(MAG_LO), math.log(MAG_HI))))
        worst = max(worst, abs(float(np.linalg.norm(f)) - mag) / mag)
    check("||f|| equals the sampled magnitude (documented draw order: z, theta, mag)",
          worst < 1e-14,
          f"max relative error {worst:.2e} over 2000 seeds "
          f"(un-normalised sampler: up to 4.4e-2)")

    logmid = float(np.mean(np.log(norms)))
    want_logmid = 0.5 * (math.log(MAG_LO) + math.log(MAG_HI))
    check("magnitude is log-uniform over the requested decade",
          abs(logmid - want_logmid) < 0.02
          and norms.min() >= MAG_LO - 1e-9 and norms.max() <= MAG_HI + 1e-9,
          f"mean log|f| = {logmid:.4f} (want {want_logmid:.4f}), "
          f"range {norms.min():.2f}..{norms.max():.2f} N")

    # ------------------------------------------------------------------- chirp
    print("\nSingleJointChirp")
    probe = SingleJointChirp(0.08, 0.5, 6.0, 2.0, rng=np.random.default_rng(5))
    ts = np.arange(0.0, 2.0, DT)
    vals = np.array([probe.value(t) for t in ts])
    others = np.delete(np.arange(N_JOINTS), probe.joint)
    check("exactly one joint is driven",
          bool(np.all(vals[:, others] == 0.0)) and float(np.abs(vals[:, probe.joint]).max()) > 0.0,
          f"joint {probe.joint} driven, other 11 identically zero")
    check("amplitude is the requested amplitude",
          abs(float(np.abs(vals[:, probe.joint]).max()) - 0.08) < 2e-3,
          f"peak |value| = {np.abs(vals[:, probe.joint]).max():.5f} rad (requested 0.080)")
    check("opens continuously and is zero outside the window",
          probe.value(0.0)[probe.joint] == 0.0
          and bool(np.all(probe.value(2.0) == 0.0))
          and bool(np.all(probe.value(-0.1) == 0.0)),
          "value(t0) = 0, value(t >= t0+T) = 0, value(t < t0) = 0")

    # Frequency really sweeps: count zero crossings in each half of the window.
    sig = vals[:, probe.joint]
    half = len(sig) // 2
    xa = int(np.sum(np.diff(np.sign(sig[:half])) != 0))
    xb = int(np.sum(np.diff(np.sign(sig[half:])) != 0))
    check("frequency sweeps upward (linear chirp, not a fixed tone)",
          xb > 2 * xa,
          f"zero crossings: {xa} in the first half, {xb} in the second")

    def _stream_after_chirp(joint) -> float:
        h = np.random.default_rng(11)
        SingleJointChirp(0.08, 0.5, 6.0, 2.0, joint=joint, rng=h)
        return float(h.standard_normal())

    check("explicit joint consumes the same stream as a drawn one (draw-count invariance)",
          _stream_after_chirp(None) == _stream_after_chirp(3),
          f"next value = {_stream_after_chirp(None):.12f} either way")

    # ----------------------------------------------------------- push schedule
    print("\nPushSchedule")
    MIN_SEG, GUARD, PUSH_S = 512, 2, 0.10

    raised = None
    try:
        PushSchedule(30.0, 2, np.random.default_rng(1), warmup_s=2.0,
                     push_duration_s=PUSH_S, guard_steps=GUARD,
                     min_segment_rows=MIN_SEG, dt=DT)
    except ValueError as exc:
        raised = str(exc)
    check("refuses 2 events in 30 s rather than emitting short segments",
          raised is not None,
          (raised or "NO EXCEPTION RAISED").split(". ")[0])

    raised3 = None
    try:
        PushSchedule(40.0, 3, np.random.default_rng(1), warmup_s=2.0,
                     push_duration_s=PUSH_S, guard_steps=GUARD,
                     min_segment_rows=MIN_SEG, dt=DT)
    except ValueError as exc:
        raised3 = str(exc)
    check("refuses 3 events in 40 s",
          raised3 is not None,
          (raised3 or "NO EXCEPTION RAISED").split(". ")[0])

    sched = PushSchedule(40.0, 2, np.random.default_rng(99), warmup_s=2.0,
                         push_duration_s=PUSH_S, guard_steps=GUARD,
                         min_segment_rows=MIN_SEG, dt=DT)
    check("2 events in 40 s is accepted",
          len(sched.event_times()) == 2,
          f"events at {[round(t, 3) for t in sched.event_times()]} s, "
          f"min separation {sched.min_separation_s:.2f} s")

    # The properties that matter, checked over many seeds rather than one lucky layout.
    SEEDS, EP_S = 300, 45.0
    n_rows = int(round(EP_S / DT))
    bad_force, bad_guard, bad_short, bad_count = 0, 0, 0, 0
    bad_warmup, bad_active, bad_selfoverlap = 0, 0, 0
    shortest, force_rows_kept = 10 ** 9, 0
    for seed in range(SEEDS):
        s = PushSchedule(EP_S, 2, np.random.default_rng(seed), warmup_s=2.0,
                         push_duration_s=PUSH_S, guard_steps=GUARD,
                         min_segment_rows=MIN_SEG, dt=DT)
        segs = s.segments(EP_S, GUARD, DT)
        if len(segs) != 3:
            bad_count += 1

        # Rows the force is actually ON for, straight off the public predicate.
        force = np.array([s.active(r * DT) for r in range(n_rows)])
        # Rows the schedule says to drop: force window plus guard on both sides.
        dropped = np.zeros(n_rows, dtype=bool)
        for lo, hi in s.drop_ranges(GUARD, DT):
            dropped[max(0, lo):max(0, hi)] = True

        kept = np.zeros(n_rows, dtype=bool)
        for lo, hi in segs:
            shortest = min(shortest, hi - lo)
            if hi - lo < MIN_SEG:
                bad_short += 1
            if kept[lo:hi].any():
                bad_selfoverlap += 1
            kept[lo:hi] = True

        force_rows_kept += int(np.count_nonzero(kept & force))
        bad_force += int(np.any(kept & force))
        bad_guard += int(np.any(kept & dropped))

        for t0 in s.event_times():
            if t0 < 2.0:
                bad_warmup += 1
            if not s.active(t0) or s.active(t0 + PUSH_S) or s.active(t0 - DT):
                bad_active += 1

    check("segments() never overlaps a push window",
          bad_force == 0 and force_rows_kept == 0,
          f"{force_rows_kept} surviving rows had the force on, across {SEEDS} schedules "
          f"({SEEDS * 3} segments, {SEEDS * n_rows:,} rows)")
    check("the guard band either side of the push is dropped too",
          bad_guard == 0,
          f"no surviving row falls in any drop range (guard_steps={GUARD})")
    check("segments do not overlap each other",
          bad_selfoverlap == 0, f"{SEEDS}/{SEEDS} schedules")
    check("every segment is at least min_segment_rows long",
          bad_short == 0,
          f"shortest segment over {SEEDS} schedules = {shortest} rows (min {MIN_SEG})")
    check("events + 1 segments, always",
          bad_count == 0,
          f"3 segments from 2 events, {SEEDS}/{SEEDS} schedules")
    check("no push lands inside the warmup exclusion",
          bad_warmup == 0,
          f"earliest event >= warmup_s in {SEEDS}/{SEEDS} schedules")
    check("active() is the force window, half-open, guard excluded",
          bad_active == 0,
          "active(t0) true, active(t0+duration) false, active(t0-dt) false")

    zero = PushSchedule(20.0, 0, np.random.default_rng(4), min_segment_rows=MIN_SEG, dt=DT)
    check("zero events yields one whole-episode segment",
          zero.segments() == [(0, 1000)] and zero.event_times() == [],
          f"segments = {zero.segments()}")

    # --------------------------------------------------------- scalar samplers
    print("\nsample_sigma / sample_initial_state")
    g = np.random.default_rng(2024)
    sig = np.array([sample_sigma(g, 0.005, 0.15) for _ in range(100_000)])
    check("sigma is log-uniform on [0.005, 0.15]",
          abs(float(np.mean(np.log(sig))) - 0.5 * (math.log(0.005) + math.log(0.15))) < 0.01
          and sig.min() >= 0.005 * (1 - 1e-12) and sig.max() <= 0.15 * (1 + 1e-12),
          f"range {sig.min():.5f}..{sig.max():.5f}, "
          f"mean log {np.mean(np.log(sig)):.4f} "
          f"(want {0.5 * (math.log(0.005) + math.log(0.15)):.4f})")

    BOUNDS = {
        "joint_offset_rad": [-0.15, 0.15],
        "base_vel_mps": [-0.4, 0.4],
        "base_tilt_rad": [-0.12, 0.12],
        "base_height_offset_m": [-0.04, 0.04],
    }
    g = np.random.default_rng(8)
    st = sample_initial_state(g, BOUNDS)
    ok_shapes = (st["joint_offset_rad"].shape == (12,)
                 and st["base_vel_mps"].shape == (3,)
                 and st["base_tilt_rad"].shape == (2,)
                 and isinstance(st["base_height_offset_m"], float))
    ok_bounds = all(np.all((np.atleast_1d(st[k]) >= BOUNDS[k][0]) &
                           (np.atleast_1d(st[k]) <= BOUNDS[k][1])) for k in BOUNDS)
    check("initial state has the declared widths and respects its bounds",
          ok_shapes and ok_bounds,
          "12 joint offsets, 3 base velocities, 2 tilts (roll, pitch), 1 height")

    g_a = np.random.default_rng(8)
    sample_initial_state(g_a, BOUNDS)
    g_b = np.random.default_rng(8)
    g_b.uniform(size=18)          # 12 + 3 + 2 + 1, in the documented order
    check("initial state consumes exactly 18 draws, in the documented order",
          float(g_a.standard_normal()) == float(g_b.standard_normal()),
          "stream position after the sampler is identical to 18 plain uniforms")

    missing_raised = False
    try:
        sample_initial_state(np.random.default_rng(0),
                             {k: v for k, v in BOUNDS.items() if k != "base_tilt_rad"})
    except ValueError:
        missing_raised = True
    check("a missing bound raises instead of skipping a draw",
          missing_raised,
          "an omitted field would change the draw count and break replay")

    print("-" * 78)
    if _FAILURES:
        print(f"FAILED: {len(_FAILURES)} check(s): {_FAILURES}")
        raise SystemExit(1)
    print("all checks passed")
