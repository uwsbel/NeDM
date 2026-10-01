"""One evaluation arm (``EvalConfig``) with its validated matrix, the TOML arm files, the machine paths (``Env``) and the
``data:`` resolver that checks every released input against the pinned release manifest.

``$NEDM_DATA`` is the release restore base, ``$NEDM_CHRONO_DATA`` the data folder of the Chrono build and
``$NEDM_RELEASE_CACHE`` the Hub mirror holding the tar items' index.csv.gz (default
``$NEDM_DATA/artifacts/hf_release/download``). A ``data:<restore path>`` reference is checked on first use against
traversing/manifests/hf_release_manifest.json; ``{arena}`` / ``{task}`` expand per task; other paths are absolute or
repo-relative. ``validate()`` returns every problem at once: refusals, combinations outside the validated matrix (run
only with ``allow_unvalidated``, stamped in every record) and, given an ``Env``, the input files.
"""

from __future__ import annotations

import csv
import fnmatch
import gzip
import hashlib
import json
import math
import os
import re
import tomllib
from collections import Counter
from dataclasses import dataclass, fields
from functools import cached_property
from pathlib import Path, PurePosixPath

from .routes import ANCHOR_SPEEDS

REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST = REPO_ROOT / 'traversing/manifests/hf_release_manifest.json'
DT = 0.05                                               # recording frame (s)
GROUNDS, UPDATES = ('rigid', 'soil'), ('cem', 'mppi')
PLANNERS = ('given', 'straight', 'cem', 'cem_grad', 'live')
CONTROLLERS = ('pid', 'pid_held', 'tracker', 'nav_pid')
LABELS = ('rollback', 'rollback_belly', 'tracker', 'mission', 'goal_belly')
MODEL_KINDS = ('ci_train', 'ga_train', 'legacy')        # what the planner adapters load; txjoint is refused
PICK_ARMS = {'straight': ('S',), 'cem': ('B',), 'cem_grad': ('B', 'G')}     # a locked pick folder's arms per planner
SOIL_CONFIGS = {'crm_main': 'data:artifacts/traverse/crm_f104_v1/configs/crm_main.json'}      # sha 90cd049e..., 1 ms
# Every key build_crm reads: recorded configs spell all out, so crm_collect.CRM_DEFAULT (0.5 ms!) is never merged in.
SOIL_KEYS = ('spacing_m depth_m step_s active_domain_m active_domain_delay_s side_walls mbs_threads tire_mesh '
             'soil.density soil.young_modulus_pa soil.poisson_ratio soil.mu_I0 soil.friction soil.average_diam_m '
             'soil.cohesion_pa sph.d0_multiplier sph.free_surface_threshold sph.artificial_viscosity '
             'sph.shifting_method sph.shifting_ppst_push sph.shifting_ppst_pull sph.num_proximity_search_steps').split()
ACTOR_V2 = 'data:artifacts/traverse/generalist_20260921/B_tracker/ppo_v2/actor.npz'           # sha ea938820...
NUM = (int, float)
TYPES = dict(name=str, vehicle=str, ground=str, soil_config=str, approach_s=NUM, decisions=str, planner=str, models=str,
             speed=NUM, rounds=int, samples=int, update=str, picks=str, replan=(str, *NUM), latency_replay=str,
             controller=str, actor=str, label=str, build_lock=str, allow_unvalidated=bool)


class ConfigError(ValueError):
    """Every problem of a config (or arm file) at once."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__('\n  - '.join(['invalid evaluation config:', *self.problems]))


class ReleaseError(ValueError):
    """An input file is missing or differs from the pinned release (or a suite lock does not recompute)."""


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_atomic(path, data):
    """`data` (str, bytes, or a function writing to the open file) to `path`: a fsynced .tmp, then os.replace."""
    tmp = Path(path).with_name(Path(path).name + '.tmp')
    with open(tmp, 'wb') as f:
        data(f) if callable(data) else f.write(data.encode() if isinstance(data, str) else data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def soil_config(ref, env) -> tuple[Path, dict]:
    """(path, content) of soil config `ref` ('crm_main' or JSON): all SOIL_KEYS, step_s divides 0.05 (crm_collect.py:201)."""
    p = env.file(SOIL_CONFIGS.get(ref, ref))
    c = json.loads(p.read_text())
    miss = [k for k in SOIL_KEYS if k.rpartition('.')[2] not in (c.get(k.partition('.')[0], {}) if '.' in k else c)]
    if not isinstance(step := c.get('step_s'), float) or step <= 0 or abs(round(DT / step) * step - DT) > 1e-12 or miss:
        raise ValueError(f'soil config {p}: missing {miss}; step_s {step!r} must be given and divide 0.05 s')
    return p, c


@dataclass(frozen=True)
class EvalConfig:
    """One arm = one results column (the switches: README.md)."""
    name: str
    vehicle: str = 'hmmwv'
    ground: str = 'rigid'
    soil_config: str | None = None
    approach_s: float = 0.0
    decisions: str | None = None
    planner: str = 'cem'
    models: str | None = None
    speed: float | None = None
    rounds: int = 4
    samples: int = 64
    update: str = 'cem'
    picks: str | None = None
    replan: str | float | None = None
    latency_replay: str | None = None
    controller: str = 'pid'
    actor: str | None = None
    label: str = 'rollback'
    build_lock: str | None = None
    allow_unvalidated: bool = False

    def __post_init__(self):            # 3 and 3.0 are one config (one sha)
        for k in ('approach_s', 'speed', 'replan'):
            if type(getattr(self, k)) is int:
                object.__setattr__(self, k, float(getattr(self, k)))

    @property
    def tag(self) -> str | None:
        """Planner rng tag. Sampling: 'n2iter_cem<R>x<N>[_fixed<v>]' (planner_arms.py:29-38); live: mode + str(period),
        the waypoint arm ran with period 2.0 (nav_runner.py:145, nav_tasks.py:6); None for the rng-free planners."""
        if self.planner in ('given', 'straight'):
            return None
        if self.planner == 'live':
            return 'waypoint2.0' if self.replan == 'waypoint' else f'periodic{self.replan}'
        return f'n2iter_cem{self.rounds}x{self.samples}' + ('' if self.speed is None else f'_fixed{self.speed:g}')

    def seed(self, task_id: str) -> int:
        """The planner rng seed of a task: md5(task id + tag)[:8] (f104_n2_iter.py:25-26; one M1 stream per mission)."""
        if self.tag is None:
            raise ValueError(f'planner {self.planner} uses no rng')
        return int(hashlib.md5((task_id + self.tag).encode()).hexdigest()[:8], 16)

    @property
    def F(self) -> int:                 # decision frame of the two-pass approach protocol
        return round(self.approach_s / DT)

    @property
    def near_stop_rule(self) -> bool:   # the 40 s near-stop rule applies (crm_collect_ext.py:330-339)
        return self.controller in ('pid_held', 'tracker')

    @property
    def validated(self) -> bool:
        return not _types(self) and not _matrix(self)[1]

    @property
    def sha(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d) -> EvalConfig:
        bad = [f'unknown key {k!r}' for k in sorted(set(d) - set(TYPES))] + ([] if 'name' in d else ['name is required'])
        if bad:
            raise ConfigError(bad)
        return cls(**d)

    def validate(self, env: Env | None = None, *, tasks=(), model_info=None) -> list[str]:
        """Every problem at once. With ``env`` also the inputs: released files exist and match (a templated reference once
        per distinct expansion over ``tasks``), soil config, decision states, locked picks and the task files the arm
        reads, and the models' kind and training ground: ``model_info`` = planner.model_info (the planner owns torch)."""
        problems = _types(self)
        if problems:
            return problems
        refused, off = _matrix(self)
        if env is not None:
            refused += _inputs(self, env, tasks, model_info)
        return refused + ([] if self.allow_unvalidated else
                          [f'outside the validated matrix (allow_unvalidated = true runs it, stamped): {m}' for m in off])


def _types(c: EvalConfig) -> list[str]:
    def wrong(v, t):                    # a bool is an int, not a number here
        return not isinstance(v, t) or isinstance(v, bool) and t is not bool
    return [f'{k} has type {type(v).__name__} ({v!r})' if wrong(v, t) else f'{k} = {v!r} is not finite'
            for k, t in TYPES.items() for v in [getattr(c, k)]
            if (v is not None or EvalConfig.__dataclass_fields__[k].default is not None)
            and (wrong(v, t) or isinstance(v, float) and not math.isfinite(v))]


def _matrix(c: EvalConfig) -> tuple[list[str], list[str]]:
    """(refused, off-matrix) from the fields alone; the types are already checked."""
    R, U = [], []

    def rule(ok, msg, to=R):
        if not ok:
            to.append(msg)

    from .vehicles import VEHICLES                      # imported here: vehicles imports this module
    v, g, p, ctl, lab, a, sc = c.vehicle, c.ground, c.planner, c.controller, c.label, c.approach_s, c.soil_config
    no_search = p in ('given', 'straight', 'live')     # planners without the sampling search
    rule(v in VEHICLES, f'vehicle {v!r} is not one of {tuple(VEHICLES)}' + (
        ': the M113 is a smoke-only tracked stand-in; recount its released drives (vehicle_smoke_test_drives)'
        if v.startswith('m113') else ''))
    for k, allowed in (('ground', GROUNDS), ('planner', PLANNERS), ('update', UPDATES), ('controller', CONTROLLERS),
                       ('label', LABELS)):
        rule(getattr(c, k) in allowed, f'{k} {getattr(c, k)!r} is not one of {allowed}' + (
            ' (MPPI: planner "cem" with update = "mppi"; gradient refinement: "cem_grad")' if k == 'planner' else ''))
    rule(re.fullmatch(r'[A-Za-z0-9]+(_[A-Za-z0-9]+)*', c.name), f'name {c.name!r}: letters, digits, single underscores')
    # vehicle x ground
    rule(not (v.startswith('polaris') and g == 'rigid'), f'{v} on rigid ground: no rigid Polaris path (ov_vehicle.py)')
    rule(not (v == 'gator' and g == 'rigid'), 'gator on rigid ground (context rows only)', U)
    rule(v not in ('polaris_4wd', 'polaris_w08'), f'{v} is a smoke-only variant', U)
    # ground
    rule(g != 'soil' or sc, 'soil needs soil_config: the collector default step (0.5 ms) is never used silently '
         '(crm_collect.py:38)')
    rule(g == 'soil' or sc is None, 'soil_config is for soil only')
    rule(sc is None or sc in SOIL_CONFIGS or sc.endswith('.json'), f'soil_config {sc!r}: "crm_main" or a JSON path')
    rule(sc in (None, 'crm_main'), f'soil config {sc} (sha recorded)', U)
    # decision
    rule(0.0 <= a < 120.0 and abs(a / DT - round(a / DT)) < 1e-6, f'approach_s {a}: a multiple of 0.05 s in [0, 120)')
    rule(a == 0.0 or not no_search, f'approach_s with planner {p}: the two-pass protocol plans at F (model planners)')
    rule(a in (0.0, 0.5, 1.0, 3.0), f'approach_s {a} (recorded: 0.5, 1, 3 s)', U)
    rule(a == 0.0 or v == 'hmmwv', f'an approach on the {v}', U)
    rule(c.decisions is None or a > 0.0, 'decisions are moving-start states: set approach_s')
    # planner
    rule(p in ('given', 'straight') or c.models, f'planner {p} needs models')
    rule(p not in ('given', 'straight') or c.models is None, f'models are not used by planner {p}')
    rule(c.speed is None or p in ('straight', 'cem'), f'speed with planner {p} (the gradient chain drives the free '
         'family only, ci_grad.py)')
    rule(c.speed is None or 0.5 <= c.speed <= 6.0, f'speed {c.speed}: route speeds are clipped to [0.5, 6] m/s')
    rule(p != 'straight' or c.speed in (None, *ANCHOR_SPEEDS), f'straight at {c.speed} m/s: the straight route is the '
         f'offset-0 anchor at one of {ANCHOR_SPEEDS} m/s (routes.straight)')
    rule(c.rounds >= 2 and c.samples >= 1, 'rounds must be >= 2 and samples >= 1 (the single-goal one-shot pools of '
         'arms A and D are dropped)')
    rule(not no_search or (c.rounds, c.samples, c.update) == (4, 64, 'cem'),
         f'rounds / samples / update are not used by planner {p}')
    rule(c.picks is None or p in PICK_ARMS, f'locked picks with planner {p}')
    rule(c.picks is None or a == 0.0 or c.decisions, 'locked picks after an approach need decisions: the released '
         'states they were planned at (Pick.locked checks each pose)')
    if p in ('cem', 'cem_grad'):
        rule((c.rounds, c.samples) == (4, 64), f'{c.rounds} x {c.samples} sampling (recorded: 4 x 64)', U)
        rule(c.update == 'cem', 'update mppi (every headline used cumulative-elite CEM)', U)
        rule(c.speed is None or (c.speed == 2.0 and g == 'rigid'),
             f'fixed-speed family at {c.speed} m/s on {g} (recorded: 2 m/s on rigid)', U)
    rule(p != 'straight' or c.speed in (None, 6.0) or (c.speed == 2.0 and g == 'rigid'),
         f'straight at {c.speed} m/s on {g} (recorded: 6 m/s; 2 m/s on rigid)', U)
    # live (M1)
    if p == 'live':
        rule(v == 'hmmwv' and g == 'rigid', f'live on the {v} on {g}: M1 ran the HMMWV on rigid ground only')
        rule(ctl == 'nav_pid', f'live with controller {ctl}: the mission loop is built around nav_pid (SpeedPI)')
        rule(c.build_lock, 'live needs build_lock (the OptiX depth-FOV fix 9e1a0448b)')
        rule(c.replan is not None, 'live needs replan: "waypoint" or a period in s')
    rule(c.replan is None or p == 'live', 'replan is for planner live only')
    rule(c.latency_replay is None or p == 'live', 'latency_replay is for planner live only')
    rule(c.replan in (None, 'waypoint') or (not isinstance(c.replan, str) and c.replan > 0.0
                                            and abs(c.replan / DT - round(c.replan / DT)) < 1e-6),
         f'replan {c.replan!r}: "waypoint" or a positive multiple of 0.05 s')
    rule(c.latency_replay is None or not isinstance(c.replan, str), 'latency_replay needs a periodic replan')
    rule(c.replan in (None, 'waypoint', 2.0, 1.0), f'replan {c.replan} (recorded: waypoint, 2 s, 1 s)', U)
    rule(c.latency_replay is None or c.replan == 1.0, 'latency replay with a period other than 1 s', U)
    # controller
    rule(ctl != 'nav_pid' or p == 'live', 'nav_pid is the M1 mission controller (planner live)')
    rule(ctl != 'tracker' or v == 'hmmwv', f'tracker on the {v}: only driven on the HMMWV (Polaris engine columns are '
         'about 1/16 scale; m3_tracker_routes.csv has no vehicle column)')
    rule((ctl == 'tracker') == (c.actor is not None), 'the tracker needs actor; actor is for the tracker only')
    rule(ctl not in ('pid_held', 'tracker') or (v, a, p) == ('hmmwv', 0.0, 'given'),
         f'{ctl} on planned routes, after an approach or off the HMMWV', U)
    rule(c.actor in (None, ACTOR_V2), 'an actor other than the released round-2 actor', U)
    # label
    rule(lab not in ('rollback_belly', 'goal_belly') or v in VEHICLES and VEHICLES[v].belly,
         f'{lab} on the {v}: it has no belly points')
    rule((lab == 'mission') == (p == 'live'), 'label mission <=> planner live')
    planned = p in ('straight', 'cem', 'cem_grad')
    ok = {'rollback': planned and v in ('hmmwv', 'gator'), 'rollback_belly': planned and v in ('polaris', 'polaris_pc'),
          'tracker': p == 'given' and v == 'hmmwv', 'goal_belly': p == 'given'}
    rule(ok.get(lab, True), f'label {lab} with planner {p} on the {v}', U)
    rule(c.build_lock, f'no build_lock: no parity-checked {g} build pinned (configs/traversing/evaluation/locks)', U)
    return R, U


def _inputs(c: EvalConfig, env: Env, tasks, model_info) -> list[str]:
    """Problems of the input files, one message per failing reference."""
    P = []

    def attempt(fn, *args):
        try:
            return fn(*args)
        except Exception as e:          # noqa: BLE001  every failure becomes one reported problem
            P.append(f'{type(e).__name__}: {e}')

    def per_task(ref):                  # a templated reference once per distinct expansion over the tasks
        if '{' in ref and not tasks:
            P.append(f'{ref} is templated: validate with the suite tasks')
        return list((attempt(lambda: {env.expand(ref, t): t for t in tasks}) or {}).values()) if '{' in ref else [None]

    world = {'soil': 'crm', 'rigid': 'rigid'}.get(c.ground)
    if bad := [t.id for t in tasks if (t.kind == 'mission') != (c.planner == 'live')]:
        P.append(f'{len(bad)} tasks: planner live drives missions, the other planners single-goal tasks (e.g. {bad[:3]})')
    if c.models:
        ckpts = attempt(env.glob, c.models) or []
        info = [i for i in (attempt(model_info, q) for q in ckpts) if i] if model_info is not None else []
        kinds, trained = Counter(k for k, _ in info), {d for _, d in info} - {None}    # legacy Net: no domain_filter
        if model_info is None:
            P.append(f'models {c.models}: kind and training ground not checked (pass model_info=planner.model_info)')
        elif len(kinds) > 1 or not set(kinds) <= set(MODEL_KINDS):
            P.append(f'models {c.models}: kinds {dict(kinds)}, need one of {MODEL_KINDS} (txjoint: context only)')
        elif kinds and c.planner == 'cem_grad' and 'ci_train' not in kinds:
            P.append(f'cem_grad needs a ci_train ensemble (refine.DiffEnsemble), got {set(kinds)}')
        elif kinds and c.planner == 'live' and 'legacy' not in kinds:
            P.append(f'live needs the legacy direct-depth ensemble, got {set(kinds)}')
        if trained - {world, 'both'}:           # ag_picks.py:191-192 (legacy models record no domain_filter)
            P.append(f'models {c.models} were trained on {sorted(trained)} (domain_filter), not for {c.ground}')
    if c.decisions:
        if not re.match(rf'poses_{world}(_all)?\.json$', PurePosixPath(c.decisions).name):
            P.append(f'decisions {c.decisions}: not a poses_{world}[_all].json file of ground {c.ground}')
        poses = attempt(lambda: json.loads(env.file(c.decisions).read_text()))
        if poses is not None:
            if set(frames := Counter(e.get('frame') for e in poses.values())) != {c.F}:
                P.append(f'decisions {c.decisions}: frames {dict(frames)}, approach_s {c.approach_s} needs F = {c.F}')
            if missing := [t.id for t in tasks if t.id not in poses]:
                P.append(f'decisions {c.decisions}: no state for {len(missing)} tasks (e.g. {missing[:3]})')
    for t in per_task(c.picks) if c.picks else ():      # the pick folder was made for this arm (Pick.locked: each pose)
        s = attempt(lambda: json.loads(env.file(c.picks.rstrip('/') + '/summary.json', t).read_text())) or {}
        made = (s.get('world'), _tail(s.get('models')), s.get('mode') if c.planner == 'straight' else None,
                tuple(s.get('arms', ())))
        if s and made != (world, _tail(c.models), f'straight{c.speed or 6.0:g}' if c.planner == 'straight' else None,
                          PICK_ARMS.get(c.planner)):
            P.append(f'picks {env.expand(c.picks, t)} were made for (world, models, mode, arms) {made}, not this arm')
    for ref in filter(None, (c.actor, c.latency_replay)):
        for t in per_task(ref):
            attempt(env.file, ref, t)
    if c.soil_config:
        attempt(soil_config, c.soil_config, env)
    for f, on in (('route', c.planner == 'given'), ('approach', c.approach_s > 0),
                  ('map', c.planner in ('cem', 'cem_grad') and not c.picks)):
        if on and (miss := [t.id for t in tasks if getattr(t, f) is None]):
            P.append(f'{len(miss)} tasks have no {f}, which this arm reads (e.g. {miss[:3]})')
    if c.build_lock and not attempt(lambda: env.path(c.build_lock).is_file()):
        P.append(f'build lock {c.build_lock} not found (written by scripts/traversing/evaluation/fingerprint.py)')
    return P


def _tail(ref):                         # a model glob from 'artifacts/' on (summaries hold absolute or repo paths)
    return ref[ref.find('artifacts/'):] if ref and 'artifacts/' in ref else ref


def load_arms(path, ground: str | None = None) -> list[EvalConfig]:
    """The arms of a TOML file: ``[defaults]`` merged under each ``[[arm]]``; (name, ground) unique within the file.
    ``ground`` keeps the arms of one ground; it is required when a name is used on both grounds (a run's output
    folders are keyed by arm name, and the two grounds run on different Chrono builds)."""
    doc, P, arms = tomllib.loads(Path(path).read_text()), [], []
    P += [f'unknown top-level key {k!r} (only [defaults] and [[arm]])' for k in sorted(set(doc) - {'defaults', 'arm'})]
    defaults, rows = doc.get('defaults', {}), doc.get('arm', [])
    if not isinstance(defaults, dict) or not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise ConfigError([f'{path}: [defaults] must be a table and the arms [[arm]] tables'])
    for i, row in enumerate(rows):
        try:
            arms.append(EvalConfig.from_dict({**defaults, **row}))
        except ConfigError as e:
            P += [f'arm {i}: {m}' for m in e.problems]
    P += [f'arm {k} appears {n} times' for k, n in Counter((c.name, c.ground) for c in arms).items() if n > 1]
    if ground is not None:
        P += [] if ground in GROUNDS else [f'ground {ground!r} is not one of {GROUNDS}']
        arms = [c for c in arms if c.ground == ground]
    elif both := sorted({c.name for c in arms if c.ground == 'soil'} & {c.name for c in arms if c.ground == 'rigid'}):
        P.append(f'arms {both} exist on both grounds: choose one (ground=, --ground rigid | soil)')
    if P or not arms:
        raise ConfigError([f'{path}: {m}' for m in P or [f'no [[arm]] tables (ground {ground})']])
    return arms


class Release:
    """(sha256, bytes) of every restore path of the pinned release: plain files from the manifest; tar members from their
    item's index.csv.gz in the Hub mirror ``cache`` (the index's own sha256 is pinned by the manifest)."""

    def __init__(self, data: Path, cache: Path, manifest: Path = MANIFEST):
        m = json.loads(Path(manifest).read_text())
        self.data, self.cache, self.prefixes = Path(data), Path(cache), m['path_remap']['prefixes']
        self.plain, self.indexes, self._members, self._ok = {}, [], {}, {}
        for name, it in m['items'].items():
            head = f"traversing/{it['bundle']}/{name}/"
            for hub in it['files']:
                meta = m['files'][hub]
                if meta['kind'] == 'file':
                    self.plain[str(PurePosixPath(it['restore_root'], hub[len(head):]))] = (meta['sha256'], meta['bytes'])
                elif hub == head + 'index.csv.gz':        # the tar member list (ids.txt.gz is kind 'index' too)
                    self.indexes.append((it['restore_root'], hub, meta['sha256']))
        self.indexes.sort(key=lambda t: -len(t[0]))        # most specific restore root first

    def record(self, rel: str) -> tuple[str, int]:
        """(sha256, bytes) of a restore path; raises if it is not released or its item index is not in the cache."""
        if rel in self.plain:
            return self.plain[rel]
        missing = []
        for root, hub, sha in self.indexes:
            if root and not rel.startswith(root + '/'):
                continue
            if hub not in self._members:
                p = self.cache / hub
                if not p.is_file():
                    missing.append(hub)
                    continue
                if sha256_file(p) != sha:
                    raise ReleaseError(f'{p}: sha256 differs from the pinned manifest')
                with gzip.open(p, 'rt', newline='') as f:
                    self._members[hub] = {r['path']: (r['sha256'], int(r['bytes'])) for r in csv.DictReader(f)}
            if rel in self._members[hub]:
                return self._members[hub][rel]
        hint = f'; item indexes not in {self.cache} (set NEDM_RELEASE_CACHE): {missing}' if missing else ''
        raise ReleaseError(f'{rel} is not a file of the pinned release{hint}')

    def file(self, rel: str) -> Path:
        """``data / rel`` after checking its size and sha256 against the release (once per file state)."""
        sha, nbytes = self.record(rel)
        p = self.data / rel
        if not p.is_file():
            raise ReleaseError(f'{p}: release file missing (restore it with download_traversing_data.py)')
        st = p.stat()
        if self._ok.get(p) != (st.st_size, st.st_mtime_ns):
            if st.st_size != nbytes or sha256_file(p) != sha:
                raise ReleaseError(f'{p}: content differs from the release record (sha256 {sha[:12]}...)')
            self._ok[p] = (st.st_size, st.st_mtime_ns)
        return p

    def glob(self, pattern: str) -> list[Path]:
        """Released files matching a glob in the last path component: every local match must be a checked release file
        and every plain release file that matches must be present (no silently smaller ensemble)."""
        d, name = pattern.rpartition('/')[::2]
        if any(ch in d for ch in '*?['):
            raise ReleaseError(f'{pattern}: wildcards only in the last path component')
        rels = {f'{d}/{q.name}' for q in (self.data / d).glob(name)} | {
            r for r in self.plain if r.rpartition('/')[0] == d and fnmatch.fnmatchcase(r.rpartition('/')[2], name)}
        if not rels:
            raise ReleaseError(f'{pattern} matches no release file')
        return [self.file(r) for r in sorted(rels)]

    def remap(self, path: str) -> str:
        """A path written in a released task file -> its restore path (path_remap.prefixes, longest key wins)."""
        k = max((k for k in self.prefixes if path == k or (k.endswith('/') and path.startswith(k))), key=len, default=None)
        return path if k is None else self.prefixes[k] + path[len(k):]


@dataclass(frozen=True)
class Env:
    """Machine paths only (``from_environ``); ``release`` checks ``data:`` files against the pinned manifest."""
    data: Path
    chrono_data: Path | None = None
    device: str = 'cuda'
    release_cache: Path | None = None

    @classmethod
    def from_environ(cls, environ=None) -> Env:
        e = os.environ if environ is None else environ
        dirs = {k: Path(e[k]) for k in ('NEDM_DATA', 'NEDM_CHRONO_DATA', 'NEDM_RELEASE_CACHE') if e.get(k)}
        P = [f'{k}={p} is not a directory' for k, p in dirs.items() if not p.is_dir()]
        P += [] if 'NEDM_DATA' in dirs else ['NEDM_DATA (the release restore base) is not set']
        if P:
            raise ConfigError(P)
        return cls(dirs['NEDM_DATA'], dirs.get('NEDM_CHRONO_DATA'), e.get('NEDM_DEVICE', 'cuda'),
                   dirs.get('NEDM_RELEASE_CACHE'))

    @cached_property
    def release(self) -> Release:
        return Release(self.data, self.release_cache or self.data / 'artifacts/hf_release/download')

    @staticmethod
    def expand(ref: str, task=None) -> str:
        """``{arena}`` / ``{task}`` from ``task`` (.arena, .id)."""
        if '{' not in ref:
            return ref
        if task is None:
            raise ConfigError([f'{ref} needs a task for its placeholders'])
        return ref.format_map({'arena': task.arena, 'task': task.id})

    def path(self, ref: str, task=None) -> Path:
        """A reference as a path (no checks)."""
        ref = self.expand(ref, task)
        return self.data / ref[5:] if ref.startswith('data:') else REPO_ROOT / ref

    def file(self, ref: str, task=None) -> Path:
        """A reference to one file: release-checked for ``data:``, else it must exist."""
        ref = self.expand(ref, task)
        if ref.startswith('data:'):
            return self.release.file(ref[5:])
        if not (p := REPO_ROOT / ref).is_file():
            raise ReleaseError(f'{p} not found')
        return p

    def glob(self, ref: str) -> list[Path]:
        """Sorted files of a glob reference (release-checked for ``data:``); no match is an error."""
        if ref.startswith('data:'):
            return self.release.glob(ref[5:])
        base = REPO_ROOT / ref
        if not (out := sorted(q for q in base.parent.glob(base.name) if q.is_file())):
            raise ReleaseError(f'{ref} matches no file')
        return out
