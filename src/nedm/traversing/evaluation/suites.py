"""Tasks (WHERE: one pair, route or mission) and the released suites, each checked against its recorded lock on load.

A ``Task`` keeps the exact original id: it seeds md5(id + tag), keys the pairing and the result tables. ``load_suite``
reads a released suite under ``Env.data`` and verifies every file it hands out; the Task pins each file's sha256 as
checked, so a worker (``Task.from_dict`` with its own Env) re-checks every file it reads (``read_case``, ``read_route``):
- ``f104_800`` (M2, M4b): SUITE_LOCKED.sha256 4327c110... over suite.json + tasks_crm.json + tasks_rigid.json
  (ga_suite.py:317-327); suite.json pins each case and route_00. The approach route of each group (the a5 and a5data
  folders hold the same content) is pinned by content hash in approach_suite_index.json (sha c2e3474a...).
- ``unseen_soil1000`` / ``unseen_rigid2000`` (M4a, M4b): ALL_LOCKED c2d0022c... (E1: g247 g251 g260 g271) and
  ALL_LOCKED_E1b e2909946... (E1b: g241 g258 g263 g268) pin each arena's SUITE_LOCKED, which pins every case and
  route (ag_suite_lock.py:34-40). Soil drives the 125 lowest md5(group) per arena, asserted equal to
  soil_unseen_subset.json (SUBSETS_LOCKED_E1b 05039510...).
- ``tracker423`` (M3): tracking_suite.json a53b0eb6... pins each case and reference route.
- ``missions30`` (M1): tasks_main.json ea670e3e... names the 30 missions; mission files are release-checked.
- ``smoke144`` (M4b smoke): sample A (sample_A.json afb0607f...) of smoke_v2_polaris.json (a4eaf300...); its cases and
  routes are release-checked (they are tar members: the item index must be in the release cache).
- ``custom:<dir>``: cases written by ``make_case`` (or copies of released cases), checked but not locked.
Every arena must be 80 m and its planner map rendered from its BMP (ag_map_check). Given, approach and custom route_00
routes must start and end within 0.25 m of the case start and goal (crm_collect.py:199-200).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field, fields
from pathlib import Path, PurePosixPath

import numpy as np

from .config import ConfigError, Env, ReleaseError, sha256_file, write_atomic
from .routes import ENDS_TOL_M, KEYS, ends_within, load_route, route_sha256

T = 'artifacts/traverse/'
F104_SUITE, APPROACH = T + 'generalist_20260921/A_adapt/suite/', T + 'crm_improve_20260922/a5data/approach_suite'
UNSEEN_LOCKS, NAV = T + 'arena_gator_20260925/suites/', T + 'fdm_f104_50h_20260909/nav_v1/'
LOCKS = dict(f104_800='4327c110b962631ca6b4f849e049ff55320d748e0a0db62df35cc1953e910763',          # SUITE_LOCKED head
             approach='c2e3474a1dc5cc86a17b00ee1e909115e265baa050408c4e8395e231fcd2a040',          # index file sha
             E1='c2d0022c5a31f44ef55129895e4eedea403e59abfe5635beb3dd98f5daf8c334',                 # ALL_LOCKED head
             E1b='e29099465eb7071c97174e821eb8f309b40901491e4c7a5cce8e02d40cf1e7fe',                # ALL_LOCKED_E1b head
             subsets='05039510ceccda33781f895c6e4ef2258001019a6a362cdb0fc0447ad79d6f25',            # SUBSETS_LOCKED_E1b
             tracker423='a53b0eb69381e7b437c6afbe3e339c1671c28c727c10f2c5912472368b9ca2e5',         # tracking_suite.json
             missions30='ea670e3ec84f4ef5664113807b50f34acaedfa92a79244068fecb4798ccd9f2b',         # tasks_main.json
             smoke_tasks='a4eaf3001355a14e925e01d1bdb73dd2a23ce00a6c43b70719f6fd21799e7cc3',        # smoke_v2_polaris
             smoke144='afb0607f659b7bbbc989c8b9428d32728bece9943839d43b3e21ab18c7dffdec')           # sample_A.json
UNSEEN = dict(g260='E1', g271='E1', g251='E1', g247='E1', g258='E1b', g268='E1b', g263='E1b', g241='E1b')
MAPS = {'f104': T + 'crm_f104_v1/maps/arena_f104_50h_v1',
        **{g: T + f'arena_gator_20260925/maps/arena_{g}' for g in ('g203', 'g217', 'g228', *sorted(UNSEEN))}}
SPLITS, SOIL_PER_ARENA, _ARENAS = ('train', 'val', 'test'), 125, {}
FILES = ('case', 'route00', 'route', 'approach', 'map')        # Task path fields, each pinned in Task.sha


@dataclass(frozen=True)
class Task:
    """WHERE: one pair (single goal) or mission. ``sha`` pins every file field as load_suite checked it (map: its
    observation.npz); ``meta`` holds table columns only; no drive reads it."""
    id: str
    arena: str                          # 'f104' -> assets/traverse/arena_f104_50h_v1, else assets/traverse/arena_<arena>
    case: Path                          # layout.start_xy/start_yaw, goal_xy, goal_radius_m, split, assets == []
    route00: Path | None = None         # planner base at a standing start (None -> base_route(layout, goal))
    route: Path | None = None           # given route (tracker, smoke, custom trajectory)
    approach: Path | None = None        # released approach route (content sha pinned)
    goals: tuple | None = None          # mission waypoints (M1)
    map: Path | None = None             # static_map_v1 folder of the arena (planner input), None if not released
    meta: dict = field(default_factory=dict, compare=False)
    sha: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        if missing := [f for f in FILES if getattr(self, f) is not None and not self.sha.get(f)]:
            raise ValueError(f'task {self.id}: no sha256 for {missing}')

    @property
    def kind(self) -> str:
        return 'mission' if self.goals else 'single_goal'

    def read_case(self) -> dict:
        """The case (a mission file for M1), its sha256 re-checked."""
        return _json(self.case, self.sha['case'])

    def read_route(self, name: str) -> dict:
        """'route00' | 'route' | 'approach' as float64 arrays, meta cleared (routes.load_route), sha re-checked."""
        return load_route(_json(getattr(self, name), self.sha[name]))

    def to_dict(self, env: Env) -> dict:
        """JSON form; a path under env.data is written as 'data:<restore path>' (portable to another machine)."""
        def rel(p):
            return f'data:{p.relative_to(env.data).as_posix()}' if p.is_relative_to(env.data) else str(p)
        return {f.name: rel(v) if isinstance(v := getattr(self, f.name), Path) else v for f in fields(self)}

    @classmethod
    def from_dict(cls, d, env: Env) -> Task:
        def path(v):
            return env.data / v[5:] if v.startswith('data:') else Path(v)
        return cls(**{k: path(v) if k in FILES and v is not None else tuple(map(tuple, v)) if k == 'goals' and v
                      else v for k, v in d.items()})


def md5hex(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def arena_dir(arena: str, env: Env) -> Path:
    return env.data / 'assets/traverse' / ('arena_f104_50h_v1' if arena == 'f104' else f'arena_{arena}')


def load_suite(name: str, subset: str | None = None, env: Env | None = None) -> list[Task]:
    """The tasks of a released suite (or ``custom:<dir>``), locks recomputed and files checked; ``subset``: see select."""
    env = env or Env.from_environ()
    if name.startswith('custom:'):
        tasks = _custom(Path(name[7:]).resolve(), env)
    elif name in SUITES:
        tasks = LOADERS[name](name, env)
    else:
        raise ConfigError([f'suite {name!r} is not one of {SUITES} or custom:<dir>'])
    if len({t.id for t in tasks}) != len(tasks):
        raise ReleaseError(f'{name}: duplicate task ids')
    return select(tasks, subset)


def select(tasks, subset: str | None) -> list[Task]:
    """``key=v1|v2,...`` keeps tasks whose id, arena or meta[key] is one of the values; ``lowest_md5=N`` keeps the N lowest
    md5(id) hex digests (the outcome-blind rule of the recorded checks, select_groups.py:14-16), ``lowest_md5_per_arena=N``
    per arena. Terms apply left to right; suite order is kept. Unknown keys, values matching nothing and N larger than
    the pool are errors."""
    tasks = list(tasks)
    for term in (subset.split(',') if subset else ()):
        k, _, v = term.partition('=')
        if k in ('lowest_md5', 'lowest_md5_per_arena'):
            n = int(v) if v.isdigit() else 0
            pools = defaultdict(list)
            for t in tasks:
                pools[t.arena if k.endswith('arena') else None].append(t)
            if n < 1 or min(map(len, pools.values()), default=0) < n:
                raise ConfigError([f'subset {term!r}: N must be in 1..{min(map(len, pools.values()), default=0)}'])
            keep = {t.id for p in pools.values() for t in sorted(p, key=lambda t: md5hex(t.id))[:n]}
        else:
            get = {'id': lambda t: t.id, 'arena': lambda t: t.arena}.get(k, lambda t: t.meta.get(k))
            want, have = set(v.split('|')), {str(get(t)) for t in tasks}
            if not v or not all(k in ('id', 'arena') or k in t.meta for t in tasks) or want - have:
                raise ConfigError([f'subset {term!r}: key {k!r} with values {sorted(want - have) or v!r} matches no task '
                                   f'(keys: id, arena, {sorted({m for t in tasks for m in t.meta})})'])
            keep = {t.id for t in tasks if str(get(t)) in want}
        tasks = [t for t in tasks if t.id in keep]
    return tasks


def blocks(tasks, size: int) -> list[list[Task]]:
    """Pair blocks: tasks sorted by md5(id) hex, chunked (ag_eval_tasks.py:203-212); every arm of a pair runs in the block
    of its task, so a block is the unit pinned to one node."""
    if size < 1 or len({t.id for t in tasks}) != len(tasks):
        raise ConfigError([f'blocks: size {size} must be >= 1 and task ids unique'])
    order = sorted(tasks, key=lambda t: md5hex(t.id))
    return [order[i:i + size] for i in range(0, len(order), size)]


def make_case(out_dir, id, arena, start_xy, start_yaw, goal_xy, *, route=None, split='test', env: Env | None = None):
    """Write a case in the released schema as <out_dir>/<id>.json (and a given route as routes/<id>/route.json) for
    ``load_suite('custom:<out_dir>')``. The arena folder under Env.data needs arena_000.bmp and arena_meta.json (80 m);
    planning also needs a released static map (MAPS), given and straight routes do not. Returns the case path."""
    env, out = env or Env.from_environ(), Path(out_dir)
    s, g = [float(x) for x in start_xy], [float(x) for x in goal_xy]
    P = [f'{k} {v!r}: letters, digits, _ . -' for k, v in (('id', id), ('arena', arena))
         if not (isinstance(v, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', v))]
    P += [] if split in SPLITS else [f'split {split!r} is not one of {SPLITS}']
    P += [] if len(s) == len(g) == 2 and np.all(np.abs(s + g) < 40.0) and np.isfinite(float(start_yaw)) else [
        'start and goal must lie inside +-40 m, the yaw be finite']
    P += [] if np.hypot(g[0] - s[0], g[1] - s[1]) > 2.5 else ['the goal lies within the 2.5 m goal radius of the start']
    P += [] if not (out / f'{id}.json').exists() else [f'{out}/{id}.json exists']
    if P:
        raise ConfigError(P)
    _arena(arena, env, custom=True)
    case = dict(id=id, split=split, arena=f'assets/traverse/{arena_dir(arena, env).name}',
                layout=dict(episode_id=id, seed=int(md5hex(id)[:8], 16), assets=[], house_xy=g, house_yaw=float(start_yaw),
                            start_xy=s, start_yaw=float(start_yaw)),
                goal_xy=g, goal_radius_m=2.5, horizon_s=120.0, arena_half_extent_m=40.0)
    if route is not None:
        _endpoints(route, case, f'route of {id}')
        (out / 'routes' / id).mkdir(parents=True, exist_ok=True)
        write_atomic(out / 'routes' / id / 'route.json', json.dumps(route, indent=1))
    out.mkdir(parents=True, exist_ok=True)
    write_atomic(out / f'{id}.json', json.dumps(case, indent=1))         # the case last: it makes the task
    return out / f'{id}.json'


# --------------------------------------------------------------------------------------------------------- checks
def _read(path, sha=None) -> bytes:
    b = Path(path).read_bytes()
    if sha is not None and hashlib.sha256(b).hexdigest() != sha:
        raise ReleaseError(f'{path}: sha256 {hashlib.sha256(b).hexdigest()[:12]}... is not the locked {sha[:12]}...')
    return b


def _json(path, sha=None):
    return json.loads(_read(path, sha))


def _digest(path, sha=None) -> str:
    return hashlib.sha256(_read(path, sha)).hexdigest()


def lock_digest(pairs) -> str:
    """sha256 over (name + digest bytes) sorted by name, from (name, hex digest) pairs: the head of every
    *LOCKED.sha256 (ga_suite.write_suite_lock, ag_suite_lock.lock_files) and PICKS_LOCKED (ga_planner.py:602-605)."""
    h = hashlib.sha256()
    for n, d in sorted(pairs):
        h.update(n.encode())
        h.update(bytes.fromhex(d))
    return h.hexdigest()


def _lock(path, head=None, sha=None) -> dict[str, str]:
    """A *LOCKED.sha256 file: line 1 = lock_digest of the listed files, then '<digest>  <name>' lines. Line 1 is
    recomputed (and must equal ``head``; the file's own sha256 must equal ``sha``); returns {name: digest}."""
    lines = _read(path, sha).decode().splitlines()
    rows = {n: d for d, _, n in (ln.partition('  ') for ln in lines[1:] if ln)}
    h = lock_digest(rows.items())
    if h != lines[0].split()[0] or head not in (None, h) or len(rows) != len(lines) - 1:
        raise ReleaseError(f'{path}: lock {lines[0].split()[0][:12]}... does not recompute ({h[:12]}..., '
                           f'pinned {(head or "-")[:12]}...)')
    return rows


def _case(path, id, arena, sha=None, mission=False) -> tuple[dict, str]:
    """The checked case and its file sha256."""
    b = _read(path, sha)
    c = json.loads(b)
    P = [] if c.get('id') == id else [f'id {c.get("id")!r} != {id!r}']
    P += [] if mission or c.get('split') in SPLITS else [f'split {c.get("split")!r} (train/val/test, gen_collect.py:244)']
    P += [] if c.get('layout', {}).get('assets') == [] else ['layout.assets must be [] (crm_collect.py:405)']
    P += [] if _arena_name(c.get('arena', '')) == arena else [f'arena {c.get("arena")!r} is not {arena}']
    if P:
        raise ReleaseError(f'{path}: ' + '; '.join(P))
    return c, hashlib.sha256(b).hexdigest()


def _arena_name(rel: str) -> str:
    n = PurePosixPath(rel).name
    return 'f104' if n == 'arena_f104_50h_v1' else n[6:] if n.startswith('arena_') else f'?{rel}'


def _endpoints(route, case, what):
    if not (set(KEYS) <= set(route) and len(route['waypoints']) >= 2
            and ends_within(route, case['layout']['start_xy'], case['goal_xy'])):
        raise ReleaseError(f'{what}: needs {KEYS} and must start and end within {ENDS_TOL_M} m of the case start, goal')


def _route(path, case, sha=None, content=None) -> str:
    """Checks a route file (endpoints, pinned file or content sha); returns its file sha256."""
    b = _read(path, sha)
    r = json.loads(b)
    _endpoints(r, case, str(path))
    if content is not None and route_sha256(r) != content:
        raise ReleaseError(f'{path}: route content {route_sha256(r)[:12]}... is not the locked {content[:12]}...')
    return hashlib.sha256(b).hexdigest()


def _arena(arena, env, custom=False) -> tuple[Path | None, str | None]:
    """Checks an arena once per process: 80 m, BMP and meta release-checked when released (custom arenas may be new),
    and its planner map (released for MAPS only) rendered from this BMP (ag_map_check). Returns the map folder and
    the sha256 of its observation.npz (None, None without a map)."""
    key, cache = (arena, env.data), _ARENAS
    if key not in cache:
        d, rel = arena_dir(arena, env), str(arena_dir(arena, env).relative_to(env.data))
        released = rel + '/arena_000.bmp' in env.release.plain
        if not (released or custom and (d / 'arena_000.bmp').is_file()):
            raise ReleaseError(f'arena {arena}: no arena_000.bmp in the release or in {d}')
        bmp, meta = [env.release.file(f'{rel}/{f}') if released else d / f for f in ('arena_000.bmp', 'arena_meta.json')]
        if json.loads(meta.read_text()).get('size_m') != 80.0:
            raise ReleaseError(f'{meta}: size_m must be 80 (gen_collect.py:248)')
        cache[key] = None, None
        if arena in MAPS:
            npz = MAPS[arena] + '/observation.npz'
            obs = json.loads(env.release.file(MAPS[arena] + '/observation.json').read_text())
            env.release.file(npz)
            if obs.get('arena_bmp_sha256') != sha256_file(bmp):
                raise ReleaseError(f'{env.data / MAPS[arena]}: planner map rendered from another BMP than {bmp}')
            cache[key] = env.data / MAPS[arena], env.release.record(npz)[0]
    return cache[key]


# --------------------------------------------------------------------------------------------------------- loaders
def _f104(name, env) -> list[Task]:
    base = env.data / F104_SUITE
    lock = _lock(base / 'SUITE_LOCKED.sha256', head=LOCKS['f104_800'])
    suite = _json(base / 'suite.json', lock['suite.json'])
    idx = _json(env.data / (APPROACH + '_index.json'), LOCKS['approach'])['groups']
    if not suite['n_groups'] == len(suite['groups']) == len(idx) == 800:
        raise ReleaseError(f'{base}/suite.json: {len(suite["groups"])} groups, {len(idx)} approach routes, not 800')
    (m, msha), tasks = _arena('f104', env), []
    for g in suite['groups']:
        gid, a = g['group'], idx[g['group']]
        cp, r00 = base / f'cases/{gid}.json', base / f'cases/routes/{gid}/route_00.json'
        ap = env.data / APPROACH / f'{gid}.json'
        case, csha = _case(cp, gid, 'f104', g['case_sha256'])
        if a['case_sha256'] != g['case_sha256']:
            raise ReleaseError(f'{gid}: the approach index was built on another case')
        tasks.append(Task(gid, 'f104', cp, r00, approach=ap, map=m,
                          meta=dict(stratum=g['stratum'], evaluation_stratum=g['evaluation_stratum']),
                          sha=dict(case=csha, route00=_digest(r00, g['route00_sha256']), map=msha,
                                   approach=_route(ap, case, content=a['route_sha256']))))
    return tasks


def _unseen(name, env) -> list[Task]:
    locks = _lock(env.data / UNSEEN_LOCKS / 'ALL_LOCKED.sha256', head=LOCKS['E1'])
    locks |= _lock(env.data / UNSEEN_LOCKS / 'ALL_LOCKED_E1b.sha256', head=LOCKS['E1b'])
    subsets = _lock(env.data / UNSEEN_LOCKS / 'SUBSETS_LOCKED_E1b.sha256', head=LOCKS['subsets'])
    declared = _json(env.data / UNSEEN_LOCKS / 'soil_unseen_subset.json',
                     subsets[UNSEEN_LOCKS + 'soil_unseen_subset.json'])['arenas']
    tasks = []
    for arena, wave in UNSEEN.items():
        rel = f'{UNSEEN_LOCKS}test_{arena}.SUITE_LOCKED.sha256'
        files, (m, msha) = _lock(env.data / rel, sha=locks[rel]), _arena(arena, env)
        cdir = T + f'arena_gator_20260925/cases/test_{arena}/cases/'
        recs = _json(env.data / cdir / 'cases.json', files[cdir + 'cases.json'])['records']
        low = sorted((r['scene_id'] for r in recs), key=md5hex)[:SOIL_PER_ARENA]
        if len(recs) != 250 or set(low) != set(declared[arena]['groups']):
            raise ReleaseError(f'{arena}: {len(recs)} groups; the 125 lowest md5 differ from soil_unseen_subset.json')
        low = set(low)
        for r in recs if name == 'unseen_rigid2000' else [r for r in recs if r['scene_id'] in low]:
            g, cp, r00 = r['scene_id'], cdir + r['case'], cdir + r['routes'][0]
            if not r00.endswith(f'{g}/route_00.json'):
                raise ReleaseError(f'{cdir}cases.json: first route of {g} is {r["routes"][0]}')
            tasks.append(Task(g, arena, env.data / cp, env.data / r00, map=m,
                              meta=dict(evaluation_stratum=r['evaluation_stratum'], family=declared[arena]['family'],
                                        wave=wave),
                              sha=dict(case=_case(env.data / cp, g, arena, files[cp])[1], map=msha,
                                       route00=_digest(env.data / r00, files[r00]))))
    return tasks


def _tracker423(name, env) -> list[Task]:
    suite = _json(env.file('data:' + T + 'generalist_20260921/B_tracker/suite/tracking_suite.json'), LOCKS[name])
    if not suite['n_routes'] == len(suite['routes']) == 423:
        raise ReleaseError(f'tracking_suite.json: {len(suite["routes"])} routes, not 423')
    (m, msha), tasks = _arena('f104', env), []
    for r in suite['routes']:
        cp, rp = env.data / r['case_file'], env.data / r['route_file']
        case, csha = _case(cp, r['group'], 'f104', r['case_sha256'])
        tasks.append(Task(r['id'], 'f104', cp, route=rp, map=m,
                          meta={k: r[k] for k in ('stratum', 'group', 'speed_profile_id', 'evaluation_stratum')},
                          sha=dict(case=csha, route=_route(rp, case, r['route_sha256']), map=msha)))
    return tasks


def _missions30(name, env) -> list[Task]:
    rows = _json(env.file(f'data:{NAV}local_luffy/tasks_main.json'), LOCKS[name])
    ids = sorted({r['mission'] for r in rows})
    if len(ids) != 30 or len(rows) != 120:
        raise ReleaseError(f'tasks_main.json: {len(ids)} missions, {len(rows)} rows (30 x 4 schedules expected)')
    tasks = []
    for mid in ids:
        p = env.file(f'data:{NAV}missions/{mid}.json')
        arena = _arena_name(json.loads(p.read_text())['arena'])
        (c, csha), (m, msha) = _case(p, mid, arena, mission=True), _arena(arena, env)
        tasks.append(Task(mid, arena, p, goals=tuple(tuple(map(float, g)) for g in c['goals']), map=m,
                          meta=dict(n_goals=len(c['goals'])), sha=dict(case=csha, map=msha)))
    return tasks


def _smoke144(name, env) -> list[Task]:
    sample = _json(env.file(f'data:{T}offroad_vehicles_20260927/scratch/S3/sample_A.json'), LOCKS[name])['rows']
    rows = _json(env.file(f'data:{T}offroad_vehicles_20260927/tasks/smoke_v2_polaris.json'), LOCKS['smoke_tasks'])
    paths = defaultdict(set)
    for r in rows:
        if r.get('sample') == 'A':
            paths[r['pair_id']].add((r['case'], r['route']))
    ids = [r['pair_id'] for r in sample]
    if len(ids) != 144 or set(ids) != set(paths) or any(len(v) != 1 for v in paths.values()):
        raise ReleaseError('smoke sample A: not 144 routes with one case and route each in smoke_v2_polaris.json')
    (m, msha), tasks = _arena('f104', env), []
    for r in sample:
        (cp, rp), = paths[r['pair_id']]
        cp, rp = (env.file('data:' + env.release.remap(p)) for p in (cp, rp))
        case, csha = _case(cp, r['group'], 'f104')
        tasks.append(Task(r['pair_id'], 'f104', cp, route=rp, map=m,
                          meta={k: r[k] for k in ('group', 'stratum', 'kind', 'profile')},
                          sha=dict(case=csha, route=_route(rp, case), map=msha)))
    return tasks


def _custom(d: Path, env) -> list[Task]:
    tasks = []
    for p in sorted(d.glob('*.json')):
        arena, rd = _arena_name(json.loads(p.read_text()).get('arena', '')), d / 'routes' / p.stem
        (case, csha), (m, msha) = _case(p, p.stem, arena), _arena(arena, env, custom=True)
        sha = dict(case=csha, map=msha, **{k: _route(rd / f, case) for k, f in (('route00', 'route_00.json'),
                                                                                ('route', 'route.json'))
                                           if (rd / f).is_file()})
        tasks.append(Task(p.stem, arena, p, rd / 'route_00.json' if 'route00' in sha else None,
                          rd / 'route.json' if 'route' in sha else None, map=m, sha=sha))
    if not tasks:
        raise ConfigError([f'custom:{d}: no case files (write them with make_case)'])
    return tasks


LOADERS = dict(f104_800=_f104, unseen_soil1000=_unseen, unseen_rigid2000=_unseen, tracker423=_tracker423,
               missions30=_missions30, smoke144=_smoke144)
SUITES = tuple(LOADERS)
