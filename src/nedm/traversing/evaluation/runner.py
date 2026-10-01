"""Run an arm on tasks: plan, drive, resume (FINAL_DESIGN 2.1, 4.3-4.7).

``TraversalEval.episode``: [pass 1, the approach driven to horizon L -> decision at F] -> pick (pick.json; planning is
never charged) -> drive (after an approach: pass 2, branching to the pick at F). ``spawn``: each drive in a fresh
process (process_env; soil binds one GPU through ROCr), ATTEMPTS attempts counted across job kills, then 'crash'; a
refusal (exit 3: a deterministic input or build problem) raises ConfigError, never a record. A run folder (``prepare``)
holds ONE evaluation of one ground: config/, tasks.jsonl, blocks.json, blocks/<i>.pin.json, runs/<task>/<arm>/,
superseded/; writes are atomic, DONE (the sha256 of every file, re-checked by ``cells``) comes last and only DONE counts.
Off its pinned node and build a block moves each unfinished pair aside and drives it again whole, keeping picks no pass
1 made (rigid physics repeats on one node only, SPEC 0.5). ``python -m`` this module = spawn's child (one drive).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields
from functools import cache
from pathlib import Path

import numpy as np

from .config import DT, ConfigError, Env, EvalConfig, sha256_file, soil_config, write_atomic
from .controllers import make_controller
from .episode import Branch, Stops, drive_episode
from .labels import UNLABELLED, drive_labels
from .routes import ENDS_TOL_M, ends_within, load_route, route_json, route_sha256
from .sim import LaunchError, Sim
from .suites import Task, arena_dir, blocks
from .vehicles import load as load_vehicle

# threads of a drive process: rigid single-threaded (gen_array_g.sbatch); soil as crm_collect.sbatch + crm_worker.py
PROCESS_ENV = dict(rigid=dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', LP_NUM_THREADS='1'),
                   soil=dict(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'))
WALL_S, ATTEMPTS, REFUSED = dict(rigid=3600., soil=2400.), 2, 3      # per-attempt wall limit; exit code of a refusal
GUARD_S = dict(rigid=400., soil=500.)       # longest drive (120 s at ~2 wall-s per sim-s + build; 4.6) + 100 s
KEEP = ('attempts.jsonl', 'drive.log', 'pick.json', 'input.json')    # survive a failed attempt; the rest is redone
CODE = ('nedm/traversing/evaluation/*.py', 'nedm/hmmwv/hmmwv_data.py',
        *(f'nedm/traversing/training/{m}.py' for m in ('state', 'risk_model', 'numpy_actor')))
BUILD_DATA = ('vehicle/hmmwv/hmmwv_chassis_col.obj', 'vehicle/hmmwv/hmmwv_tire_coarse*.obj',
              'vehicle/gator/gator_chassis_col.obj', 'vehicle/Polaris/*.json')    # Chrono data the physics reads
_PLAN = threading.Lock()                    # one plan at a time per process


def process_env(ground, gpu=None) -> dict:
    """Environment of a drive process: PROCESS_ENV and on soil one GPU, filtered at the ROCr layer and seen by HIP as
    device 0 (crm_worker.py:69-76)."""
    one = {} if ground == 'rigid' else {'ROCR_VISIBLE_DEVICES': str(int(gpu)), 'HIP_VISIBLE_DEVICES': '0'}
    return {**PROCESS_ENV[ground], **one}


def env_problems(ground, e) -> list[str]:
    """How the environment `e` differs from a process_env(ground, gpu)."""
    one = ground == 'rigid' or e.get('HIP_VISIBLE_DEVICES', '0') == '0' and str(e.get('ROCR_VISIBLE_DEVICES')).isdigit()
    return [f'{k}={e.get(k)} (need {v})' for k, v in PROCESS_ENV[ground].items() if e.get(k) != v] + (
        [] if one else ['one GPU: ROCR_VISIBLE_DEVICES=<gpu> HIP_VISIBLE_DEVICES=0'])


def _json(obj, **kw) -> str:                # numpy values through .tolist(); any other unknown type raises
    return json.dumps(obj, default=lambda o: o.tolist(), **kw)


def _sha(obj) -> str:
    return hashlib.sha256(_json(obj, sort_keys=True).encode()).hexdigest()


def _files(d) -> dict:                      # sha256 of every file of a run folder but DONE
    return {p.relative_to(d).as_posix(): sha256_file(p) for p in sorted(d.rglob('*')) if p.is_file() and p.name != 'DONE'}


@dataclass
class Record:
    """record.json (these fields) + trajectory.npz in the collectors' key names [+ crm_extra.npz, vehicle_extra.npz].
    status: a collector status | launch_failed | no_route | crash."""
    task: str
    arm: str
    status: str
    validated: bool
    frames: int = 0
    elapsed_s: float = 0.
    goal_time_s: float | None = None
    positive_work_kj: float = 0.
    near_stop_fired: bool | None = None     # None: the rule is off (native controllers)
    branch: dict | None = None              # frame, pose, start_error_m, route_sha256 of a branched drive
    route_sha256: str | None = None
    pick: dict | None = None                # set by episode(): the pick driven, or why there was no decision
    provenance: dict = field(default_factory=dict)
    arrays: dict = field(default_factory=dict, repr=False)
    extras: dict = field(default_factory=dict, repr=False)     # {'crm_extra': arrays} -> crm_extra.npz

    def save(self, out, routes=()):
        """The (name, route) files, the npz files, then record.json, each atomically; never over a finished drive."""
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        if (out / 'record.json').exists():
            raise FileExistsError(f'{out}/record.json exists: a finished drive is never overwritten')
        for name, r in routes:
            write_atomic(out / name, json.dumps(route_json(r)))
        for name, a in ([('trajectory', self.arrays)] if self.arrays else []) + list(self.extras.items()):
            write_atomic(out / f'{name}.npz', lambda f, a=a: np.savez_compressed(f, **a))
        self.write_meta(out)

    def write_meta(self, out):
        write_atomic(Path(out) / 'record.json', _json({f.name: getattr(self, f.name) for f in fields(self)
                                                       if f.name not in ('arrays', 'extras')}, indent=1))

    @classmethod
    def load(cls, out):
        rec, npz = cls(**json.loads((Path(out) / 'record.json').read_text())), {}
        for p in sorted(Path(out).glob('*.npz')):
            with np.load(p) as z:
                npz[p.stem] = {k: z[k] for k in z.files}
        rec.arrays, rec.extras = npz.pop('trajectory', {}), npz
        return rec


def drive(cfg: EvalConfig, task: Task, route: dict, *, branch=None, horizon_s=120., env: Env | None = None) -> Record:
    """One Chrono episode of `cfg` on `task` in this process, following `route` from the case's layout pose; branch =
    (F, route) swaps to the picked route at frame F. Every deterministic problem (config, build lock, release files,
    route ends, soil config, actor, vehicle files; then the source pins and CRM counts in Sim.build) raises ConfigError
    before any physics. A failed launch check is a Record with status 'launch_failed' (no arrays); a Chrono abort, a
    soil fall-through or a non-finite state raises (spawn retries it)."""
    t0, env = time.time(), env or Env.from_environ()
    try:
        P = cfg.validate() + ([] if task.kind == 'single_goal' else [f'task {task.id}: missions (M1) are not ported']) + (
            [] if env.chrono_data else ['NEDM_CHRONO_DATA (the data folder of the Chrono build) is not set']) + [
            f'a {cfg.ground} drive runs in process_env({cfg.ground!r}, gpu): {b}' for b in env_problems(cfg.ground,
                                                                                                        os.environ)]
        if P := P or lock_problems([cfg], env):
            raise ConfigError(P)
        case = task.read_case()
        goal = np.asarray(case['goal_xy'], float)
        if not ends_within(route, case['layout']['start_xy'], goal):
            raise ConfigError([f'{task.id}: the route must start and end within {ENDS_TOL_M} m of the case start, goal'])
        sp, soil = soil_config(cfg.soil_config, env) if cfg.soil_config else (None, None)
        hook = None if branch is None else Branch(branch[0], branch[1], goal, soil=bool(soil))
        ctrl, vehicle = make_controller(cfg, env), load_vehicle(cfg.vehicle, env)
    except ConfigError:
        raise
    except (ValueError, KeyError, OSError, ImportError) as e:     # a release, file or route problem: also a refusal
        raise ConfigError([f'{task.id} / {cfg.name}: {type(e).__name__}: {e}']) from e
    sim = Sim.build(case, arena_dir(task.arena, env), vehicle, env.chrono_data, soil)
    stops = Stops(goal, case.get('goal_radius_m', 2.5), near_stop=cfg.near_stop_rule, horizon_s=horizon_s,
                  breakthrough_m=sim.breakthrough_m)
    try:
        ep = drive_episode(sim, route, ctrl, stops, hook)
        status, n, error = ep.status, len(ep.state), None
    except LaunchError as e:
        ep, status, n, error = None, 'launch_failed', 0, str(e)
    import pychrono
    prov = dict(host=platform.node(), slurm={k: os.environ.get(k) for k in ('SLURM_JOB_ID', 'SLURM_JOB_PARTITION')},
                python=sys.version.split()[0], numpy=np.__version__, pychrono=pychrono.__file__,
                chrono_data=str(env.chrono_data), process_env={k: os.environ.get(k) for k in (
                    *PROCESS_ENV[cfg.ground], 'ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES')},
                config_sha=cfg.sha, config=cfg.to_dict(), task_sha=task.sha, horizon_s=horizon_s, launch=sim.launch,
                scene=sim.info, vehicle=sim.vehicle_info, controller=ctrl.info, error=error, wall_s=time.time() - t0)
    if soil:
        prov['soil'] = dict(config=str(sp), config_sha256=sha256_file(sp), physics_dt_s=sim.dt,
                            max_sinkage_m=stops.max_sinkage)
    extras = {'crm_extra': sim.crm_extra(ep.extra)} if soil and ep else {}
    if sim.belly is not None and ep:
        extras['vehicle_extra'] = sim.belly.arrays()
    return Record(task.id, cfg.name, status, cfg.validated, frames=n, elapsed_s=n * DT,
                  goal_time_s=n * DT if status == 'goal_reached' else None, positive_work_kj=ep.total_work if ep else 0.,
                  near_stop_fired=stops.near_stop_fired if cfg.near_stop_rule else None, branch=ep and ep.branch,
                  route_sha256=route_sha256(route), provenance=prov, arrays=ep.arrays() if ep else {}, extras=extras)


def spawn(cfg: EvalConfig, task: Task, route: dict, out, *, branch=None, horizon_s=120., env: Env | None = None,
          gpu=0) -> Record:
    """`drive` in a fresh process with process_env(cfg.ground, gpu); a finished drive in `out` is loaded, not re-run.
    The attempts are counted in drive.log (refusals not), so a drive killed with its job twice becomes a 'crash'."""
    out, env = Path(out), env or Env.from_environ()
    if (out / 'record.json').exists():
        return Record.load(out)
    out.mkdir(parents=True, exist_ok=True)
    inp = dict(task=task.to_dict(env), arm=cfg.to_dict(), route=route_json(route), horizon_s=horizon_s,
               branch=branch and [branch[0], route_json(branch[1])])
    write_atomic(out / 'input.json', _json(inp))
    cmd = [sys.executable, '-P', '-u', '-m', __spec__.name, '--input', str(out / 'input.json'), '--out', str(out)]
    penv = {**os.environ, **{k: str(v) for k, v in (('NEDM_DATA', env.data), ('NEDM_CHRONO_DATA', env.chrono_data),
                                                     ('NEDM_RELEASE_CACHE', env.release_cache)) if v},
            **process_env(cfg.ground, gpu)}
    wipe = lambda: [f.unlink() for f in out.iterdir() if f.is_file() and f.name not in KEEP]  # noqa: E731
    log = out / 'drive.log'
    text = log.read_text() if log.exists() else ''
    for attempt in range(1 + text.count('# attempt ') - text.count('# refused'), ATTEMPTS + 1):
        wipe()                                            # a failed attempt's partial files; the drive restarts
        t0 = time.time()
        with open(log, 'a') as f:
            f.write(f'# attempt {attempt}: {" ".join(cmd)}\n')
            f.flush()
            try:
                rc = subprocess.run([*cmd, '--attempt', str(attempt)], stdout=f, stderr=subprocess.STDOUT,
                                    timeout=WALL_S[cfg.ground], env=penv).returncode
            except subprocess.TimeoutExpired:
                rc = 'wall_timeout'
        with open(out / 'attempts.jsonl', 'a') as f:
            f.write(json.dumps(dict(attempt=attempt, rc=rc, wall_s=round(time.time() - t0, 1), host=platform.node(),
                                    gpu=None if cfg.ground == 'rigid' else gpu)) + '\n')
        if rc == 0 and (out / 'record.json').exists():
            return Record.load(out)
        if rc == REFUSED:
            raise ConfigError([f'{task.id} / {cfg.name}: the drive refused its inputs (see {log})'])
    wipe()
    last = [ln for ln in log.read_text().splitlines() if ln.strip()][-1:]
    rec = Record(task.id, cfg.name, 'crash', cfg.validated, route_sha256=route_sha256(route), provenance=dict(
        attempts=ATTEMPTS, log=str(log), last_log_line=last and last[0], host=platform.node()))
    rec.save(out)
    return rec


class TraversalEval:
    """One arm (FINAL_DESIGN 2.1): plan() is Chrono-free, drive() runs ONE Chrono episode in this process, episode()
    and run() orchestrate pass 1 -> decision -> plan -> drive, one process per drive, resumable."""
    node: dict = {}                         # set by run_block: the node provenance of every run it finishes

    def __init__(self, cfg: EvalConfig, out, env: Env | None = None):
        if problems := cfg.validate():
            raise ConfigError([f'{cfg.name}: {p}' for p in problems])
        self.cfg, self.out, self.env = cfg, Path(out), env or Env.from_environ()

    def plan(self, task, decision=None):
        from .planner import plan
        with _PLAN:
            return plan(self.cfg, task, decision, self.env)

    def drive(self, task, route, *, branch=None, horizon_s=120.) -> Record:
        return drive(self.cfg, task, route, branch=branch, horizon_s=horizon_s, env=self.env)

    def run(self, tasks, *, stage='all', workers=1, gpus=()):
        return run_arms([self], tasks, stage=stage, workers=workers, gpus=gpus)

    def episode(self, task, stage='all', gpu=0) -> Record | None:
        """The DONE run of `task` in runs/<task>/<arm> (stage 'plan': stop once pick.json is written)."""
        c, d = self.cfg, self.out / 'runs' / task.id / self.cfg.name
        if (d / 'DONE').exists():
            return Record.load(d)
        if c.planner == 'live':
            raise NotImplementedError(f'{c.name}: planner live (M1) plans inside the drive; nav.py is not ported yet')
        from .planner import Decision, Pick
        go = lambda sub, route, **kw: spawn(c, task, route, d / sub, env=self.env, gpu=gpu, **kw)  # noqa: E731
        dec = None
        if not (c.picks or c.planner == 'given'):
            if not c.approach_s:
                dec = Decision.standing(task)
            elif c.decisions:
                dec = Decision.from_release(task, c.decisions, self.env)
            else:                                        # pass 1: the native approach drive, horizon L
                p1 = go('pass1', task.read_route('approach'), horizon_s=c.approach_s)
                if p1.frames < c.F:                      # ended before F: no decision; its own status is the label
                    return self._finish(task, d, p1, dict(decided=False, pass1_status=p1.status), gpu)
                dec = Decision.after_approach(task, p1, c.F)
        if (d / 'pick.json').exists():
            pick = Pick.from_dict(json.loads((d / 'pick.json').read_text()))
        else:
            pick = self.plan(task, dec)
            d.mkdir(parents=True, exist_ok=True)
            write_atomic(d / 'pick.json', _json(pick.to_dict(), indent=1))
        if stage == 'plan':
            return None
        about = dict(arm=pick.arm, route_sha256=pick.route_sha256, z_mean=pick.z_mean, z_pess=pick.z_pess,
                     decision=dec.source if dec else c.decisions, picks=pick.record.get('picks'))
        if pick.route is None:
            return self._finish(task, d, Record(task.id, c.name, 'no_route', c.validated), about, gpu)
        rec = go('', task.read_route('approach'), branch=(c.F, pick.route)) if c.approach_s else go('', pick.route)
        return self._finish(task, d, rec, about, gpu)

    def _finish(self, task, d, rec, pick, gpu):
        """record.json with the pick and the run provenance (soil: the GPU index into node gpu_uuids), then DONE (the
        sha256 of every file of the run), last."""
        if not (d / 'record.json').exists():
            rec.save(d)                                  # a no-route record, or the pass-1 drive that ended early
        rec.pick = pick
        rec.provenance['run'] = dict(self.node, config_sha=self.cfg.sha, gpu=gpu if self.cfg.ground == 'soil' else None)
        rec.write_meta(d)
        write_atomic(d / 'DONE', _json(dict(task=task.id, task_sha=_sha(task.to_dict(self.env)), code_sha=code_sha(),
                                            config_sha=self.cfg.sha, files=_files(d)), indent=1))
        return rec


@cache
def code_sha() -> str:
    """sha256 of the package code a run executes (CODE)."""
    src, h = Path(__file__).resolve().parents[3], hashlib.sha256()
    for p in sorted(q for g in CODE for q in src.glob(g)):
        h.update(p.relative_to(src).as_posix().encode() + b'\0' + p.read_bytes())
    return h.hexdigest()


def fingerprint(env: Env) -> dict:
    """The build (FINAL_DESIGN 4.2): sha256 of pychrono's native modules and the Chrono libraries of its build (found
    without importing pychrono) and of BUILD_DATA under NEDM_CHRONO_DATA."""
    spec, cd = importlib.util.find_spec('pychrono'), env.chrono_data
    if spec is None or cd is None:
        raise ConfigError(['the build fingerprint needs pychrono on the path and NEDM_CHRONO_DATA'])
    b = Path(spec.submodule_search_locations[0]).parents[1]              # the build (bin/pychrono, lib/)
    return {f'{k}:{p.relative_to(r).as_posix()}': sha256_file(p) for k, r, ps in (
        ('build', b, [*b.glob('*/pychrono/*.so'), *b.glob('lib/libChrono*.so*')]),
        ('data', cd, [q for g in BUILD_DATA for q in cd.glob(g)])) for p in sorted(ps)}


def lock_problems(cfgs, env: Env, fp=None) -> list[str]:
    """The arms whose build_lock (written by fingerprint.py on a parity-checked build) is not this build."""
    locked = [c for c in cfgs if c.build_lock]
    fp = fingerprint(env) if locked and fp is None else fp
    return [f'{c.name}: this build differs from the lock {c.build_lock}' for c in locked
            if json.loads(env.path(c.build_lock).read_text())['fingerprint'] != fp]


def node_info(env: Env) -> dict:
    """Where a block runs: host, SLURM ids, CPU, the GPUs (name; UUIDs in ROCr order, which a soil drive's
    ROCR_VISIBLE_DEVICES indexes), the code sha and the build fingerprint."""
    import torch
    fp, n = fingerprint(env), torch.cuda.device_count() if torch.cuda.is_available() else 0
    cpu = [ln.split(':', 1)[1].strip() for ln in open('/proc/cpuinfo') if ln.startswith('model name')][:1] or [None]
    slurm = {k[6:].lower(): os.environ.get(k) for k in ('SLURM_JOB_ID', 'SLURM_JOB_PARTITION', 'SLURM_ARRAY_TASK_ID')}
    return dict(host=platform.node(), **slurm, cpu=cpu[0], code_sha=code_sha(), build_sha=_sha(fp), fingerprint=fp,
                gpu=torch.cuda.get_device_name(0) if n else None,
                gpu_uuids=[str(torch.cuda.get_device_properties(k).uuid) for k in range(n)])


def prepare(out, arms, tasks, block_size=None) -> list[list[str]]:
    """Writes <out>/config, tasks.jsonl + TASKS.sha256 and blocks.json (md5-sorted blocks of whole pairs), or checks
    that an existing run folder holds exactly these. Every arm is validated with its inputs on these tasks."""
    out, env, names = Path(out), arms[0].env, [a.cfg.name for a in arms]
    if len(set(names)) < len(names) or len({a.cfg.ground for a in arms}) > 1 or {a.out for a in arms} != {out}:
        raise ConfigError([f'arms {[(a.cfg.name, a.cfg.ground, str(a.out)) for a in arms]}: a run folder holds arms of '
                           'one ground (one build and thread recipe) with distinct names'])
    info = None
    if any(a.cfg.models for a in arms):
        from .planner import model_info as info
    if P := [f'{a.cfg.name}: {m}' for a in arms for m in a.cfg.validate(env, tasks=tasks, model_info=info)]:
        raise ConfigError(P)
    bl = [[t.id for t in b] for b in blocks(tasks, block_size or len(tasks))]
    lines = ''.join(_json(t.to_dict(env), sort_keys=True) + '\n' for t in tasks)
    files = {f'config/{a.cfg.name}.json': json.dumps(dict(a.cfg.to_dict(), sha=a.cfg.sha), indent=1) for a in arms}
    files |= {'tasks.jsonl': lines, 'TASKS.sha256': hashlib.sha256(lines.encode()).hexdigest() + '\n',
              'blocks.json': json.dumps(dict(arms=names, blocks=bl), indent=1)}
    if bad := [n for n, text in files.items() if (out / n).exists() and (out / n).read_text() != text]:
        raise ConfigError([f'{out}: {bad} differ; a run folder holds one evaluation (choose a new --out)'])
    for n, text in files.items():
        (out / n).parent.mkdir(parents=True, exist_ok=True)
        write_atomic(out / n, text)
    return bl


def load_run(out, env: Env | None = None):
    """(arms, tasks by id, blocks) of a prepared run folder; the task list and every arm file checked by their sha."""
    out, env = Path(out), env or Env.from_environ()
    meta, lines = json.loads((out / 'blocks.json').read_text()), (out / 'tasks.jsonl').read_text()
    cfgs = [json.loads((out / 'config' / f'{n}.json').read_text()) for n in meta['arms']]
    arms = [TraversalEval(EvalConfig.from_dict({k: v for k, v in c.items() if k != 'sha'}), out, env) for c in cfgs]
    if hashlib.sha256(lines.encode()).hexdigest() != (out / 'TASKS.sha256').read_text().strip() or \
            [a.cfg.sha for a in arms] != [c['sha'] for c in cfgs]:
        raise ConfigError([f'{out}: tasks.jsonl or an arm file was edited (sha)'])
    return arms, {t['id']: Task.from_dict(t, env) for t in map(json.loads, lines.splitlines())}, meta['blocks']


def run_block(out, i, *, stage='all', workers=1, gpus=(), deadline=None, accept_code_change=False, env=None):
    """Block i of a prepared run folder on this node: pin, pair-atomic resume, then the pairs in `workers` threads (on
    soil each owns one GPU of `gpus`); no pair starts within len(arms) x GUARD_S of `deadline` (epoch s). Stage 'plan'
    neither pins nor moves (picks are planned where the planner is bitwise and driven on the cluster). Returns this
    call's records; raises at the end if a pair failed or the guard left pairs for a resubmission."""
    out = Path(out)
    arms, tasks, bl = load_run(out, env)
    ground, node, pp, job = arms[0].cfg.ground, node_info(arms[0].env), out / f'blocks/{i}.pin.json', out / 'job'
    pin = json.loads(pp.read_text()) if pp.exists() else {}
    want = json.loads((job / 'code_sha.json').read_text())['code_sha'] if (job / 'code_sha.json').exists() else None
    P = [] if stage == 'plan' else lock_problems([a.cfg for a in arms], arms[0].env, node['fingerprint'])
    if ground == 'soil' and workers > len(gpus):
        P.append(f'soil: {workers} workers for GPUs {list(gpus)} (one drive process per GPU)')
    if want not in (None, node['code_sha']):
        P.append(f'code sha {node["code_sha"]} is not the job\'s {want} (job/code_sha.json)')
    if pin and pin['code_sha'] != node['code_sha'] and not accept_code_change:
        P.append(f'block {i} was pinned with code {pin["code_sha"]} (--accept-code-change)')
    if P:
        raise ConfigError(P)
    if stage != 'plan':
        if pin and (pin['host'], pin['build_sha']) != (node['host'], node['build_sha']):      # pair-atomic resume
            for t in bl[i]:
                runs = [out / 'runs' / t / a.cfg.name for a in arms]
                for r in [] if all((r / 'DONE').exists() for r in runs) else [r for r in runs if r.exists()]:
                    (s := out / 'superseded' / t).mkdir(parents=True, exist_ok=True)
                    r.rename(m := s / f'{r.name}__{time.strftime("%Y%m%d_%H%M%S")}')
                    if (m / 'pick.json').exists() and not (m / 'pass1').exists():   # a pick no drive of this node made
                        r.mkdir()
                        shutil.copy2(m / 'pick.json', r / 'pick.json')
        hist = pin.pop('history', []) + ([{k: v for k, v in pin.items() if k != 'fingerprint'}] if pin else [])
        pp.parent.mkdir(exist_ok=True)
        write_atomic(pp, _json(dict(node, accepted_code_change=accept_code_change, history=hist), indent=1))
    recs, late, errors = [], [], {}
    for a in arms:
        a.node = {k: v for k, v in node.items() if k != 'fingerprint'}

    def worker(w):                               # pairs dealt round-robin; on soil worker w owns gpus[w]
        for t in bl[i][w::workers]:
            if deadline and time.time() + len(arms) * GUARD_S[ground] > deadline:
                late.append(t)
                continue
            try:
                recs.extend(a.episode(tasks[t], stage, gpu=gpus[w] if ground == 'soil' else 0) for a in arms)
            except Exception as e:  # noqa: BLE001  every failed pair is reported below, the others still run
                errors[t] = e
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(worker, range(workers)))
    if errors or late:
        raise RuntimeError(f'block {i}: {len(errors)} pairs failed {errors}; walltime guard: {len(late)} pairs left '
                           f'for a resubmission {late}')
    return [r for r in recs if r is not None]


def run_arms(arms, tasks, *, stage='all', block=None, block_size=None, workers=1, gpus=()):
    """Every arm of a pair in the same block; the blocks run in turn here (slurm.py runs one per array task)."""
    bl = prepare(arms[0].out, arms, tasks, block_size)
    return [r for i in ([block] if block is not None else range(len(bl)))
            for r in run_block(arms[0].out, i, stage=stage, workers=workers, gpus=gpus, env=arms[0].env)]


def cells(out, env: Env | None = None) -> dict:
    """{task: {arm: drive_labels(run, the arm's label) | 'crash' | 'launch_failed'}} of the DONE runs only, each
    re-checked against its DONE file list; an unfinished arm is absent ('-' in a table; labels.ON_MISSING decides an
    unlabelled cell)."""
    arms, tasks, _ = load_run(out, env)
    res = {t: {} for t in tasks}
    for t in tasks:
        for a in arms:
            if ((d := a.out / 'runs' / t / a.cfg.name) / 'DONE').exists():
                if json.loads((d / 'DONE').read_text())['files'] != _files(d):
                    raise RuntimeError(f'{d}: its files changed after DONE was written')
                s = json.loads((d / 'record.json').read_text())['status']
                res[t][a.cfg.name] = s if s in UNLABELLED else drive_labels(d, a.cfg.label)
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description='One drive in this process (what spawn runs; input.json: task, arm, route, '
                                'horizon_s, branch [F, route]), in process_env(ground, gpu).')
    p.add_argument('--input', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--attempt', type=int, default=1, help='recorded in the provenance')
    a = p.parse_args(argv)
    try:
        env, j = Env.from_environ(), json.loads(Path(a.input).read_text())
        task, cfg, route = Task.from_dict(j['task'], env), EvalConfig.from_dict(j['arm']), load_route(j['route'])
        branch = j['branch'] and (j['branch'][0], load_route(j['branch'][1]))
        rec = drive(cfg, task, route, branch=branch or None, horizon_s=j['horizon_s'], env=env)
    except (ConfigError, NotImplementedError) as e:
        print(f'# refused: {e}', flush=True)
        sys.exit(REFUSED)
    rec.provenance['attempt'] = a.attempt
    rec.save(a.out, [('route.json', route)] + ([('branch_route.json', branch[1])] if branch else []))
    print(json.dumps(dict(task=rec.task, arm=rec.arm, status=rec.status, frames=rec.frames,
                          wall_s=round(rec.provenance['wall_s'], 1))))


if __name__ == '__main__':
    main()
