#!/usr/bin/env python3
"""Shard pool for rigid rows run on the idle CPU cores of the soil allocations (arena_gator_20260925, module E3b2).

One process per job step (`srun --overlap --jobid=<soil job> ... bash ag_rigid_pool.sh`). It claims whole shards of a
gen_runner-style task file one at a time and runs each with the unchanged-logic runner ag_rigid_runner.py (a copy of
gen_runner_g.py that keeps rich telemetry) as a subprocess, SLURM_ARRAY_TASK_ID = the shard. A shard runs on one node,
so all routes of a group share a node (every group sits in one shard, scripts/ag_rigid_tasks_v2.py).

Shard classes (fixed here, chosen by shard number):
  1000-1999  Gator f104 pool rows : GEN_ROOT = G3,    collector G3/source/scripts/ag_gen_collect_ext.py,
             FDM_RUNTIME_FINGERPRINT = G3/runtime/gator_runtime_fingerprint.json (the rows also pass it)
  2000-2999  HMMWV spread test rows: GEN_ROOT = G3/r2, collector G3/r2/source/scripts/gen_collect_ext.py (unchanged
             file; the r2 tree differs from G3/source only in scripts/gen_arenas.json, which lists the spread arenas),
             FDM_RUNTIME_FINGERPRINT = the f104 HMMWV fingerprint (gen_array_g.sbatch's)
Claims: mkdir <out>/pool/claims/<shard>; the owner refreshes its mtime every 60 s; a claim older than AG_POOL_STALE_S
(900 s) whose shard has no done record is taken over (the step that held it was killed). Before a shard runs, run
directories of its rows without episode_complete.json are moved to <out>/pool/moved/ (the rigid collector refuses to
write into a used directory). After the runner exits, incomplete rows are moved aside the same way and the shard is run
once more (second attempt); what is still incomplete is listed in the done record. After each shard,
rich_telemetry.npz / rich_telemetry.json of completed rows are deleted (0.9 MB per run); rich_intervals.npz (chassis
contact maxima) is kept.
Stops claiming after AG_POOL_DEADLINE (epoch seconds) or when <out>/pool/STOP exists.
Environment: AG_POOL_TASKS, AG_POOL_OUT, AG_POOL_SHARDS (e.g. "1000-1299,2000-2124", tried in this order),
AG_POOL_WORKERS, AG_POOL_DEADLINE; the Chrono environment is set by ag_rigid_pool.sh.
"""
import json, os, random, shutil, subprocess, sys, threading, time
from pathlib import Path

G3 = '/work1/dannegrut/harry/experiments/arena_gator_20260925'
HMMWV_FP = '/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909/pilot_runtime_412394.json'
CLASSES = {
    'gator': dict(lo=1000, hi=1999, root=G3, collector=f'{G3}/source/scripts/ag_gen_collect_ext.py',
                  runner=f'{G3}/source/scripts/ag_rigid_runner.py', fingerprint=f'{G3}/runtime/gator_runtime_fingerprint.json',
                  pythonpath=f'{G3}/source/scripts:{G3}/source/src'),
    'spread_hmmwv': dict(lo=2000, hi=2999, root=f'{G3}/r2', collector=f'{G3}/r2/source/scripts/gen_collect_ext.py',
                         runner=f'{G3}/r2/source/scripts/ag_rigid_runner.py', fingerprint=HMMWV_FP,
                         pythonpath=f'{G3}/r2/source/scripts:{G3}/r2/source/src'),
}
TASKS = os.environ['AG_POOL_TASKS']
OUT = Path(os.environ['AG_POOL_OUT'])
WORKERS = int(os.environ.get('AG_POOL_WORKERS', '12'))
DEADLINE = float(os.environ.get('AG_POOL_DEADLINE', '1e12'))
STALE_S = float(os.environ.get('AG_POOL_STALE_S', '900'))
JOB = os.environ.get('SLURM_JOB_ID', 'local') + '.' + os.environ.get('SLURM_STEP_ID', 'x')
HOST = os.uname().nodename
POOL = OUT / 'pool'
for d in ('claims', 'done', 'moved', 'stale', 'logs', 'steps'):
    (POOL / d).mkdir(parents=True, exist_ok=True)


def parse_shards(spec):
    out = []
    for part in spec.split(','):
        lo, _, hi = part.partition('-')
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def klass(shard):
    for name, c in CLASSES.items():
        if c['lo'] <= shard <= c['hi']:
            return name, c
    raise ValueError(f'shard {shard} has no class')


rows = [t for t in json.load(open(TASKS)) if t.get('run', True)]
by_shard = {}
for t in rows:
    by_shard.setdefault(t['shard'], []).append(t)
SHARDS = [s for s in parse_shards(os.environ['AG_POOL_SHARDS']) if s in by_shard]


def log(msg):
    line = f'{time.strftime("%H:%M:%S")} {HOST} {JOB} {msg}'
    print(line, flush=True)


def done(shard):
    return (POOL / 'done' / f'{shard}.json').exists()


def try_claim(shard):
    if done(shard):
        return False
    claim = POOL / 'claims' / str(shard)
    try:
        os.mkdir(claim)
    except FileExistsError:
        try:
            if time.time() - claim.stat().st_mtime < STALE_S or done(shard):
                return False
            moved = POOL / 'stale' / f'{shard}.{int(time.time())}.{JOB}.{os.getpid()}.{random.random():.6f}'
            os.rename(claim, moved)
            if time.time() - moved.stat().st_mtime < STALE_S:
                os.rename(moved, claim)
                return False
            os.mkdir(claim)
        except OSError:
            return False
    (claim / 'owner').write_text(f'{JOB} {HOST} {time.time():.0f}\n')
    return True


def complete(t):
    return (OUT / 'runs' / t['id'] / 'episode_complete.json').exists()


def move_incomplete(shard, tag):
    n = 0
    for t in by_shard[shard]:
        d = OUT / 'runs' / t['id']
        if d.exists() and not complete(t):
            os.rename(d, POOL / 'moved' / f"{t['id']}.{tag}.{int(time.time())}.{JOB}")
            n += 1
    return n


def run_shard(shard):
    name, c = klass(shard)
    env = dict(os.environ)
    env.pop('NEDM_VEHICLE', None)
    env.update(GEN_ROOT=c['root'], GEN_COLLECTOR=c['collector'], GEN_TASKS=TASKS, GEN_OUT=str(OUT),
               GEN_WORKERS=str(WORKERS), GEN_CHRONO_DATA='/work1/dannegrut/harry/nrd/chrono-build/data',
               FDM_RUNTIME_FINGERPRINT=c['fingerprint'], AG_KEEP_RICH='1', SLURM_ARRAY_TASK_ID=str(shard),
               PYTHONPATH=c['pythonpath'] + ':' + os.environ.get('PYTHONPATH', ''))
    claim = POOL / 'claims' / str(shard)
    stop = threading.Event()

    def beat():
        while not stop.wait(60):
            try:
                os.utime(claim)
            except OSError:
                pass
    th = threading.Thread(target=beat, daemon=True); th.start()
    t0 = time.time()
    moved0 = move_incomplete(shard, 'pre')
    rcs = []
    for attempt in (1, 2):
        with open(POOL / 'logs' / f'shard_{shard}.a{attempt}.{JOB}.log', 'w') as lg:
            rc = subprocess.run([env.get('NRD_PYTHON', sys.executable), '-P', '-u', c['runner']], env=env,
                                stdout=lg, stderr=subprocess.STDOUT).returncode
        rcs.append(rc)
        left = [t['id'] for t in by_shard[shard] if not complete(t)]
        if not left:
            break
        move_incomplete(shard, f'a{attempt}')
    stop.set()
    freed = 0
    for t in by_shard[shard]:
        if complete(t):
            for f in ('rich_telemetry.npz', 'rich_telemetry.json'):
                p = OUT / 'runs' / t['id'] / f
                if p.exists():
                    freed += p.stat().st_size
                    p.unlink()
    left = [t['id'] for t in by_shard[shard] if not complete(t)]
    rec = dict(shard=shard, klass=name, host=HOST, job=JOB, rows=len(by_shard[shard]), complete=len(by_shard[shard]) - len(left),
               incomplete=left, runner_rc=rcs, moved_before=moved0, start=t0, end=time.time(), wall_s=time.time() - t0,
               workers=WORKERS, collector=c['collector'], freed_rich_bytes=freed)
    json.dump(rec, open(POOL / 'done' / f'{shard}.json', 'w'), indent=1)
    log(f'shard {shard} ({name}) {rec["complete"]}/{rec["rows"]} in {rec["wall_s"]:.0f} s rc={rcs}')


def main():
    status = POOL / 'steps' / f'{JOB}_{HOST}.json'
    n_done, t_start = 0, time.time()
    log(f'pool start: {len(SHARDS)} shards, workers {WORKERS}, deadline {time.strftime("%H:%M", time.localtime(DEADLINE))}')
    while True:
        progressed = False
        for s in SHARDS:
            if time.time() > DEADLINE or (POOL / 'STOP').exists():
                log('deadline or STOP: no more claims'); return
            if try_claim(s):
                run_shard(s)
                n_done += 1; progressed = True
                json.dump(dict(job=JOB, host=HOST, shards=n_done, wall_s=time.time() - t_start, last=s, updated=time.time()),
                          open(status, 'w'))
        if all(done(s) for s in SHARDS):
            log('all shards done'); return
        if not progressed:
            time.sleep(120)   # shards held by others: wait, then look for stale claims again


if __name__ == '__main__':
    main()
