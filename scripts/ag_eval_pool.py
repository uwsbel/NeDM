#!/usr/bin/env python3
"""Shard pool for the rigid EVALUATION rows (arena_gator_20260925, module E6b), run on idle CPU cores of the soil
allocations (`srun --overlap`), like scripts/ag_rigid_pool.py (E3b2, frozen; it knows only shards 1000-2999).

Differences from ag_rigid_pool.py:
  * Task files: AG_EVAL_TASKLIST names a text file with one task file path per line; it is re-read before every claim
    round, so a later task file (e.g. task B's fixed 2 m/s rows) is picked up by running steps. A shard number must occur
    in exactly one listed file (a shard found in two files is never run and is logged).
  * One shard class for every evaluation shard (3000-8999): GEN_ROOT = G3/r2 (the tree whose gen_arenas.json lists all
    14 arenas), collector G3/r2/source/scripts/ag_gen_collect_ext.py (frozen dispatcher, rows carry --vehicle),
    runner G3/r2/source/scripts/ag_rigid_runner.py (= gen_runner_g.py keeping rich telemetry), FDM_RUNTIME_FINGERPRINT =
    the f104 HMMWV fingerprint (Gator rows pass their own --runtime-fingerprint). NEDM_VEHICLE is removed.
  * One node per shard (VERIFY_E6a point 2: Chrono rigid is deterministic per node only, and every arm of a group sits
    in one shard): when a stale claim is taken over, EVERY run directory of the shard (complete or not) is moved to
    <out>/pool/moved/ and the whole shard is driven again on the new node. A second attempt inside the same step (same
    node) re-drives only the incomplete rows, as before. After the shard, the host recorded by each completed run
    (simulation_provenance.json) is compared with this node; the done record lists hosts and any mismatch.
Claims, stale takeover (AG_POOL_STALE_S, 900 s), deadline (AG_POOL_DEADLINE), STOP file (<out>/pool/STOP), rich telemetry
clean-up (rich_telemetry.* deleted after the shard, rich_intervals.npz kept) are as in ag_rigid_pool.py.
  * AG_POOL_CONCURRENT (default 1) shards run at once in one step, each with AG_POOL_WORKERS / AG_POOL_CONCURRENT
    collector processes (the node's process count stays AG_POOL_WORKERS): a shard's wall time is set by its slowest
    episode (up to the 120 s horizon, about 270 s wall), so one shard at a time left most workers idle (first shard:
    median episode 59 s, shard 274 s). Every shard still runs entirely on this node.
Environment: AG_EVAL_TASKLIST, AG_POOL_OUT, AG_POOL_WORKERS, AG_POOL_CONCURRENT, AG_POOL_DEADLINE; Chrono environment
from ag_eval_pool.sh.
"""
import json, os, random, subprocess, sys, threading, time
from pathlib import Path

G3 = '/work1/dannegrut/harry/experiments/arena_gator_20260925'
HMMWV_FP = '/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909/pilot_runtime_412394.json'
EVAL = dict(lo=3000, hi=8999, root=f'{G3}/r2', collector=f'{G3}/r2/source/scripts/ag_gen_collect_ext.py',
            runner=f'{G3}/r2/source/scripts/ag_rigid_runner.py', fingerprint=HMMWV_FP,
            pythonpath=f'{G3}/r2/source/scripts:{G3}/r2/source/src')
TASKLIST = os.environ['AG_EVAL_TASKLIST']
OUT = Path(os.environ['AG_POOL_OUT'])
WORKERS_TOTAL = int(os.environ.get('AG_POOL_WORKERS', '12'))
CONCURRENT = max(1, int(os.environ.get('AG_POOL_CONCURRENT', '1')))
WORKERS = max(1, WORKERS_TOTAL // CONCURRENT)
DEADLINE = float(os.environ.get('AG_POOL_DEADLINE', '1e12'))
STALE_S = float(os.environ.get('AG_POOL_STALE_S', '900'))
JOB = os.environ.get('SLURM_JOB_ID', 'local') + '.' + os.environ.get('SLURM_STEP_ID', 'x')
HOST = os.uname().nodename
POOL = OUT / 'pool'
for d in ('claims', 'done', 'moved', 'stale', 'logs', 'steps'):
    (POOL / d).mkdir(parents=True, exist_ok=True)
_CACHE = {}


def log(msg):
    print(f'{time.strftime("%H:%M:%S")} {HOST} {JOB} {msg}', flush=True)


def load_tasks():
    """{shard: (task file, rows)} over every listed task file; shards found in two files are dropped (and logged)."""
    files = [l.strip() for l in open(TASKLIST) if l.strip() and not l.startswith('#')]
    by, owner, dup = {}, {}, set()
    for f in files:
        st = os.stat(f)
        key = (f, st.st_mtime, st.st_size)
        if key not in _CACHE:
            _CACHE[key] = [t for t in json.load(open(f)) if t.get('run', True)]
        for t in _CACHE[key]:
            s = int(t['shard'])
            assert EVAL['lo'] <= s <= EVAL['hi'], f'{f}: shard {s} outside the evaluation range'
            if s in owner and owner[s] != f:
                dup.add(s); continue
            owner[s] = f
            by.setdefault(s, []).append(t)
    for s in dup:
        log(f'ERROR shard {s} occurs in two task files; not run'); by.pop(s, None)
    return {s: (owner[s], rows) for s, rows in by.items()}


def done(shard):
    return (POOL / 'done' / f'{shard}.json').exists()


def try_claim(shard):
    """(claimed, takeover)."""
    if done(shard):
        return False, False
    claim = POOL / 'claims' / str(shard)
    takeover = False
    try:
        os.mkdir(claim)
    except FileExistsError:
        try:
            if time.time() - claim.stat().st_mtime < STALE_S or done(shard):
                return False, False
            moved = POOL / 'stale' / f'{shard}.{int(time.time())}.{JOB}.{os.getpid()}.{random.random():.6f}'
            os.rename(claim, moved)
            if time.time() - moved.stat().st_mtime < STALE_S:
                os.rename(moved, claim)
                return False, False
            os.mkdir(claim)
            takeover = True
        except OSError:
            return False, False
    (claim / 'owner').write_text(f'{JOB} {HOST} {time.time():.0f}\n')
    return True, takeover


def complete(t):
    return (OUT / 'runs' / t['id'] / 'episode_complete.json').exists()


def move_runs(rows, tag, everything):
    n = 0
    for t in rows:
        d = OUT / 'runs' / t['id']
        if d.exists() and (everything or not complete(t)):
            os.rename(d, POOL / 'moved' / f"{t['id']}.{tag}.{int(time.time())}.{JOB}")
            n += 1
    return n


def run_host(t):
    p = OUT / 'runs' / t['id'] / 'simulation_provenance.json'
    try:
        j = json.load(open(p))
    except (OSError, ValueError):
        return None

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k.lower() in ('host', 'hostname', 'node') and isinstance(v, str):
                    return v
                r = walk(v)
                if r:
                    return r
        return None
    return walk(j)


def run_shard(shard, tfile, rows, takeover):
    c = EVAL
    env = dict(os.environ)
    env.pop('NEDM_VEHICLE', None)
    env.update(GEN_ROOT=c['root'], GEN_COLLECTOR=c['collector'], GEN_TASKS=tfile, GEN_OUT=str(OUT),
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
    # a takeover (or any complete run left from another step) re-drives the WHOLE shard on this node
    foreign = [t['id'] for t in rows if complete(t) and (run_host(t) or '').split('.')[0] != HOST.split('.')[0]]
    moved0 = move_runs(rows, 'takeover' if takeover else 'pre', everything=bool(takeover or foreign))
    rcs = []
    for attempt in (1, 2):
        with open(POOL / 'logs' / f'shard_{shard}.a{attempt}.{JOB}.log', 'w') as lg:
            rc = subprocess.run([env.get('NRD_PYTHON', sys.executable), '-P', '-u', c['runner']], env=env,
                                stdout=lg, stderr=subprocess.STDOUT).returncode
        rcs.append(rc)
        left = [t['id'] for t in rows if not complete(t)]
        if not left:
            break
        move_runs(rows, f'a{attempt}', everything=False)
    stop.set()
    freed = 0
    for t in rows:
        if complete(t):
            for f in ('rich_telemetry.npz', 'rich_telemetry.json'):
                p = OUT / 'runs' / t['id'] / f
                if p.exists():
                    freed += p.stat().st_size
                    p.unlink()
    left = [t['id'] for t in rows if not complete(t)]
    hosts = {}
    for t in rows:
        if complete(t):
            h = (run_host(t) or '?').split('.')[0]
            hosts[h] = hosts.get(h, 0) + 1
    mism = sorted(h for h in hosts if h != HOST.split('.')[0])
    rec = dict(shard=shard, tasks=tfile, klass='eval', host=HOST, job=JOB, rows=len(rows), complete=len(rows) - len(left),
               incomplete=left, runner_rc=rcs, takeover=takeover, foreign_complete_before=foreign, moved_before=moved0,
               run_hosts=hosts, host_mismatch=mism, start=t0, end=time.time(), wall_s=time.time() - t0, workers=WORKERS,
               collector=c['collector'], runner=c['runner'], root=c['root'], freed_rich_bytes=freed)
    json.dump(rec, open(POOL / 'done' / f'{shard}.json', 'w'), indent=1)
    log(f'shard {shard} {rec["complete"]}/{rec["rows"]} in {rec["wall_s"]:.0f} s rc={rcs} takeover={takeover} hosts={hosts}')


def loop(k):
    status = POOL / 'steps' / f'{JOB}_{HOST}_t{k}.json'
    n_done, t_start = 0, time.time()
    while True:
        tasks = load_tasks()
        shards = sorted(tasks)
        progressed = False
        for s in shards:
            if time.time() > DEADLINE or (POOL / 'STOP').exists():
                log(f't{k} deadline or STOP: no more claims'); return
            ok, takeover = try_claim(s)
            if ok:
                run_shard(s, tasks[s][0], tasks[s][1], takeover)
                n_done += 1; progressed = True
                json.dump(dict(job=JOB, host=HOST, thread=k, shards=n_done, wall_s=time.time() - t_start, last=s, updated=time.time()),
                          open(status, 'w'))
                break          # re-read the task list after every shard (new files, lowest open shard first)
        if shards and all(done(s) for s in shards):
            log(f't{k} all {len(shards)} listed shards done'); return
        if not progressed:
            time.sleep(120)


def main():
    log(f'eval pool start: {CONCURRENT} concurrent shards x {WORKERS} workers, deadline {time.strftime("%H:%M", time.localtime(DEADLINE))}, '
        f'tasklist {TASKLIST}')
    ths = [threading.Thread(target=loop, args=(k,)) for k in range(CONCURRENT)]
    for i, th in enumerate(ths):
        th.start(); time.sleep(2 if i + 1 < len(ths) else 0)
    for th in ths:
        th.join()
    log('eval pool step exits')


if __name__ == '__main__':
    main()
