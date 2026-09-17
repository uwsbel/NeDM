"""Cluster array runner for gen_v1 single-route episodes (test arms and data collection).

Each task row: id, case (path relative to the gen_v1 root), route (relative), shard. All rows of a shard run in one
array task with a thread pool, so every arm of a test group shares a node (Chrono is deterministic per node only).
Rich telemetry is deleted after a successful episode: labels need only trajectory.npz / outcome.json /
command_reference.npz / case.json / anchor_state.npz.
"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
C = os.environ['GEN_ROOT']
TASKS = os.environ['GEN_TASKS']
OUT = os.environ['GEN_OUT']
PY_ = os.environ.get('NRD_PYTHON', 'python3')
aid = int(os.environ['SLURM_ARRAY_TASK_ID'])
rows = [t for t in json.load(open(TASKS)) if t['shard'] == aid and t.get('run', True)]
os.makedirs(OUT + '/runs', exist_ok=True); os.makedirs(OUT + '/logs', exist_ok=True)
print(f'shard {aid}: {len(rows)} episodes', flush=True)
t0 = time.time()


def run(t):
    d = f"{OUT}/runs/{t['id']}"
    if os.path.exists(d + '/episode_complete.json'):
        return t['id'], 'cached'
    cmd = [PY_, '-P', '-u', C + '/gen_collect.py', '--source-root', C + '/source',
           '--case', C + '/' + t['case'], '--route', C + '/' + t['route'], '--out', d,
           '--chrono-data', '/work1/dannegrut/harry/nrd/chrono-build/data', '--horizon-s', '120']
    with open(f"{OUT}/logs/{t['id']}.log", 'w') as lg:
        r = subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, timeout=3600)
    if r.returncode == 0:
        for f in ('rich_telemetry.npz', 'rich_telemetry.json', 'rich_intervals.npz'):
            try: os.remove(f'{d}/{f}')
            except FileNotFoundError: pass
    return t['id'], ('ok' if r.returncode == 0 else f'rc={r.returncode}')


with ThreadPoolExecutor(max_workers=int(os.environ.get('GEN_WORKERS', '12'))) as ex:
    bad = [(i, s) for i, s in ex.map(run, rows) if s not in ('ok', 'cached')]
for i, s in bad:
    print('  FAIL', i, s, flush=True)
print(f'shard {aid} done in {time.time() - t0:.0f}s, {len(bad)} failures', flush=True)
