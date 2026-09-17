"""Cluster array runner for nav_v1: one shard = a list of (mission, arm) runs executed with a thread pool."""
import json, os, subprocess, time
from concurrent.futures import ThreadPoolExecutor

N = os.environ['NAV_ROOT']
OUT = os.environ['NAV_OUT']
shard = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
rows = [t for t in json.load(open(os.environ['NAV_TASKS'])) if t['shard'] == shard]
os.makedirs(OUT + '/runs', exist_ok=True); os.makedirs(OUT + '/logs', exist_ok=True)
print(f'shard {shard}: {len(rows)} runs', flush=True)
t0 = time.time()


def run(t):
    d = f"{OUT}/runs/{t['mission']}__{t['arm']}"
    if os.path.exists(d + '/mission_outcome.json'):
        return t, 'cached'
    cmd = [os.environ['NRD_PYTHON'], '-P', '-u', N + '/code/nav_runner.py',
           '--mission', f"{N}/missions/{t['mission']}.json", '--mode', t['mode'],
           '--period', str(t['period']), '--out', d,
           '--source-root', os.environ['NAV_SOURCE'],
           '--chrono-data', os.environ['NAV_CHRONO_DATA'],
           '--torch-threads', os.environ.get('NAV_TORCH_THREADS', '1')]
    for k, flag in (('latency_s', '--latency-s'), ('path_heights', '--path-heights'), ('margin_m', '--margin-m'),
                    ('switch_margin', '--switch-margin'), ('sense_radius_m', '--sense-radius-m'),
                    ('pick', '--pick')):
        if k in t:
            cmd += [flag, str(t[k])]
    if t.get('keep_current'):
        cmd += ['--keep-current']
    if t.get('no_mask'):
        cmd += ['--no-mask']
    if t.get('save_frames'):
        cmd += ['--save-frames']
    with open(f"{OUT}/logs/{t['mission']}__{t['arm']}.log", 'w') as lg:
        r = subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, timeout=int(os.environ.get('NAV_TIMEOUT', '10800')))
    return t, ('ok' if r.returncode == 0 else f'rc={r.returncode}')


with ThreadPoolExecutor(max_workers=int(os.environ.get('NAV_WORKERS', '4'))) as ex:
    res = list(ex.map(run, rows))
bad = [(t, s) for t, s in res if s not in ('ok', 'cached')]
for t, s in bad:
    print('  FAIL', t['mission'], t['arm'], s, flush=True)
print(f'done in {time.time() - t0:.0f}s, {len(bad)} failures', flush=True)
