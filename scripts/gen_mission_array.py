"""Cluster array runner for the five-goal missions: all arms of a mission run in the same array task."""
import json, os, subprocess, time
from concurrent.futures import ThreadPoolExecutor
C = os.environ['GEN_ROOT']
rows = [t for t in json.load(open(os.environ['GEN_TASKS'])) if t['shard'] == int(os.environ['SLURM_ARRAY_TASK_ID'])]
OUT = os.environ['GEN_OUT']
os.makedirs(OUT + '/runs', exist_ok=True); os.makedirs(OUT + '/logs', exist_ok=True)
print(f"shard {os.environ['SLURM_ARRAY_TASK_ID']}: {len(rows)} mission runs", flush=True)
t0 = time.time()


def run(t):
    d = f"{OUT}/runs/{t['mission_id']}__{t['arm']}"
    if os.path.exists(d + '/mission_outcome.json'):
        return t, 'cached'
    cmd = [os.environ['NRD_PYTHON'], '-P', '-u', C + '/planner/gen_mission_runner.py', '--mission', C + '/' + t['mission'],
           '--arm', t['arm'], '--out', d, '--source-root', C + '/source',
           '--chrono-data', '/work1/dannegrut/harry/nrd/chrono-build/data']
    if os.environ.get('GEN_SENSOR_MAPS'):   # sensor_v1: plan every leg from the arena's captured RGB-D image
        m = json.load(open(C + '/' + t['mission']))
        cmd += ['--sensor-map', os.environ['GEN_SENSOR_MAPS'] + '/' + os.path.basename(m['arena'])]
    with open(f"{OUT}/logs/{t['mission_id']}__{t['arm']}.log", 'w') as lg:
        r = subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, timeout=7200)
    return t, ('ok' if r.returncode == 0 else f'rc={r.returncode}')


with ThreadPoolExecutor(max_workers=int(os.environ.get('GEN_WORKERS', '12'))) as ex:
    bad = [(t, s) for t, s in ex.map(run, rows) if s not in ('ok', 'cached')]
for t, s in bad:
    print('  FAIL', t['mission_id'], t['arm'], s, flush=True)
print(f'done in {time.time() - t0:.0f}s, {len(bad)} failures', flush=True)
