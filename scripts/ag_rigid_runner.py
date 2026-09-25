"""Rigid runner for the arena_gator_20260925 Gator pilot (E3b1): a copy of gen_runner_g.py whose only change is
AG_KEEP_RICH=1 (default here): rich telemetry (chassis contact forces, rich_intervals.npz) is kept after a
successful episode instead of deleted, so the pilot can check chassis grounding.  Everything else is gen_runner_g.py:

Cluster array runner for the generalist rigid drives (copy of gen_runner.py parameterised by environment variables).

  GEN_ROOT       experiment root (default /work1/dannegrut/harry/experiments/generalist_20260921); GEN_ROOT/source holds the
                 rsync'd src/, scripts/, assets/ and source_manifest.json
  GEN_COLLECTOR  collector script (default GEN_ROOT/source/scripts/gen_collect_ext.py)
  GEN_TASKS      tasks json: rows {id, case, route, shard[, run, mode, extra: [...]]}; case/route relative to GEN_ROOT unless
                 absolute; 'mode' becomes --mode; 'extra' is appended verbatim (absolute paths for --recorded / --branch-route /
                 --actor)
  GEN_OUT        output root: runs/<id>, logs/<id>.log
  GEN_WORKERS    thread-pool size (one collector subprocess each; branch_auto rows run their subprocesses sequentially
                 unless 'extra' carries --cont-parallel)
  GEN_TIMEOUT_S  per-task timeout (default 3600; branch_auto rows get 4x: replay + 3 continuations)
  GEN_CHRONO_DATA chrono data dir (default /work1/dannegrut/harry/nrd/chrono-build/data, exactly gen_array.sbatch's)
All rows of a shard run in one array task with a thread pool, so every arm of a group shares a node (Chrono is
deterministic per node only).  Rich telemetry is deleted after a successful episode from runs/<id> and from the
branch_auto sibling dirs runs/<id>__replay, runs/<id>__c*.
"""
import glob, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
C = os.environ.get('GEN_ROOT', '/work1/dannegrut/harry/experiments/generalist_20260921')
COLLECTOR = os.environ.get('GEN_COLLECTOR', C + '/source/scripts/gen_collect_ext.py')
CHRONO_DATA = os.environ.get('GEN_CHRONO_DATA', '/work1/dannegrut/harry/nrd/chrono-build/data')
TASKS = os.environ['GEN_TASKS']
OUT = os.environ['GEN_OUT']
PY_ = os.environ.get('NRD_PYTHON', 'python3')
TIMEOUT = float(os.environ.get('GEN_TIMEOUT_S', '3600'))
aid = int(os.environ['SLURM_ARRAY_TASK_ID'])
rows = [t for t in json.load(open(TASKS)) if t['shard'] == aid and t.get('run', True)]
os.makedirs(OUT + '/runs', exist_ok=True); os.makedirs(OUT + '/logs', exist_ok=True)
print(f'shard {aid}: {len(rows)} episodes; collector {COLLECTOR}; root {C}', flush=True)
t0 = time.time()


def _abs(p):
    return p if os.path.isabs(p) else C + '/' + p


def run(t):
    d = f"{OUT}/runs/{t['id']}"
    if os.path.exists(d + '/episode_complete.json'):
        return t['id'], 'cached'
    cmd = [PY_, '-P', '-u', COLLECTOR, '--source-root', C + '/source',
           '--case', _abs(t['case']), '--route', _abs(t['route']), '--out', d,
           '--chrono-data', CHRONO_DATA, '--horizon-s', '120']
    mode = t.get('mode')
    if mode:
        cmd += ['--mode', mode]
    cmd += [str(x) for x in t.get('extra', [])]
    timeout = TIMEOUT * (1 + int(t.get('n_cont', 3))) if mode == 'branch_auto' else TIMEOUT
    with open(f"{OUT}/logs/{t['id']}.log", 'w') as lg:
        lg.write(' '.join(cmd) + '\n'); lg.flush()
        try:
            r = subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, timeout=timeout)
            rc = r.returncode
        except subprocess.TimeoutExpired:
            rc = 'timeout'
    if rc == 0 and os.environ.get('AG_KEEP_RICH', '1') != '1':
        for dd in [d] + glob.glob(d + '__*'):
            for f in ('rich_telemetry.npz', 'rich_telemetry.json', 'rich_intervals.npz'):
                try: os.remove(f'{dd}/{f}')
                except FileNotFoundError: pass
    return t['id'], ('ok' if rc == 0 else f'rc={rc}')


with ThreadPoolExecutor(max_workers=int(os.environ.get('GEN_WORKERS', '12'))) as ex:
    bad = [(i, s) for i, s in ex.map(run, rows) if s not in ('ok', 'cached')]
for i, s in bad:
    print('  FAIL', i, s, flush=True)
print(f'shard {aid} done in {time.time() - t0:.0f}s, {len(bad)} failures', flush=True)
