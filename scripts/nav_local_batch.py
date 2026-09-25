"""Run a nav_v1 task list on luffy: every (mission, arm) through scripts/nav_local.py, N at a time.

  python3 scripts/nav_local_batch.py --tasks TASKS.json --out DIR [--workers 6] [--video]
"""
import argparse, json, os, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MISSIONS = REPO / 'artifacts/traverse/fdm_f104_50h_20260909/nav_v1/missions'
MODELS = str(REPO / 'artifacts/traverse/fdm_f104_50h_20260909/sensor_v2/matched/matched_Dabs_s*.pt')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--video', action='store_true', help='capture RGB video (overhead + chase) for every task')
    a = ap.parse_args()
    out = Path(a.out); (out / 'runs').mkdir(parents=True, exist_ok=True); (out / 'logs').mkdir(exist_ok=True)
    tasks = json.load(open(a.tasks))
    env = dict(os.environ, PYTHONPATH='/home/harry/chrono/build/bin', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def run(t):
        d = out / 'runs' / f"{t['mission']}__{t['arm']}"
        if (d / 'mission_outcome.json').exists():
            return t, 'cached', 0.
        cmd = ['/usr/bin/python3.12', '-u', str(REPO / 'scripts/nav_local.py'),
               '--mission', str(MISSIONS / f"{t['mission']}.json"), '--mode', t['mode'], '--period', str(t['period']),
               '--out', str(d), '--source-root', str(REPO), '--chrono-data', '/home/harry/chrono/data',
               '--models', MODELS, '--device', 'cuda', '--torch-threads', '1']
        for k, flag in (('latency_s', '--latency-s'), ('sense_radius_m', '--sense-radius-m'), ('pick', '--pick'),
                        ('switch_margin', '--switch-margin'), ('latency_replay', '--latency-replay')):
            if k in t:
                cmd += [flag, str(t[k])]
        if t.get('keep_current'):
            cmd += ['--keep-current']
        if a.video:
            cmd += ['--video-dir', str(d / 'video'), '--video-every', '2', '--chase-cam']
        t0 = time.time()
        with open(out / 'logs' / f"{t['mission']}__{t['arm']}.log", 'w') as lg:
            r = subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, env=env, cwd=str(REPO), timeout=4 * 3600)
        return t, ('ok' if r.returncode == 0 else f'rc={r.returncode}'), time.time() - t0

    t_start = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for t, status, wall in ex.map(run, tasks):
            done += 1
            print(f"[{done}/{len(tasks)}] {t['mission']}__{t['arm']}: {status} ({wall:.0f} s, elapsed {time.time() - t_start:.0f} s)",
                  flush=True)
    print('batch done', round(time.time() - t_start), 's', flush=True)


if __name__ == '__main__':
    main()
