#!/usr/bin/env python3
"""GPU worker pool for CRM episode collection: one worker process per visible GPU, all pulling from one task list.

  tasks.json rows: {"id", "case", "route"[, "config"]} with paths relative to CRM_ROOT.
  Claim    = mkdir of <out>/claims/<id> (never flock); the owner refreshes its mtime while the episode runs. A claim whose
             mtime is older than CRM_STALE_S and whose run has no episode_complete.json is taken over (killed job).
  Complete = <out>/runs/<id>/episode_complete.json (written atomically by crm_collect.py) -> never run again.
  Failure  = non-zero exit: attempt counted in <out>/failed/<id>.json; retried up to CRM_MAX_ATTEMPTS by any worker.

Any number of jobs on any partitions may run this at the same time against the same <out>; it is safe to cancel and
resubmit (resumable). GPUs are bound with HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES (CUDA_VISIBLE_DEVICES locally).
"""
import json, os, random, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path(os.environ['CRM_ROOT'])
TASKS = Path(os.environ['CRM_TASKS'])
OUT = Path(os.environ['CRM_OUT'])
PY = os.environ.get('NRD_PYTHON', sys.executable)
CHRONO_DATA = os.environ['CRM_CHRONO_DATA']
CONFIG = os.environ.get('CRM_CONFIG', '')
N_GPU = int(os.environ.get('CRM_GPUS', '1'))
STALE_S = float(os.environ.get('CRM_STALE_S', '1500'))
MAX_ATTEMPTS = int(os.environ.get('CRM_MAX_ATTEMPTS', '2'))
DEADLINE = time.time() + float(os.environ.get('CRM_BUDGET_S', '1e9'))   # stop claiming new work after this
EP_TIMEOUT = float(os.environ.get('CRM_EPISODE_TIMEOUT_S', '2400'))
JOB = os.environ.get('SLURM_JOB_ID', 'local')
HOST = os.uname().nodename

rows = [t for t in json.load(open(TASKS)) if t.get('run', True)]
for d in ('runs', 'claims', 'failed', 'logs', 'workers', 'stale'):
    (OUT / d).mkdir(parents=True, exist_ok=True)


def attempts(tid):
    try:
        return json.load(open(OUT / 'failed' / f'{tid}.json'))['attempts']
    except (FileNotFoundError, ValueError):
        return 0


def try_claim(tid):
    if (OUT / 'runs' / tid / 'episode_complete.json').exists() or attempts(tid) >= MAX_ATTEMPTS:
        return False
    claim = OUT / 'claims' / tid
    try:
        os.mkdir(claim)  # mkdir exclusivity is the primitive that held across nodes on this filesystem
    except FileExistsError:
        try:
            if time.time() - claim.stat().st_mtime < STALE_S:
                return False
            # stale: its job died. Exactly one renamer wins; the winner re-creates the claim.
            moved = OUT / 'stale' / f'{tid}.{int(time.time())}.{JOB}.{os.getpid()}.{random.random():.6f}'
            os.rename(claim, moved)
            if time.time() - moved.stat().st_mtime < STALE_S:  # lost a race: that was somebody's fresh claim
                os.rename(moved, claim)
                return False
            os.mkdir(claim)
        except OSError:
            return False
    try:
        (claim / 'owner').write_text(f'{JOB} {HOST} {time.time():.0f}\n')
    except OSError:
        pass
    return True


def worker(gpu):
    env = dict(os.environ)
    if os.environ.get('CRM_CUDA'):
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    elif os.environ.get('CRM_NO_GPU_BIND'):  # CPU workers (rigid Chrono): no device filtering
        pass
    else:  # filter at the ROCr layer so the process sees exactly one device, which HIP then calls device 0
        env['ROCR_VISIBLE_DEVICES'], env['HIP_VISIBLE_DEVICES'] = str(gpu), '0'
    env['OMP_NUM_THREADS'] = env.get('CRM_OMP', '4')
    order = list(range(len(rows)))
    random.Random(f'{JOB}:{HOST}:{gpu}').shuffle(order)
    order.sort(key=lambda i: rows[i].get('tier', 0))  # stable: tier by tier, random within a tier
    done = sim_s = streak = 0
    t0 = time.time()
    status_path = OUT / 'workers' / f'{JOB}_{HOST}_gpu{gpu}.json'
    for index in order:
        if time.time() > DEADLINE or (OUT / 'STOP_CLAIMS').exists():
            break
        t = rows[index]
        tid = t['id']
        if not try_claim(tid):
            continue
        run_dir = OUT / 'runs' / tid
        collector = os.environ.get('CRM_COLLECTOR', str(ROOT / 'source/scripts/crm_collect.py'))
        cmd = [PY, '-P', '-u', collector, '--source-root', str(ROOT / 'source'),
               '--case', str(ROOT / t['case']), '--route', str(ROOT / t['route']), '--out', str(run_dir),
               '--chrono-data', CHRONO_DATA, '--horizon-s', '120'] + list(t.get('extra', []))
        if t.get('episode_seed') is not None:
            cmd += ['--episode-seed', str(t['episode_seed'])]
        config = t.get('config') or CONFIG
        if config and 'crm_collect' in collector:
            cmd += ['--crm-config', str(ROOT / config)]
        started = time.time()
        with open(OUT / 'logs' / f'{tid}.log', 'a') as lg:
            proc = subprocess.Popen(cmd, stdout=lg, stderr=subprocess.STDOUT, env=env)
            while True:
                try:
                    rc = proc.wait(timeout=120)
                    break
                except subprocess.TimeoutExpired:
                    try:
                        os.utime(OUT / 'claims' / tid)  # heartbeat (directory mtime)
                    except OSError:
                        pass
                    if time.time() - started > EP_TIMEOUT:
                        proc.kill(); rc = -9
                        break
        if rc == 0 and (run_dir / 'episode_complete.json').exists():
            done += 1; streak = 0
            try:
                sim_s += json.load(open(run_dir / 'episode_complete.json'))['actual_elapsed_s']
            except Exception:
                pass
        else:
            n = attempts(tid) + 1
            json.dump({'attempts': n, 'rc': rc, 'job': JOB, 'host': HOST, 'gpu': gpu, 'time': time.time()},
                      open(OUT / 'failed' / f'{tid}.json', 'w'))
            try:  # let another worker retry
                os.rename(OUT / 'claims' / tid, OUT / 'stale' / f'{tid}.failed.{int(time.time())}.{JOB}.{gpu}')
            except OSError:
                pass
        if not (rc == 0 and (run_dir / 'episode_complete.json').exists()):
            streak += 1
            if streak >= 3:  # a broken GPU/node must not burn the retry budget of the whole queue
                print(f'worker gpu{gpu}: 3 consecutive failures, last {tid} rc={rc}; stopping this worker', flush=True)
                break
        json.dump({'job': JOB, 'host': HOST, 'gpu': gpu, 'episodes': done, 'sim_s': sim_s,
                   'wall_s': time.time() - t0, 'last': tid, 'updated': time.time()}, open(status_path, 'w'))
    print(f'worker gpu{gpu}: {done} episodes, {sim_s:.0f} sim-s in {time.time() - t0:.0f} s', flush=True)


def guarded(gpu):
    # several sweeps: tasks released by failures or dead jobs are picked up again; transient filesystem errors
    # must not retire a GPU for the rest of the job
    for sweep in range(int(os.environ.get('CRM_SWEEPS', '3'))):
        try:
            worker(gpu)
        except Exception as exc:  # noqa: BLE001
            print(f'worker gpu{gpu}: sweep {sweep} aborted by {type(exc).__name__}: {exc}', flush=True)
            time.sleep(30)
        if time.time() > DEADLINE or (OUT / 'STOP_CLAIMS').exists():
            break


threads = [threading.Thread(target=guarded, args=(g,)) for g in range(N_GPU)]
for th in threads:
    th.start()
for th in threads:
    th.join()
print('all workers done', flush=True)
