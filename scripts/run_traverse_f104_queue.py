#!/usr/bin/env python3
"""Shared AMD F104 queue; quota counts only verified, completed 50 ms intervals.

Initialize once, then run one process per allocation with configurable local
subprocess concurrency. A durable completion ledger repairs interrupted state
updates. Claimed tasks are never silently recycled; retries keep separate files.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
import traceback

DT = .05
CORE_SOURCE = ("scripts/traverse_fdm_rgbd_diverse_chrono.py", "src/nedm/traverse/scene.py",
    "src/nedm/hmmwv_data.py", "src/nedm/traverse/terrain.py", "src/nedm/traverse/layout.py",
    "src/nedm/traverse/fdm_data.py", "src/nedm/traverse/fdm_diverse_data.py",
    "src/nedm/traverse/fdm_rich_telemetry.py", "src/nedm/training/constants.py")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+f'.{os.getpid()}.{threading.get_ident()}.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n')
        f.flush(); os.fsync(f.fileno())
        return f.tell()


def checked(path, expected):
    path = Path(path).resolve()
    require(path.is_file() and sha(path) == expected, f'Hash mismatch: {path}')
    return path


def path_from(base, value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (Path(base)/p).resolve()


def load_contract(path):
    path = Path(path).resolve()
    c = read(path)
    for key in ('campaign_root', 'source_root', 'source_manifest_sha256', 'collector_path',
                'collector_sha256', 'task_file', 'task_sha256', 'target_seconds', 'chrono_data'):
        require(key in c, f'Missing contract field {key}')
    for key in ('campaign_root', 'source_root', 'collector_path', 'task_file', 'chrono_data'):
        c[key] = str(path_from(path.parent, c[key]))
    require(abs(float(c.get('dt', DT))-DT) < 1e-12, 'Recording dt must be exactly .05 s')
    require(0 < float(c.get('horizon', c.get('horizon_s', 120.))) <= 120., 'Horizon outside (0,120]')
    require(math.isfinite(float(c['target_seconds'])) and float(c['target_seconds']) > 0, 'Invalid target')
    c['_contract_path'], c['_contract_sha256'] = str(path), sha(path)
    c['_target_frames'] = math.ceil(float(c['target_seconds'])/DT-1e-8)
    return c


def load_tasks(c):
    path = checked(c['task_file'], c['task_sha256'])
    value = read(path)
    tasks = value if isinstance(value, list) else value.get('tasks')
    require(isinstance(tasks, list) and len(tasks) > 0, 'Task file must contain a nonempty list or tasks list')
    ids = set()
    for task in tasks:
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', task['id']) is not None, 'Unsafe task ID')
        require(task['id'] not in ids, 'Duplicate task ID'); ids.add(task['id'])
        require(task['split'] in ('train', 'val', 'test') and task.get('group_id'), 'Missing group/split')
        for field in ('case', 'route'):
            require(re.fullmatch('[0-9a-f]{64}', task[field+'_sha256']) is not None, 'Invalid input SHA')
            task[field] = str(path_from(path.parent, task[field]))
    return tasks


class Queue:
    def __init__(self, contract, tasks):
        self.c, self.tasks = contract, tasks
        self.root = Path(contract['campaign_root'])
        self.q = self.root/'queue'
        self.local_lock = threading.Lock()

    @contextmanager
    def locked(self):
        # flock is additionally serialized within this Python process.
        with self.local_lock:
            with (self.q/'queue.lock').open('a+') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)

    def initialize(self):
        self.q.mkdir(parents=True, exist_ok=True)
        with self.locked():
            require(not (self.q/'state.json').exists(), 'Queue already initialized; refusing to reset quota')
            require(not (self.q/'completed_ledger.jsonl').exists(), 'Existing ledger; refusing reset')
            state = dict(schema='f104_shared_queue_v1', contract_sha256=self.c['_contract_sha256'],
                task_sha256=self.c['task_sha256'], task_count=len(self.tasks), next_index=0,
                target_frames=self.c['_target_frames'], target_seconds=float(self.c['target_seconds']),
                completed_frames=0, completed_seconds=0., completed_count=0, failures=0,
                ledger_offset=0, inflight={}, completed_by_split={}, created_unix_s=time.time())
            (self.q/'completed_ledger.jsonl').touch(exist_ok=False)
            atomic_json(self.q/'state.json', state)
        return state

    def state(self):
        state = read(self.q/'state.json')
        require(state['contract_sha256'] == self.c['_contract_sha256'], 'Queue contract changed')
        require(state['task_sha256'] == self.c['task_sha256'], 'Queue task set changed')
        # A completion is journaled before the state commit. Replay only the
        # uncommitted tail, so a crash at either boundary neither loses nor adds time.
        with (self.q/'completed_ledger.jsonl').open('rb') as f:
            f.seek(state['ledger_offset'])
            for line in f:
                require(line.endswith(b'\n'), 'Incomplete durable ledger line: stop for explicit recovery')
                entry = json.loads(line)
                self.apply_completion(state, entry)
                state['ledger_offset'] += len(line)
        return state

    @staticmethod
    def apply_completion(state, entry):
        require(entry['sequence'] == state['completed_count']+1, 'Invalid completion sequence')
        claim = state['inflight'].pop(entry['id'], None)
        require(claim is not None and claim['index'] == entry['index'], 'Completion does not own an active claim')
        state['completed_count'] += 1
        state['completed_frames'] += entry['validated_frames']
        state['completed_seconds'] = state['completed_frames']*DT
        split = state['completed_by_split'].setdefault(entry['split'], {'count':0, 'frames':0, 'seconds':0.})
        split['count'] += 1; split['frames'] += entry['validated_frames']; split['seconds'] = split['frames']*DT
        state['last_completion_unix_s'] = entry['completed_unix_s']

    def claim(self, node):
        with self.locked():
            state = self.state()
            if state['completed_frames'] >= state['target_frames'] or state['next_index'] >= len(self.tasks):
                atomic_json(self.q/'state.json', state)
                return None
            index = state['next_index']; state['next_index'] += 1
            task = self.tasks[index]
            state['inflight'][task['id']] = {'index':index, 'node':node, 'claimed_unix_s':time.time()}
            atomic_json(self.q/'state.json', state)
            return index, task

    def complete(self, index, task, result):
        with self.locked():
            state = self.state()
            entry = dict(result, sequence=state['completed_count']+1, id=task['id'], index=index,
                group_id=task['group_id'], split=task['split'], completed_unix_s=time.time())
            offset = append(self.q/'completed_ledger.jsonl', entry)
            self.apply_completion(state, entry); state['ledger_offset'] = offset
            atomic_json(self.q/'state.json', state)
            return state['completed_seconds']

    def fail(self, index, task, error):
        with self.locked():
            state = self.state()
            claim = state['inflight'].pop(task['id'], None)
            require(claim is not None and claim['index'] == index, 'Failure does not own active claim')
            state['failures'] += 1
            append(self.q/'failures.jsonl', dict(id=task['id'], index=index, node=claim['node'], error=error,
                failed_unix_s=time.time(), counted_frames=0, retry_policy='Retain all attempts; no automatic queue reset/reclaim'))
            atomic_json(self.q/'state.json', state)


def bind_node(c, node_dir):
    source = Path(c['source_root'])
    manifest = read(checked(source/'source_manifest.json', c['source_manifest_sha256']))
    mapping = manifest['files']
    for name in CORE_SOURCE:
        require(name in mapping, 'Missing mandatory source: '+name)
    actual_sources = {}
    for name, value in mapping.items():
        if name.endswith('.py'):
            checked(source/name, value); actual_sources[name] = value
    checked(c['collector_path'], c['collector_sha256'])
    if c.get('worker_sha256'):
        checked(__file__, c['worker_sha256'])
    runtime = c.get('runtime_files_sha256')
    if runtime is None:
        fp = checked(c['runtime_fingerprint'], c['runtime_fingerprint_sha256'])
        runtime = read(fp)['runtime_sha256']
    require(runtime and any('_vehicle.so' in x for x in runtime) and any('/vehicle/hmmwv/' in x for x in runtime), 'Incomplete native/vehicle runtime inventory')
    for name, value in runtime.items():
        checked(path_from(c['chrono_data'], name), value)
    runtime = {str(path_from(c['chrono_data'], name)):value for name,value in runtime.items()}
    runtime_path = node_dir/'runtime_fingerprint.json'
    atomic_json(runtime_path, {'runtime_sha256':runtime})
    env = os.environ.copy()
    env['FDM_RUNTIME_FINGERPRINT'] = str(runtime_path)
    env['PYTHONPATH'] = str(source/'src')+os.pathsep+str(source/'scripts')+os.pathsep+env.get('PYTHONPATH','')
    sys.path[:0] = [str(source/'src'), str(source/'scripts')]
    checker = source/'scripts/check_traverse_fdm_rich_telemetry.py'
    require(checker.is_file() and 'scripts/check_traverse_fdm_rich_telemetry.py' in actual_sources, 'Telemetry checker must be in frozen source manifest')
    spec = importlib.util.spec_from_file_location('f104_frozen_rich_check', checker)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    atomic_json(node_dir/'node_binding.json', {'contract_sha256':c['_contract_sha256'],
        'source_manifest_sha256':c['source_manifest_sha256'], 'source_files_sha256':actual_sources,
        'runtime_files_sha256':runtime, 'runtime_fingerprint_sha256':sha(runtime_path),
        'collector_sha256':c['collector_sha256'], 'worker_sha256':sha(__file__),
        'python':sys.executable, 'hostname':socket.gethostname(),
        'thread_environment':{k:env.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LP_NUM_THREADS')},
        'verified_unix_s':time.time()})
    return env, module.verify, runtime


def verify_episode(out, task, c, verify_rich, runtime):
    import numpy as np
    marker_path = out/'episode_complete.json'
    marker = read(marker_path)
    require(marker['schema'] == 'f104_episode_complete_v1', 'Wrong completion schema')
    required = {'trajectory.npz','rich_telemetry.npz','rich_intervals.npz','rich_telemetry.json',
                'outcome.json','f104_episode.json','collection_request.json','command_reference.npz','case.json','reference.json'}
    require(required <= set(marker['artifacts_sha256']), 'Incomplete marker payload')
    for name, value in marker['artifacts_sha256'].items():
        path = (out/name).resolve()
        require(out.resolve() in path.parents and path != marker_path.resolve(), 'Unsafe marker payload path')
        checked(path, value)
    request = read(out/'collection_request.json')
    require(sha(out/'collection_request.json') == marker['request_sha256'], 'Request hash mismatch')
    for key in ('case_sha256','route_sha256'):
        require(request[key] == task[key], 'Task/request input mismatch: '+key)
    require(request['source_manifest_sha256'] == c['source_manifest_sha256'] and request['wrapper_sha256'] == c['collector_sha256'], 'Collector/source mismatch')
    require(request['runtime_sha256'] == runtime and request['split'] == task['split'], 'Runtime/split mismatch')
    require(request['scene_id'] == task['group_id'], 'Group/scene mismatch')
    summary, outcome = read(out/'f104_episode.json'), read(out/'outcome.json')
    require(summary['initial_state_valid'] and summary['native_height_valid'], 'Failed physical/terrain startup validation')
    with np.load(out/'trajectory.npz', allow_pickle=False) as data:
        n = len(data['state'])
        require(n > 0 and n == outcome['frames'] == summary['interval_count'], 'Frame count mismatch')
        require(abs(float(data['dt_s'])-DT) < 1e-8, 'Wrong dt (allow float32 storage encoding only)')
        require(data['state'].shape == (n,17) and data['action'].shape == (n,3) and data['pose'].shape == (n,3), 'Trajectory shapes mismatch')
        require(data['terminal_state'].shape == (17,) and data['terminal_pose'].shape == (3,), 'Missing measured terminal')
        for key in ('state','action','pose','terminal_state','terminal_pose','power_kw','positive_work_kj_per_interval'):
            require(np.isfinite(data[key]).all(), 'Nonfinite trajectory '+key)
    for value in (marker['actual_elapsed_s'], summary['actual_elapsed_s'], outcome['elapsed_s']):
        require(abs(float(value)-n*DT) < 1e-8, 'Measured duration mismatch')
    require(n*DT <= float(c.get('horizon', c.get('horizon_s',120.)))+1e-8, 'Elapsed exceeds declared horizon')
    rich = verify_rich(out, trajectory=out/'trajectory.npz', require_solver_steps=True)
    require(rich['intervals'] == n, 'Rich/trajectory count mismatch')
    with np.load(out/'rich_intervals.npz', allow_pickle=False) as intervals:
        for key in ('engine_interface_positive_work_kj','max_abs_roll_rad','max_abs_pitch_rad',
                    'max_chassis_contact_resultant_n','max_asset_contact_max_resultant_n'):
            require(np.isfinite(intervals[key]).all(), 'Missing required interval field '+key)
        np.testing.assert_allclose(intervals['duration_s'], DT, rtol=0, atol=1e-9)
    with np.load(out/'rich_telemetry.npz', allow_pickle=False) as values:
        np.testing.assert_allclose(values['time_s'], np.arange(n+1)*DT, rtol=0, atol=1e-8)
        for wheel in ('fl','fr','rl','rr'):
            for suffix in ('longitudinal_slip','force_world_vertical_n','force_projected_terrain_normal_n'):
                key = f'tire_{wheel}_{suffix}'
                require(np.isfinite(values[key]).all(), 'Missing required tire field '+key)
    validation = {'passed':True, 'validated_frames':n, 'actual_elapsed_s':n*DT, 'rich_check':rich,
        'episode_complete_sha256':sha(marker_path), 'all_marker_payload_hashes_verified':True}
    atomic_json(out/'queue_validation.json', validation)
    return {'validated_frames':n, 'actual_elapsed_s':n*DT, 'out':str(out), 'status':outcome['status'],
        'episode_complete_sha256':validation['episode_complete_sha256'],
        'queue_validation_sha256':sha(out/'queue_validation.json')}


def run(c, tasks, workers):
    require(workers > 0, 'Workers must be positive')
    q = Queue(c,tasks)
    node = re.sub('[^A-Za-z0-9_.-]', '_', f"{os.environ.get('SLURM_JOB_ID','local')}_{socket.gethostname()}_{os.getpid()}")
    node_dir = q.root/'node_status'/node
    node_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(node_dir/'status.json', {'node':node,'status':'binding','workers':workers,'started_unix_s':time.time()})
    env, rich_verify, runtime = bind_node(c,node_dir)
    stop = threading.Event(); stats_lock = threading.Lock()
    stats = {'node':node, 'workers':workers, 'status':'running', 'completed':0, 'failures':0,
             'consecutive_errors':0, 'validated_frames':0, 'started_unix_s':time.time()}
    atomic_json(node_dir/'status.json',stats)
    def worker():
        while not stop.is_set():
            claim = q.claim(node)
            if claim is None:
                return
            index, task = claim
            error = None
            attempts = int(c.get('max_attempts',1))
            require(attempts >= 1, 'max_attempts must be positive')
            for attempt in range(attempts):
                out = q.root/'runs'/(task['id'] if attempt == 0 else task['id']+f'__retry_{attempt:03d}')
                committing = False
                try:
                    require(not out.exists(), f'Existing output preserved; refusing reuse: {out}')
                    out.mkdir(parents=True)
                    for field in ('case','route'):
                        checked(task[field],task[field+'_sha256'])
                    cmd = [sys.executable,'-P','-u',c['collector_path'],'--source-root',c['source_root'],
                        '--source-manifest-sha256',c['source_manifest_sha256'],'--case',task['case'],
                        '--case-sha256',task['case_sha256'],'--route',task['route'],'--route-sha256',task['route_sha256'],
                        '--out',str(out),'--chrono-data',c['chrono_data'],'--horizon-s',str(c.get('horizon',c.get('horizon_s',120.)))]
                    atomic_json(out/'queue_launch.json',{'task':task,'index':index,'attempt':attempt,'node':node,
                        'argv':cmd,'contract_sha256':c['_contract_sha256'],'launched_unix_s':time.time()})
                    with (out/'collector.log').open('w') as log:
                        process = subprocess.run(cmd,env=env,cwd=c['source_root'],stdout=log,stderr=subprocess.STDOUT,
                            timeout=float(c.get('episode_wall_timeout_s',1800.)),check=False)
                    require(process.returncode == 0, f'Collector exit {process.returncode}; see {out}/collector.log')
                    result = verify_episode(out,task,c,rich_verify,runtime)
                    # Quota commit errors are fatal, not collector retries: the
                    # durable ledger may already contain this completion.
                    committing = True
                    total = q.complete(index,task,result)
                    with stats_lock:
                        stats['completed'] += 1; stats['validated_frames'] += result['validated_frames']; stats['consecutive_errors'] = 0
                        stats['last_global_completed_seconds'] = total; stats['updated_unix_s'] = time.time()
                        atomic_json(node_dir/'status.json',stats)
                    error = None
                    break
                except Exception as exc:
                    if committing:
                        stop.set()
                        raise
                    error = ''.join(traceback.format_exception(type(exc),exc,exc.__traceback__))
                    atomic_json(node_dir/f'error_{index:06d}_{attempt:03d}.json',{'id':task['id'],'out':str(out),'error':error,'counted_frames':0})
            if error is not None:
                q.fail(index,task,error)
                with stats_lock:
                    stats['failures'] += 1; stats['consecutive_errors'] += 1; stats['updated_unix_s'] = time.time()
                    if stats['consecutive_errors'] >= 5:
                        stop.set(); stats['stop_reason'] = 'Five consecutive implementation/process/validation errors'
                    atomic_json(node_dir/'status.json',stats)
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(worker) for _ in range(workers)]
            for future in futures:
                future.result()
        stats['status'] = 'failed' if stop.is_set() else 'complete'
    except BaseException:
        stats['status'] = 'failed'; stats['fatal_error'] = traceback.format_exc()
        raise
    finally:
        stats['finished_unix_s'] = time.time()
        atomic_json(node_dir/'status.json',stats)
    print(json.dumps(stats))
    return 1 if stats['status']=='failed' else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--initialize',action='store_true')
    mode.add_argument('--status',action='store_true',help='Read atomic state only; no imports, mutations or full inventory scan')
    args = p.parse_args(); c = load_contract(args.contract)
    if args.status:
        print(json.dumps(read(Path(c['campaign_root'])/'queue/state.json'),indent=2)); return
    tasks = load_tasks(c)
    if args.initialize:
        print(json.dumps(Queue(c,tasks).initialize(),indent=2)); return
    raise SystemExit(run(c,tasks,args.workers))


if __name__ == '__main__':
    main()
