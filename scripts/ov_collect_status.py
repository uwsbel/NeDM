#!/usr/bin/env python3
"""Progress of an offroad_vehicles_20260927 soil launch (smoke or collection), for the monitor (module M3).
Plain python3 (3.9+), json only; run on the cluster login node:

  python3 -B ov_collect_status.py --tasks $G4/tasks/smoke_v1.json --out $G4/soil_v1 [--tiers 0-6] [--json f] [--stale-min 25]

Key = the arm of a row (row field 'arm', else the id prefix before '__', else 'hmmwv'), so polaris, gatorctl, m113,
polaris_pc, bitid_hmmwv, ... are counted apart; the sample (A / B / bitid / collection) is shown next to it.
Per key: rows (run: true), complete (episode_complete.json), failed ids (failed/<id>.json, attempts), ids out of
attempts, launch-check failures (collection_failure.json naming the settled-launch check, or a log line saying so),
status mix, simulated hours, mean simulated seconds per complete run, wall seconds per simulated second (outcome.json
wall_s / elapsed_s), claims in flight, and STALLS = claims without a complete run whose heartbeat (claim mtime) is older
than --stale-min minutes (a killed job; another worker re-takes them after 25 min) or drives running longer than their
row's timeout. Per tier: complete / rows (+ in flight); 'complete up to tier K'. Also: retired workers (log lines "3
consecutive failures"), Tracebacks in job logs, STOP_CLAIMS present, worker status files updated in the last 10 min.
"""
import argparse, glob, json, os, time
from collections import Counter, defaultdict


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def key_of(r):
    if r.get('arm'):
        return r['arm']
    return r['id'].split('__', 1)[0] if '__' in r['id'] else 'hmmwv'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tiers', default=None, help='A-B: rows left in this tier range per key')
    ap.add_argument('--stale-min', type=float, default=25.0)
    ap.add_argument('--max-attempts', type=int, default=2)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    now = time.time()
    tasks = load(a.tasks)
    if tasks is None:
        raise SystemExit(f'cannot read task file {a.tasks}')
    rows = [r for r in tasks if r.get('run', True)]
    runs, claims_dir, failed_dir, logs_dir = (os.path.join(a.out, d) for d in ('runs', 'claims', 'failed', 'logs'))
    have = set(os.listdir(runs)) if os.path.isdir(runs) else set()
    claims = set(os.listdir(claims_dir)) if os.path.isdir(claims_dir) else set()
    failed = {n[:-5]: (load(os.path.join(failed_dir, n)) or {}) for n in os.listdir(failed_dir)} if os.path.isdir(failed_dir) else {}
    K = defaultdict(lambda: dict(rows=0, complete=0, failed_ids=0, out_of_attempts=0, launch_fail=0, in_flight=0, stalled=0,
                                 sim_s=0.0, wall_s=0.0, status=Counter(), sample=Counter(), stalled_ids=[]))
    T = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    for r in rows:
        k = key_of(r); b = K[k]; b['rows'] += 1; b['sample'][r.get('sample', 'collection')] += 1
        tid = r['id']; t = int(r.get('tier', 0)); T[k][t][0] += 1
        d = os.path.join(runs, tid)
        if tid in have and os.path.exists(os.path.join(d, 'episode_complete.json')):
            b['complete'] += 1; T[k][t][1] += 1
            o = load(os.path.join(d, 'outcome.json')) or {}
            b['status'][o.get('status', '?')] += 1
            b['sim_s'] += float(o.get('elapsed_s', 0) or 0); b['wall_s'] += float(o.get('wall_s', 0) or 0)
            continue
        if tid in failed:
            b['failed_ids'] += 1
            if int(failed[tid].get('attempts', 0) or 0) >= a.max_attempts:
                b['out_of_attempts'] += 1
        cf = load(os.path.join(d, 'collection_failure.json')) if tid in have else None
        launch = bool(cf and 'launch' in str(cf.get('error', '')).lower())
        if not launch and tid in failed:
            try:
                with open(os.path.join(logs_dir, f'{tid}.log'), errors='replace') as f:
                    launch = 'Invalid settled launch' in f.read()
            except OSError:
                pass
        b['launch_fail'] += int(launch)
        if tid in claims:
            b['in_flight'] += 1; T[k][t][2] += 1
            try:
                age = now - os.stat(os.path.join(claims_dir, tid)).st_mtime
                owner = open(os.path.join(claims_dir, tid, 'owner')).read().split()
                started = float(owner[2]) if len(owner) > 2 else None
            except (OSError, ValueError):
                age, started = 0.0, None
            too_long = started is not None and now - started > float(r.get('timeout_s', 2400)) + 300
            if age > a.stale_min * 60 or too_long:
                b['stalled'] += 1
                if len(b['stalled_ids']) < 5:
                    b['stalled_ids'].append(tid)
    retired = tracebacks = 0
    for f in glob.glob(os.path.join(logs_dir, '*.out')):
        try:
            s = open(f, errors='replace').read()
        except OSError:
            continue
        retired += s.count('3 consecutive failures'); tracebacks += int('Traceback' in s)
    workers = [load(f) or {} for f in glob.glob(os.path.join(a.out, 'workers', '*.json'))]
    live = [w for w in workers if now - float(w.get('updated', 0) or 0) < 600]
    tot = dict(time=time.strftime('%F %T'), tasks=a.tasks, out=a.out, rows=len(rows), complete=sum(b['complete'] for b in K.values()),
               failed_ids=sum(b['failed_ids'] for b in K.values()), stalled=sum(b['stalled'] for b in K.values()),
               in_flight=sum(b['in_flight'] for b in K.values()), sim_h=round(sum(b['sim_s'] for b in K.values()) / 3600, 2),
               retired_workers=retired, job_logs_with_traceback=tracebacks, worker_files=len(workers), workers_updated_10min=len(live),
               stop_claims=os.path.exists(os.path.join(a.out, 'STOP_CLAIMS')))
    print(json.dumps(tot))
    lo = hi = None
    if a.tiers:
        lo, _, hi = a.tiers.partition('-'); lo, hi = int(lo), int(hi or lo)
    res = dict(total=tot, keys={})
    for k in sorted(K, key=lambda k: (min(T[k]), k)):
        b = K[k]; tiers = T[k]
        upto = None
        for t in sorted(tiers):
            if tiers[t][1] == tiers[t][0]:
                upto = t
            else:
                break
        wps = b['wall_s'] / b['sim_s'] if b['sim_s'] else None
        mean = b['sim_s'] / b['complete'] if b['complete'] else None
        line = ' '.join(f'{t}:{v[1]}/{v[0]}' + (f'(+{v[2]})' if v[2] else '') for t, v in sorted(tiers.items()))
        left = sum(v[0] - v[1] for t, v in tiers.items() if lo is not None and lo <= t <= hi) if lo is not None else None
        print(f"{k:14s} {'/'.join(sorted(b['sample'])):10s} {b['complete']:6d}/{b['rows']:<6d} failed {b['failed_ids']:3d} "
              f"(out of attempts {b['out_of_attempts']}, launch {b['launch_fail']})  flight {b['in_flight']:3d} stalled {b['stalled']:2d}  "
              f"sim {b['sim_s'] / 3600:6.2f} h  mean {mean if mean is None else round(mean, 1)} s  wall/sim {wps if wps is None else round(wps, 2)}  "
              + ' '.join(f'{s}={n}' for s, n in b['status'].most_common())
              + (f'  left in {a.tiers}: {left}' if left is not None else '') + (f"  stalled e.g. {b['stalled_ids']}" if b['stalled_ids'] else ''))
        print(f"{'':14s} tiers {line}  complete up to tier {upto}")
        res['keys'][k] = dict({x: v for x, v in b.items() if x not in ('status', 'sample')}, status=dict(b['status']), sample=dict(b['sample']),
                              tiers={str(t): dict(rows=v[0], complete=v[1], in_flight=v[2]) for t, v in sorted(tiers.items())},
                              complete_up_to_tier=upto, wall_per_sim=wps, mean_sim_s=mean, rows_left_in_range=left)
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
