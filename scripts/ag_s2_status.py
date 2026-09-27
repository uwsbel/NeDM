#!/usr/bin/env python3
"""Progress of the soil evaluation rows of soil_v3 (arena_gator_20260925, soil track step 2). Login node, python3,
standard library only.
Per tier (declared priority group) and per first arm / vehicle: rows, complete (episode_complete.json), claimed but not
complete (running or dead), failed attempts, simulated seconds of the complete ones; also the reused soil_v2 drives the
mappings point to, and the soil_v2 training tiers 7-12 still open.
  python3 ag_s2_status.py --tasks $G3/tasks/soil_v3.json --out $G3/soil_v1 [--json f]
"""
import argparse, json, os, time
from collections import Counter, defaultdict
from pathlib import Path

NAMES = {-9: 'A+B primary', -8: 'M3b, M2', -7: 'anchors', -6: 'in distribution', -5: 'extra straight6 held-out'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--json')
    a = ap.parse_args()
    out = Path(a.out)
    rows = json.load(open(a.tasks))
    ev = [r for r in rows if r.get('kind') == 'eval']
    tr = [r for r in rows if r.get('tier', -1) >= 7]
    tot = defaultdict(Counter)
    sim = Counter()
    by_arm = defaultdict(Counter)
    for r in ev:
        t = r['tier']
        d = out / 'runs' / r['id']
        done = (d / 'episode_complete.json').exists()
        claimed = (out / 'claims' / r['id']).exists()
        failed = (out / 'failed' / f"{r['id']}.json").exists()
        tot[t]['rows'] += 1
        tot[t]['complete'] += done
        tot[t]['claimed_open'] += (claimed and not done)
        tot[t]['failed_attempt'] += failed
        k = f"{t}|{r['vehicle']}|{r['arms'][0]}"
        by_arm[k]['rows'] += 1; by_arm[k]['complete'] += done
        if done:
            try:
                sim[t] += json.load(open(d / 'episode_complete.json')).get('actual_elapsed_s', 0.0)
            except Exception:
                pass
    trc = Counter()
    for r in tr:
        trc['rows'] += 1
        trc['complete'] += (out / 'runs' / r['id'] / 'episode_complete.json').exists()
    res = dict(time=time.strftime('%H:%M:%S'), tiers={str(t): dict(tot[t], name=NAMES.get(t, ''), sim_h=round(sim[t] / 3600, 2))
                                                   for t in sorted(tot)},
               eval_rows=len(ev), eval_complete=sum(v['complete'] for v in tot.values()),
               training_tiers_7_12=dict(trc), by_arm={k: dict(v) for k, v in sorted(by_arm.items())},
               failed_files=len(os.listdir(out / 'failed')) if (out / 'failed').exists() else None)
    print(f"{res['time']}  evaluation rows complete {res['eval_complete']}/{res['eval_rows']}; training tiers 7-12 {trc['complete']}/{trc['rows']}")
    for t in sorted(tot):
        v = tot[t]
        print(f"  tier {t:3d} {NAMES.get(t, ''):26s} {v['complete']:5d}/{v['rows']:5d} complete, {v['claimed_open']:4d} running, "
              f"{v['failed_attempt']:3d} with a failed attempt, {sim[t] / 3600:6.2f} sim-h")
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
