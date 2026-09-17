"""Cut a `perf record -k CLOCK_MONOTONIC` profile of render_split_probe.py into its regimes.

perf script must have been produced with:  perf script -i DATA -F tid,time,ip,sym,dso --no-demangle
Reports CPU seconds (samples / frequency) per regime split by thread (main vs other) and by shared object, plus the
top symbols of the main thread and of the worker threads.
"""
import argparse, collections, json, re, subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--script', required=True)
ap.add_argument('--probe', required=True)
ap.add_argument('--freq', type=float, required=True)
ap.add_argument('--main-tid', type=int, default=None)
ap.add_argument('--out', required=True)
a = ap.parse_args()
probe = json.loads(Path(a.probe).read_text())
windows = []  # (start, end, name)
for size, rec in probe['sizes'].items():
    ff = rec['first_frame']['mono']; windows.append((ff[0], ff[1], f'{size}/first'))
    for regime, r in rec['regimes'].items():
        for f in r['frames']:
            windows.append((f['mono'][0], f['mono'][1], f'{size}/{regime}'))
windows.sort()
wall = collections.defaultdict(float)
for s, e, n in windows:
    wall[n] += e - s
line_re = re.compile(r'^\s*(\d+)\s+(\d+\.\d+):\s+([0-9a-f]+)\s+(.*)$')
agg = {}
tids = collections.Counter()
main_tid = a.main_tid if a.main_tid is not None else probe.get('pid')
import bisect
starts = [w[0] for w in windows]
with open(a.script, errors='replace') as fh:
    for line in fh:
        m = line_re.match(line)
        if not m:
            continue
        tid, t, rest = int(m.group(1)), float(m.group(2)), m.group(4)
        tids[tid] += 1
        i = bisect.bisect_right(starts, t) - 1
        if i < 0 or t > windows[i][1]:
            continue
        name = windows[i][2]
        k = rest.find(' (')
        sym, dso = (rest[:k], rest[k + 2:].rstrip(')')) if k >= 0 else (rest, '?')
        dso_short = dso.split('/')[-1] if dso.startswith('/') and 'memfd' not in dso else ('JIT/anon:' + dso)
        g = agg.setdefault(name, {'samples': 0, 'by_thread': collections.Counter(), 'by_dso_main': collections.Counter(),
                                  'by_dso_other': collections.Counter(), 'sym_main': collections.Counter(),
                                  'sym_other': collections.Counter(), 'tids': set()})
        g['samples'] += 1; g['tids'].add(tid)
        cls = 'main' if tid == main_tid else 'other'
        g['by_thread'][cls] += 1
        g[f'by_dso_{cls}'][dso_short] += 1
        g[f'sym_{cls}'][f'{sym} [{dso_short}]'] += 1

def demangle(names):
    try:
        res = subprocess.run(['c++filt'], input='\n'.join(names), capture_output=True, text=True, timeout=60).stdout.split('\n')
        return res[:len(names)]
    except Exception:
        return names

out = {'freq': a.freq, 'main_tid': main_tid, 'regimes': {}}
lines = []
for name in sorted(agg, key=lambda n: (int(n.split('/')[0]), n)):
    g = agg[name]; s = 1. / a.freq
    rec = {'wall_s': wall[name], 'cpu_s': g['samples'] * s, 'n_threads_sampled': len(g['tids']),
           'cpu_s_by_thread': {k: v * s for k, v in g['by_thread'].items()},
           'cpu_s_by_dso_main': {k: v * s for k, v in g['by_dso_main'].most_common(12)},
           'cpu_s_by_dso_other': {k: v * s for k, v in g['by_dso_other'].most_common(12)}}
    for cls, n in (('main', 30), ('other', 15)):
        top = g[f'sym_{cls}'].most_common(n)
        names = demangle([t[0].split(' [')[0] for t in top])
        rec[f'top_sym_{cls}'] = [[nm[:160] + ' [' + t[0].split(' [')[-1], t[1] * s] for nm, t in zip(names, top)]
    out['regimes'][name] = rec
    lines.append(f"== {name}: wall {rec['wall_s']:.2f} s, cpu {rec['cpu_s']:.2f} s, threads {rec['n_threads_sampled']}, "
                 f"main {rec['cpu_s_by_thread'].get('main', 0):.2f} s, other {rec['cpu_s_by_thread'].get('other', 0):.2f} s")
    lines.append('  main dso: ' + ', '.join(f'{k} {v:.2f}' for k, v in rec['cpu_s_by_dso_main'].items()))
    lines.append('  other dso: ' + ', '.join(f'{k} {v:.2f}' for k, v in rec['cpu_s_by_dso_other'].items()))
    for nm, v in rec['top_sym_main'][:15]:
        lines.append(f'    main {v:7.2f}  {nm}')
    for nm, v in rec['top_sym_other'][:6]:
        lines.append(f'    other {v:7.2f}  {nm}')
out['tid_sample_counts'] = dict(tids.most_common(40))
Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
Path(a.out).with_suffix('.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
