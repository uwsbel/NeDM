"""Collect the AMD render-latency results (harness + split probe) into one table (summary.json / summary.csv)."""
import csv, json, re
from pathlib import Path

A = Path(__file__).resolve().parent.parent
rows = []
for f in sorted((A / 'results').glob('lvp_*.json')):
    if f.name.endswith('_decision.json'):
        continue
    d = json.loads(f.read_text())
    m = re.match(r'lvp_(\d+)_lp(\d+)_(\d+)(_rgb)?$', d['label'])
    part, lp = 'mi' + m.group(1) + 'x', int(m.group(2))
    for size, r in d['sizes'].items():
        mv, st = r['regimes']['moving'], r['regimes']['static']
        row = {'label': d['label'], 'partition': part, 'host': d['host'], 'threads': lp, 'size': int(size),
               'with_rgb': d['with_rgb'], 'scene_build_s': r['scene_build_s'],
               'first_frame_s': r['first_frame']['update_s'] + r['first_frame']['take_s'],
               'static_median_s': st['total_median_s'], 'moving_median_s': mv['total_median_s'],
               'moving_min_s': mv['total_min_s'], 'moving_max_s': mv['total_max_s'],
               'update_median_s': mv['update_median_s'], 'take_median_s': mv['take_median_s'],
               'valid_fraction': r['valid_fraction']}
        pf = A / 'probe' / (d['label'].replace('lvp_', 'probe_') + '.json')
        if pf.exists():
            p = json.loads(pf.read_text())['sizes'].get(size)
            if p:
                g = p['regimes']
                row.update({'probe_frozen_median_s': g['frozen']['total_median_s'],
                            'probe_moving_median_s': g['moving']['total_median_s'],
                            'probe_rebuild_extra_s': g['moving']['total_median_s'] - g['frozen']['total_median_s'],
                            'probe_moving_main_cpu_s': g['moving']['main_cpu_median_s'],
                            'probe_frozen_worker_cpu_s': g['frozen']['worker_cpu_median_s'],
                            'probe_trace_cpu_us_per_pixel': 1e6 * g['frozen']['worker_cpu_median_s'] / int(size) ** 2,
                            'probe_rebuilds': {k: v['rebuilt_count'] for k, v in g.items()},
                            'probe_threads_in_process': p['n_threads']})
        rows.append(row)
for f in sorted((A / 'results').glob('*_decision.json')):
    rows.append({'label': json.loads(f.read_text())['label'], 'decision': json.loads(f.read_text())})
extra = []
for f in sorted((A / 'probe').glob('probe_*.json')):
    if re.match(r'probe_\d+_lp\d+_\d+(_rgb)?\.json$', f.name):
        continue
    d = json.loads(f.read_text())
    for size, p in d['sizes'].items():
        g = p['regimes']
        extra.append({'label': d['label'], 'host': d['host'], 'threads': int(d['env']['LP_NUM_THREADS']), 'size': int(size),
                      'hide_vehicle': d.get('hide_vehicle', False), 'vector_width': d['env'].get('LP_NATIVE_VECTOR_WIDTH'),
                      'frozen_median_s': g['frozen']['total_median_s'], 'static_median_s': g['static']['total_median_s'],
                      'moving_median_s': g['moving']['total_median_s'],
                      'moving_main_cpu_s': g['moving']['main_cpu_median_s'],
                      'frozen_worker_cpu_s': g['frozen']['worker_cpu_median_s'],
                      'threads_in_process': p['n_threads']})
(A / 'summary.json').write_text(json.dumps({'harness_rows': rows, 'probe_only_rows': extra}, indent=1) + '\n')
keys = sorted({k for r in rows if 'size' in r for k in r if k != 'probe_rebuilds'})
with open(A / 'summary.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore'); w.writeheader()
    for r in rows:
        if 'size' in r:
            w.writerow(r)
fmt = lambda v: f'{v:.3f}' if isinstance(v, float) else str(v)
for r in rows:
    if 'size' in r:
        print(r['partition'], r['threads'], r['size'], 'rgb' if r['with_rgb'] else '   ',
              *[fmt(r.get(k, '')) for k in ('scene_build_s', 'first_frame_s', 'static_median_s', 'moving_median_s',
                                           'update_median_s', 'take_median_s', 'probe_frozen_median_s',
                                           'probe_rebuild_extra_s', 'probe_moving_main_cpu_s', 'probe_trace_cpu_us_per_pixel')])
    else:
        print(r)
for e in extra:
    print('probe-only', e['label'], e['threads'], e['size'], e['hide_vehicle'], e['vector_width'],
          *[fmt(e[k]) for k in ('frozen_median_s', 'static_median_s', 'moving_median_s', 'moving_main_cpu_s',
                                'frozen_worker_cpu_s', 'threads_in_process')])
