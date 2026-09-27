#!/usr/bin/env python3
"""arena_gator_20260925 E5a (resumed): tables from scripts/ag_offline_auc.py outputs.

For every model (ensemble) and evaluation file: within-group AUC of unsafe (W_unsafe, the ranking the planner uses)
from a standing start (startup rows) and from moving starts (established rows), on
  val      the val split of the arena (never fitted by any model of this study),
  dev+val  the dev fold (ga_train's md5(group) % 5 == 0 training groups) plus val, only where the model fitted none of
           those groups (holdout-mode models; deploy models on arenas they were not trained on).
Plus the members' startup W_unsafe range (training-noise indication) and the lowest-risk-route failure rate.

  python scripts/ag_auc_table.py e5/offline/auc_*.json --out-md e5/offline/auc_table.md --out-json e5/offline/auc_summary.json
numpy-free (json only)."""
import argparse, json, os


def fmt(x, nd=3):
    return '-' if x is None else f'{x:.{nd}f}'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+'); ap.add_argument('--out-md'); ap.add_argument('--out-json')
    ap.add_argument('--order', default='', help='comma-separated model order (others follow alphabetically)')
    ap.add_argument('--domain', default='rigid')
    a = ap.parse_args(argv)
    models, evals, src = {}, {}, {}
    for p in a.files:
        r = json.load(open(p))
        for en, ev in r['evals'].items():
            if en in evals:
                assert evals[en]['sha256'] == ev['sha256'], f'evaluation {en} differs between files ({p})'
            evals[en] = ev
        for mn, m in r['models'].items():
            assert mn not in models, f'model {mn} appears twice ({src.get(mn)} and {p})'
            models[mn] = dict(meta=m, res=r['results'][mn]); src[mn] = p
    order = [x for x in a.order.split(',') if x in models] + sorted(x for x in models if x not in a.order.split(','))
    enames = list(evals)
    summ = dict(files={p: os.path.basename(p) for p in a.files}, evals=evals, models={})
    lines = ['| model | mode | fitted groups | ' + ' | '.join(f'{e} val: start / moving (groups) | {e} dev+val start' for e in enames) + ' |',
             '|---|---|---|' + '---|---|' * len(enames)]
    for mn in order:
        m = models[mn]; meta = m['meta']
        row = [mn, ','.join(meta['modes']), str(meta['fitted_groups'])]
        ms = dict(modes=meta['modes'], fitted_groups=meta['fitted_groups'], ds=meta['ds'],
                  checkpoints=[dict(path=c['path'], sha256=c['sha256'], seed=c['seed']) for c in meta['checkpoints']],
                  evals={})
        # ag_offline_auc's --check-trainer pairs row sets by row count only; keep the pairs per evaluation file, because an
        # evaluation of the other vehicle on the same arena can have the same count (then it compares different rows)
        tc = {}
        for k, c in meta.get('trainer_check', {}).items():
            en = k.split('|')[1]
            tc[en] = max(tc.get(en, 0.0), c['diff'])
        ms['trainer_check_max_diff_by_eval'] = tc
        for en in enames:
            blk = m['res'].get(en, {})
            out = {}
            for sname in ('val', 'dev+val'):
                b = blk.get(sname, {})
                ens = b.get('ensemble', {}).get(a.domain)
                if not ens or (sname == 'dev+val' and b.get('partly_fitted')):
                    out[sname] = None; continue
                mem = b.get('members_W_unsafe', {}).get('startup', [])
                out[sname] = dict(groups=b['groups'], rows=b['rows'] - b['rows_dropped_fitted'],
                                  W_unsafe_start=ens['startup']['W_unsafe'], W_unsafe_moving=ens['established']['W_unsafe'],
                                  W_fail_start=ens['startup']['W_fail'], pick_fail_unsafe_start=ens['startup']['pick_fail_unsafe'],
                                  random_fail_unsafe_start=ens['startup']['random_fail_unsafe'], rate_unsafe_start=ens['startup']['rate_unsafe'],
                                  members_W_unsafe_start=[min(mem), max(mem)] if mem else None)
            ms['evals'][en] = out
            v, dv = out['val'], out['dev+val']
            row.append(f"{fmt(v['W_unsafe_start'])} / {fmt(v['W_unsafe_moving'])} ({v['groups']})" if v else '-')
            row.append(f"{fmt(dv['W_unsafe_start'])} ({dv['groups']})" if dv else 'fitted')
        summ['models'][mn] = ms
        lines.append('| ' + ' | '.join(row) + ' |')
    txt = '\n'.join(lines) + '\n'
    print(txt)
    if a.out_md:
        open(a.out_md, 'w').write(txt)
    if a.out_json:
        json.dump(summ, open(a.out_json, 'w'), indent=1)


if __name__ == '__main__':
    main()
