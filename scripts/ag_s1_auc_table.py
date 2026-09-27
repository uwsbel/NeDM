#!/usr/bin/env python3
"""arena_gator_20260925 soil stage 1: plain-text / markdown table of scripts/ag_offline_auc.py output.

  python scripts/ag_s1_auc_table.py <auc.json> [--set val|dev+val|auto] [--md out.md]

Per model and evaluation file: within-group AUC of 'unsafe' (W_unsafe) from a standing start and from moving starts,
the failure rate of the lowest-risk offline pick per group (standing start), rows and groups. 'auto' = dev+val for
holdout-mode models (dev fold not fitted), val for deploy-mode models (their dev fold was fitted). Cells whose rows were
partly fitted by the model are marked '*' and use only the unfitted rows.
"""
import argparse, json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('auc'); ap.add_argument('--set', default='auto', choices=['auto', 'val', 'dev+val', 'dev'])
    ap.add_argument('--md', default=None)
    a = ap.parse_args()
    J = json.load(open(a.auc))
    evals = list(J['evals'])
    lines = ['| model | mode | rows used | ' + ' | '.join(f'{e}: start AUC / moving AUC / pick fail (groups)' for e in evals) + ' |',
             '|---|---|---|' + '---|' * len(evals)]
    for m, info in J['models'].items():
        mode = '/'.join(info['modes'])
        s = a.set if a.set != 'auto' else ('dev+val' if mode == 'holdout' else 'val')
        cells = []
        for e in evals:
            blk = J['results'][m][e][s]
            ens = blk.get('ensemble')
            if not ens:
                cells.append('-'); continue
            dk = next(iter(ens))
            st, mv = ens[dk]['startup'], ens[dk]['established']
            star = '*' if blk['partly_fitted'] else ''
            cells.append(f"{st['W_unsafe']:.3f} / {mv['W_unsafe']:.3f} / {st.get('pick_fail_unsafe', float('nan')):.3f} ({st.get('n_groups', blk['groups'])}){star}")
        lines.append(f'| {m} | {mode} | {s} | ' + ' | '.join(cells) + ' |')
    txt = '\n'.join(lines)
    print(txt)
    if a.md:
        open(a.md, 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
