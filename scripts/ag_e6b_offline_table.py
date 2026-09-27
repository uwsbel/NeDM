#!/usr/bin/env python3
"""Offline within-group ranking on the 8 unseen test arenas (arena_gator_20260925, module E6b; REVIEW_R1 5c).

Reads the per-row ensemble logits written by scripts/ag_score_offline.py --save-logits (e6/offline/<tag>.logits.npz)
and the labels of the evaluation-only files (e4/evalonly/<arena>_rigid_test). Standing-start rows only (anchor frame 0:
the 12 designed routes of each group, the decision the planner makes). Per model: within-group AUC of unsafe and of
fail (ga_train.cell_auc: route pairs inside a group, pooled over groups), per arena, near / spread / all; the lowest-risk
route's unsafe and fail rate (random route for reference). Paired model contrasts with a group bootstrap (2,000
resamples of the 2,000 groups; composite M1 = mean of the M1a / M1b values, M3 = mean of M3a / M3b).
  PYTHONPATH=src:scripts python scripts/ag_e6b_offline_table.py --out $K3/e6/offline/offline_unseen
"""
import argparse, glob, json, os
from pathlib import Path
import numpy as np

K3 = Path('artifacts/traverse/arena_gator_20260925')
NEAR = ('g260', 'g271', 'g251', 'g247'); SPREAD = ('g258', 'g268', 'g263', 'g241')


def per_group(y, s, g):
    """{group: (concordant, pairs)} as ga_train.cell_auc (ties count 1/2)."""
    out = {}
    for c in np.unique(g):
        m = g == c; yy, ss = y[m], s[m]
        if yy.min() == yy.max():
            out[c] = (0.0, 0); continue
        d = ss[yy == 1][:, None] - ss[yy == 0][None, :]
        out[c] = (float((d > 0).sum() + 0.5 * (d == 0).sum()), int(d.size))
    return out


def load(tag):
    L = np.load(K3 / 'e6' / 'offline' / f'{tag}.logits.npz', allow_pickle=True)
    files = [str(f) for f in L['files']]
    rows = []
    for i, f in enumerate(files):
        z = np.load(f, allow_pickle=True)
        ids = L[f'id_{i}'].astype(str); assert (ids == z['id'].astype(str)).all(), f
        st = z['anchor_frame'].astype(int) == 0
        rows.append(dict(arena=str(z['arena'][0]), group=z['group'].astype(str)[st], unsafe=z['unsafe'][st].astype(int),
                         fail=z['fail'][st].astype(int), logit=L[f'ensemble_logit_{i}'][st].astype(float)))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--boot', type=int, default=2000)
    a = ap.parse_args()
    tags = [Path(p).name[:-len('.logits.npz')] for p in sorted(glob.glob(str(K3 / 'e6/offline/*.logits.npz')))]
    stats, groups_ref = {}, None
    for t in tags:
        per = {}
        for r in load(t):
            for lab in ('unsafe', 'fail'):
                for g, v in per_group(r[lab], r['logit'], r['group']).items():
                    per.setdefault(lab, {})[g] = v
            # lowest-risk route per group (highest logit = highest risk; the planner picks the lowest)
            for g in np.unique(r['group']):
                m = r['group'] == g; k = np.argmin(r['logit'][m])
                per.setdefault('pick_unsafe', {})[g] = float(r['unsafe'][m][k]); per.setdefault('pick_fail', {})[g] = float(r['fail'][m][k])
                per.setdefault('rand_unsafe', {})[g] = float(r['unsafe'][m].mean()); per.setdefault('rand_fail', {})[g] = float(r['fail'][m].mean())
        stats[t] = per
        gs = sorted(per['unsafe'])
        groups_ref = gs if groups_ref is None else groups_ref
        assert gs == groups_ref, f'{t}: other groups'
    G = np.array(groups_ref); arena = np.array([g[:4] for g in G])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(G), (a.boot, len(G)))

    def arr(t, key):
        return np.array([stats[t][key][g] for g in G], float) if key.startswith(('pick', 'rand')) else \
            (np.array([stats[t][key][g][0] for g in G]), np.array([stats[t][key][g][1] for g in G], float))

    def W(t, lab, sel):
        ok, n = arr(t, lab); return float(ok[sel].sum() / n[sel].sum())
    sets = dict(all=np.ones(len(G), bool), near=np.isin(arena, NEAR), spread=np.isin(arena, SPREAD), **{ar: arena == ar for ar in NEAR + SPREAD})
    table = {}
    for t in tags:
        table[t] = {s: dict(W_unsafe=W(t, 'unsafe', m), W_fail=W(t, 'fail', m), pick_unsafe=100 * arr(t, 'pick_unsafe')[m].mean(),
                            pick_fail=100 * arr(t, 'pick_fail')[m].mean(), random_unsafe=100 * arr(t, 'rand_unsafe')[m].mean(),
                            random_fail=100 * arr(t, 'rand_fail')[m].mean(), groups=int(m.sum())) for s, m in sets.items()}
    comp = {'M1': ['M1a_deploy', 'M1b_deploy'], 'M3': ['M3a_deploy', 'M3b_deploy'], 'M2': ['M2_deploy'], 'A3': ['A3_deploy'],
            'M1a': ['M1a_deploy'], 'M1b': ['M1b_deploy'], 'M3a': ['M3a_deploy'], 'M3b': ['M3b_deploy']}

    contr = {}
    for name, (x, y) in dict(M3_vs_M1=('M3', 'M1'), A3_vs_M1=('A3', 'M1'), M2_vs_M1=('M2', 'M1'), A3_vs_M3=('A3', 'M3'),
                             seed_M1a_vs_M1b=('M1a', 'M1b'), seed_M3a_vs_M3b=('M3a', 'M3b')).items():
        if not all(t in stats for t in comp[x] + comp[y]):
            continue
        res = {}
        for s in ('all', 'near', 'spread'):
            m = sets[s]; Gi = np.where(m)[0]
            I = Gi[rng.integers(0, len(Gi), (a.boot, len(Gi)))]
            out = {}
            for lab in ('unsafe', 'fail'):
                def wboot(members):
                    b = []
                    for t in members:
                        ok, n = arr(t, lab); b.append(ok[I].sum(1) / n[I].sum(1))
                    return np.mean(b, 0)
                def wpt(members):
                    return float(np.mean([W(t, lab, m) for t in members]))
                d = wpt(comp[x]) - wpt(comp[y]); db = wboot(comp[x]) - wboot(comp[y])
                out[f'W_{lab}'] = dict(test=wpt(comp[x]), ref=wpt(comp[y]), diff=d, ci95=[float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))])
            for key in ('pick_unsafe', 'pick_fail'):
                vx = np.mean([arr(t, key) for t in comp[x]], 0); vy = np.mean([arr(t, key) for t in comp[y]], 0)
                dd = 100 * (vx - vy)
                out[key] = dict(test=100 * vx[m].mean(), ref=100 * vy[m].mean(), diff_pts=float(dd[m].mean()),
                                ci95=[float(np.percentile(dd[I].mean(1), 2.5)), float(np.percentile(dd[I].mean(1), 97.5))])
            res[s] = out
        contr[name] = res
    json.dump(dict(tool='scripts/ag_e6b_offline_table.py', models=tags, groups=len(G), rows='standing-start rows (12 designed routes per group)',
                   table=table, contrasts=contr), open(a.out + '.json', 'w'), indent=1)
    L = ['| model | all 8 unseen: W unsafe | W fail | near | spread | lowest-risk route unsafe (random) | ' + ' | '.join(NEAR + SPREAD) + ' |',
         '|---|---|---|---|---|---|' + '---|' * 8]
    for t in tags:
        T = table[t]
        L.append(f"| {t} | {T['all']['W_unsafe']:.3f} | {T['all']['W_fail']:.3f} | {T['near']['W_unsafe']:.3f} | {T['spread']['W_unsafe']:.3f} | "
                 f"{T['all']['pick_unsafe']:.1f} % ({T['all']['random_unsafe']:.1f} %) | " + ' | '.join(f"{T[ar]['W_unsafe']:.3f}" for ar in NEAR + SPREAD) + ' |')
    L.append('')
    for n, r in contr.items():
        for s, o in r.items():
            L.append(f"{n} {s}: W_unsafe {o['W_unsafe']['test']:.4f} vs {o['W_unsafe']['ref']:.4f} diff {o['W_unsafe']['diff']:+.4f} "
                     f"[{o['W_unsafe']['ci95'][0]:+.4f}, {o['W_unsafe']['ci95'][1]:+.4f}]; W_fail diff {o['W_fail']['diff']:+.4f} "
                     f"[{o['W_fail']['ci95'][0]:+.4f}, {o['W_fail']['ci95'][1]:+.4f}]; pick unsafe {o['pick_unsafe']['test']:.2f} vs {o['pick_unsafe']['ref']:.2f} %")
    Path(a.out + '.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
