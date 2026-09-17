"""Night-2: data-scaling curve + final models on the merged dataset.

Answers the user's question 1 directly: hold architecture, labels and protocol fixed and vary how much data the
model is trained on. Two axes:
  quantity : 25 / 50 / 75 / 100% of the designed routes (old campaign train groups + wave A1)
  kind     : + wave A2 (routes drawn from the planner's own proposal distribution)
Metrics on the dev fold (groups held out of fitting):
  G_unsafe      same-group same-speed-profile ranking AUC (night-1's selection metric)
  pick_unsafe   the planner-relevant one: rank each dev group's designed routes, take the model's top choice,
                and report how often that choice was unsafe in Chrono (lower is better)
"""
import argparse, hashlib, json, os, sys
import numpy as np, torch
sys.path.insert(0, 'scripts')
from f104_n2_train import Net, Data, train_one, predict, CTX_COLS
from f104_night_train import metrics, dev_group

ROOT = 'artifacts/traverse/fdm_f104_50h_20260909'


def pick_quality(s, d, mask):
    """For each group in the mask: the model's lowest-risk route among that group's routes -> was it unsafe?"""
    grp = d['group'][mask].astype(str); uns = d['unsafe'][mask].astype(int); fl = d['fail'][mask].astype(int)
    sc = s[mask]
    out_u, out_f = [], []
    for g in np.unique(grp):
        m = grp == g
        k = int(np.argmin(sc[m]))
        out_u.append(uns[m][k]); out_f.append(fl[m][k])
    return float(np.mean(out_u)), float(np.mean(out_f)), len(out_u)


def subset(D, frac, sources, seed=0):
    """Fraction of GROUPS (not rows) from the designed pool, plus the requested extra sources."""
    src = D.d['source'].astype(str); grp = D.d['group'].astype(str)
    base = D.fit & np.isin(src, ['designed'])
    groups = np.unique(grp[base])
    rng = np.random.default_rng(seed); keep = set(rng.permutation(groups)[:max(1, int(round(frac * len(groups))))])
    m = base & np.array([g in keep for g in grp])
    for s in sources:
        m = m | (D.fit & (src == s) & np.array([g in keep for g in grp]))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', default=ROOT + '/night2_v1/station_ds_all.npz')
    ap.add_argument('--ctx', default='none')
    ap.add_argument('--arch', default='gru')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--curve', default='0.25,0.5,0.75,1.0')
    ap.add_argument('--final-seeds', type=int, default=5)
    ap.add_argument('--out', default=ROOT + '/night2_v1')
    a = ap.parse_args()
    D = Data(a.ds, a.ctx)
    src = D.d['source'].astype(str)
    print('rows', D.n, '| sources', {s: int((src == s).sum()) for s in np.unique(src)},
          '| fit', int(D.fit.sum()), 'dev', int(D.dev.sum()), flush=True)
    dev_designed = D.dev & (src == 'designed')
    rows = []
    for sources, tag in (([], 'designed'), (['on_policy'], 'designed+on_policy')):
        for frac in [float(x) for x in a.curve.split(',')]:
            if sources and frac != 1.0: continue
            m = subset(D, frac, sources)
            for seed in range(a.seeds):
                model, met = train_one(D, arch=a.arch, seed=seed, epochs=a.epochs, fit_mask=m)
                sc = predict(model, D, np.arange(D.n))
                pu, pf, ng = pick_quality(sc, D.d, dev_designed)
                rows.append(dict(tag=tag, frac=frac, n_rows=int(m.sum()),
                                 n_groups=len(np.unique(D.d['group'][m].astype(str))),
                                 pick_unsafe=pu, pick_fail=pf, pick_groups=ng, **met))
                print(f"  {tag:20s} frac {frac:4.2f} s{seed}  rows {int(m.sum()):6d}  G_unsafe {met['G_unsafe']:.3f}  "
                      f"G_fail {met['G_fail']:.3f}  pick_unsafe {pu:.3f}  pick_fail {pf:.3f}", flush=True)
                json.dump(rows, open(a.out + '/scaling.json', 'w'), indent=1)
    print('wrote', a.out + '/scaling.json')


if __name__ == '__main__':
    main()
