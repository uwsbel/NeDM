"""Ensemble-level offline read-out for the matched v2 variants (the planner uses the ensemble, not a single seed).

For each variant: average the per-seed route logits, then report route choice at matched speed on the held-out arenas
(the review's primary offline metric), plus ranking AUCs, per arena and pooled, with a paired bootstrap over
start/goal groups for the depth-minus-height difference.
"""
import argparse, glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_riskmodel import Net, route_logit
from sensor_train_v2 import Data, VARIANTS, route_choice, CH
from sensor_train import metrics

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def ens_scores(files, variant, ck_glob, train_arenas, eval_arenas):
    D = Data(files, variant, train_arenas, eval_arenas)
    zs = []
    for p in sorted(glob.glob(ck_glob)):
        ck = torch.load(p, map_location=DEV, weights_only=False)
        assert ck['channels'] == VARIANTS[variant], (ck['channels'], variant)
        m = Net(ck['cin'], ck['nctx'], arch=ck['arch'], layers=ck.get('layers', 2)).to(DEV)
        m.load_state_dict(ck['state']); m.eval()
        out = []
        with torch.no_grad():
            for i in range(0, D.n, 1024):
                out.append(route_logit(m(torch.tensor(D.X[i:i + 1024], device=DEV),
                                         torch.tensor(D.ctx[i:i + 1024], device=DEV))).float().cpu().numpy())
        zs.append(np.concatenate(out))
    return D, np.mean(zs, 0), len(zs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', required=True); ap.add_argument('--ck-dir', required=True)
    ap.add_argument('--variants', default='H,H0,Drel,Dabs'); ap.add_argument('--tag', default='matched')
    ap.add_argument('--train-arenas', default='f104,g228,g203,g217'); ap.add_argument('--eval-arenas', default='g216,g231')
    ap.add_argument('--out', required=True); ap.add_argument('--reference', default='H')
    a = ap.parse_args()
    res, picks = {}, {}
    for v in a.variants.split(','):
        D, z, n = ens_scores(a.files, v, f'{a.ck_dir}/{a.tag}_{v}_s*.pt', a.train_arenas.split(','), a.eval_arenas.split(','))
        m = D.eval_designed
        res[v] = dict(seeds=n, pooled=route_choice(z, D.d, m), rank=metrics(z[m], D.d, m))
        for ar in sorted(set(D.d['arena'][D.eval].astype(str))):
            res[v][f'choice_{ar}'] = route_choice(z, D.d, m & (D.d['arena'].astype(str) == ar))
        # per-cell pick outcome, for the paired bootstrap
        grp = D.d['group'][m].astype(str); prof = D.d['profile'][m].astype(int); y = D.d['unsafe'][m].astype(int); s = z[m]
        cells = {}
        for i, (g, p) in enumerate(zip(grp, prof)):
            cells.setdefault((g, int(p)), []).append(i)
        pk = {k: (int(y[np.asarray(ix)[int(np.argmin(s[np.asarray(ix)]))]]), int(y[np.asarray(ix)].min() == 0))
              for k, ix in cells.items() if len(ix) >= 2}
        picks[v] = pk
        c = res[v]['pooled']
        print(f"{v:6s} ({n} seeds)  picked-unsafe {100*c['picked_unsafe']:5.2f}%  avoidable {100*c['avoidable_unsafe']:5.2f}%  "
              f"rank G {res[v]['rank']['G_unsafe']:.3f} W {res[v]['rank']['W_unsafe']:.3f}", flush=True)
    ref = a.reference
    rng = np.random.default_rng(0)
    keys = sorted(set.intersection(*[set(p) for p in picks.values()]))
    groups = sorted({k[0] for k in keys})
    by_group = {g: [k for k in keys if k[0] == g] for g in groups}
    for v in picks:
        if v == ref: continue
        d = np.array([picks[v][k][0] - picks[ref][k][0] for k in keys], float)
        boots = []
        for _ in range(4000):
            gs = rng.choice(groups, len(groups))
            idx = [k for g in gs for k in by_group[g]]
            boots.append(100 * np.mean([picks[v][k][0] - picks[ref][k][0] for k in idx]))
        b = int(sum(1 for k in keys if picks[v][k][0] > picks[ref][k][0])); c_ = int(sum(1 for k in keys if picks[v][k][0] < picks[ref][k][0]))
        res[v][f'vs_{ref}'] = dict(cells=len(keys), diff_points=100 * d.mean(),
                                   ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                   worse=b, better=c_)
        print(f"  {v} - {ref}: {100*d.mean():+.2f} points [{np.percentile(boots,2.5):+.2f}, {np.percentile(boots,97.5):+.2f}]  ({b} worse vs {c_} better cells)")
    json.dump(res, open(a.out, 'w'), indent=1, default=float)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
