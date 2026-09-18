"""Offline read-out on CRM-labelled routes: CRM-trained ensemble vs the frozen rigid N2 ensemble vs simple rules.

Rows = held-out start-goal groups of the CRM dataset (case split val/test; never fitted by the CRM model; NOTE the
rigid ensemble was fitted on the rigid twins of these routes). Metrics: pooled and within-group AUC for goal-not-reached,
and route choice: failure rate of each scorer's lowest-risk route per group (vs random and oracle).
  python scripts/crm_offline_compare.py --ds <npz> --crm-logits <CRM_N2_<mode>_logits.npz> [--rows heldout|dev]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))


def auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    if len(y) == 0 or y.min() == y.max(): return float('nan')
    r = np.argsort(np.argsort(s)) + 1.0; n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def within(y, s, g):
    ok = tot = 0.0
    for c in np.unique(g):
        m = g == c; yy, ss = y[m], s[m]
        if yy.min() == yy.max(): continue
        d = ss[yy == 1][:, None] - ss[yy == 0][None, :]
        ok += (d > 0).sum() + 0.5 * (d == 0).sum(); tot += d.size
    return ok / tot if tot else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True); ap.add_argument('--crm-logits', required=True)
    ap.add_argument('--rows', default='heldout'); ap.add_argument('--out', default=None)
    ap.add_argument('--rigid', default='artifacts/traverse/fdm_f104_50h_20260909/night2_v1/final/N2_s*.pt')
    a = ap.parse_args()
    import gen_planner as P
    d = np.load(a.ds, allow_pickle=True); L = np.load(a.crm_logits, allow_pickle=True)
    assert (d['id'] == L['id']).all()
    m = L[a.rows].astype(bool)
    X = d['X'][m].astype(np.float32); ctx = d['ctx'][m][:, 17:22]
    y = d['fail'][m].astype(float); g = d['group'][m].astype(str)
    z_rigid = P.RiskModel(a.rigid).score(X, ctx)[0]
    z_crm = L['ensemble_logit'][m]
    speed = X[:, 3].mean((1, 2))
    rng = np.random.default_rng(0)
    scorers = {'crm_trained': z_crm, 'rigid_frozen': z_rigid, 'faster_is_safer': -speed, 'random': rng.random(len(y))}
    out = dict(rows=a.rows, n_routes=int(len(y)), n_groups=int(len(np.unique(g))), fail_rate=float(y.mean()), scorers={})
    for k, s in scorers.items():
        picks = [y[g == c][np.argmin(s[g == c])] for c in np.unique(g)]
        out['scorers'][k] = dict(pooled_auc=auc(y, s), within_group_auc=float(within(y, s, g)), picked_route_fail_rate=float(np.mean(picks)))
    out['oracle_picked_fail_rate'] = float(np.mean([y[g == c].min() for c in np.unique(g)]))
    boot = []
    cs = np.unique(g); pc = np.array([y[g == c][np.argmin(z_crm[g == c])] for c in cs]); pr = np.array([y[g == c][np.argmin(z_rigid[g == c])] for c in cs])
    for _ in range(4000):
        j = rng.integers(0, len(cs), len(cs)); boot.append(pc[j].mean() - pr[j].mean())
    out['picked_fail_crm_minus_rigid'] = dict(diff=float(pc.mean() - pr.mean()), ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                                              crm_worse=int(((pc == 1) & (pr == 0)).sum()), rigid_worse=int(((pc == 0) & (pr == 1)).sum()))
    print(json.dumps(out, indent=1))
    if a.out: json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
