"""Offline metrics of the route-risk training record (the numbers in the per-seed rows of ``<tag>.json``).

For each ground type (rigid, crm, and 'both' pooled) x {startup (anchor_frame 0), established, all}:
  P_/W_/G_{unsafe,fail}   pooled AUC / within-group AUC (cells 'group|ground') / same-group-and-speed-profile AUC
  pick_fail[_unsafe]      failure rate of the lowest-risk route per group cell, with random_ and oracle_ references
  brier_/ece_             calibration of P(unsafe) = 1 - exp(-exp(route logit)) (10 equal-width bins)
  S_*                     the same ranking / pick numbers inside decision states (several routes driven from one
                          identical start state: episode|anchor_frame|ground)
``all_metrics`` adds the blocks by row source and by anchor-frame bucket.
"""

from __future__ import annotations

import numpy as np

from risk_model import DOMAIN_NAME

ANCHOR_BUCKETS = [('k0', 0, 0), ('k10_30', 10, 30), ('k40p', 40, 10 ** 9)]


def rank_avg(s):
    """1-based ranks with ties given their average rank."""
    _, inv, cnt = np.unique(np.asarray(s, float), return_inverse=True, return_counts=True)
    first = np.cumsum(cnt) - cnt + 1.0
    return first[inv] + (cnt[inv] - 1) / 2.0


def auc(y, s):
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    if len(y) == 0 or y.min() == y.max():
        return float('nan')
    r = rank_avg(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def cell_auc(y, s, cells):
    """AUC over positive/negative pairs inside the same cell; (auc, number of pairs)."""
    ok = tot = 0.0
    for c in np.unique(cells):
        m = cells == c
        yy, ss = y[m], s[m]
        if len(yy) == 0 or yy.min() == yy.max():
            continue
        d = ss[yy == 1][:, None] - ss[yy == 0][None, :]
        ok += (d > 0).sum() + 0.5 * (d == 0).sum()
        tot += d.size
    return (ok / tot if tot else float('nan')), int(tot)


def calibration(p, y, nbin=10):
    """Brier score and equal-width expected calibration error."""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    if len(p) == 0:
        return float('nan'), float('nan')
    brier = float(np.mean((p - y) ** 2))
    b = np.minimum((p * nbin).astype(int), nbin - 1)
    ece = 0.0
    for i in range(nbin):
        m = b == i
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return brier, float(ece)


def regime_metrics(s, d, m):
    """Ranking, pick and calibration metrics on rows m (s = logits of those rows)."""
    out = {'n': int(m.sum())}
    if out['n'] == 0:
        return out
    grp = np.array([f'{g}|{DOMAIN_NAME.get(int(x), x)}' for g, x in zip(d['group'][m].astype(str), d['domain'][m])])
    prof = d['profile'][m].astype(int) if 'profile' in d else np.full(out['n'], -1)
    des = prof >= 0
    gcell = np.array([f'{g}|{p}' for g, p in zip(grp, prof)])
    p = 1.0 - np.exp(-np.exp(np.asarray(s, float)))
    for lab in ('unsafe', 'fail'):
        y = d[lab][m].astype(float)
        out[f'P_{lab}'] = auc(y, s)
        out[f'W_{lab}'], out[f'Wn_{lab}'] = cell_auc(y, s, grp)
        out[f'G_{lab}'], out[f'Gn_{lab}'] = cell_auc(y[des], s[des], gcell[des]) if des.any() else (float('nan'), 0)
        out[f'brier_{lab}'], out[f'ece_{lab}'] = calibration(p, y)
        out[f'rate_{lab}'] = float(y.mean())
    out['mean_p'] = float(p.mean())
    out['unsafe_not_fail_rate'] = float(((d['unsafe'][m] == 1) & (d['fail'][m] == 0)).mean())
    for lab in ('fail', 'unsafe'):   # route choice: lowest-risk route per group|ground cell vs random / oracle
        y = d[lab][m].astype(float)
        picks, rand, orac = [], [], []
        for g in np.unique(grp):
            k = np.where(grp == g)[0]
            if len(k) < 2:
                continue
            picks.append(y[k[np.argmin(s[k])]])
            rand.append(y[k].mean())
            orac.append(y[k].min())
        suf = '' if lab == 'fail' else '_unsafe'
        out.update({f'pick_fail{suf}': float(np.mean(picks)) if picks else float('nan'),
                    f'random_fail{suf}': float(np.mean(rand)) if rand else float('nan'),
                    f'oracle_fail{suf}': float(np.mean(orac)) if orac else float('nan')})
        if lab == 'fail':
            out['n_groups'] = len(picks)
    return out


def cell_index(cells):
    """Row-index arrays of the cells with >= 2 rows."""
    _, inv, cnt = np.unique(cells, return_inverse=True, return_counts=True)
    order = np.argsort(inv, kind='stable')
    starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    return [order[s:s + c] for s, c in zip(starts, cnt) if c >= 2]


def state_metrics(s, d, m):
    """Decision-state cells episode|anchor_frame|ground (singleton cells skipped): within-state AUC S_* and the
    lowest-risk pick failure per state."""
    if 'episode' not in d or m.sum() == 0:
        return {'S_cells': 0}
    cells = np.array([f'{e}|{k}|{x}' for e, k, x in zip(d['episode'][m].astype(str), d['anchor_frame'][m].astype(int),
                                                        d['domain'][m].astype(int))])
    idx = cell_index(cells)
    out = {'S_cells': len(idx), 'S_rows': int(sum(len(i) for i in idx))}
    for lab in ('unsafe', 'fail'):
        y = d[lab][m].astype(float)
        ok = tot = 0.0
        picks, rand, orac = [], [], []
        for k in idx:
            yy, ss = y[k], s[k]
            picks.append(yy[np.argmin(ss)])
            rand.append(yy.mean())
            orac.append(yy.min())
            if yy.min() == yy.max():
                continue
            dd = ss[yy == 1][:, None] - ss[yy == 0][None, :]
            ok += (dd > 0).sum() + 0.5 * (dd == 0).sum()
            tot += dd.size
        suf = '' if lab == 'fail' else '_unsafe'
        out[f'S_{lab}'], out[f'Sn_{lab}'] = (ok / tot if tot else float('nan')), int(tot)
        out.update({f'S_pick_fail{suf}': float(np.mean(picks)) if picks else float('nan'),
                    f'S_random_fail{suf}': float(np.mean(rand)) if rand else float('nan'),
                    f'S_oracle_fail{suf}': float(np.mean(orac)) if orac else float('nan')})
    return out


def regime_plus(s, d, m):
    out = regime_metrics(s, d, m)
    if out['n']:
        out.update(state_metrics(s, d, m))
    return out


def metrics(s, d, m):
    """{ground: {startup, established, all}} for the grounds present in rows m (s = logits of ALL rows), plus 'both'
    (pooled; cells still keyed group|ground) when more than one ground is present."""
    dom = d['domain'].astype(int)
    startup = d['anchor_frame'].astype(int) == 0
    assert len(s) == len(dom) == len(m), 'metrics() takes the full logit vector plus a row mask'
    out = {}
    doms = sorted(np.unique(dom[m]).tolist())
    blocks = [(DOMAIN_NAME.get(x, str(x)), m & (dom == x)) for x in doms] + ([('both', m)] if len(doms) > 1 else [])
    for key, dm in blocks:
        out[key] = {name: regime_plus(s[mm], d, mm)
                    for name, mm in (('startup', dm & startup), ('established', dm & ~startup), ('all', dm))}
    return out


def anchor_masks(d):
    k = d['anchor_frame'].astype(int)
    out = [(name, (k >= lo) & (k <= hi)) for name, lo, hi in ANCHOR_BUCKETS]
    other = ~np.any([mm for _, mm in out], 0)
    if other.any():
        out.append(('k_other', other))
    return out


def all_metrics(s, d, m):
    """{'all': metrics, 'by_source': {source: metrics}, 'by_anchor': {bucket: metrics}} on rows m."""
    src = d['source'].astype(str)
    return dict(all=metrics(s, d, m), by_source={x: metrics(s, d, m & (src == x)) for x in sorted(np.unique(src[m]))},
                by_anchor={name: metrics(s, d, m & mm) for name, mm in anchor_masks(d) if (m & mm).any()})


def domain_head_metrics(dl, d, m):
    """Ground-type head read-out (hist_aux): AUC / accuracy of its logit on established and startup rows."""
    startup = d['anchor_frame'].astype(int) == 0
    out = {}
    for name, mm in (('established', m & ~startup), ('startup', m & startup)):
        if mm.sum() == 0:
            continue
        y = d['domain'][mm].astype(float)
        out[f'{name}_auc'] = auc(y, dl[mm])
        out[f'{name}_acc'] = float(((dl[mm] > 0) == (y > 0.5)).mean())
        out[f'{name}_n'] = int(mm.sum())
    return out
