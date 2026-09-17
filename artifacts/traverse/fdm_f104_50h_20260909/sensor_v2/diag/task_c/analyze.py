"""TASK C step 2+3: what changes when the same checkpoints score v2 corridors instead of v1 corridors.

Reads scores/<arena>_all.npz (replay_pools.py) and driven_outcomes.json (collect_outcomes.py).
Writes analysis.json in this directory and prints the tables.

Diagnostic only: every checkpoint was TRAINED on v1 corridors, so v2 is an input shift, not a deployment result.
"""
import json, sys
from math import comb
from pathlib import Path
import numpy as np


def mcnemar(b, c):
    """Exact two-sided sign test on discordant counts (same helper as scripts/f104_n2_analyze)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

HERE = Path(__file__).resolve().parent
ARENAS = (sys.argv[1].split(',') if len(sys.argv) > 1 else ['f104', 'g203', 'g216', 'g217', 'g228', 'g231'])
MODELS = ['n2', 'e0', 'd']
POOLS = ['proposal', 'fixed2']
ARM_MODEL = {'n2': ('n2', 'proposal'), 'e0': ('e0', 'proposal'), 'd': ('d', 'proposal'),
             'straight6': ('n2', 'proposal'),
             'n2_fixed2': ('n2', 'fixed2'), 'e0_fixed2': ('e0', 'fixed2'), 'd_fixed2': ('d', 'fixed2')}


def prob(z):
    return 1 - np.exp(-np.exp(z))


def spearman(a, b):
    ra = np.argsort(np.argsort(a, 1), 1).astype(float)
    rb = np.argsort(np.argsort(b, 1), 1).astype(float)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))


def auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    if y.min() == y.max():
        return float('nan')
    r = np.argsort(np.argsort(s)) + 1.0
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def boot_ci(fn, n, rng, reps=4000):
    v = [fn(rng.integers(0, n, n)) for _ in range(reps)]
    v = [x for x in v if np.isfinite(x)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (float('nan'),) * 2


def load():
    D = {}
    groups = []
    for a in ARENAS:
        z = np.load(HERE / f'scores/{a}_all.npz')
        D[a] = {k: z[k] for k in z}
        groups += [(a, g) for g in z['group']]
    return D, groups


def main():
    D, groups = load()
    diag_names = list(D['f104']['diag_names'].astype(str))
    out = {'n_groups': len(groups), 'per_arena': {a: int(len(D[a]['group'])) for a in ARENAS}}
    print(f"{len(groups)} groups: " + ', '.join(f'{a} {len(D[a]["group"])}' for a in ARENAS))

    cat = lambda key: np.concatenate([D[a][key] for a in ARENAS], 0)
    Z = {(p, m, v): cat(f'{p}_{m}_{v}') for p in POOLS for m in MODELS for v in ('v1', 'v2')}
    DG = {p: cat(f'{p}_diag') for p in POOLS}

    # ---------------------------------------------------------------- 1. how much scores move
    print('\n== 1. score movement (all candidates, 256 per group per pool) ==')
    print(f'{"pool":9s} {"model":5s} {"mean dz":>9s} {"sd dz":>8s} {"p50|dz|":>8s} {"p95|dz|":>8s} '
          f'{"sd within":>9s} {"|dz|/sd":>8s} {"P v1":>9s} {"P v2":>9s}')
    out['score_movement'] = {}
    for p in POOLS:
        for m in MODELS:
            z1, z2 = Z[(p, m, 'v1')], Z[(p, m, 'v2')]
            dz = z2 - z1
            dzc = dz - dz.mean(1, keepdims=True)          # what can actually change a ranking
            sdw = z1.std(1).mean()
            r = dict(mean_dz=float(dz.mean()), sd_dz=float(dz.std()), p50_abs_dz=float(np.median(np.abs(dz))),
                     p95_abs_dz=float(np.percentile(np.abs(dz), 95)),
                     p50_abs_dz_centred=float(np.median(np.abs(dzc))),
                     p95_abs_dz_centred=float(np.percentile(np.abs(dzc), 95)),
                     mean_within_group_sd_z=float(sdw),
                     ratio_p50_centred_to_sd=float(np.median(np.abs(dzc)) / sdw),
                     mean_prob_v1=float(prob(z1).mean()), mean_prob_v2=float(prob(z2).mean()),
                     median_prob_v1=float(np.median(prob(z1))), median_prob_v2=float(np.median(prob(z2))))
            out['score_movement'][f'{p}/{m}'] = r
            print(f'{p:9s} {m:5s} {r["mean_dz"]:+9.4f} {r["sd_dz"]:8.4f} {r["p50_abs_dz"]:8.4f} '
                  f'{r["p95_abs_dz"]:8.4f} {sdw:9.4f} {r["ratio_p50_centred_to_sd"]:8.3f} '
                  f'{r["median_prob_v1"]:9.2e} {r["median_prob_v2"]:9.2e}')

    # ---------------------------------------------------------------- 2. rank correlation
    print('\n== 2. within-group rank correlation over the 256 candidates ==')
    print(f'{"pool":9s} {"model":5s} {"mean rho":>9s} {"median":>8s} {"p5":>8s} {"min":>8s} '
          f'{"top10 ovl":>10s} {"top1in10":>9s}')
    out['rank_correlation'] = {}
    for p in POOLS:
        for m in MODELS:
            z1, z2 = Z[(p, m, 'v1')], Z[(p, m, 'v2')]
            rho = spearman(z1, z2)
            o1 = np.argsort(z1, 1)[:, :10]; o2 = np.argsort(z2, 1)[:, :10]
            ov = np.array([len(set(a) & set(b)) for a, b in zip(o1, o2)]) / 10.0
            t1in10 = np.array([o1[i, 0] in set(o2[i]) for i in range(len(o1))])
            r = dict(mean=float(rho.mean()), median=float(np.median(rho)), p5=float(np.percentile(rho, 5)),
                     min=float(rho.min()), top10_overlap=float(ov.mean()), v1_top1_in_v2_top10=float(t1in10.mean()))
            out['rank_correlation'][f'{p}/{m}'] = r
            print(f'{p:9s} {m:5s} {r["mean"]:9.4f} {r["median"]:8.4f} {r["p5"]:8.4f} {r["min"]:8.4f} '
                  f'{r["top10_overlap"]:10.3f} {r["v1_top1_in_v2_top10"]:9.3f}')

    # ---------------------------------------------------------------- 3. argmin changes
    print('\n== 3. how often the driven route (argmin) changes ==')
    print(f'{"pool":9s} {"model":5s} {"flip%":>7s} {"rank of v1 pick under v2 (median/p90)":>38s} '
          f'{"dz1 flip":>9s} {"dP":>10s}')
    out['argmin'] = {}
    flips = {}
    for p in POOLS:
        for m in MODELS:
            z1, z2 = Z[(p, m, 'v1')], Z[(p, m, 'v2')]
            i1 = z1.argmin(1); i2 = z2.argmin(1)
            fl = i1 != i2
            flips[(p, m)] = fl
            g = np.arange(len(z1))
            r1_under2 = (z2 < z2[g, i1][:, None]).sum(1)      # rank of the v1 pick in the v2 ordering
            r2_under1 = (z1 < z1[g, i2][:, None]).sum(1)
            gap1 = z1[g, i2] - z1[g, i1]                      # how much worse the v2 pick looks to v1
            dP = prob(z1[g, i2]) - prob(z1[g, i1])
            r = dict(flip_rate=float(fl.mean()), n_flip=int(fl.sum()),
                     median_rank_v1pick_under_v2=float(np.median(r1_under2)),
                     p90_rank_v1pick_under_v2=float(np.percentile(r1_under2, 90)),
                     median_rank_v2pick_under_v1=float(np.median(r2_under1)),
                     p90_rank_v2pick_under_v1=float(np.percentile(r2_under1, 90)),
                     median_v1_logit_gap_on_flips=float(np.median(gap1[fl])) if fl.any() else None,
                     median_v1_prob_gap_on_flips=float(np.median(dP[fl])) if fl.any() else None)
            out['argmin'][f'{p}/{m}'] = r
            print(f'{p:9s} {m:5s} {100 * r["flip_rate"]:6.1f}% '
                  f'{r["median_rank_v1pick_under_v2"]:17.0f} / {r["p90_rank_v1pick_under_v2"]:<18.0f} '
                  f'{r["median_v1_logit_gap_on_flips"] or 0:+9.4f} {r["median_v1_prob_gap_on_flips"] or 0:+10.2e}')
    # per-arena flip rate
    out['argmin_per_arena'] = {}
    idx = np.concatenate([[a] * len(D[a]['group']) for a in ARENAS])
    for p in POOLS:
        for m in MODELS:
            out['argmin_per_arena'][f'{p}/{m}'] = {a: float(flips[(p, m)][idx == a].mean()) for a in ARENAS}
    print('\n  flip rate per arena (proposal / fixed2):')
    for m in MODELS:
        print(f'   {m:3s} ' + '  '.join(f'{a} {100 * out["argmin_per_arena"][f"proposal/{m}"][a]:4.1f}/'
                                        f'{100 * out["argmin_per_arena"][f"fixed2/{m}"][a]:<4.1f}' for a in ARENAS))

    # ---------------------------------------------------------------- 4. where do changes concentrate
    print('\n== 4. do changed picks concentrate on high-slope / image-edge regions? ==')
    out['concentration'] = {}
    key_cols = ['p95_abs_grade', 'mean_abs_cross', 'mean_radius_m', 'max_radius_m', 'mean_v1_disp_m',
                'max_v1_disp_m', 'mean_abs_delev', 'valid_frac_v1']
    for p in POOLS:
        Dg = DG[p]
        gstat = {c: Dg[:, :, diag_names.index(c)].mean(1) for c in key_cols}       # group-level: pool mean
        for m in MODELS:
            fl = flips[(p, m)]
            rec = {}
            for c in key_cols:
                v = gstat[c]
                order = np.argsort(v, kind='stable')                       # equal-sized quartiles by rank
                qi = np.empty(len(v), int); qi[order] = (np.arange(len(v)) * 4) // len(v)
                rec[c] = dict(mean_flip=float(v[fl].mean()), mean_noflip=float(v[~fl].mean()),
                              flip_rate_by_quartile=[float(fl[qi == k].mean()) for k in range(4)],
                              quartile_edges=[float(x) for x in np.quantile(v, [0, .25, .5, .75, 1.0])],
                              n_by_quartile=[int((qi == k).sum()) for k in range(4)])
            out['concentration'][f'{p}/{m}'] = rec
        print(f'  -- pool {p}: flip rate by quartile of the group mean statistic --')
        hdr = f'    {"statistic":16s}' + ''.join(f'{m:>26s}' for m in MODELS)
        print(hdr)
        for c in key_cols:
            row = f'    {c:16s}'
            for m in MODELS:
                q = out['concentration'][f'{p}/{m}'][c]['flip_rate_by_quartile']
                row += '  ' + ' '.join(f'{100 * x:5.1f}' for x in q)
            print(row)

    # on flipped groups, does the pick itself move to gentler / more central terrain?
    print('\n  on flipped groups, statistics of the v1 pick vs the v2 pick (mean over flips):')
    out['pick_shift'] = {}
    for p in POOLS:
        Dg = DG[p]
        for m in MODELS:
            z1, z2 = Z[(p, m, 'v1')], Z[(p, m, 'v2')]
            i1 = z1.argmin(1); i2 = z2.argmin(1); fl = flips[(p, m)]
            g = np.arange(len(z1))
            rec = {}
            for c in key_cols + ['mean_speed', 'route_len_m']:
                col = diag_names.index(c)
                a1 = Dg[g, i1, col][fl]; a2 = Dg[g, i2, col][fl]
                rec[c] = dict(v1_pick=float(a1.mean()), v2_pick=float(a2.mean()),
                              delta=float((a2 - a1).mean()),
                              frac_v2_lower=float((a2 < a1).mean()))
            out['pick_shift'][f'{p}/{m}'] = rec
            print(f'   {p:9s} {m:3s} ' + '  '.join(
                f'{c}: {rec[c]["v1_pick"]:.3f}->{rec[c]["v2_pick"]:.3f}'
                for c in ('p95_abs_grade', 'mean_v1_disp_m', 'mean_radius_m', 'mean_speed')))

    # candidate-level: |centred dz| vs the candidate's own geometry error
    print('\n  candidate-level Pearson r of |centred dz| with corridor statistics:')
    out['candidate_corr'] = {}
    for p in POOLS:
        Dg = DG[p]
        for m in MODELS:
            dz = Z[(p, m, 'v2')] - Z[(p, m, 'v1')]
            dzc = np.abs(dz - dz.mean(1, keepdims=True)).ravel()
            rec = {}
            for c in ['mean_v1_disp_m', 'max_v1_disp_m', 'p95_abs_grade', 'mean_abs_delev', 'mean_radius_m']:
                v = Dg[:, :, diag_names.index(c)].ravel().astype(np.float64)
                rec[c] = float(np.corrcoef(dzc, v)[0, 1])
            out['candidate_corr'][f'{p}/{m}'] = rec
            print(f'   {p:9s} {m:3s} ' + '  '.join(f'{c}={rec[c]:+.3f}' for c in rec))

    # ---------------------------------------------------------------- 5. driven routes and their outcomes
    print('\n== 5. driven routes: does v2 rank the observed-unsafe ones worse? ==')
    drv = json.load(open(HERE / 'driven_outcomes.json'))
    gidx = {}
    for a in ARENAS:
        for i, g in enumerate(D[a]['group'].astype(str)):
            gidx[g] = (a, i)
    rows = []
    for g, rec in drv.items():
        if g not in gidx:
            continue
        a, i = gidx[g]
        for arm, v in rec['arms'].items():
            m, p = ARM_MODEL[arm]
            assert v['pool'] == p, (arm, v['pool'])
            z1 = D[a][f'{p}_{m}_v1'][i]; z2 = D[a][f'{p}_{m}_v2'][i]
            j = v['index']
            rows.append(dict(group=g, arena=a, arm=arm, model=m, pool=p, index=j,
                             unsafe=v['unsafe'], fail=v['fail'], back_s=v['back_s'], max_tilt=v['max_tilt'],
                             z1=float(z1[j]), z2=float(z2[j]),
                             r1=int((z1 < z1[j]).sum()), r2=int((z2 < z2[j]).sum()),
                             still_argmin_v2=bool(z2.argmin() == j),
                             was_argmin_v1=bool(z1.argmin() == j),
                             saved_risk=v.get(f'risk_{m}')))
    out['n_driven_rows'] = len(rows)
    rng = np.random.default_rng(0)
    print(f'{"arm":11s} {"n":>5s} {"unsafe":>7s} {"d rank (unsafe)":>16s} {"d rank (safe)":>14s} '
          f'{"AUC v1":>8s} {"AUC v2":>8s} {"dAUC [95% CI]":>24s}')
    out['driven'] = {}
    for arm in ARM_MODEL:
        R = [r for r in rows if r['arm'] == arm]
        if not R:
            continue
        y = np.array([r['unsafe'] for r in R]); dr = np.array([r['r2'] - r['r1'] for r in R], float)
        z1 = np.array([r['z1'] for r in R]); z2 = np.array([r['z2'] for r in R])
        a1, a2 = auc(y, z1), auc(y, z2)
        ci = boot_ci(lambda ix: auc(y[ix], z2[ix]) - auc(y[ix], z1[ix]), len(R), rng)
        gap = lambda ix: (dr[ix][y[ix] == 1].mean() - dr[ix][y[ix] == 0].mean()) if y[ix].any() and (y[ix] == 0).any() else np.nan
        gci = boot_ci(gap, len(R), rng, reps=2000)
        rec = dict(n=len(R), n_unsafe=int(y.sum()), auc_v1=a1, auc_v2=a2, dauc=a2 - a1, dauc_ci=ci,
                   drank_gap_unsafe_minus_safe=float(dr[y == 1].mean() - dr[y == 0].mean()) if y.any() else None,
                   drank_gap_ci=gci,
                   mean_drank_unsafe=float(dr[y == 1].mean()) if y.any() else None,
                   mean_drank_safe=float(dr[y == 0].mean()),
                   median_drank_unsafe=float(np.median(dr[y == 1])) if y.any() else None,
                   median_drank_safe=float(np.median(dr[y == 0])),
                   frac_unsafe_no_longer_argmin=float(np.mean([not r['still_argmin_v2'] for r in R if r['unsafe'] and r['was_argmin_v1']])) if any(r['unsafe'] and r['was_argmin_v1'] for r in R) else None,
                   frac_safe_no_longer_argmin=float(np.mean([not r['still_argmin_v2'] for r in R if not r['unsafe'] and r['was_argmin_v1']])))
        out['driven'][arm] = rec
        print(f'{arm:11s} {rec["n"]:5d} {rec["n_unsafe"]:7d} {rec["mean_drank_unsafe"] or 0:16.1f} '
              f'{rec["mean_drank_safe"]:14.1f} {a1:8.3f} {a2:8.3f} '
              f'{a2 - a1:+8.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]   gap {rec["drank_gap_unsafe_minus_safe"] or 0:+6.1f} '
              f'[{gci[0]:+.1f},{gci[1]:+.1f}]')

    # paired, label-carrying comparison: within a group, pairs of DRIVEN candidates with opposite outcomes
    print('\n  within-group discordant pairs of driven routes (one unsafe, one safe), scored by one model:')
    print(f'{"pool":9s} {"model":5s} {"pairs":>6s} {"v1 correct":>11s} {"v2 correct":>11s} {"both/only v1/only v2":>22s}')
    out['pairs'] = {}
    bygroup = {}
    for r in rows:
        bygroup.setdefault((r['group'], r['pool']), []).append(r)
    for p in POOLS:
        for m in MODELS:
            n = c1 = c2 = only1 = only2 = 0
            for (g, pp), R in bygroup.items():
                if pp != p:
                    continue
                a, i = gidx[g]
                z1 = D[a][f'{p}_{m}_v1'][i]; z2 = D[a][f'{p}_{m}_v2'][i]
                seen = {}
                for r in R:
                    seen.setdefault(r['index'], r['unsafe'])
                idxs = list(seen)
                for x in idxs:
                    for y2 in idxs:
                        if seen[x] == 1 and seen[y2] == 0:
                            n += 1
                            o1 = z1[x] > z1[y2]; o2 = z2[x] > z2[y2]
                            c1 += o1; c2 += o2; only1 += (o1 and not o2); only2 += (o2 and not o1)
            out['pairs'][f'{p}/{m}'] = dict(pairs=int(n), v1_correct=int(c1), v2_correct=int(c2),
                                            only_v1=int(only1), only_v2=int(only2),
                                            sign_test_p=mcnemar(int(only1), int(only2)))
            if n:
                print(f'{p:9s} {m:5s} {n:6d} {c1 / n:11.3f} {c2 / n:11.3f} '
                      f'{n - only1 - only2:>10d}/{only1}/{only2}   p={mcnemar(int(only1), int(only2)):.4f}')

    json.dump(out, open(HERE / 'analysis.json', 'w'), indent=1)
    print('\nwrote', HERE / 'analysis.json')


if __name__ == '__main__':
    main()
