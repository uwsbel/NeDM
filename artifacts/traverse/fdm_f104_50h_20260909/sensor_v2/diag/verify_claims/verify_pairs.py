"""Re-verify TASK C's label-carrying pair test with group clustering and multiplicity accounting.

Reproduces analysis.json['pairs'] exactly, then recomputes with (a) one pair per group, (b) a group-level
cluster bootstrap on the discordant difference, (c) a group-clustered permutation/sign test, and reports how
many independent groups actually carry the discordant pairs.
"""
import json, sys
from math import comb
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
TC = HERE.parent / 'task_c'
ARENAS = ['f104', 'g203', 'g216', 'g217', 'g228', 'g231']
MODELS = ['n2', 'e0', 'd']
POOLS = ['proposal', 'fixed2']
ARM_MODEL = {'n2': ('n2', 'proposal'), 'e0': ('e0', 'proposal'), 'd': ('d', 'proposal'),
             'straight6': ('n2', 'proposal'),
             'n2_fixed2': ('n2', 'fixed2'), 'e0_fixed2': ('e0', 'fixed2'), 'd_fixed2': ('d', 'fixed2')}


def sign_test(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


D = {}
gidx = {}
for a in ARENAS:
    z = np.load(TC / f'scores/{a}_all.npz')
    D[a] = {k: z[k] for k in z}
    for i, g in enumerate(z['group'].astype(str)):
        gidx[g] = (a, i)

drv = json.load(open(TC / 'driven_outcomes.json'))

# group/pool -> {candidate index: unsafe}
bygroup = {}
for g, rec in drv.items():
    if g not in gidx:
        continue
    for arm, v in rec['arms'].items():
        m, p = ARM_MODEL[arm]
        bygroup.setdefault((g, p), {}).setdefault(v['index'], v['unsafe'])

out = {}
rng = np.random.default_rng(0)
for p in POOLS:
    for m in MODELS:
        pairs = []          # (group, o1, o2)
        for (g, pp), seen in bygroup.items():
            if pp != p:
                continue
            a, i = gidx[g]
            z1 = D[a][f'{p}_{m}_v1'][i]; z2 = D[a][f'{p}_{m}_v2'][i]
            un = [k for k in seen if seen[k] == 1]
            sa = [k for k in seen if seen[k] == 0]
            for x in un:
                for y in sa:
                    pairs.append((g, bool(z1[x] > z1[y]), bool(z2[x] > z2[y])))
        n = len(pairs)
        o1 = sum(1 for _, a1, a2 in pairs if a1 and not a2)
        o2 = sum(1 for _, a1, a2 in pairs if a2 and not a1)
        c1 = sum(1 for _, a1, _ in pairs if a1); c2 = sum(1 for _, _, a2 in pairs if a2)
        groups = sorted({g for g, _, _ in pairs})
        gd = {}
        for g, a1, a2 in pairs:
            gd.setdefault(g, []).append((a1, a2))
        n_disc_groups = sum(1 for g in gd if any(a1 != a2 for a1, a2 in gd[g]))
        # (a) one pair per group: first unsafe x first safe, deterministic
        first = {}
        for g, a1, a2 in pairs:
            first.setdefault(g, (a1, a2))
        fo1 = sum(1 for g in first if first[g][0] and not first[g][1])
        fo2 = sum(1 for g in first if first[g][1] and not first[g][0])
        # (b) cluster bootstrap on the accuracy difference (v2 - v1), resampling GROUPS
        gl = [np.array([[a1, a2] for a1, a2 in gd[g]], float) for g in groups]
        def stat(ix):
            s = np.concatenate([gl[j] for j in ix], 0)
            return s[:, 1].mean() - s[:, 0].mean()
        bs = np.array([stat(rng.integers(0, len(gl), len(gl))) for _ in range(4000)])
        # (c) cluster sign test: flip the sign of every discordant pair in a group together
        obs = (c2 - c1)
        gdisc = np.array([sum(int(a2) - int(a1) for a1, a2 in gd[g]) for g in groups], float)
        nz = gdisc[gdisc != 0]
        perm = rng.choice([-1.0, 1.0], size=(20000, len(nz))) * np.abs(nz)
        pperm = float((np.abs(perm.sum(1)) >= abs(obs) - 1e-9).mean())
        out[f'{p}/{m}'] = dict(
            pairs=n, v1_correct_frac=c1 / n, v2_correct_frac=c2 / n, only_v1=o1, only_v2=o2,
            naive_sign_p=sign_test(o1, o2),
            n_groups_with_pairs=len(groups), n_groups_with_any_discordant_pair=n_disc_groups,
            one_pair_per_group=dict(n=len(first), only_v1=fo1, only_v2=fo2, p=sign_test(fo1, fo2),
                                    v1_correct_frac=sum(1 for g in first if first[g][0]) / len(first),
                                    v2_correct_frac=sum(1 for g in first if first[g][1]) / len(first)),
            cluster_bootstrap_diff=dict(mean=float(bs.mean()), ci=[float(np.percentile(bs, 2.5)),
                                                                   float(np.percentile(bs, 97.5))],
                                        p_two_sided=float(2 * min((bs <= 0).mean(), (bs >= 0).mean()))),
            cluster_sign_test_p=pperm,
            groups_net_v2_better=int((gdisc > 0).sum()), groups_net_v1_better=int((gdisc < 0).sum()))
        r = out[f'{p}/{m}']
        print(f"{p:9s} {m:3s} pairs={n:4d} groups={len(groups):4d} disc-groups={n_disc_groups:3d} "
              f"v1={c1/n:.3f} v2={c2/n:.3f} only1/only2={o1}/{o2} naive p={r['naive_sign_p']:.4f} "
              f"cluster-sign p={pperm:.4f} boot diff {bs.mean():+.4f} "
              f"[{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] "
              f"| 1/group: {fo1}/{fo2} p={r['one_pair_per_group']['p']:.4f}")

json.dump(out, open(HERE / 'verify_pairs.json', 'w'), indent=1)
print('wrote', HERE / 'verify_pairs.json')
