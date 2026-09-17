"""Independent re-analysis of the Task C replay, from the saved per-candidate scores.

This verifies the ANALYSIS (flip rates, rank correlation, score movement, outcome-ordering tests).
The scores themselves are re-scored independently for a subsample by v_taskC_rescore.py.
  /home/harry/miniconda3/envs/nedm/bin/python <this file>
"""
import json, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
TC = HERE.parent / "task_c"
ARENAS = ["f104", "g203", "g216", "g217", "g228", "g231"]
POOLS = ["proposal", "fixed2"]
MODELS = ["n2", "e0", "d"]


def spearman(a, b):
    ra = np.argsort(np.argsort(a, axis=1), axis=1).astype(float)
    rb = np.argsort(np.argsort(b, axis=1), axis=1).astype(float)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))


def main():
    S = {a: dict(np.load(TC / "scores" / f"{a}_all.npz", allow_pickle=True)) for a in ARENAS}
    groups = np.concatenate([S[a]["group"] for a in ARENAS])
    out = {"n_groups": int(groups.size), "per_arena_groups": {a: int(S[a]["group"].size) for a in ARENAS}}

    res = {}
    for pool in POOLS:
        for m in MODELS:
            v1 = np.concatenate([S[a][f"{pool}_{m}_v1"] for a in ARENAS])
            v2 = np.concatenate([S[a][f"{pool}_{m}_v2"] for a in ARENAS])
            a1 = v1.argmin(1); a2 = v2.argmin(1)
            flips = a1 != a2
            rho = spearman(v1, v2)
            dz = v2 - v1
            cen = dz - dz.mean(1, keepdims=True)
            # rank of the v1 pick inside the v2 order, on flipped groups
            r_v1_in_v2 = np.argsort(np.argsort(v2, axis=1), axis=1)[np.arange(len(a1)), a1] + 1
            r_v2_in_v1 = np.argsort(np.argsort(v1, axis=1), axis=1)[np.arange(len(a2)), a2] + 1
            top10_v1 = (np.argsort(v2, axis=1)[:, :10] == a1[:, None]).any(1)
            res[f"{pool}_{m}"] = dict(
                flip_rate=float(flips.mean()), flips=int(flips.sum()),
                per_arena_flip={a: float((S[a][f"{pool}_{m}_v1"].argmin(1) != S[a][f"{pool}_{m}_v2"].argmin(1)).mean())
                                for a in ARENAS},
                spearman_mean=float(rho.mean()), spearman_p5=float(np.quantile(rho, .05)), spearman_min=float(rho.min()),
                dz_mean=float(dz.mean()), dz_sd=float(dz.std()),
                centred_abs_dz_p50=float(np.quantile(np.abs(cen), .50)),
                centred_abs_dz_p95=float(np.quantile(np.abs(cen), .95)),
                within_group_sd_v1=float(v1.std(1).mean()),
                v1top1_in_v2top10=float(top10_v1.mean()),
                flip_rank_v1pick_in_v2_median=float(np.median(r_v1_in_v2[flips])) if flips.any() else None,
                flip_rank_v2pick_in_v1_median=float(np.median(r_v2_in_v1[flips])) if flips.any() else None,
                flip_v1pick_in_v2top10=float(((r_v1_in_v2 <= 10) & flips).sum() / max(flips.sum(), 1)),
                top10_overlap=float(np.mean([len(set(np.argsort(v1[i])[:10]) & set(np.argsort(v2[i])[:10])) / 10
                                             for i in range(len(v1))])),
            )
    out["score_movement"] = res

    # ---- outcome-linked tests ----
    DO = json.load(open(TC / "driven_outcomes.json"))
    idx = {}
    for a in ARENAS:
        for i, g in enumerate(S[a]["group"]):
            idx[str(g)] = (a, i)
    arms = ["n2", "e0", "d", "straight6", "n2_fixed2", "e0_fixed2", "d_fixed2"]
    arm_model = {"n2": ("proposal", "n2"), "e0": ("proposal", "e0"), "d": ("proposal", "d"),
                 "straight6": ("proposal", "n2"),
                 "n2_fixed2": ("fixed2", "n2"), "e0_fixed2": ("fixed2", "e0"), "d_fixed2": ("fixed2", "d")}
    unsafe_rate = {}
    rank_move = {}
    aucs = {}
    for arm in arms:
        ys, r1, r2, s1, s2 = [], [], [], [], []
        pool, m = arm_model[arm]
        for g, rec in DO.items():
            if arm not in rec["arms"] or g not in idx:
                continue
            a, i = idx[g]
            v = rec["arms"][arm]
            k = v.get("index")
            if k is None or v.get("pool") is None:
                continue
            p = v["pool"]
            if p not in POOLS:
                continue
            V1 = S[a][f"{p}_{m}_v1"][i]; V2 = S[a][f"{p}_{m}_v2"][i]
            ys.append(int(v["unsafe"]))
            r1.append(int(np.argsort(np.argsort(V1))[k]) + 1)
            r2.append(int(np.argsort(np.argsort(V2))[k]) + 1)
            s1.append(float(V1[k])); s2.append(float(V2[k]))
        ys = np.array(ys); r1 = np.array(r1); r2 = np.array(r2); s1 = np.array(s1); s2 = np.array(s2)
        unsafe_rate[arm] = [int(ys.size), float(ys.mean())]
        d = r2 - r1
        rank_move[arm] = dict(unsafe=float(d[ys == 1].mean()), safe=float(d[ys == 0].mean()),
                              gap=float(d[ys == 1].mean() - d[ys == 0].mean()))

        def auc(y, s):
            r = np.argsort(np.argsort(s)) + 1.0
            n1 = y.sum(); n0 = len(y) - n1
            return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))
        aucs[arm] = dict(v1=auc(ys, s1), v2=auc(ys, s2), n=int(ys.size))
    out["driven_unsafe_rate"] = unsafe_rate
    out["rank_move_of_driven_route"] = rank_move
    out["pooled_auc"] = aucs

    # ---- within-group unsafe-vs-safe pairs of driven routes ----
    pairs = {}
    for pool in POOLS:
        arms_here = ["n2", "e0", "d"] if pool == "proposal" else ["n2_fixed2", "e0_fixed2", "d_fixed2"]
        # a "pair" = a group where two arms drove routes from this pool with different outcomes
        for m in MODELS:
            n = c1 = c2 = only1 = only2 = 0
            for g, rec in DO.items():
                if g not in idx:
                    continue
                a, i = idx[g]
                cand = [(v["index"], v["unsafe"]) for arm, v in rec["arms"].items()
                        if arm in arms_here and v.get("pool") == pool and v.get("index") is not None]
                # unique candidate indices with known outcome
                seen = {}
                for k, u in cand:
                    seen.setdefault(k, set()).add(int(u))
                ks = [k for k, s in seen.items() if len(s) == 1]
                lab = {k: list(seen[k])[0] for k in ks}
                pos = [k for k in ks if lab[k] == 1]; neg = [k for k in ks if lab[k] == 0]
                if not pos or not neg:
                    continue
                V1 = S[a][f"{pool}_{m}_v1"][i]; V2 = S[a][f"{pool}_{m}_v2"][i]
                for kp in pos:
                    for kn in neg:
                        n += 1
                        o1 = V1[kp] > V1[kn]; o2 = V2[kp] > V2[kn]
                        c1 += o1; c2 += o2
                        only1 += (o1 and not o2); only2 += (o2 and not o1)
            if n:
                pairs[f"{pool}_{m}"] = dict(n=int(n), v1=float(c1 / n), v2=float(c2 / n),
                                            only_v1=int(only1), only_v2=int(only2),
                                            sign_p=float(sign_test(int(only1), int(only2))))
    out["within_group_pairs"] = pairs

    json.dump(out, open(HERE / "v_taskC.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("n_groups", "driven_unsafe_rate", "rank_move_of_driven_route",
                                          "pooled_auc", "within_group_pairs")}, indent=1))
    for k, v in res.items():
        print(f"{k}: flips {100*v['flip_rate']:.1f}% ({v['flips']})  rho {v['spearman_mean']:.4f}  "
              f"dz mean {v['dz_mean']:+.3f} sd {v['dz_sd']:.3f}  |cen dz| p50 {v['centred_abs_dz_p50']:.3f}  "
              f"top10ov {v['top10_overlap']:.3f}  v1top1in v2top10 {v['v1top1_in_v2top10']:.3f}")


def sign_test(a, b):
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    s = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * s / 2 ** n)


if __name__ == "__main__":
    main()
