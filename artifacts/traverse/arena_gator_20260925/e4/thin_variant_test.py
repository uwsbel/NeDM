# Thin (one-world) variant == the rigid / soil half of the ga_build_mixed file (same episodes, same arrays)
import sys, json, numpy as np, os
sys.path.insert(0, 'scripts')
import ag_build_ds as B
E = 'artifacts/traverse/arena_gator_20260925/e4/f104_hmmwv'
A = 'artifacts/traverse'
full = np.load(f'{E}/ci_f104_hmmwv_both.npz', allow_pickle=True)
res = {}
for w, roots in [('rigid', [A + '/fdm_f104_50h_20260909/production_v3/runs', A + '/fdm_f104_50h_20260909/production_v4/runs']), ('crm', [A + '/crm_f104_v1/collect_v1/runs'])]:
    out = f'/tmp/ag_e4_thin_{w}.npz'
    tier = dict(zip(zip(full['domain'].astype(int), full['episode'].astype(str)), full['tier'].astype(int)))
    def extra(d):
        return dict(arena=np.array(['f104'] * len(d['id']), object), vehicle=np.array(['hmmwv'] * len(d['id']), object),
                    tier=np.array([tier[(int(x), e)] for x, e in zip(d['domain'], d['episode'].astype(str))], np.int16))
    r = B.ci_one_world(f'{E}/reanchor_f104_hmmwv_both_{w}.npz', w, roots, out, extra, 8)
    t = np.load(out, allow_pickle=True); code = B.GBM.DOMAIN_CODE[w]; m = full['domain'] == code
    eq = {}
    for k in sorted(set(t.files) | set(full.files)):
        if k not in t.files or k not in full.files: eq[k] = 'missing'; continue
        a, b = t[k], full[k]
        if k in ('hist_cols', 'priv_names'): eq[k] = bool(np.array_equal(a.astype(str) if a.dtype == object else a, b.astype(str) if b.dtype == object else b)); continue
        b = b[m]
        eq[k] = bool(np.array_equal(a.astype(str), b.astype(str))) if a.dtype == object else bool(np.array_equal(a, b, equal_nan=np.issubdtype(a.dtype, np.floating)))
    res[w] = dict(build=r, rows=int(len(t['id'])), rows_in_mixed=int(m.sum()), all_equal=all(v is True for v in eq.values()), arrays=eq)
    print(w, res[w]['rows'], res[w]['rows_in_mixed'], res[w]['all_equal'], [k for k, v in eq.items() if v is not True], flush=True)
    os.remove(out)
json.dump(res, open(f'{E}/thin_variant_check.json', 'w'), indent=1)
