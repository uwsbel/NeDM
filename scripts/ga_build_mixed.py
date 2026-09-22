"""Mixed rigid + CRM re-anchored dataset with measured history windows and a privileged teacher context (PLAN A1).

Input: the two night-2 re-anchored files (reanchor_rigid.npz, reanchor_crm.npz: 15,024 episodes each, anchors k = 0 and
multiples of 40 frames). Output: one npz with every original array kept (rows: rigid first, then CRM, each in its
original order) plus
  domain     (n,) int8      0 rigid, 1 crm
  id         (n,) object    original id suffixed '@rigid' / '@crm'; group / episode unchanged
  hist       (n,40,15) f16  causal window ending at the anchor k: hist[t] = [state[k-39+t][cols 0-6, 11-15], action[k-40+t]]
  hmask      (n,40) bool    True where the action frame k-40+t >= 0 (k = 0 rows are all-masked; masked steps are 0)
  privileged (n,8) f32      means over the state frames max(0, k-39)..k of tyre Fz fl/fr/rl/rr (cols 7-10), motorshaft torque
                            (col 16), and for CRM the wheel-mean slip ratio and the wheel-mean spindle height above the BMP
                            ground (crm_extra.npz: spindle_z_m - bmp_ground_z_m; zeros for rigid), then is_crm
  hist_cols  (12,) int16    the state columns used in hist; priv_names (8,) str.
Raw episodes are read from the local run dirs (CRM collect_v1/runs; rigid production_v3/runs + production_v4/runs) by the
row's `episode` (= run dir name) and `anchor_frame` (= k). Asserts uniqueness of (episode, anchor_frame, domain) and that no
id / group / episode matches a planner-suite pattern. Reports counts per domain x split x startup/established and the
fraction of rows whose raw episode was found.

  PYTHONPATH=src:scripts python scripts/ga_build_mixed.py --out artifacts/traverse/generalist_20260921/A_adapt/datasets/mixed_reanchor.npz
"""
import argparse, fnmatch, json, os, resource, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np

T = 40
HIST_STATE_COLS = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15]   # vx, vy, roll, pitch, roll rate, pitch rate, yaw rate, 4 spindle omegas, engine speed
PRIV_NAMES = ['fz_fl_mean_n', 'fz_fr_mean_n', 'fz_rl_mean_n', 'fz_rr_mean_n', 'torque_mean_nm', 'slip_ratio_mean', 'spindle_above_bmp_mean_m', 'is_crm']
BLACKLIST = ['f104_crm_eval_group_*', 'f104_g1_test_group_*', 'f104_pair_group_*']
DOMAIN_CODE = {'rigid': 0, 'crm': 1}
ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'artifacts', 'traverse')
DEFAULTS = dict(rigid=ART + '/crm_night2_v1/datasets/reanchor_rigid.npz', crm=ART + '/crm_night2_v1/datasets/reanchor_crm.npz',
                rigid_runs=[ART + '/fdm_f104_50h_20260909/production_v3/runs', ART + '/fdm_f104_50h_20260909/production_v4/runs'],
                crm_runs=[ART + '/crm_f104_v1/collect_v1/runs'])


def find_run(episode, roots):
    for r in roots:
        d = os.path.join(r, episode)
        if os.path.isdir(d):
            return d
    return None


def cut_episode(item):
    """One raw episode -> (hist, hmask, privileged) for every anchor k in `ks`. Returns (episode, None) when the run dir is missing."""
    episode, ks, roots, is_crm = item
    d = find_run(episode, roots)
    if d is None:
        return episode, None
    z = np.load(d + '/trajectory.npz')
    state = z['state'].astype(np.float32); act = z['action'].astype(np.float32); n = len(state)
    assert act.shape == (n, 3) and state.shape[1] == 17, (episode, state.shape, act.shape)
    if is_crm:
        e = np.load(d + '/crm_extra.npz')
        slip = e['slip_ratio'].astype(np.float32).mean(1)
        sink = (e['spindle_z_m'].astype(np.float32) - e['bmp_ground_z_m'].astype(np.float32)[:, None]).mean(1)
        assert len(slip) == n and len(sink) == n, (episode, n, len(slip))
    m = len(ks)
    hist = np.zeros((m, T, 15), np.float16); hmask = np.zeros((m, T), bool); priv = np.zeros((m, 8), np.float32)
    t = np.arange(T)
    for j, k in enumerate(ks):
        k = int(k); assert 0 <= k < n, (episode, k, n)
        si = k - (T - 1) + t; ai = k - T + t; ok = ai >= 0
        hmask[j] = ok
        if ok.any():
            hist[j, ok] = np.concatenate([state[si[ok]][:, HIST_STATE_COLS], act[ai[ok]]], 1).astype(np.float16)
        w = slice(max(0, k - (T - 1)), k + 1)
        priv[j, 0:4] = state[w, 7:11].mean(0); priv[j, 4] = state[w, 16].mean()
        if is_crm:
            priv[j, 5] = slip[w].mean(); priv[j, 6] = sink[w].mean(); priv[j, 7] = 1.0
    return episode, (hist, hmask, priv)


def blacklisted(s):
    return any(fnmatch.fnmatch(s, p) for p in BLACKLIST)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rigid', default=DEFAULTS['rigid']); ap.add_argument('--crm', default=DEFAULTS['crm'])
    ap.add_argument('--rigid-runs', nargs='+', default=DEFAULTS['rigid_runs']); ap.add_argument('--crm-runs', nargs='+', default=DEFAULTS['crm_runs'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit-episodes', type=int, default=0, help='self-test: keep only the first N episodes (sorted) of each domain')
    ap.add_argument('--no-compress', action='store_true', help='np.savez instead of savez_compressed (faster, ~2x larger)')
    a = ap.parse_args()
    a.workers = max(1, min(a.workers, 8))
    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    # ---- metadata (everything but X) from both files; rigid (domain 0) first, then CRM (domain 1)
    srcs = [('rigid', a.rigid, a.rigid_runs), ('crm', a.crm, a.crm_runs)]
    meta = {}; keep = {}; n_by = {}
    for name, path, _ in srcs:
        z = np.load(path, allow_pickle=True)
        keys = [k for k in z.files if k != 'X']
        d = {k: z[k] for k in keys}
        sel = np.arange(len(d['id']))
        if a.limit_episodes:
            eps = sorted(set(d['episode'].astype(str)))[:a.limit_episodes]
            sel = np.flatnonzero(np.isin(d['episode'].astype(str), eps))
            d = {k: v[sel] for k, v in d.items()}
        meta[name] = d; keep[name] = sel; n_by[name] = len(sel)
        print(f'{name}: {len(sel)} rows, {len(set(d["episode"].astype(str)))} episodes from {path}', flush=True)
    keys = list(meta['rigid'].keys()); assert keys == list(meta['crm'].keys()), 'key sets differ between the two files'
    out = {}
    for k in keys:
        out[k] = np.concatenate([meta['rigid'][k], meta['crm'][k]])
    n = len(out['id'])
    domain = np.concatenate([np.zeros(n_by['rigid'], np.int8), np.ones(n_by['crm'], np.int8)])
    out['domain'] = domain
    out['id'] = np.array([f'{i}@{"crm" if dm else "rigid"}' for i, dm in zip(out['id'].astype(str), domain)], object)
    episode = out['episode'].astype(str); group = out['group'].astype(str); af = out['anchor_frame'].astype(int); split = out['split'].astype(str)

    # ---- contracts: uniqueness and blacklist
    trip = list(zip(episode, af.tolist(), domain.tolist()))
    assert len(set(trip)) == n, f'(episode, anchor_frame, domain) not unique: {n - len(set(trip))} duplicates'
    assert len(set(out['id'].astype(str))) == n, 'ids not unique'
    bad = [s for s in set(out['id'].astype(str)) | set(group) | set(episode) if blacklisted(s)]
    assert not bad, f'blacklisted ids/groups present: {bad[:10]}'
    print(f'{n} rows; uniqueness and blacklist assertions passed', flush=True)

    # ---- history windows + privileged context from the raw episodes (one task per episode per domain)
    hist = np.zeros((n, T, 15), np.float16); hmask = np.zeros((n, T), bool); priv = np.zeros((n, 8), np.float32)
    tasks = []
    for name, _, roots in srcs:
        dm = DOMAIN_CODE[name]; by_ep = defaultdict(list)
        for i in np.flatnonzero(domain == dm):
            by_ep[episode[i]].append(int(i))
        tasks += [(ep, [af[i] for i in idx], roots, dm == 1, idx) for ep, idx in sorted(by_ep.items())]
    found = np.zeros(n, bool); misses = []
    with ProcessPoolExecutor(a.workers) as ex:
        for (ep, ks, roots, is_crm, idx), (ep2, res) in zip(tasks, ex.map(cut_episode, [t[:4] for t in tasks], chunksize=32)):
            assert ep == ep2
            if res is None:
                misses.append((ep, 'crm' if is_crm else 'rigid')); continue
            h, m, p = res; idx = np.asarray(idx)
            hist[idx] = h; hmask[idx] = m; priv[idx] = p; found[idx] = True
    print(f'episodes cut in {time.time() - t0:.0f} s; rows with raw episode found: {found.sum()}/{n} ({100 * found.mean():.2f} %)', flush=True)
    if misses:
        print(f'MISSING {len(misses)} episodes (first 20): {misses[:20]}', flush=True)

    # ---- invariants: startup rows all-masked; established rows fully valid (anchors are multiples of 40 here) and the last
    #      window step equals the anchor state stored in ctx (float16 rounding)
    k0 = af == 0; est = ~k0
    assert not hmask[k0].any(), 'k = 0 rows must be all-masked'
    assert np.all(hmask[found & est, :].sum(1) == np.minimum(af[found & est], T)), 'hmask count != min(k, 40)'
    ref = out['ctx'][found & est][:, HIST_STATE_COLS].astype(np.float32); got = hist[found & est, T - 1, :12].astype(np.float32)
    tol = 2e-3 * np.maximum(np.abs(ref), 1.0)
    assert np.all(np.abs(ref - got) <= tol), f'hist last step != ctx state at anchor (max err {np.abs(ref - got).max():.4g})'
    assert np.all(priv[found & (domain == 1), 7] == 1) and np.all(priv[domain == 0, 5:8] == 0)
    out['hist'] = hist; out['hmask'] = hmask; out['privileged'] = priv
    out['hist_cols'] = np.asarray(HIST_STATE_COLS, np.int16); out['priv_names'] = np.asarray(PRIV_NAMES, object)

    # ---- counts
    table = {}
    for dm_name, dm in DOMAIN_CODE.items():
        for sp in ['train', 'val', 'test']:
            for kind, msk in [('startup', k0), ('established', est)]:
                table[f'{dm_name}|{sp}|{kind}'] = int(((domain == dm) & (split == sp) & msk).sum())
    pairs = defaultdict(set)
    for e, k, dm in trip:
        pairs[(e, k)].add(dm)
    both = [(e, k) for (e, k), s in pairs.items() if len(s) == 2]
    summary = dict(out=os.path.abspath(a.out), rows=n, rows_rigid=n_by['rigid'], rows_crm=n_by['crm'], episodes_rigid=len(set(episode[domain == 0])),
                   episodes_crm=len(set(episode[domain == 1])), groups=len(set(group)), counts=table,
                   raw_found_rows=int(found.sum()), raw_found_fraction=float(found.mean()), missing_episodes=misses,
                   coinciding_pairs=len(both), coinciding_pairs_established=sum(1 for e, k in both if k > 0),
                   priv_percentiles={nm: np.percentile(priv[domain == 1, j], [1, 50, 99]).round(4).tolist() for j, nm in enumerate(PRIV_NAMES)},
                   hist_T=T, hist_state_cols=HIST_STATE_COLS, hmask_rule='action frame k-40+t >= 0', priv_window='state frames max(0,k-39)..k',
                   sources=dict(rigid=a.rigid, crm=a.crm, rigid_runs=a.rigid_runs, crm_runs=a.crm_runs), workers=a.workers, limit_episodes=a.limit_episodes)
    print('counts (domain|split|kind):'); [print(f'  {k:28s} {v:7d}') for k, v in table.items()]
    print(f'coinciding (episode, anchor_frame) pairs: {len(both)} ({summary["coinciding_pairs_established"]} established)', flush=True)

    # ---- corridors: preallocate the output and fill from one source at a time (peak ~ out + one input)
    X = np.zeros((n, 5, 96, 32), np.float16); o = 0
    for name, path, _ in srcs:
        z = np.load(path, allow_pickle=True); xs = z['X']
        xs = xs[keep[name]] if a.limit_episodes else xs
        assert xs.dtype == np.float16 and xs.shape[0] == n_by[name]
        X[o:o + len(xs)] = xs; o += len(xs); del xs, z
    out['X'] = X
    print(f'X assembled {X.shape} in {time.time() - t0:.0f} s', flush=True)
    (np.savez if a.no_compress else np.savez_compressed)(a.out, **out)
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6; rc = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6
    summary.update(elapsed_s=round(time.time() - t0, 1), peak_rss_gb_main=round(ru, 2), peak_rss_gb_child_max=round(rc, 2), size_gb=round(os.path.getsize(a.out) / 1e9, 3))
    with open(os.path.splitext(a.out)[0] + '_build.json', 'w') as f:
        json.dump(summary, f, indent=1)
    print(f'wrote {a.out} ({summary["size_gb"]} GB) in {summary["elapsed_s"]} s; peak RSS main {ru:.1f} GB, child max {rc:.1f} GB', flush=True)
    if misses:
        print(f'WARNING: {len(misses)} episodes without a raw run dir (their hist/privileged are zeros, hmask False)', flush=True)
        sys.exit(2)


if __name__ == '__main__':
    main()
