#!/usr/bin/env python3
"""VERIFY E4: the subset tool's rules on the presets that have no real data yet (M2, M3, A3, LOAO*, H, tier cut),
recomputed with my own code on a synthetic three-arena light file made from the f104 columns:
  f104 as is; 'g203' = f104 renamed, soil rows of tiers > 9 dropped (a partial arena); 'g228' = f104 renamed with the
  groups whose md5 % 7 == 0 dropped and 540 training groups kept at most (fewer than 545).
ag_subset.py runs with --list-only (manifest only); its selected groups and row counts are compared with mine."""
import hashlib, json, os, subprocess, sys, tempfile
import numpy as np

K3 = '/home/harry/NeDM-traverse_mppi/artifacts/traverse/arena_gator_20260925'
REPO = '/home/harry/NeDM-traverse_mppi'
PY = '/home/harry/miniconda3/envs/nedm/bin/python'
LIGHT = ('id', 'group', 'split', 'domain', 'arena', 'vehicle', 'tier', 'episode', 'anchor_frame')
z = np.load(K3 + '/e4/f104_hmmwv/ci_f104_hmmwv_both.npz', allow_pickle=True)
L = {k: z[k] for k in LIGHT}
md5 = lambda s: hashlib.md5(s.encode()).hexdigest()
tmp = tempfile.mkdtemp(prefix='ve4_syn_')


def renamed(name, keep):
    d = {k: v[keep] for k, v in L.items()}
    for k in ('id', 'group', 'episode'):
        d[k] = np.array([s.replace('f104_', f'{name}_', 1) for s in d[k].astype(str)], object)
    d['arena'] = np.array([name] * keep.sum(), object)
    return d


dom = L['domain'].astype(int); tier = L['tier'].astype(int); grp = L['group'].astype(str); sp = L['split'].astype(str)
files = {}
files['f104'] = {k: v for k, v in L.items()}
files['g203'] = renamed('g203', ~((dom == 1) & (tier > 9)))
tr_groups = sorted({g for g, s in zip(grp, sp) if s == 'train' and int(md5(g), 16) % 7 != 0})[:540]
keep228 = np.array([(s != 'train' or g in set(tr_groups)) and int(md5(g), 16) % 7 != 0 for g, s in zip(grp, sp)])
files['g228'] = renamed('g228', keep228)
paths = {}
for n, d in files.items():
    paths[n] = f'{tmp}/syn_{n}.npz'; np.savez(paths[n], **d)
# ids file for H: 90 % of the soil episodes, written with the Gator prefix
rng = np.random.default_rng(3)
eps = sorted(set(L['episode'].astype(str)[dom == 1]))
keep_ids = set(rng.choice(eps, int(0.9 * len(eps)), replace=False))
open(f'{tmp}/ids.txt', 'w').write('\n'.join('gator__' + e for e in sorted(keep_ids)))

allf = [paths['f104'], paths['g203'], paths['g228']]


def mine(world, groups, tiers=None, ids=None, keep_dev=False, allow_short=False):
    code = 0 if world == 'rigid' else 1
    res = {}
    rows = 0; fit_h = 0
    for arena, n in groups.items():
        d = files[arena]
        m = d['domain'].astype(int) == code
        t = d['tier'].astype(int)
        f = m.copy()
        if tiers: f &= (t >= tiers[0]) & (t <= tiers[1])
        e = d['episode'].astype(str)
        if ids is not None: f &= np.array([x in ids for x in e])
        g = d['group'].astype(str); s = d['split'].astype(str)
        cand = sorted(set(g[f & (s == 'train')]), key=md5)
        k = len(cand) if n == 'all' else n
        if k > len(cand):
            if not allow_short: res[arena] = 'SHORT'; continue
            k = len(cand)
        sel = set(cand[:k]); devk = {x for x in cand if int(md5(x), 16) % 5 == 0} if keep_dev else set()
        ev = m.copy()
        if ids is not None: ev &= np.array([x in ids for x in e])
        keep = (f & (s == 'train') & np.isin(g, list(sel | devk))) | (ev & np.isin(s, ['val', 'test']))
        rows += int(keep.sum()); fit_h += int((keep & (s == 'train') & np.array([int(md5(x), 16) % 5 != 0 for x in g])).sum())
        res[arena] = dict(sel=sel, rows=int(keep.sum()), train_rows=int((keep & (s == 'train')).sum()))
    return res, rows, fit_h


def tool(world, preset=None, groups=None, tiers=None, ids=False, extra=()):
    out = f'{tmp}/o_{preset or "g"}_{world}.npz'
    cmd = [PY, f'{REPO}/scripts/ag_subset.py', '--ds', *allf, '--world', world, '--out', out, '--list-only']
    cmd += ['--preset', preset] if preset else ['--groups', groups]
    if tiers: cmd += ['--tiers', tiers]
    if ids: cmd += ['--ids-file', f'{tmp}/ids.txt']
    cmd += list(extra)
    p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=f'{REPO}/src:{REPO}/scripts'))
    if p.returncode != 0:
        return None, p.stderr.strip().splitlines()[-1]
    # --list-only prints the manifest (without selected groups): re-run the selection through main() in process
    sys.path.insert(0, f'{REPO}/scripts')
    import ag_subset
    argv = cmd[2:]
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        man = ag_subset.main(argv)
    return man, None


tests = [
    ('M2', 'crm', dict(f104=545, g203=545), None, False, False),
    ('M3', 'crm', dict(f104=363, g203=363, g228=363), None, False, False),
    ('M3', 'rigid', dict(f104=363, g203=363, g228=363), None, False, False),
    ('A3', 'crm', dict(f104='all', g203='all', g228='all'), None, False, False),
    ('LOAO2_f104_g228', 'rigid', dict(f104=273, g228=272), None, False, False),
    ('LOAO1_g228', 'crm', dict(g228=545), None, False, False),
    ('M3', 'crm', dict(f104=363, g203=363, g228=363), (0, 9), False, False),
    ('H', 'crm', dict(f104='all'), None, True, False),
    ('LC545', 'rigid', dict(f104=545), None, False, True),
]
report = []
for preset, world, groups, tiers, ids, kd in tests:
    man, err = tool(world, preset=preset, tiers=f'{tiers[0]}-{tiers[1]}' if tiers else None, ids=ids)
    me, rows, fit_h = mine(world, groups, tiers, keep_ids if ids else None, keep_dev=kd)
    short = any(v == 'SHORT' for v in me.values())
    r = dict(preset=preset, world=world, tiers=tiers, ids=ids)
    if short:
        r.update(expect='refuse (short)', tool_refused=man is None, tool_error=err)
    else:
        r.update(tool_ok=man is not None, err=err)
        if man is not None:
            r.update(rows_tool=man['rows'], rows_mine=rows, fit_holdout_tool=man['fit_rows_holdout'], fit_holdout_mine=fit_h,
                     groups_equal={a: set(man['selected_groups'][a]) == me[a]['sel'] for a in groups},
                     n_selected={a: len(me[a]['sel']) for a in groups},
                     train_rows_equal={a: man['per_arena'][a]['train_rows'] == me[a]['train_rows'] for a in groups})
    report.append(r)
# nesting and cross-world identity of M3
m3c, _, _ = mine('crm', dict(f104=363, g203=363, g228=363)); m3r, _, _ = mine('rigid', dict(f104=363, g203=363, g228=363))
m2c, _, _ = mine('crm', dict(f104=545, g203=545))
extra = dict(M3_same_groups_both_worlds={a: m3c[a]['sel'] == m3r[a]['sel'] for a in m3c},
             M3_inside_M2={a: m3c[a]['sel'] <= m2c[a]['sel'] for a in ('f104', 'g203')},
             g228_train_groups_synthetic=len(tr_groups))
print(json.dumps(dict(tests=report, extra=extra), indent=1, default=str))
json.dump(dict(tests=report, extra=extra), open(K3 + '/verify_e4/v6_subset_rules.json', 'w'), indent=1, default=str)
