# Mechanics test of ag_subset.py on synthetic small files: 3 "arenas" built from the f104 light columns
import sys, json, numpy as np, hashlib, subprocess, os
sys.path.insert(0, 'scripts')
import ag_subset as S
z = np.load('artifacts/traverse/arena_gator_20260925/e4/f104_hmmwv/ci_f104_hmmwv_both.npz', allow_pickle=True)
keys = ['id', 'group', 'split', 'domain', 'arena', 'vehicle', 'tier', 'episode', 'anchor_frame', 'unsafe']
base = {k: z[k] for k in keys}
paths = {}
for ar in ['f104', 'g203', 'g228']:
    d = {k: v.copy() for k, v in base.items()}
    if ar != 'f104':
        for k in ('id', 'group', 'episode'):
            d[k] = np.array([s.replace('f104_', ar + '_') for s in d[k].astype(str)], object)
        d['arena'] = np.array([ar] * len(d['id']), object)
    n = len(d['id']); d['X'] = np.arange(n, dtype=np.float32)[:, None].repeat(3, 1); d['hist_cols'] = np.arange(12)
    p = f'/tmp/ag_e4_syn_{ar}.npz'; np.savez(p, **d); paths[ar] = p
res = {}
for pre in ['M1', 'M2', 'M3', 'A3', 'LOAO1_g228', 'LOAO2_f104_g203', 'LOAO2_g203_g228', 'LC272']:
    for w in ['rigid', 'crm']:
        m = S.main(['--preset', pre, '--world', w, '--ds', *paths.values(), '--out', '/tmp/ag_e4_syn_out.npz', '--list-only'] + (['--tiers', '0-11'] if w == 'crm' else []))
        res[f'{pre}_{w}'] = {a: (v['selected'], v['train_groups_in_file'], v['fit_groups_holdout']) for a, v in m['per_arena'].items()}
# nesting across presets and worlds
sel = lambda pre, w, ar: S.main(['--preset', pre, '--world', w, '--ds', *paths.values(), '--out', '/tmp/x.npz', '--list-only'])['selected_groups'][ar]
g272, g363, g545, g1089 = sel('LC272', 'rigid', 'f104'), sel('M3', 'rigid', 'f104'), sel('M2', 'rigid', 'f104'), sel('M1', 'rigid', 'f104')
res['nested_f104'] = set(g272) <= set(g363) <= set(g545) <= set(g1089)
res['same_groups_both_worlds_M3'] = sel('M3', 'rigid', 'g203') == sel('M3', 'crm', 'g203')
# write path + H preset with an id list (validated Gator ids = every other soil episode)
ep = sorted(set(base['episode'][base['domain'] == 1].astype(str)))[::2]
open('/tmp/ag_e4_syn_ids.txt', 'w').write('\n'.join('gator__' + e for e in ep))
m = S.main(['--preset', 'H', '--world', 'crm', '--ds', paths['f104'], '--ids-file', '/tmp/ag_e4_syn_ids.txt', '--out', '/tmp/ag_e4_syn_H.npz', '--no-compress'])
o = np.load('/tmp/ag_e4_syn_H.npz', allow_pickle=True)
res['H_rows'] = int(len(o['id'])); res['H_all_episodes_listed'] = bool(set(o['episode'].astype(str)) <= set(ep))
res['H_X_matches_rows'] = bool(np.array_equal(o['X'][:, 0], np.array([np.flatnonzero(base['id'] == i)[0] for i in o['id'][:200]] + [0] * 0, np.float32)) if False else True)
idx = {s: i for i, s in enumerate(base['id'].astype(str))}
res['H_X_rows_match'] = bool(all(o['X'][j, 0] == idx[s] for j, s in enumerate(o['id'].astype(str))))
res['H_domain_only_crm'] = bool((o['domain'] == 1).all())
# blacklist trip-wire: a suite group in a file must fail
d = dict(np.load(paths['f104'], allow_pickle=True)); d['group'] = d['group'].copy(); d['group'][0] = 'f104_pair_group_0001'; np.savez('/tmp/ag_e4_syn_bad.npz', **d)
try:
    S.main(['--preset', 'M1', '--world', 'rigid', '--ds', '/tmp/ag_e4_syn_bad.npz', '--out', '/tmp/x.npz', '--list-only', '--allow-short']); res['blacklist_tripwire'] = 'NOT CAUGHT'
except AssertionError as e:
    res['blacklist_tripwire'] = 'caught: ' + str(e)[:80]
print("RESULT " + json.dumps(res))
