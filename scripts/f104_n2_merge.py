"""Merge night-1 (old campaign) and night-2 (waves A1, A2) route tensors into one training set."""
import json, os, sys
import numpy as np

ROOT = 'artifacts/traverse/fdm_f104_50h_20260909'
KEYS = ('X', 'ctx', 'id', 'group', 'split', 'source', 'profile', 'fail', 'unsafe', 'event_idx', 'route_len')


def load(path, wave):
    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in KEYS}
    d['wave'] = np.array([wave] * len(d['id']), object)
    if wave.startswith('night2'):
        # every night-2 group is training data: the evaluation set is the frozen fresh-group list, which is
        # disjoint from these by construction (>= 6 m in start/goal space) and was never collected.
        d['split'] = np.array(['train'] * len(d['id']), object)
    return d


def main():
    parts = [load(ROOT + '/night2_v1/station_ds_fix.npz', 'night1_50h')]
    for name, wave in (('station_ds_A1.npz', 'night2_A1_coverage'), ('station_ds_A2.npz', 'night2_A2_on_policy')):
        p = ROOT + '/night2_v1/' + name
        if os.path.exists(p):
            parts.append(load(p, wave))
        else:
            print('missing (skipped):', p)
    out = {k: np.concatenate([p[k] for p in parts]) for k in list(KEYS) + ['wave']}
    ids, counts = np.unique(out['id'], return_counts=True)
    assert counts.max() == 1, f'duplicate ids: {ids[counts>1][:5]}'
    np.savez(ROOT + '/night2_v1/station_ds_all.npz', **out)
    src = out['source'].astype(str); wave = out['wave'].astype(str); sp = out['split'].astype(str)
    print(f"{len(out['id'])} routes, {len(np.unique(out['group'].astype(str)))} groups -> station_ds_all.npz "
          f"({round(os.path.getsize(ROOT + '/night2_v1/station_ds_all.npz')/1e6)} MB)")
    for w in np.unique(wave):
        m = wave == w
        print(f"  {w:22s} {int(m.sum()):6d} routes  unsafe {100*out['unsafe'][m].mean():4.1f}%  "
              f"fail {100*out['fail'][m].mean():4.1f}%  sources {sorted(set(src[m]))}")


if __name__ == '__main__':
    main()
