#!/usr/bin/env python3
"""VERIFY_gator_full (independent checker of task B stage 2, arena_gator_20260925): read-only extraction on the login node.

  runs   <out> <runs root>   every run folder whose name holds an f104 suite group (f104_pair_group_ / f104_crm_eval_group_):
                             scene, vehicle block, reference.json sha256 (bytes and content = the 4 driven fields), collection_request fields (route / case sha256,
                             collector and wrapper sha256, crm config name, host), outcome status / goal_reached / elapsed,
                             episode_complete present, launch check (initial_state_validation passed / finite), outcome.json sha256, mtimes of collection_request.json and outcome.json
                             -> <out>/runs_suite.jsonl
  light  <out> <npz ...>     light per-row columns of training / evaluation npz files (+ their key list) -> <out>/<name>_light.npz
Nothing is written outside <out>; nothing imported from the study's tools."""
import hashlib, json, os, sys, time
import numpy as np


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


ROUTE_FIELDS = ('waypoints', 'speeds', 'stations', 'headings')


def content_sha(route):
    """route content = the four driven fields (meta excluded), canonical JSON"""
    return hashlib.sha256(json.dumps({k: route.get(k) for k in ROUTE_FIELDS}, sort_keys=True).encode()).hexdigest()


def runs(out, root):
    os.makedirs(out, exist_ok=True)
    names = sorted(n for n in os.listdir(root) if 'f104_pair_group_' in n or 'f104_crm_eval_group_' in n)
    t0 = time.time()
    with open(os.path.join(out, 'runs_suite.jsonl'), 'w') as fo:
        for i, n in enumerate(names):
            d = os.path.join(root, n)
            r = dict(run=n)
            try:
                cr = json.load(open(os.path.join(d, 'collection_request.json')))
                v = cr.get('vehicle')
                r.update(scene=cr.get('scene_id'), route_sha256_req=cr.get('route_sha256'), case_sha256=cr.get('case_sha256'),
                         collector_sha256=cr.get('collector_sha256'), crm_config=(cr.get('crm_config') or {}).get('name'),
                         host=cr.get('host'), vehicle=(v.get('name') if isinstance(v, dict) else v),
                         wrapper_sha256=(v.get('wrapper_sha256') if isinstance(v, dict) else None),
                         ag_vehicle_sha256=(v.get('ag_vehicle_sha256') if isinstance(v, dict) else None),
                         req_mtime=os.path.getmtime(os.path.join(d, 'collection_request.json')))
            except FileNotFoundError:
                r['no_request'] = True
            rp = os.path.join(d, 'reference.json')
            r['reference_sha256'] = sha(rp) if os.path.exists(rp) else None
            if os.path.exists(rp):
                ref = json.load(open(rp))
                r['reference_content_sha256'] = content_sha(ref)
            op = os.path.join(d, 'outcome.json')
            if os.path.exists(op):
                o = json.load(open(op))
                r.update(status=o.get('status'), goal_reached=o.get('goal_reached'), elapsed_s=o.get('elapsed_s'),
                         goal_time_s=o.get('goal_time_s'), outcome_sha256=sha(op), out_mtime=os.path.getmtime(op),
                         case_id=o.get('case_id'))
            r['complete'] = os.path.exists(os.path.join(d, 'episode_complete.json'))
            ip = os.path.join(d, 'initial_state_validation.json')
            if os.path.exists(ip):
                iv = json.load(open(ip)); r['launch_passed'] = iv.get('passed'); r['launch_finite'] = iv.get('finite')
            fo.write(json.dumps(r) + '\n')
            if i % 2000 == 0:
                print(i, len(names), round(time.time() - t0, 1), flush=True)
    print('runs done', len(names), round(time.time() - t0, 1), flush=True)


KEYS = ('id', 'group', 'split', 'domain', 'arena', 'vehicle', 'tier', 'episode', 'anchor_frame', 'source', 'status',
        'profile', 'fail', 'unsafe', 'route_len')


def light(out, files):
    os.makedirs(out, exist_ok=True)
    for p in files:
        t0 = time.time()
        z = np.load(p, allow_pickle=True)
        d = {}
        for k in KEYS:
            if k in z.files:
                v = z[k]
                d[k] = v.astype(str) if v.dtype.kind in 'OUS' else v
        d['_keys'] = np.array(sorted(z.files))
        name = os.path.basename(p)[:-4]
        np.savez_compressed(os.path.join(out, name + '_light.npz'), **d)
        print(name, len(d['id']), round(time.time() - t0, 1), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'runs':
        runs(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'light':
        light(sys.argv[2], sys.argv[3:])
    else:
        raise SystemExit(__doc__)
