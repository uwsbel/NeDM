#!/usr/bin/env python3
"""Verifier (VERIFY_rigid_results.md): read-only login-node extraction, per rigid evaluation run, of the host
(simulation_provenance.json, regex), the drive start (collection_request.json mtime), the end (episode_complete.json
mtime) and the sha256 of reference.json (the route actually driven). Run: ssh amd python3 - < this_file > out.json"""
import os, re, json, hashlib, sys
root = '/work1/dannegrut/harry/experiments/arena_gator_20260925/rigid_eval/runs'
pat = re.compile(r'"host"\s*:\s*"([^"]+)"')
out = {}
for rid in sorted(os.listdir(root)):
    d = os.path.join(root, rid)
    rec = {}
    try:
        with open(os.path.join(d, 'simulation_provenance.json')) as f:
            m = pat.search(f.read(4000)); rec['h'] = m.group(1).split('.')[0] if m else None
    except OSError:
        rec['h'] = None
    try:
        rec['t0'] = os.stat(os.path.join(d, 'collection_request.json')).st_mtime
    except OSError:
        rec['t0'] = None
    try:
        rec['ref'] = hashlib.sha256(open(os.path.join(d, 'reference.json'), 'rb').read()).hexdigest()
    except OSError:
        rec['ref'] = None
    try:
        rec['t1'] = os.stat(os.path.join(d, 'episode_complete.json')).st_mtime
    except OSError:
        rec['t1'] = None
    out[rid] = rec
json.dump(out, sys.stdout)
