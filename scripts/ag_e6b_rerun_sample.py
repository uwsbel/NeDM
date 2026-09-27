#!/usr/bin/env python3
"""E6b pick determinism sample (arena_gator_20260925): re-plan the 5 lowest-md5 groups of every model pick directory
(e6/picks/rigid/<arena>/<model>_<free|fixed2>) in a new process into a scratch folder and compare the route files byte
for byte (file sha256) with the production picks. Writes e6/picks/rerun_sample.json.
  PYTHONPATH=src:scripts python scripts/ag_e6b_rerun_sample.py [--parallel 10]
"""
import argparse, glob, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

K3 = Path('artifacts/traverse/arena_gator_20260925')
S = Path('/tmp/ag_e6b_rerun')


def one(d):
    m = json.load(open(f'{d}/ag_picks.json'))
    gs = open(f'{d}/groups.txt').read().split()[:5]
    o = S / d.replace('/', '_')
    gf = Path(str(o) + '.groups'); gf.write_text('\n'.join(gs) + '\n')
    cmd = [sys.executable, 'scripts/ag_picks.py', '--arena', m['arena'], '--world', m['world'], '--mode', m['mode'], '--cases', m['cases_dir'],
           '--groups', f'@{gf}', '--models', m['models_glob'], '--model-tag', m['model_tag'], '--map-root', m['map_root'], '--out', str(o), '--force']
    env = dict(os.environ, OMP_NUM_THREADS='2'); env.pop('NEDM_VEHICLE', None)
    rc = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    if rc:
        return dict(dir=d, rc=rc)
    r = json.load(open(o / 'ag_picks.json'))['picks']
    return dict(dir=d, groups=len(gs), identical=sum(1 for g in gs if r[g]['file_sha256'] == m['picks'][g]['file_sha256']))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--parallel', type=int, default=10); a = ap.parse_args()
    S.mkdir(parents=True, exist_ok=True)
    dirs = sorted(d for d in glob.glob(str(K3 / 'e6/picks/rigid/*/*_free')) + glob.glob(str(K3 / 'e6/picks/rigid/*/*_fixed2'))
                  if os.path.exists(f'{d}/ag_picks.json'))
    with ThreadPoolExecutor(a.parallel) as ex:
        res = list(ex.map(one, dirs))
    out = dict(dirs=len(res), groups=sum(r.get('groups', 0) for r in res), identical=sum(r.get('identical', 0) for r in res),
               failed=[r for r in res if 'rc' in r], differing=[r for r in res if r.get('identical') != r.get('groups')], per_dir=res)
    json.dump(out, open(K3 / 'e6/picks/rerun_sample.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'per_dir'}))


if __name__ == '__main__':
    main()
