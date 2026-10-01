"""Write (--write) or check (--check) a build lock: runner.fingerprint, the sha256 of pychrono's
native modules, the Chrono libraries of its build and the Chrono data its vehicles' physics reads (NEDM_CHRONO_DATA).
Lock only a build that passed the parity checks (configs/traversing/evaluation/locks); an arm with build_lock refuses
any other build.

    PYTHONPATH=src python scripts/traversing/evaluation/fingerprint.py --write LOCK.json --note 'B1/B2 bitwise, job N'
        [--render]     (M1: also the run-time OptiX programs and the HMMWV meshes the depth camera renders)
    PYTHONPATH=src python scripts/traversing/evaluation/fingerprint.py --check LOCK.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

from nedm.traversing.evaluation.config import Env, write_atomic
from nedm.traversing.evaluation.runner import code_sha, fingerprint, node_info

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
g = p.add_mutually_exclusive_group(required=True)
g.add_argument('--write', type=Path)
g.add_argument('--check', type=Path)
p.add_argument('--note', default='', help='the parity checks this build passed')
p.add_argument('--render', action='store_true', help='a render lock: runner.RENDER files included (M1)')
a, env = p.parse_args(), Env.from_environ()
if a.write:
    if a.write.exists():
        sys.exit(f'{a.write} exists: a lock is never overwritten')
    node, fp = node_info(env), fingerprint(env, a.render)
    write_atomic(a.write, json.dumps(dict(fingerprint=fp, render=a.render, build_sha=node['build_sha'], note=a.note,
                                          chrono_data=str(env.chrono_data), host=node['host'], code_sha=code_sha(),
                                          written=time.strftime('%Y-%m-%dT%H:%M:%S')), indent=1))
    print(f'wrote {a.write}: build {node["build_sha"]} ({len(fp)} files, render {a.render})')
else:
    lock = json.loads(a.check.read_text())
    lock, fp = lock['fingerprint'], fingerprint(env, bool(lock.get('render')))
    diff = sorted(k for k in lock.keys() | fp.keys() if lock.get(k) != fp.get(k))
    print(json.dumps(dict(lock=str(a.check), equal=not diff, differ=diff)))
    sys.exit(1 if diff else 0)
