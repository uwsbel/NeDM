"""Shared helpers of the arena_gator_20260925 task builders (E3a): scripts/ag_soil_tasks.py, scripts/ag_rigid_tasks.py.

group_routes() reproduces the route order of scripts/crm_tasks.py (the f104 night-2 soil pool) for any group: the 12
designed routes (route_00..11) then the 8 on-policy routes (op_00..07), shuffled by random.Random(md5(group)[:8]);
position k in the shuffled list is tier k. random.shuffle's permutation depends only on the generator and the list
length, so the order equals crm_tasks.py's for the same group name (checked against the f104 tasks_train.json).
Episode ids are <group>_route_NN / <group>_op_NN (rigid and soil twins share them); episode_seed = md5(id)[:8].
Paths in the rows are relative to the cluster experiment root G3 (= CRM_ROOT = GEN_ROOT); local checks map them to K3.
"""
import hashlib, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
K3 = ROOT / 'artifacts/traverse/arena_gator_20260925'
G3 = '/work1/dannegrut/harry/experiments/arena_gator_20260925'
K3_REL = 'artifacts/traverse/arena_gator_20260925/'


def md5_int(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def train_dirs(arena):
    """(designed case dir, on-policy dir), relative to G3 / K3."""
    return f'cases/train_{arena}/cases', f'cases/train_{arena}_onpolicy'


def records(case_dir):
    """cases.json records of a gen_cases.py folder (relative to K3): {scene_id: record}."""
    return {r['scene_id']: r for r in json.load(open(K3 / case_dir / 'cases.json'))['records']}


def group_ids(case_dir, n=None):
    ids = sorted(records(case_dir))
    return ids[:n] if n else ids


def onpolicy_index(arena):
    """{(group, k): route path relative to G3} from routes.json (repo-relative paths there)."""
    d = train_dirs(arena)[1]
    out = {}
    for r in json.load(open(K3 / d / 'routes.json')):
        p = r['route']
        assert p.startswith(K3_REL), p
        out[(r['group'], int(r['index']))] = p[len(K3_REL):]
    return out


def group_routes(g, designed_dir, onpolicy_dir, op_index=None):
    """[(tier, id, route path, kind)] in tier order (crm_tasks.py shuffle)."""
    routes = [(f'{g}_route_{k:02d}', f'{designed_dir}/routes/{g}/route_{k:02d}.json', 'designed') for k in range(12)]
    ops = []
    for k in range(8):
        p = f'{onpolicy_dir}/routes/{g}/op_{k:02d}.json'
        if op_index is not None:
            assert op_index[(g, k)] == p, (g, k, op_index.get((g, k)), p)
        ops.append((f'{g}_op_{k:02d}', p, 'on_policy'))
    routes += ops
    random.Random(md5_int(g)).shuffle(routes)
    return [(tier, rid, route, kind) for tier, (rid, route, kind) in enumerate(routes)]


def case_split(case_rel):
    return json.load(open(K3 / case_rel))['split']


def local_path(rel):
    """cluster path relative to G3 (or absolute G3 path) -> local K3 path; None for paths outside G3."""
    if rel.startswith('/'):
        return K3 / rel[len(G3) + 1:] if rel.startswith(G3 + '/') else None
    return K3 / rel
