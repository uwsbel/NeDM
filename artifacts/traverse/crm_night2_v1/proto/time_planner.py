import sys, time, json, glob, os
import numpy as np
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src')
import gen_planner as GP, f104_n2_dataset as DS, f104_n2_sampler as S
root = 'artifacts/traverse/crm_f104_v1/map_root'
DS.init_map(root)
print('map', DS.G['npx'], DS.G['mpp'], DS.G['elev_scale'])
cases = sorted([p for p in glob.glob("artifacts/traverse/crm_f104_v1/cases_eval/cases/*.json") if not p.endswith("/cases.json")])[:8]
model = GP.RiskModel(pattern='artifacts/traverse/crm_f104_v1/train_v1/deploy/CRM_N2_s*.pt')
print('device', model.dev, 'members', len(model.members))
rows = []
for cp in cases:
    c = json.load(open(cp))
    pose = np.array([c['layout']['start_xy'][0], c['layout']['start_xy'][1], c['layout']['start_yaw']], float)
    goal = np.asarray(c['goal_xy'], float)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter(); base = GP.base_route(pose, goal); t1 = time.perf_counter()
    cands, tries = GP.proposal_pool(base, pose, rng); t2 = time.perf_counter()
    X, L = GP.corridors(cands); t3 = time.perf_counter()
    z, p = model.score(X, GP.geom_ctx(pose[:2], goal, pose[2], L)); t4 = time.perf_counter()
    # time the pieces of proposal: sample_one vs validate
    ts = time.perf_counter(); [S.sample_one(base, rng) for _ in range(256)]; ts1 = time.perf_counter()
    r = S.sample_one(base, rng); tv = time.perf_counter(); [GP.safe_validate(r, [], GP.CFG, pose) for _ in range(256)]; tv1 = time.perf_counter()
    n_anchor = sum(1 for r in cands if r['meta'].get('candidate') == 'n2_anchor')
    rows.append(dict(case=os.path.basename(cp), L=float(base['stations'][-1]), n_wp=len(base['waypoints']), n=len(cands), tries=tries, anchors=n_anchor,
                     acc=(len(cands)-n_anchor)/max(tries,1), base_ms=(t1-t0)*1e3, propose_ms=(t2-t1)*1e3, corridor_ms=(t3-t2)*1e3, score_ms=(t4-t3)*1e3,
                     sample_only_ms_per=(ts1-ts)*1e3/256, validate_ms_per=(tv1-tv)*1e3/256, zmin=float(z.min()), zmed=float(np.median(z)), pmin=float(p.min())))
    print(rows[-1])
print('MEAN', {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'case'})
# second call of score to exclude warmup
t=time.perf_counter(); z,p = model.score(X, GP.geom_ctx(pose[:2], goal, pose[2], L)); print('score warm ms', (time.perf_counter()-t)*1e3)
# rejection reasons on 512 raw samples
from nedm.traverse.fdm_mppi import validate_reference
import collections
reasons = collections.Counter(); nval = 0
for cp in cases[:4]:
    c = json.load(open(cp)); pose = np.array([c['layout']['start_xy'][0], c['layout']['start_xy'][1], c['layout']['start_yaw']], float); goal = np.asarray(c['goal_xy'], float)
    base = GP.base_route(pose, goal); rng = np.random.default_rng(1)
    for _ in range(512):
        r = S.sample_one(base, rng)
        try: v = validate_reference(r, [], GP.CFG, pose)
        except ValueError as e: v = {'valid': False, 'reasons': ['degenerate']}
        if v['valid']: nval += 1
        for rs in v['reasons']: reasons[rs] += 1
print('raw acceptance', nval / (4*512), 'reasons', dict(reasons))
