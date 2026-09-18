"""Rebuild one evaluation group's 256-candidate pool (same md5 seed as crm_pools.py) and score it with the CRM-trained
ensemble; save everything the scoring picture and the demo drives need."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import crm_pools as CP
import gen_planner as P

ap = argparse.ArgumentParser(); ap.add_argument('--group', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
K = 'artifacts/traverse/crm_f104_v1'
CP._init(K + '/map_root')
g, case_path, tries, pools = CP.build(f'{K}/cases_eval/cases/{a.group}.json')
crm = P.RiskModel(f'{K}/train_v1/deploy/CRM_N2_s*.pt'); rigid = P.RiskModel()
pl = pools['proposal']; X = pl['X'].astype(np.float32)
zc = CP.member_logits(crm, X, pl['ctx']); zr = CP.member_logits(rigid, X, pl['ctx'])
z = zc.mean(0); p = 1 - np.exp(-np.exp(z))
cands = pl['cands']
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / 'pool.npz', logit_crm=z, p_crm=p, logit_crm_members=zc, logit_rigid=zr.mean(0),
                    wp=np.stack([np.asarray(c['waypoints']) for c in cands]), sp=np.stack([np.asarray(c['speeds']) for c in cands]),
                    st=np.stack([np.asarray(c['stations']) for c in cands]), hd=np.stack([np.asarray(c['headings']) for c in cands]),
                    anchor=np.array([c.get('meta', {}).get('candidate') == 'n2_anchor' for c in cands]))
pick = json.load(open(f'{K}/eval_v1/picks/{a.group}.json'))['arms']
assert int(np.argmin(z)) == pick['crm']['index'], (int(np.argmin(z)), pick['crm']['index'])
order = np.argsort(z)
print('n', len(z), 'argmin', int(order[0]), 'matches the locked evaluation pick')
ms = np.array([np.asarray(c['speeds'])[1:-1].mean() for c in cands]); L = np.array([c['stations'][-1] for c in cands])
for r in list(range(0, 12)) + list(range(20, 256, 20)) + [255]:
    i = order[r]; print(f'rank {r:3d} idx {i:3d} P {p[i]:.3f} logit {z[i]:+.2f} mean speed {ms[i]:.2f} length {L[i]:.1f} anchor {bool(cands[i].get("meta",{}).get("candidate")=="n2_anchor")}')
print('P quantiles', np.round(np.quantile(p, [0, .1, .25, .5, .75, .9, 1]), 3))
