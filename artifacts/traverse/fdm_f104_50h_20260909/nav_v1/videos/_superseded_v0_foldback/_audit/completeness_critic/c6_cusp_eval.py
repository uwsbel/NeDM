import json, glob, os, numpy as np
B='/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_f104_50h_20260909/nav_v1/'
for label, root in [('local eval', B+'local_luffy/runs/'), ('AMD main', B+'main/runs/')]:
    tab={}
    for d in sorted(glob.glob(root+'*__*')):
        if not os.path.exists(d+'/mission_outcome.json'): continue
        o=json.load(open(d+'/mission_outcome.json')); rr=json.load(open(d+'/routes.json'))
        n=0; last=None
        for r in rr:
            w=np.array(r['waypoints'])
            if len(w)<3: continue
            h=np.degrees(np.arctan2(np.diff(w[:,1]),np.diff(w[:,0]))); dh=np.abs((np.diff(h)+180)%360-180)
            if dh.max()>150: n+=1; last=r['frame']
        arm=os.path.basename(d).split('__')[1]; st=o['status']
        k=(st, n>0); tab.setdefault(k,[]).append(os.path.basename(d)+(f'({n})' if n else ''))
    print('==',label)
    for k in sorted(tab): print(k, len(tab[k]), tab[k] if k[1] or k[0]!='mission_complete' else '')
