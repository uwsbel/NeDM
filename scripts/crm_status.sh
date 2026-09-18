#!/bin/bash
# one-line status of a CRM collect dir on the cluster
D=${1:-/work1/dannegrut/harry/experiments/crm_f104_20260916/collect_v1}
ssh -o ConnectTimeout=20 amd "cd $D && python3 - <<'PY'
import json,glob,os,time
w=[json.load(open(f)) for f in glob.glob('workers/*.json')]
ep=sum(x['episodes'] for x in w); sim=sum(x['sim_s'] for x in w)/3600
live=[x for x in w if time.time()-x['updated']<900]
rate=sum(x['sim_s']/max(x['wall_s'],1) for x in live)
print(time.strftime('%H:%M'), 'episodes', ep, 'sim_h %.2f'%sim, 'failed', len(os.listdir('failed')), 'claims', len(os.listdir('claims')), 'live_workers', len(live), 'rate_sim_h_per_h %.1f'%rate)
PY
squeue -u harry -h | awk '{print \$2,\$5}' | sort | uniq -c | tr '\n' ';'"
