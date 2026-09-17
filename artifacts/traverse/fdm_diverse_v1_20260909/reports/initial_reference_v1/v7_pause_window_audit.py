from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
root=Path('/work1/dannegrut/harry/experiments/fdm_diverse_v1_20260909/online_full_validation_grid_v2')
report={'observed_utc':datetime.now(timezone.utc).isoformat(),'scope':'Supplemental command-overlap diagnostic, primary event/success definitions unchanged. No simulation/model changes.','timing':'Planner paused and desired speed requests reconstructed at recorded new_request_first_applied_time_s on2ms grid. Counts concern half-open2s windows of40recorded intervals; do not relabel measured bounded events.','trials':[]}
for family in ['ridge_passes','rough_mosaic']:
 p=root/f'diverse_v1_val_{family}_00_time';outcome=json.loads((p/'outcome.json').read_text());decisions=[json.loads(f.read_text()) for f in sorted((p/'decisions').glob('*.json'))]
 n=round(outcome['elapsed_s']/.002);paused=np.zeros(n,bool);desired=np.full(n,np.nan)
 for i,row in enumerate(decisions):
  begin=round(row['new_request_first_applied_time_s']/.002)
  end=round(decisions[i+1]['new_request_first_applied_time_s']/.002) if i+1<len(decisions) else n
  paused[begin:min(end,n)]=row['paused'];desired[begin:min(end,n)]=row['new_desired_speed_mps']
 windows=outcome['bounded_blockage_v1_windows'];begins=np.asarray(windows,dtype=int)*25;ends=begins+1000
 conditions={'planner_paused':paused,'desired_speed_zero':np.isfinite(desired)&(np.abs(desired)<1e-9),'either_paused_or_zero':paused|(np.isfinite(desired)&(np.abs(desired)<1e-9))}
 counts={}
 for key,values in conditions.items():
  prefix=np.r_[0,np.cumsum(values)];samples=prefix[ends]-prefix[begins]
  counts[key]={'any':int((samples>0).sum()),'all':int((samples==1000).sum()),'none':int((samples==0).sum()),'partly':int(((samples>0)&(samples<1000)).sum()),'overlap_fraction_mean':float(samples.mean()/1000) if len(samples) else None,'whole_run_duration_s':float(values.sum()*.002)}
 report['trials'].append({'scene_id':outcome['case_id'],'bounded_window_count':len(windows),'counts':counts,'first_window':None if not windows else {'start_s':windows[0]*.05,'end_s':windows[0]*.05+2,'paused_overlap_s':float(paused[begins[0]:ends[0]].sum()*.002),'zero_desired_overlap_s':float(conditions['desired_speed_zero'][begins[0]:ends[0]].sum()*.002)},'primary_outcome_sha256':hashlib.sha256((p/'outcome.json').read_bytes()).hexdigest()})
print(json.dumps(report,indent=2,allow_nan=False))
