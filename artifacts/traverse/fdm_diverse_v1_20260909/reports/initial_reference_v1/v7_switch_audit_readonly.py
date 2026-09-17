from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
base=Path('/work1/dannegrut/harry/experiments/fdm_diverse_v1_20260909')
root=base/'online_full_validation_grid_v2'
case_root=base/'snapshots/campaign_v2/artifacts/traverse/fdm_diverse_v1_20260909/cases'
records=json.loads((case_root/'cases.json').read_text())['records']
report={'observed_utc':datetime.now(timezone.utc).isoformat(),'scope':'Completed validation ridge_time and rough_time of source online_v7/job412098 only; read-only decision/trajectory audit, no simulation or model changes.','trials':[]}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def point_dist(points,line):return np.linalg.norm(points[:,None,:]-line[None,:,:],axis=-1).min(1)
for family,initial_index in [('ridge_passes',14),('rough_mosaic',2)]:
 scene=f'diverse_v1_val_{family}_00';directory=root/(scene+'_time')
 record=next(r for r in records if r['scene_id']==scene and r['split']=='val')
 initial=json.loads((case_root/record['routes'][initial_index]).read_text());initial_xy=np.asarray(initial['waypoints'])
 start=initial_xy[0];goal=initial_xy[-1];u=(goal-start)/np.linalg.norm(goal-start);normal=np.array([-u[1],u[0]]);length=np.linalg.norm(goal-start)
 with np.load(directory/'trajectory.npz',allow_pickle=False) as f:positions=np.vstack([f['pose'],f['terminal_pose']]);states=f['state'];actions=f['action']
 distance_to_initial=point_dist(positions[:,:2],initial_xy)
 outcome=json.loads((directory/'outcome.json').read_text())
 with np.load(directory/'rich_intervals.npz',allow_pickle=False) as f:
  contact=np.maximum(f['max_asset_contact_max_resultant_n'],f['max_chassis_contact_resultant_n'])
  roll=f['max_abs_roll_rad'];pitch=f['max_abs_pitch_rad']
 contacts=np.flatnonzero(contact>1)
 trial={'scene_id':scene,'initial_fixed_winner_index':initial_index,'initial_fixed_winner_offset_m':initial['meta']['lateral_offset_m'],'initial_fixed_winner_physically_safe':family=='ridge_passes',
 'source':str(directory),'files':{name:sha(directory/name) for name in ['outcome.json','online_protocol.json','trajectory.npz','rich_intervals.npz']},
 'outcome':{k:outcome.get(k) for k in ['status','goal_reached','elapsed_s','goal_progress_m','final_goal_distance_m','path_length_m','sustained_near_stop','longest_consecutive_effortful_near_zero_speed_s','planning_decisions','planning_abstentions']},
 'first_contact_s':float((contacts[0]+1)*.05) if len(contacts) else None,
 'first_bounded_window_endpoint_s':float((min(outcome['bounded_blockage_v1_windows'])+40)*.05) if outcome['bounded_blockage_v1_windows'] else None,
 'first_distance_above_2m_from_initial_s':float(np.flatnonzero(distance_to_initial>2)[0]*.05) if (distance_to_initial>2).any() else None,
 'max_distance_from_initial_m':float(distance_to_initial.max()),'max_abs_roll_deg':float(np.degrees(roll.max())),'max_abs_pitch_deg':float(np.degrees(pitch.max())),'decisions':[]}
 for file in sorted((directory/'decisions').glob('*.json')):
  row=json.loads(file.read_text());decision=row['decision'];idx=decision.get('selected_family_index');selected=decision.get('route');families=row['candidate_references'];pose=np.asarray(row['anchor_pose'])
  entry={'time_s':row['time_s'],'frame':row['frame'],'pose':row['anchor_pose'],'vx_mps':row['anchor_state17'][0],'paused':row['paused'],'selected_family_index':idx,'selected_cost':decision.get('cost'),'distance_to_initial_m':float(point_dist(pose[None,:2],initial_xy)[0]),'decision_sha256':sha(file)}
  if selected:
   selected_xy=np.asarray(selected['waypoints']);family_meta=families[idx]['meta'];source_xy=np.asarray(families[idx]['waypoints']);ell=(selected_xy-start)@u;lat=(selected_xy-start)@normal
   order=np.argsort(ell);valid=np.diff(ell[order])>1e-7
   entry.update(selected_source_meta=family_meta,selected_route_meta=selected['meta'],source_midpoint_lateral_m=float(np.interp(length*.5,(source_xy-start)@u,(source_xy-start)@normal)),
    selected_midpoint_lateral_m=float(np.interp(length*.5,ell[order],lat[order])),
    selected_lateral_quarterpoints_m=[float(np.interp(length*q,ell[order],lat[order])) for q in [.25,.5,.75]],
    selected_vs_source_max_distance_m=float(point_dist(selected_xy,source_xy).max()),
    selected_vs_initial_max_distance_m=float(point_dist(selected_xy,initial_xy).max()),new_desired_speed_mps=row['new_desired_speed_mps'])
  scores=decision.get('family_scores',[])
  chosen_records=[]
  for score in scores:
   origin=score['reference_meta'].get('candidate_origin')
   if score['family_index']==idx or origin in ['active_reference',f'original_family_{initial_index:02d}']:
    chosen_records.append({k:score[k] for k in ['family_index','cost','unfiltered_cost','allowed_by_predicted_risk','estimated_time_to_goal_s','predicted_goal_progress_m','contact_probability','low_progress_probability','reference_meta']})
  entry['selected_active_initial_scores']=chosen_records
  entry['initial_global_family_available']=any(r['meta'].get('candidate_origin')==f'original_family_{initial_index:02d}' for r in families)
  trial['decisions'].append(entry)
 report['trials'].append(trial)
print(json.dumps(report,indent=2,allow_nan=False))
