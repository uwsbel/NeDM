"""POST-HOC DIAGNOSTIC. Frozen weights/source; no optimization or physics."""
from pathlib import Path
import sys,json,hashlib,platform,copy,time
import numpy as np
ROOT=Path(__file__).resolve().parent
CAMPAIGN=ROOT.parents[1]/'fdm_diverse_v1_20260909'
SOURCE=CAMPAIGN/'demos/protected_v1/cross_time/source_online_v9'
CHECKPOINT=CAMPAIGN/'demos/protected_v1/cross_time/inputs/checkpoint.pt'
sys.path.insert(0,str(SOURCE/'src'))
import torch
from nedm.traverse.fdm_diverse_model import load_rgbd_checkpoint
from nedm.traverse.fdm_diverse_planner import RGBDReferenceScorer,RGBDCostConfig
from nedm.traverse.fdm_online_candidates import locate_reference,reference_fingerprint,POLICY
from nedm.traverse.fdm_diverse_data import build_command_features
from nedm.traverse.fdm_diverse_targets import prepare_episode_labels,build_diverse_targets
from nedm.traverse.fdm_data import build_history

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def write(p,d):Path(p).write_text(json.dumps(clean(d),indent=2,allow_nan=False)+'\n')
def peak_time(mask,times):
 i=np.flatnonzero(mask);return None if not len(i) else float(times[i[0]])
def path_tracking(xy,route):
 wp=np.asarray(route['waypoints']);d=np.diff(wp,axis=0);norm=(d*d).sum(1)
 offsets=xy[:,None,:]-wp[None,:-1,:];fraction=np.clip((offsets*d).sum(-1)/norm,0.,1.)
 distance=np.linalg.norm(offsets-fraction[...,None]*d,axis=-1)
 return distance.min(1)
def row_cost(out,i):return {k:v[i] for k,v in out.items()}

def fixed_reference_context(scene,launch,obs):
 support=read(CAMPAIGN/'reports/protected_test_offline_final_v2/fixed_reference_support.json')
 audited={x['episode_id']:x for x in support['routes']}
 result=[]
 for p in sorted((ROOT/'inputs/fixed_references'/scene).glob('family_*')):
  ref=read(p/'reference.json');outcome=read(p/'outcome.json');a=arrays(p/'anchor_state.npz')
  certified=audited[scene+'__'+p.name];assert sha(p/'outcome.json')==certified['source_sha256']['outcome.json']
  diffs={k:float(np.max(np.abs(np.asarray(a[k])-np.asarray(obs[k])))) for k in ('state','pose','history','goal_xy')}
  result.append({'family':p.name,'reference':ref,'reference_sha256':sha(p/'reference.json'),'outcome_sha256':sha(p/'outcome.json'),
    'safe_goal':bool(certified['schema_safe_goal']), 'safe_goal_evidence':'Previously verified protected fixed-reference support including solver-step chassis contact, asset contact, rollover and bounded motion',
    'status':outcome['status'],'elapsed_s':outcome['elapsed_s'],'positive_work_kj':outcome.get('positive_work_kj'),
    'anchor_max_abs_differences':diffs,'anchor_arrays_exact':all(np.array_equal(a[k],obs[k]) for k in diffs)})
 return result

def sibling_match(route,fixed):
 choices=[]
 for x in fixed:
  ref=x['reference']
  if np.asarray(ref['waypoints']).shape!=np.asarray(route['waypoints']).shape:continue
  diffs={k:float(np.max(np.abs(np.asarray(route[k])-np.asarray(ref[k])))) for k in ('waypoints','stations','headings','speeds')}
  choices.append((diffs['waypoints']+diffs['speeds'],x,diffs))
 if not choices:return None
 _,x,diffs=min(choices,key=lambda t:t[0])
 return {k:v for k,v in x.items() if k!='reference'}|{'reference_max_abs_differences':diffs,'reference_arrays_exact':all(v==0 for v in diffs.values()),
   'interpretation':'Prior launch-only fixed-reference outcome. Different collector/controller provenance; approximate geometry matches are not exact same-policy counterfactuals from launch or later states.'}

start=time.time();torch.set_num_threads(4)
assert sha(CHECKPOINT)=='0251cb87ddd25b6dfd8c628680f3470983ca4ca36ef65a05bb966021a4691a20'
model,checkpoint=load_rgbd_checkpoint(CHECKPOINT,device='cpu')
assert int(checkpoint['step'])==5000
results=[]
source_plan=read(CAMPAIGN/'demos/protected_v1/cross_time/replay_plan.json')
source_hashes=dict(source_plan['source_files'])
assert len(source_hashes)==86
for rel,expected in source_hashes.items():assert sha(SOURCE/rel)==expected,(rel,expected)

for raw in sorted((ROOT/'inputs/raw').iterdir()):
 if not raw.is_dir():continue
 trial=raw.name;scene=trial.rsplit('_selected_rgbd_',1)[0]
 outdir=ROOT/'trials'/trial;outdir.mkdir(parents=True,exist_ok=True)
 protocol=read(raw/'online_protocol.json');outcome=read(raw/'outcome.json');decision=read(raw/'decisions/decision_00000.json')
 trajectory=arrays(raw/'trajectory.npz');rich=arrays(raw/'rich_telemetry.npz');interval=arrays(raw/'rich_intervals.npz')
 obs=arrays(ROOT/'inputs/observations'/scene/'observation.npz')
 assert sha(ROOT/'inputs/observations'/scene/'observation.npz')==protocol['scene_observation_sha256']
 assert protocol['model_checkpoint_sha256']==sha(CHECKPOINT)
 assert protocol['planning_mode']=='plan_once' and outcome['planning_decisions']==1 and not decision['paused']
 for rel,expected in protocol['source_sha256'].items():
  if rel.startswith(('src/','scripts/')):
   actual=sha(SOURCE/rel);assert actual==expected,(rel,actual,expected);source_hashes[rel]=actual
 final=decision['decision']['route'];base_index=decision['decision']['selected_family_index'];families=decision['candidate_references'];parent=families[base_index]
 assert all(np.array_equal(np.asarray(final[k]),np.asarray(outcome['final_reference'][k])) for k in ('waypoints','stations','headings','speeds'))
 n=len(trajectory['state']);times=np.arange(n+1)*.05
 poses=np.concatenate([trajectory['pose'],trajectory['terminal_pose'][None]])
 states=np.concatenate([trajectory['state'],trajectory['terminal_state'][None]])
 parked=np.r_[trajectory['parked'],trajectory['terminal_parked']].astype(bool)
 ep=prepare_episode_labels(poses,states,trajectory['action'],parked,interval,goal_xy=obs['goal_xy'],goal_radius_m=outcome['goal_radius_m'])
 contact=(ep['contact']>1.).any(1);asset=ep['contact'][:,0]>1.;chassis=ep['contact'][:,1]>1.
 first_contact=peak_time(contact,times[:-1]);first_asset=peak_time(asset,times[:-1]);first_bound=peak_time(ep['bounded_endpoints'],times)
 end_audit=min(n-1,int(round(((first_contact if first_contact is not None else 30.)+2.)/.05)))
 frames=set(range(0,end_audit+1,40));frames.add(end_audit)
 for event in (first_contact,first_asset,first_bound):
  if event is not None:
   for t in (event-12.,event):
    for delta in (-.05,0.,.05):
     frame=int(round((t+delta)/.05))
     if 0<=frame<=end_audit:frames.add(frame)
 fixed=fixed_reference_context(scene,decision,obs)
 candidates0=[final,*families]
 launch_rows=[];records={};outputs={};features={};labels={}
 def evaluate_frame(frame):
  if frame in records:return records[frame]
  pose=poses[frame]
  history=build_history(trajectory['state'][:frame+1],trajectory['action'][:frame+1],trajectory['pose'][:frame+1],frame)
  if frame==0:
   assert hashlib.sha256(history.tobytes()).hexdigest()==decision['history_sha256']
   assert np.array_equal(pose,np.asarray(decision['anchor_pose']))
   routes=[copy.deepcopy(x) for x in candidates0]
  else:routes=[locate_reference(r,pose,'continued_final' if i==0 else 'launch_sibling_'+str(i-1)) for i,r in enumerate(candidates0)]
  scorer=RGBDReferenceScorer(model,obs['rgbd'],history,pose,obs['goal_xy'],elapsed_s=frame*.05,cost_config=RGBDCostConfig(**protocol['cost_config']),control=protocol['image_intervention'])
  pred=scorer.predict(routes);cost=scorer.cost_breakdown(pred)
  target=build_diverse_targets(ep,frame,horizon=model.config.horizon,output_dt=model.config.dt,attitude=rich['roll_rad'][:,None]*np.array([[1,0]])+rich['pitch_rad'][:,None]*np.array([[0,1]]))
  assert model.config.progress_event_definition=='bounded_motion'
  target['events'][:,2]=target['bounded_motion'][:,0];target['event_mask'][:,2]=target['bounded_motion_mask'][:,0]
  feat=build_command_features(routes[0],pose,station=routes[0].get('meta',{}).get('fdm_station'),elapsed_s=frame*.05,horizon=model.config.horizon,output_dt=model.config.dt)
  valid=target['trajectory_mask'][:,0].astype(bool);xyerr=np.linalg.norm(pred['trajectory'][0,:,:2]-target['trajectory'][:,:2],axis=-1)
  c,s=np.cos(pose[2]),np.sin(pose[2]);local=pred['trajectory'][0,:,:2];world=pose[:2]+np.stack((c*local[:,0]-s*local[:,1],s*local[:,0]+c*local[:,1]),axis=-1)
  distance=np.linalg.norm(world-obs['goal_xy'],axis=1);reached=distance<=scorer.cost_config.goal_radius_m
  first=int(np.argmax(reached)) if reached.any() else len(distance)-1
  prefix=np.arange(len(distance))<=first;stall_eligible=prefix & ((np.arange(len(distance))+1)*model.config.dt>=2.-1e-6)
  if reached.any():stall_eligible &= np.arange(len(distance))<first
  observed_hazard=target['events'][-1].astype(bool)&target['event_mask'][-1].astype(bool)
  r={'frame':frame,'time_s':frame*.05,'history_sha256':hashlib.sha256(history.tobytes()).hexdigest(),'continued_reference_sha256':reference_fingerprint(routes[0]),
    'final_reference_cost':row_cost(cost,0),'parent_reference_cost':row_cost(cost,base_index+1),
    'raw_bounded_peak_probability':float(pred['event_probability'][0,:,2].max()),'raw_bounded_peak_horizon_s':float((pred['event_probability'][0,:,2].argmax()+1)*model.config.dt),
    'predicted_first_goal_arrival_s':None if not reached.any() else float((first+1)*model.config.dt),
    'planner_stall_eligible_horizon_s':(np.flatnonzero(stall_eligible)+1)*model.config.dt,
    'truth_horizon_event':observed_hazard,'truth_horizon_mask':target['event_mask'][-1].astype(bool),
    'hazard_within_12s':bool(observed_hazard.any()),'contact_already_seen':bool(contact[:frame].any()),'bounded_already_confirmed':bool(ep['bounded_endpoints'][:frame+1].any()),
    'first_contact_relative_s':None if first_contact is None else first_contact-frame*.05,
    'first_bounded_confirmation_relative_s':None if first_bound is None else first_bound-frame*.05,
    'continued_reference_tracking_distance_m':float(path_tracking(pose[None,:2],final)[0]),
    'continued_reference_heading_error_deg':float(locate_reference(final,pose,'measurement')['meta']['current_reference_heading_error_deg']),
    'forecast_valid_steps':int(valid.sum()),'forecast_xy_ade_m':float(xyerr[valid].mean()),'forecast_xy_fde_m':float(xyerr[np.flatnonzero(valid)[-1]]),
    'forecast_work_error_kj_at_last_measured':float(pred['work'][0,np.flatnonzero(valid)[-1],0]-target['work'][np.flatnonzero(valid)[-1],0]),
    'actual_future_tracking_max_m':float(path_tracking(poses[frame:min(frame+240,n)+1,:2],final).max()),
    'allowed_launch_siblings':[],'eligible_allowed_launch_siblings':[]}
  for i in range(1,len(routes)):
   located=locate_reference(routes[i],pose,'diagnostic_sibling')
   nearby=located['meta']['current_reference_distance_m']<=POLICY['nearby_distance_m'] and located['meta']['current_reference_heading_error_deg']<=POLICY['nearby_heading_deg']
   if bool(cost['allowed_by_predicted_risk'][i]):
    r['allowed_launch_siblings'].append(i-1)
    if nearby:r['eligible_allowed_launch_siblings'].append(i-1)
  if frame==0:
   for i,route in enumerate(candidates0):
    match=sibling_match(route,fixed)
    launch_rows.append({'role':'final_refined' if i==0 else ('selected_parent' if i-1==base_index else 'sibling'),'family_index':None if i==0 else i-1,'reference_sha256':reference_fingerprint(route),
      'meta':route.get('meta',{}),'cost':row_cost(cost,i),'nearest_previous_fixed_reference':match})
   saved={row['family_index']:row for row in decision['decision']['family_scores']}
   differences=[]
   for idx,row in saved.items():
    for key in ('unfiltered_cost','contact_probability','low_progress_probability','rollover_probability'):
     if key in row and row[key] is not None:differences.append(abs(float(row[key])-float(cost[key][idx+1])))
   r['maximum_saved_base_score_replay_difference']=max(differences,default=0.)
   assert r['maximum_saved_base_score_replay_difference']<1.e-3
  records[frame]=r;outputs[frame]=pred;labels[frame]=target;features[frame]=feat
  return r
 for frame in sorted(frames):evaluate_frame(frame)
 # Refine the first audited allowed->rejected transition to a measured20Hz boundary.
 ordered=sorted(records);first_reject=next((f for f in ordered if not records[f]['final_reference_cost']['allowed_by_predicted_risk']),None)
 if first_reject is not None and first_reject>0:
  left=max(f for f in ordered if f<first_reject);right=first_reject
  while right-left>1:
   mid=(left+right)//2
   if evaluate_frame(mid)['final_reference_cost']['allowed_by_predicted_risk']:left=mid
   else:right=mid
  for f in (left-1,left,right,right+1):
   if 0<=f<=end_audit:evaluate_frame(f)
 ordered=sorted(records)
 for f in ordered:
  records[f]['interpretation']=('already experienced contact' if records[f]['contact_already_seen'] else 'pre-contact')+('; accepted despite measured hazard in next12s' if records[f]['final_reference_cost']['allowed_by_predicted_risk'] and records[f]['hazard_within_12s'] else '')
 np.savez_compressed(outdir/'causal_forecasts.npz',frames=np.array(ordered),times_s=np.array(ordered)*.05,
  **{key:np.stack([outputs[f][key] for f in ordered]) for key in outputs[ordered[0]]},
  **{'target_'+key:np.stack([labels[f][key] for f in ordered]) for key in labels[ordered[0]]},
  **{'selected_'+key:np.stack([features[f][key] for f in ordered]) for key in features[ordered[0]]})
 write(outdir/'anchor_records.json',[records[f] for f in ordered]);write(outdir/'launch_candidates.json',launch_rows)
 fixed_summary=[{k:v for k,v in x.items() if k!='reference'} for x in fixed]
 trial_result={'trial_id':trial,'scene_id':scene,'first_contact_s':first_contact,'first_asset_contact_s':first_asset,'first_bounded_confirmed_s':first_bound,
  'outcome':{k:outcome.get(k) for k in ('status','elapsed_s','goal_progress_m','schema_safe_goal_reached','positive_work_kj','max_abs_roll_deg','max_abs_pitch_deg')},
  'selected_family_index':base_index,'parent_metadata':parent.get('meta',{}),'final_parameters':final.get('meta',{}).get('parameters'),
  'final_max_lateral_vs_parent_m':float(np.linalg.norm(np.asarray(final['waypoints'])-np.asarray(parent['waypoints']),axis=1).max()),
  'final_max_speed_vs_parent_mps':float(np.max(np.abs(np.asarray(final['speeds'])-np.asarray(parent['speeds'])))),
  'final_reference_arrays_equal_parent':all(np.array_equal(np.asarray(final[k]),np.asarray(parent[k])) for k in ('waypoints','stations','headings','speeds')),
  'first_audited_model_rejection_s':next((records[f]['time_s'] for f in ordered if not records[f]['final_reference_cost']['allowed_by_predicted_risk']),None),
  'audited_anchors':len(ordered),'anchors':[records[f] for f in ordered],'launch_candidates':launch_rows,'prior_fixed_reference_outcomes':fixed_summary}
 write(outdir/'summary.json',trial_result);results.append(trial_result)
 print(json.dumps({'trial':trial,'anchors':len(ordered),'first_contact_s':first_contact,'first_model_reject_s':trial_result['first_audited_model_rejection_s'],'launch_cost':clean(records[0]['final_reference_cost'])}),flush=True)
provenance={'scope':'POST-HOC DIAGNOSTIC; no new test selection/training/physics and no original evaluation changes',
 'script_sha256':sha(__file__),'checkpoint_sha256':sha(CHECKPOINT),'checkpoint_step':5000,'source_root':str(SOURCE),'source_sha256':source_hashes,'source_manifest_sha256':source_plan['source_manifest_sha256'],
 'download_manifest_sha256':sha(ROOT/'inputs/download_manifest.json'),'fixed_reference_support_sha256':sha(CAMPAIGN/'reports/protected_test_offline_final_v2/fixed_reference_support.json'),'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'cpu_threads':torch.get_num_threads(),
 'elapsed_wall_s':time.time()-start,'model_supported_events':model.supported_events.cpu().numpy(),'model_config':vars(model.config),
 'truth_boundary':'Only saved pre-drive RGB-D, causal state/action/pose prefixes, supplied goal and commanded references enter predictions. Later raw outcomes/terrain-free telemetry are label/report only.',
 'alternative_boundary':'Saved launch siblings scored conditionally at later measured states are not physically tested counterfactuals. Nearby distance<=2m and heading<=15deg are logged separately. Prior fixed-route outcomes require exact reference+launch+controller parity before causal interpretation.'}
write(ROOT/'provenance.json',provenance);write(ROOT/'audit.json',{'provenance':provenance,'trials':results})
