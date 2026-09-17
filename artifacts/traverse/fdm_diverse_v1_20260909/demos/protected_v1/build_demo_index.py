from pathlib import Path
import hashlib,json,sys
from types import SimpleNamespace
import torch
import numpy as np
ROOT=Path(__file__).resolve().parent
CAMPAIGN=ROOT.parents[1]
sys.path.insert(0,str(ROOT/'cross_time/source_online_v9/src'))
from nedm.traverse.fdm_diverse_planner import RGBDReferenceScorer,RGBDCostConfig
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def link(label,path):return f'[{label}]({Path(path).resolve()})'
def event_details(raw,detail,outcome):
 with np.load(raw/'rich_intervals.npz',allow_pickle=False) as interval,np.load(raw/'rich_telemetry.npz',allow_pickle=False) as rich:
  start,end=interval['start_time_s'],interval['end_time_s'];n=len(start)
  def spans(mask):
   edges=np.flatnonzero(np.diff(np.r_[False,np.asarray(mask,bool),False].astype(np.int8)))
   return [[float(start[a]),float(end[b-1])] for a,b in edges.reshape(-1,2)]
  asset=interval['max_asset_contact_max_resultant_n']>1.;chassis=interval['max_chassis_contact_resultant_n']>1.
  union=np.zeros(n,bool);windows=outcome['bounded_blockage_v1_windows']
  for first in windows:union[first:first+40]=True
  pause=rich['command_planner_paused'][:-1] if 'command_planner_paused' in rich.files else None
  result={'contact_threshold_n':1.,'asset_contact_intervals_s':spans(asset),'chassis_contact_intervals_s':spans(chassis),'combined_contact_intervals_s':spans(asset|chassis),
   'bounded_window_count':len(windows),'bounded_window_duration_s':2.,'first_bounded_window_s':None if not windows else [float(start[windows[0]]),float(end[windows[0]+39])],
   'union_of_qualifying_bounded_windows_s':spans(union),'planner_pause_intervals_s':None if pause is None else spans(pause),
   'planner_pause_duration_s':None if pause is None else float(np.dot(pause,interval['duration_s'])),
   'bounded_definition':'41 measured endpoints with XY diameter <=0.25 m over2 s, all40 throttle intervals >0.3, no deliberate parking; onset is confirmed only after the full window.'}
 decision=read(raw/'decisions/decision_00000.json');choice=decision['decision'];index=choice.get('selected_family_index')
 family=next((row for row in choice.get('family_scores',[]) if row['family_index']==index),None)
 selected=choice.get('route');candidate=None if index is None else decision['candidate_references'][index]
 same=selected is not None and candidate is not None and all(np.array_equal(np.asarray(selected[k]),np.asarray(candidate[k])) for k in ('waypoints','stations','headings','speeds'))
 result['launch_abstained']=choice.get('abstained',False);result['selected_reference_matches_saved_base_family']=same
 fields=('contact_probability','low_progress_probability','rollover_probability','contact_risk_supported','rollover_risk_supported','stall_risk_supported')
 result['saved_launch_base_family_risk']=None if family is None else {key:family.get(key) for key in fields}
 result['saved_risk_scope']='Saved launch score for the executed reference' if same else 'Saved launch score for the selected base family before MPPI refinement; not a recorded score for the final refined reference'
 with np.load(detail/'selected_forecast_replay.npz',allow_pickle=False) as prediction:
  probabilities=np.asarray(prediction['event_probability']);maximum=np.max(probabilities,axis=0)
  support=np.asarray([family is not None and family.get(key,False) for key in ('contact_risk_supported','rollover_risk_supported','stall_risk_supported')],bool)
  times=prediction['time_s'];pose=np.asarray(decision['anchor_pose']);delta=prediction['world_xy']-pose[:2];c,s=np.cos(pose[2]),np.sin(pose[2])
  local=np.stack((c*delta[:,0]+s*delta[:,1],-s*delta[:,0]+c*delta[:,1]),-1)
  scorer=object.__new__(RGBDReferenceScorer)
  scorer.anchor_pose=pose;scorer.goal_xy=np.asarray(selected['waypoints'][-1]);scorer.cost_config=RGBDCostConfig(**choice['cost_config'])
  scorer.model=SimpleNamespace(config=SimpleNamespace(dt=float(times[0]-decision['time_s'])),supported_events=torch.from_numpy(support))
  breakdown=scorer.cost_breakdown({'trajectory':local[None],'work':prediction['positive_work_kj'][None,:,None],'event_probability':probabilities[None],'attitude':prediction['attitude_rad'][None]})
  result['recomputed_selected_reference_12s_risk']={name:float(breakdown[key][0]) if support[i] else None for i,(name,key) in enumerate((('contact','contact_probability'),('rollover','rollover_probability'),('bounded_motion','low_progress_probability')))}
  result['recomputed_selected_reference_allowed_by_predicted_risk']=bool(breakdown['allowed_by_predicted_risk'][0])
  result['raw_bounded_output_diagnostic']={'maximum':float(maximum[2]),'time_s':float(times[np.argmax(probabilities[:,2])]),'scope':'Raw head output includes unsupervised t<2 s; not the planner-eligible bounded-motion risk'}
 result['recomputed_risk_scope']='Frozen-checkpoint CPU forecast using only launch RGB-D, causal history and selected reference; existing frozen scorer cost_breakdown applies goal-prefix and bounded-motion >=2 s eligibility masks. Finite12 s forecast, not a full-route guarantee.'
 result['final_mean_gate']='Frozen ReferenceMPPI.optimize evaluates the final weighted mean through the same scorer and retains it only if finite and no worse than the best allowed sample.'
 return result
entries=[]
for name,title,job in [('cross_time','Cross-slope: time + risk',412134),('cross_energy','Cross-slope: time + risk + work',412133),('rolling_failure_1hz','Rolling hills: failure, time + risk + work',412145)]:
 root=ROOT/name;plan=read(root/'replay_plan.json');raw=root/'raw'/plan['video_trial']['id'];report=root/'report';detail=report/'selected_trial'
 parity=read(root/'parity.json');video=read(detail/'video_provenance.json');outcome=read(raw/'outcome.json');camera=read(raw/'video_camera.json')
 assert parity['passed'] and parity['array_count']==241 and len(parity['decisions'])==1
 assert video['available'] and video['frames']==parity['frame_count'] and video['maximum_timestamp_error_s']<=.002
 assert sha(detail/'actual_chrono.mp4')==video['sha256']
 assert outcome['planning_mode']=='plan_once' and outcome['planning_decisions']==1
 assert outcome['positive_work_kj']>=0 and np.isfinite(outcome['positive_work_kj'])
 entry={'directory':name,'title':title,'trial_id':plan['trial']['id'],'native_video_job':job,'video_sample_hz':camera['nominal_fps'],'physical_duration_s':outcome['elapsed_s'],'actual_frames':video['frames'],'parity_array_count':parity['array_count'],'status':outcome['status'],'schema_safe_goal_reached':outcome['schema_safe_goal_reached'],'schema_contact':outcome['schema_contact'],'schema_rollover':outcome['schema_rollover'],'bounded_stall':outcome['bounded_blockage_v1'],'positive_engine_interface_work_kj':outcome['positive_work_kj'],'goal_progress_m':outcome['goal_progress_m'],'max_abs_roll_deg':outcome['max_abs_roll_deg'],'max_abs_pitch_deg':outcome['max_abs_pitch_deg'],'maximum_encoded_timestamp_error_s':video['maximum_timestamp_error_s'],'checkpoint_sha256':read(raw/'online_protocol.json')['model_checkpoint_sha256'],'paths':{'video':str(detail/'actual_chrono.mp4'),'overview':str(detail/'overview.png'),'overview_pdf':str(detail/'overview.pdf'),'telemetry':str(report/'telemetry.png'),'telemetry_pdf':str(report/'telemetry.pdf'),'slip':str(report/'slip_diagnostics.png'),'parity':str(root/'parity.json'),'report':str(detail/'report.json'),'video_provenance':str(detail/'video_provenance.json')},'sha256':{}}
 for key,value in entry['paths'].items():entry['sha256'][key]=sha(value)
 entry['measured_events_and_launch_prediction']=event_details(raw,detail,outcome)
 entries.append(entry)
pair={'work_saving_fraction':1-entries[1]['positive_engine_interface_work_kj']/entries[0]['positive_engine_interface_work_kj'],'time_change_fraction':entries[1]['physical_duration_s']/entries[0]['physical_duration_s']-1,'scope':'One rule-selected illustrative safe pair. No cohort-wide energy-efficiency claim; validation energy gate did not pass.'}
lines=['**Actual Chrono demonstration videos**','',
 'Three complete replays of the frozen held-out evaluation. The RGB-D FDM and MPPI make one decision at launch; Chrono\'s native PID follows the selected full reference. Each overview distinguishes the full reference, the finite 12-second forecast, and the measured traversal.','',
 '| Demonstration | Physical duration | Camera sampling | Measured result | Positive mechanical work | Artifacts |','|---|---:|---:|---|---:|---|']
for e in entries:
 p=e['paths'];hazards=[label for condition,label in ((e['schema_contact'],'measured contact'),(e['bounded_stall'],'bounded motion'),(e['schema_rollover'],'rollover')) if condition]
 result='Safe full goal' if e['schema_safe_goal_reached'] else e['status']+('; '+', '.join(hazards) if hazards else '')
 media=' · '.join([link('Video',p['video']),link('Overview',p['overview']),link('Telemetry',p['telemetry'])])
 lines.append(f"| {e['title']} | {e['physical_duration_s']:.2f} s | {e['video_sample_hz']:g} Hz | {result} | {e['positive_engine_interface_work_kj']:.2f} kJ | {media} |")
lines += ['',f"The safe pair differs by **{pair['work_saving_fraction']:.2%}** in measured positive mechanical work, with {entries[0]['physical_duration_s']:.2f} s versus {entries[1]['physical_duration_s']:.2f} s completion. This small difference is illustrative; the broader energy gate did not pass. Failed-run consumption is reported, not treated as efficiency.",'',
 link('Compare both safe-route telemetry traces',ROOT/'cross_pair_report/telemetry.png')+'. '+link('All 30 frozen evaluation trials',CAMPAIGN/'reports/online_protected_test_v1_cohort_01/report.md')+'.','',
 'Every replay matched **all 241 stored arrays and every planning decision** against its original headless trial; only wall-clock durations were excluded. Video frames are actual Chrono camera pixels registered to the measured pose, state, actions and timestamps. Encoded timestamps passed verification; no invented motion or interpolated frames are present. The terminal still is held for one millisecond.','',
 'The failure video uses **1 Hz** sampling to reduce graphics time; its full 180-second physics run, 20 Hz telemetry and solver-step measurements are unchanged. The original partial 5 Hz render was preserved when job 412135 was cancelled under the documented media amendment. This did not change any evaluation result.','',
 link('Media cadence amendment',ROOT/'media_policy_v2.json')+' · '+link('Preserved partial-render inventory',ROOT/'rolling_failure/partial_before_cancel/inventory.json')+' · '+link('Machine-readable media manifest',ROOT/'complete.json')]
failure=entries[-1]['measured_events_and_launch_prediction']
def first_span(values):return 'unavailable' if values is None else 'none' if not values else f'{values[0][0]:.2f}–{values[0][1]:.2f} s'
def risk_text(values,keys):return ', '.join(f'{label} '+('unavailable' if values.get(key) is None else f'{values[key]:.1%}') for label,key in keys)
saved_risk=risk_text(failure['saved_launch_base_family_risk'] or {},(('contact','contact_probability'),('bounded motion','low_progress_probability'),('rollover','rollover_probability')))
replayed_risk=risk_text(failure['recomputed_selected_reference_12s_risk'],(('contact','contact'),('bounded motion','bounded_motion'),('rollover','rollover')))
lines += ['',f"Failure details: the camera shows the vehicle blocked against a cube obstacle. First measured contact interval **{first_span(failure['combined_contact_intervals_s'])}**; first qualifying bounded-motion window **{first_span([] if failure['first_bounded_window_s'] is None else [failure['first_bounded_window_s']])}**. Planner-pause intervals: **{first_span(failure['planner_pause_intervals_s'])}**. All contact intervals and the union of qualifying bounded-motion windows are retained in the machine-readable manifest; these measurements should not be collapsed into a generic terrain-stall label.",
 f"Launch risk provenance: {failure['saved_risk_scope']}. Saved probabilities: {saved_risk}. The separately recomputed selected-reference **planner-eligible** 12-second probabilities are {replayed_risk}; they do not assess the entire 180-second attempt. The frozen scorer excludes bounded-motion outputs before 2 seconds and after predicted goal arrival. MPPI also re-scores its final weighted mean through this same eligibility gate; it does not execute an unchecked mean."]
(ROOT/'index.md').write_text('\n'.join(lines)+'\n')
manifest={'complete':True,'all_three_native_parity_passed':True,'all_three_transfer_hashes_verified':True,'physics_source_manifest_sha256':'c2ec8b81acd066c75e46cf658c8aa5e67661f92cf3a35a6da97575b03812382a','core_freeze_sha256':'29684ce59cdd68e9b1f7310dcd2e63c55d021f6ff6712b83f99d286722efeb33','selected_checkpoint_step':5000,'planning_mode':'plan_once','passive_demo_camera_is_never_model_input':True,'model_observation':'One frozen pre-drive RGB-D snapshot; separate from the passive demo camera','entries':entries,'illustrative_pair':pair,'index_sha256':sha(ROOT/'index.md'),'builder_sha256':sha(__file__),'media_policy_sha256':sha(ROOT/'media_policy_v2.json'),'visual_qa':'Individual overviews, telemetry plots and actual camera frames visually inspected; encoded timestamps and frame inventories verified.'}
(ROOT/'complete.json').write_text(json.dumps(manifest,indent=2)+'\n')
(ROOT/'media_progress.json').write_text(json.dumps({'status':'complete','entries':entries},indent=2)+'\n')
print(json.dumps({'complete':True,'index':str(ROOT/'index.md'),'manifest_sha256':sha(ROOT/'complete.json'),'entries':[{k:e[k] for k in ['directory','physical_duration_s','actual_frames','video_sample_hz','status']} for e in entries]},indent=2))
