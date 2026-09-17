import json, hashlib, sys
from pathlib import Path
import numpy as np
root=Path('/home/harry/NeDM-traverse_mppi'); base=root/'artifacts/traverse/fdm_diverse_v1_20260909'
snapshot=base/'snapshots/online_v5'; folder=base/'online_pilot_smoke_v5'
sys.path.insert(0,str(root/'scripts')); sys.path.insert(0,str(snapshot/'src'))
from check_traverse_fdm_rich_telemetry import verify
from nedm.traverse.fdm_data import build_history
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:{k:v for k,v in np.load(p).items()}
runs={}
for name in ('rolling_headless','rolling_video','rolling_longer','mixed_energy'):
 p=folder/name; a=load(p/'trajectory.npz'); r=load(p/'rich_telemetry.npz'); q=load(p/'rich_intervals.npz')
 result=json.loads((p/'outcome.json').read_text()); protocol=json.loads((p/'online_protocol.json').read_text())
 verified=verify(p,trajectory=p/'trajectory.npz',require_solver_steps=True)
 count=0
 for f in sorted((p/'decisions').glob('decision_*.json')):
  d=json.loads(f.read_text()); i=d['frame']; h=build_history(a['state'][:i+1],a['action'][:i+1],a['pose'][:i+1],i)
  assert hashlib.sha256(h.tobytes()).hexdigest()==d['history_sha256']
  assert np.array_equal(d['anchor_pose'],a['pose'][i])
  assert np.array_equal(np.asarray(d['anchor_state17'],np.float32),a['state'][i])
  count+=1
 assert result['frames']==len(a['state']) and np.isclose(result['elapsed_s'],r['time_s'][-1])
 assert np.array_equal(a['terminal_pose'],[r[k][-1] for k in ('pos_world_x_m','pos_world_y_m','yaw_rad')])
 assert np.isclose(result['positive_work_kj'],q['engine_interface_positive_work_kj'].sum())
 contact=np.maximum(q['max_chassis_contact_resultant_n'],q['max_asset_contact_max_resultant_n'])
 assert result['schema_contact']==bool((contact>1.).any()) and np.isclose(result['max_schema_contact_n'],contact.max())
 assert json.loads((p/'anchor_equality.json').read_text())['matched']
 assert sha(base/'pilot_w8/observations'/result['case_id']/'observation.npz')==protocol['scene_observation_sha256']
 assert all(sha(snapshot/f)==s for f,s in protocol['source_sha256'].items())
 runs[name]={'rich':verified,'all_decision_history_and_anchors_verified':count,'endpoints_and_outcomes_complete':True,
  'observation_and_source_hashes_verified':True,'process_result':json.loads((folder/(name+'.result.json')).read_text()),
  **{k:result[k] for k in ('frames','elapsed_s','goal_progress_m','positive_work_kj','status')}}
a,b=(load(folder/n/'trajectory.npz') for n in ('rolling_headless','rolling_video'))
parity={k:bool(np.array_equal(a[k],b[k],equal_nan=True)) for k in a if np.issubdtype(a[k].dtype,np.number) or a[k].dtype==np.bool_}
assert all(parity.values())
rows=json.loads((folder/'rolling_video/frame_metadata.json').read_text())['frames']
ts=np.array([r['simulation_time_s'] for r in rows]); rec=np.array([r['recording_time_s'] for r in rows])
assert np.all(np.diff(ts)>0.) and np.allclose(ts-ts[0],rec,atol=1e-8)
assert rows[-1]['terminal'] and rec[-1]==runs['rolling_video']['elapsed_s']
poses=np.vstack([b['pose'],b['terminal_pose']])
for row in rows:
 assert np.array_equal(row['actual_pose'],poses[row['telemetry_frame']]) and (folder/'rolling_video'/row['file']).is_file()
report={'passed_artifact_validation':True,'job_id':'412081','job_state':'FAILED',
 'failure_scope':'Final stdout JSON ndarray serialization failed after complete artifacts. This validation does not relabel the failed job.',
 'runs':runs,'headless_video_numeric_arrays_exact':parity,
 'headless_video_trajectory_file_sha_equal':sha(folder/'rolling_headless/trajectory.npz')==sha(folder/'rolling_video/trajectory.npz'),
 'video':{'actual_frame_poses_and_timestamps_verified':True,'frame_count':len(rows),'duration_s':float(rec[-1]),'terminal_frame_present':True},
 'scope':'Development implementation smoke, no generalization claim'}
Path(__file__).with_name('validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':True,'runs':{k:{f:v[f] for f in ('frames','elapsed_s','goal_progress_m','status')} for k,v in runs.items()},'video':report['video']},indent=2))
