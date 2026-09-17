"""Read-only geometry query: construct the frozen native spline, never a vehicle/system."""
from pathlib import Path
import json,sys,hashlib,numpy as np
BASE=Path('/work1/dannegrut/harry/experiments/fdm_diverse_v1_20260909')
SOURCE=BASE/'snapshots/online_v9';sys.path.insert(0,str(SOURCE/'src'))
import pychrono as chrono
from nedm.traverse.fdm_online_driver import reference_knots,native_spline_control_points
from nedm.traverse.terrain import TerrainMap

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def distance(xy,wp):
 d=np.diff(wp,axis=0);norm=(d*d).sum(1);result=[]
 for start in range(0,len(xy),32):
  offsets=xy[start:start+32,None,:]-wp[None,:-1,:];fraction=np.clip((offsets*d).sum(-1)/norm,0.,1.)
  result.extend(np.linalg.norm(offsets-fraction[...,None]*d,axis=-1).min(1).tolist())
 return np.asarray(result)
rows=[]
for task in json.loads((BASE/'online_tasks/protected_test_online_v1.json').read_text())['tasks']:
 if task['scene_id'] not in ['diverse_v1_test_'+s+'_00' for s in ('rolling_hills','rough_mosaic','cross_slopes')] or task['arm'] not in ('selected_rgbd_time','selected_rgbd_energy'):continue
 raw=BASE/'online_protected_test_v1'/task['id'];case_path=Path(task['arguments']['case']);case=json.loads(case_path.read_text());terrain=TerrainMap.from_dir(SOURCE/case['arena'])
 decision=json.loads((raw/'decisions/decision_00000.json').read_text());ref=decision['decision']['route'];knots=reference_knots(ref,terrain)
 _,_,_,curve=native_spline_control_points(chrono,knots)
 with np.load(raw/'rich_intervals.npz') as z,np.load(raw/'trajectory.npz') as trajectory:
  contact=(z['max_asset_contact_max_resultant_n']>1.)|(z['max_chassis_contact_resultant_n']>1.)
  indices=np.flatnonzero(contact);limit=int(indices[0]) if len(indices) else min(640,len(contact)-1)
  actual=trajectory['pose'][:limit+1,:2]
  samples=[]
  for i in range(len(knots)-1):
   for u in np.linspace(0.,1.,101)[:-1]:
    v=curve.Eval(i,float(u));samples.append([v.x,v.y])
  samples.append(knots[-1,:2].tolist());dense=np.asarray(samples)
  native=distance(actual,dense);poly=distance(actual,np.asarray(ref['waypoints']));curve_vs_ref=distance(dense,np.asarray(ref['waypoints']))
  rows.append({'trial_id':task['id'],'measured_end_time_s':limit*.05,'includes_first_contact_onset_pose':bool(len(indices)),
   'native_spline_knots':len(knots),'native_dense_points':len(dense),'native_max_sample_chord_m':float(np.linalg.norm(np.diff(dense,axis=0),axis=1).max()),
   'actual_to_native_spline_xy_max_m':float(native.max()),'actual_to_native_spline_xy_p95_m':float(np.quantile(native,.95)),
   'actual_to_native_spline_xy_at_contact_onset_m':float(native[-1]),'max_offset_time_s':float(native.argmax()*.05),
   'actual_to_command_polyline_xy_max_m':float(poly.max()),'native_spline_to_command_polyline_xy_max_m':float(curve_vs_ref.max()),
   'half_width_m_for_planner_footprint':1.3,'case_sha256':sha(case_path),'decision_sha256':sha(raw/'decisions/decision_00000.json'),
   'native_curve_prior_verified_reconstruction_error_m':decision['driver_update']['native_interpolant_max_error_m']})
print(json.dumps({'scope':'POST-HOC read-only native geometry, no system construction, stepping, training or rendering','driver_source_sha256':sha(SOURCE/'src/nedm/traverse/fdm_online_driver.py'),'chrono_core_sha256':sha('/work1/dannegrut/harry/nrd/chrono-build/bin/pychrono/_core.so'),'sampled_distance_method':'Closest point on100 uniform-parameter chords per native cubic interval; retains maximum chord spacing, XY projected only','trials':rows},indent=2))
