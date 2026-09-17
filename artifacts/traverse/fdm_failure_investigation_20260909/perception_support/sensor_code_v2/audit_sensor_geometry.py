#!/usr/bin/env python3
"""Read-only terrain queries for already-unsealed frozen observations."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,hashlib
import numpy as np
campaign=Path('/work1/dannegrut/harry/experiments/fdm_diverse_v1_20260909');root=campaign/'snapshots/online_v9';sys.path[:0]=[str(root/'src'),str(root/'scripts')]
from check_traverse_fdm_diverse_geometry import construct_terrain,check_observation
from nedm.traverse.terrain import TerrainMap
from nedm.traverse.camera import CameraModel
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
freeze=campaign/'protected_test_freeze_v1.json';assert sha(freeze)=='29684ce59cdd68e9b1f7310dcd2e63c55d021f6ff6712b83f99d286722efeb33' and json.loads(freeze.read_text())['test_unseal_authorized'] is True
base=Path('/work1/dannegrut/harry/experiments/fdm_failure_investigation_20260909/perception_support');out=base/'sensor_geometry_v2';out.mkdir(parents=True,exist_ok=False);contacts=json.loads((base/'contact_points.json').read_text())
report={'script_sha256':sha(__file__),'geometry_helper_sha256':sha(Path(__file__).parent/'check_traverse_fdm_diverse_geometry.py'),'scope':'Read-only existing fixed test observations after explicit unseal; no new simulation rollout, model forward, fitting or renderer changes.','scenes':{}}
for family in ['rough_mosaic','rolling_hills']:
 scene='diverse_v1_test_'+family+'_00';case_path=campaign/'snapshots/campaign_v2/artifacts/traverse/fdm_diverse_v1_20260909/cases'/f'{scene}.json';case=json.loads(case_path.read_text());arena=campaign/'snapshots/campaign_v2'/case['arena'];tmap=TerrainMap.from_dir(arena);system,terrain,query=construct_terrain(arena,tmap)
 observation=campaign/'protected_test_cohort_v2/observations'/scene;args=SimpleNamespace(camera_json=None,depth_samples=2048,seed=20260909)
 audit,samples=check_observation(case,tmap,query,observation,args);arrays=dict(samples);profile=[]
 for i,contact in enumerate([c for c in contacts if c['scene_id']==scene]):
  pose=np.asarray(contact['pose']);xs,ys=np.meshgrid(np.linspace(-10,10,81),np.linspace(-10,10,81));xy=np.c_[xs.ravel()+pose[0],ys.ravel()+pose[1]];zz=query(xy);arrays[f'contact_{i}_xy']=xy;arrays[f'contact_{i}_chrono_z']=zz;profile.append({**contact,'array_prefix':f'contact_{i}','grid_shape':[81,81]})
 with np.load(observation/'observation.npz') as f:depth=f['depth_m'];rgb=f['rgb']
 camera=json.loads((observation/'observation.json').read_text())['camera'];cam=CameraModel(width=camera['width'],height=camera['height'],hfov_rad=camera['hfov_rad'],cam_height_m=camera['cam_height_m']);wx,wy,wz=cam.depth_to_world(depth,convention='ray',ray_scale=1.)
 assets=[]
 for i,a in enumerate(case['layout']['assets']):
  dx,dy=wx-a['x_m'],wy-a['y_m'];c,s=np.cos(a['yaw_rad']),np.sin(a['yaw_rad']);u,v=c*dx+s*dy,-s*dx+c*dy;half=a['dims']['edge_m']/2
  top=float(tmap.height(a['x_m'],a['y_m']))+a['dims']['height_m']-.15
  mask=(abs(u)<=half+.05)&(abs(v)<=half+.05)&(abs(wz-top)<.08)
  expanded=(abs(dx)<half+1)&(abs(dy)<half+1)&np.isfinite(wz)&(depth>0)&(depth<600)
  rr,cc=np.nonzero(expanded);terrain_z=query(np.c_[wx[expanded],wy[expanded]])
  arrays[f'asset_{i}_raw_rc']=np.c_[rr,cc];arrays[f'asset_{i}_hit_xyz']=np.c_[wx[expanded],wy[expanded],wz[expanded]];arrays[f'asset_{i}_chrono_z']=terrain_z
  assets.append({'index':i,**a,'top_elevation_m':top,'raw_top_face_depth_hits':int(mask.sum()),'raw_top_face_z_max_error_m':float(abs(wz[mask]-top).max()) if mask.any() else None,'raw_top_face_rgb_mean':rgb[mask].mean(0).tolist() if mask.any() else None,'raw_projected_edge_pixels':a['dims']['edge_m']*camera['width']/2/np.tan(camera['hfov_rad']/2)/(camera['cam_height_m']-top)})
 report['scenes'][scene]={'camera_audit':audit,'contact_profiles':profile,'assets':assets,'case_sha256':sha(case_path),'physics_source_manifest_sha256':sha(root/'source_manifest.json')}
 np.savez_compressed(out/(scene+'_samples.npz'),**arrays)
 print(json.dumps({'scene':scene,'depth_error':audit['depth_elevation_vs_chrono_m']['all'],'roof_pixel_error':audit['rgb_marker_alignment'].get('pixel_error')}),flush=True)
(out/'sensor_geometry.json').write_text(json.dumps(report,indent=2)+'\n')
