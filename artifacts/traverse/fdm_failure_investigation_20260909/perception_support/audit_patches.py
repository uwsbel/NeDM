#!/usr/bin/env python3
"""Deterministic image/coordinate sampling only; no neural forward or training."""
from pathlib import Path
from types import SimpleNamespace,MethodType
import json,sys,math,hashlib
import numpy as np
import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4]
CODE=ROOT/'artifacts/traverse/fdm_diverse_v1_20260909/snapshots/online_v9'
sys.path[:0]=[str(CODE/'src'),str(CODE/'scripts')]
from nedm.traverse.fdm_diverse_model import RGBDFDMConfig,RGBDFiniteHorizonFDM
from nedm.traverse.terrain import TerrainMap
BASE=Path(__file__).resolve().parent
INPUT=BASE.parent/'decision_audit'
OUT=BASE/'patch_geometry_v1';OUT.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
config=json.loads((ROOT/'artifacts/traverse/fdm_diverse_v1_20260909/runs/full_learning_h60_v3/patch_rgbd_s11/config.json').read_text())['model']
config=RGBDFDMConfig(**config)
# Only coordinate/normalization buffers are needed; no NN module is constructed.
offsets=torch.linspace(4,-4,16);ff,ll=torch.meshgrid(offsets,offsets,indexing='ij')
model=SimpleNamespace(config=config,patch_forward_m=ff.contiguous(),patch_left_m=ll.contiguous(),pixel_mean=torch.tensor([.5,.5,.5,0]).reshape(1,4,1,1),pixel_std=torch.tensor([.5,.5,.5,1]).reshape(1,4,1,1))
model.normalize_pixels=MethodType(RGBDFiniteHorizonFDM.normalize_pixels,model)
def sample(rgbd,nominal,global_features):
 with torch.no_grad():return RGBDFiniteHorizonFDM.candidate_patch_pixels(model,torch.from_numpy(rgbd)[None],torch.from_numpy(nominal),torch.from_numpy(global_features)).numpy()
def world(nominal,g):
 s,c=g[:,6,None],g[:,7,None]
 return np.stack([g[:,4,None]*40+c*nominal[:,:,0]-s*nominal[:,:,1],g[:,5,None]*40+s*nominal[:,:,0]+c*nominal[:,:,1]],axis=-1)
def project(xy,z,size=1024):
 f=size/2/math.tan(math.radians(47)/2)
 return np.stack([(size-1)/2+f*xy[...,0]/(400-z),(size-1)/2-f*xy[...,1]/(400-z)],axis=-1)
def local_to_pixel(point,center,heading):
 delta=point-center;c,s=np.cos(heading),np.sin(heading);forward=c*delta[0]+s*delta[1];left=-s*delta[0]+c*delta[1]
 return np.array([(4-left)*15/8,(4-forward)*15/8])
report={'scope':'Actual frozen RGB-D and causal command feature sampling. No checkpoint weights, neural forward, optimizer or simulator truth enters these pixel crops. BMP heights appear only in labeled audit overlays.','source_sha256':sha(__file__),'model_source_sha256':sha(CODE/'src/nedm/traverse/fdm_diverse_model.py'),'torch_version':torch.__version__,'runtime':'local CPU deterministic coordinate/image transforms, no learned inference','scenes':{},'trials':{}}
scene_data={}
for family in ['rough_mosaic','rolling_hills']:
 scene='diverse_v1_test_'+family+'_00';p=INPUT/'inputs/observations'/scene
 with np.load(p/'observation.npz') as z:obs={k:z[k] for k in ['rgb','depth_m','rgbd','pose']}
 case=json.loads((ROOT/'artifacts/traverse/fdm_diverse_v1_20260909/cases'/f'{scene}.json').read_text());tmap=TerrainMap.from_dir(ROOT/case['arena'])
 camera=json.loads((p/'observation.json').read_text())['camera'];n=obs['depth_m'].shape[0];u,v=np.meshgrid(np.arange(n),np.arange(n));f=n/2/math.tan(math.radians(47)/2);sec=np.sqrt(1+((u-(n-1)/2)/f)**2+((v-(n-1)/2)/f)**2);depth_z=400-obs['depth_m']/sec
 global_rgbd=F.interpolate(torch.from_numpy(obs['rgbd'])[None],size=(128,128),mode='area')[0].numpy()
 mpp=lambda h,w:(h)*2*math.tan(math.radians(47)/2)/w
 report['scenes'][scene]={'observation_sha256':sha(p/'observation.npz'),'raw_ground_m_per_pixel':mpp(400,1024),'candidate_input_ground_m_per_pixel':mpp(400,512),'global_context_ground_m_per_pixel':mpp(400,128),'patch_sample_spacing_m':8/15,'patch_span_m':8,'depth_normalization_m':40,'assets':[]}
 for i,a in enumerate(case['layout']['assets']):
  center=np.array([a['x_m'],a['y_m']]);top=float(tmap.height(*center))+a['dims']['height_m']-.15
  report['scenes'][scene]['assets'].append({'index':i,'edge_m':a['dims']['edge_m'],'height_m':a['dims']['height_m'],'raw_projected_edge_pixels':a['dims']['edge_m']/mpp(400-top,1024),'candidate_input_edge_pixels':a['dims']['edge_m']/mpp(400-top,512),'global_context_edge_pixels':a['dims']['edge_m']/mpp(400-top,128),'flat_ground_depth_channel_contrast':a['dims']['height_m']/40})
 scene_data[scene]=(obs,case,tmap,depth_z,global_rgbd)
 # Resolve actual offending rock separately from earlier terrain/chassis contact.
 asset_index=1 if family=='rough_mosaic' else 0;a=case['layout']['assets'][asset_index];point=np.array([a['x_m'],a['y_m']]);top=float(tmap.height(*point))+a['dims']['height_m']-.15
 fig,ax=plt.subplots(1,4,figsize=(13,3.4),constrained_layout=True)
 panels=[(obs['rgb'],1024,'Actual raw RGB (1024²)','rgb'),(depth_z,1024,'Raw depth → elevation','z'),(obs['rgbd'][:3].transpose(1,2,0),512,'Candidate source RGB (512²)','rgb'),(global_rgbd[:3].transpose(1,2,0),128,'Global encoder RGB (128²)','rgb')]
 for axis,(array,size,title,kind) in zip(ax,panels):
  uv=project(point,top,size);rad=max(3,int(math.ceil(5/mpp(400-top,size))));x,y=np.round(uv).astype(int);crop=array[max(0,y-rad):y+rad+1,max(0,x-rad):x+rad+1]
  if kind=='z':im=axis.imshow(crop,cmap='terrain',interpolation='nearest');fig.colorbar(im,ax=axis,fraction=.046,label='m')
  else:axis.imshow(crop,interpolation='nearest')
  axis.set_title(title,fontsize=10);axis.set_xticks([]);axis.set_yticks([]);axis.plot([rad],[rad],'+',color='red',ms=9)
 fig.suptitle(f'{family}: rock {asset_index}, edge {a["dims"]["edge_m"]:.2f} m; red cross = projected roof center',fontsize=11)
 fig.savefig(OUT/f'{family}_rock_resolution.png',dpi=180);plt.close(fig)

for folder in sorted((INPUT/'trials').glob('*')):
 if not any(f in folder.name for f in ['rough_mosaic','rolling_hills']):continue
 scene=folder.name.split('_selected_')[0];obs,case,tmap,depth_z,global_rgbd=scene_data[scene]
 with np.load(folder/'causal_forecasts.npz') as z:features={k:z[k] for k in ['frames','times_s','selected_commands','selected_nominal_pose','selected_global_features']}
 rawdir=INPUT/'inputs/raw'/folder.name
 with np.load(rawdir/'trajectory.npz') as z:poses=np.vstack([z['pose'],z['terminal_pose']]);states=np.vstack([z['state'],z['terminal_state']])
 with np.load(rawdir/'rich_intervals.npz') as z:chassis=z['max_chassis_contact_resultant_n'];asset=z['max_asset_contact_max_resultant_n']
 first=lambda a:int(np.flatnonzero(a>1)[0])+1 if np.any(a>1) else None
 ei=first(asset);ci=first(chassis);failure=poses[ci,:2];asset_index=1 if 'rough' in scene else 0;rock=case['layout']['assets'][asset_index];rock_xy=np.array([rock['x_m'],rock['y_m']])
 g=features['selected_global_features'];nom=features['selected_nominal_pose'];patches=sample(obs['rgbd'],nom,g);centers=world(nom,g)
 heading=np.arctan2(nom[:,:,2],nom[:,:,3])+np.arctan2(g[:,6],g[:,7])[:,None]
 rows=[]
 for i,t in enumerate(features['times_s']):
  minimum=np.linalg.norm(centers[i]-rock_xy,axis=1);j=int(minimum.argmin());hitpixels=local_to_pixel(rock_xy,centers[i,j],heading[i,j]); actualframes=features['frames'][i]+np.arange(1,61)*4;valid=actualframes<len(poses)
  error=poses[actualframes[valid],:2]-centers[i,valid];c,s=np.cos(heading[i,valid]),np.sin(heading[i,valid]);long=c*error[:,0]+s*error[:,1];lateral=-s*error[:,0]+c*error[:,1]
  row={'anchor_s':float(t),'nearest_rock_patch_index':j,'nearest_rock_patch_nominal_future_s':(j+1)*.2,'rock_distance_to_patch_center_m':float(minimum[j]),'rock_center_patch_pixel_uv':hitpixels.tolist(),'rock_center_inside_patch':bool(((hitpixels>=0)&(hitpixels<=15)).all()),'rock_nearest_patch_elevation_range_m':[float(patches[i,j,3].min()*40),float(patches[i,j,3].max()*40)],'patch_invalid_fraction':float((patches[i,:,3]<-1.5).mean()),'remaining_reference_m':float(g[i,0]),'actual_future_same_time_center_error_median_m':float(np.linalg.norm(error,axis=1).mean()),'actual_future_same_time_max_lateral_m':float(abs(lateral).max()),'actual_future_same_time_max_longitudinal_m':float(abs(long).max()),'actual_future_inside_same_time_patch_fraction':float(((abs(long)<=4)&(abs(lateral)<=4)).mean()),'first_chassis_location_min_distance_to_any_patch_m':float(np.linalg.norm(centers[i]-failure,axis=1).min())}
  if i:
   row['change_from_previous_anchor']={'dt_s':float(t-features['times_s'][i-1]),'remaining_reference_m':float(g[i,0]-g[i-1,0]),'patch_mean_abs_normalized_change':float(abs(patches[i]-patches[i-1]).mean()),'patch_max_abs_normalized_change':float(abs(patches[i]-patches[i-1]).max()),'patch_centers_max_displacement_m':float(np.linalg.norm(centers[i]-centers[i-1],axis=1).max())}
  rows.append(row)
 report['trials'][folder.name]={'feature_sha256':sha(folder/'causal_forecasts.npz'),'raw_trajectory_sha256':sha(rawdir/'trajectory.npz'),'first_chassis_positive_interval_start_s':(ci-1)*.05,'first_chassis_positive_interval_end_s':ci*.05,'first_asset_positive_interval_start_s':(ei-1)*.05,'first_asset_positive_interval_end_s':ei*.05,'actual_chassis_contact_endpoint_xy':failure.tolist(),'actual_asset_contact_endpoint_xy':poses[ei,:2].tolist(),'asset_index_visualized':asset_index,'anchors':rows}
 np.savez_compressed(OUT/(folder.name+'_patches.npz'),times_s=features['times_s'],patches=patches,centers=centers,headings=heading)
 selected_times=[0,.45,.5,2,4] if 'rough' in scene and folder.name.endswith('energy') else ([0,2,4,10,12] if 'rough' in scene else [0,10,11.45,12,20])
 fig,axs=plt.subplots(2,len(selected_times),figsize=(13,5.1),constrained_layout=True)
 for col,t in enumerate(selected_times):
  i=int(abs(features['times_s']-t).argmin());j=rows[i]['nearest_rock_patch_index'];patch=patches[i,j];rgb=np.clip((patch[:3].transpose(1,2,0)+1)/2,0,1);elevation=patch[3]*40
  axs[0,col].imshow(rgb,interpolation='nearest');im=axs[1,col].imshow(elevation,cmap='terrain',interpolation='nearest');uv=rows[i]['rock_center_patch_pixel_uv']
  for row in range(2):axs[row,col].plot(*uv,'+',color='red');axs[row,col].set_xticks([0,7.5,15],labels=['+4','0','−4']);axs[row,col].set_yticks([0,7.5,15],labels=['+4','0','−4'])
  axs[0,col].set_title(f'Anchor {features["times_s"][i]:g} s\nCommand +{(j+1)*.2:g} s',fontsize=10);fig.colorbar(im,ax=axs[1,col],fraction=.05,label='elevation m')
 fig.suptitle(folder.name.replace('diverse_v1_test_','')+'\nExact 8 m × 8 m, 16² command patches; top = forward, left = vehicle-left; red = rock center',fontsize=11)
 fig.savefig(OUT/(folder.name+'_patches.png'),dpi=170);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(12,6),constrained_layout=True)
for ax,(scene,(obs,case,tmap,depth_z,glob)) in zip(axes,scene_data.items()):
 ax.imshow(obs['rgb']);ax.set_xlim(130,900);ax.set_ylim(900,130)
 for trial,color in [('time','tab:blue'),('energy','tab:orange')]:
  name=scene+'_selected_rgbd_'+trial;p=INPUT/'inputs/raw'/name
  with np.load(p/'trajectory.npz') as z:poses=np.vstack([z['pose'],z['terminal_pose']])
  rr=report['trials'][name];end=int(round((rr['first_asset_positive_interval_end_s']+1)/.05));xy=poses[:end,:2];zz=tmap.height(xy[:,0],xy[:,1]);uv=project(xy,zz);ax.plot(uv[:,0],uv[:,1],color=color,lw=1.7,label=trial+' actual')
  for kind,marker in [('chassis','x'),('asset','o')]:
   xy=np.asarray(rr['actual_'+kind+'_contact_endpoint_xy']);z=tmap.height(*xy);u,v=project(xy,z);ax.plot(u,v,marker,color=color,ms=7)
 for i,a in enumerate(case['layout']['assets']):
  xy=np.array([a['x_m'],a['y_m']]);z=tmap.height(*xy)+a['dims']['height_m']-.15;u,v=project(xy,z);ax.text(u+4,v-4,str(i),color='red',fontsize=10)
 ax.set_title(scene.replace('diverse_v1_test_','')+'\n× first chassis-contact endpoint; ○ asset-contact endpoint');ax.legend(loc='lower left');ax.set_xlabel('raw camera column');ax.set_ylabel('raw camera row')
fig.savefig(OUT/'actual_contacts_on_rgb.png',dpi=180);plt.close(fig)
(OUT/'patch_geometry.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'out':str(OUT),'trials':len(report['trials']),'scenes':len(report['scenes'])}))
