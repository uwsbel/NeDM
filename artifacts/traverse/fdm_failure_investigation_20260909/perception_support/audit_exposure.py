#!/usr/bin/env python3
"""Read-only exposure audit; no optimizer or test-based model selection."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,hashlib,json,sys,time
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def onset(ep):
 from nedm.traverse.fdm_diverse_targets import prepare_episode_labels
 p=Path(ep['source'])
 with np.load(p/'trajectory.npz') as z: raw={k:z[k] for k in ['pose','terminal_pose','state','terminal_state','action','parked','terminal_parked']}
 with np.load(p/'rich_intervals.npz') as z: rich={k:z[k] for k in ['max_asset_contact_max_resultant_n','max_chassis_contact_resultant_n','max_abs_roll_rad','max_abs_pitch_rad','engine_interface_positive_work_kj']}
 meta=read(p/'collection_meta.json');out=read(p/'outcome.json')
 labels=prepare_episode_labels(np.vstack([raw['pose'],raw['terminal_pose']]),np.vstack([raw['state'],raw['terminal_state']]),raw['action'],np.r_[raw['parked'],raw['terminal_parked']].astype(bool),rich,goal_xy=np.asarray(meta['route']['waypoints'][-1]),goal_radius_m=out['goal_radius_m'])
 first=lambda a:int(np.flatnonzero(a)[0]) if np.any(a) else None
 return {'contact':first(np.r_[False,np.any(labels['contact']>1,axis=1)]),'rollover':first(np.r_[False,np.any(labels['attitude_peaks']>np.pi/3,axis=1)]),'bounded':first(labels['bounded_endpoints']),'sustained':first(labels['sustained_endpoints'])}
def summarize(data,records,onsets,draws,batch):
 n=len(data['anchor']);prior={name:np.array([onsets[e][name] is not None and a>=onsets[e][name] for e,a in zip(data['episode_index'],data['anchor'])]) for name in ['contact','rollover','bounded','sustained']}
 groups={'all':np.ones(n,bool),'anchor0':data['anchor']==0,'anchor_lt2s':data['anchor']<40,'pre_first_any_failure':~np.logical_or.reduce(list(prior.values())),'prior_any_failure':np.logical_or.reduce(list(prior.values()))}
 groups.update({'before_first_'+k:~v for k,v in prior.items()})
 arrays={'contact':(data['events'][:,:,0],data['event_mask'][:,:,0]),'rollover':(data['events'][:,:,1],data['event_mask'][:,:,1]),'bounded':(data['bounded_motion'][:,:,0],data['bounded_motion_mask'][:,:,0])}
 out={'windows':n,'episodes':len(records),'scenes':len(set(e['scene_id'] for e in records)),'optimizer_draws':draws,'expected_draws_per_window':draws/n,'expected_unseen_window_fraction_under_uniform_sampling':float((1-1/n)**draws),'groups':{},'per_family':{},'episode_onsets':[{'episode':e['id'],**o} for e,o in zip(records,onsets)]}
 for name,g in groups.items():
  item={'windows':int(g.sum()),'fraction':float(g.mean()),'expected_optimizer_draws':float(draws*g.mean()),'scenes':len(set(records[e]['scene_id'] for e in data['episode_index'][g])),'horizons':{}}
  for sec in (4,8,12):
   ix=sec*5-1
   if ix>=data['events'].shape[1]:continue
   horizon={}
   for event,(label,mask) in arrays.items():
    valid=g&(mask[:,ix]>0);positive=valid&(label[:,ix]>0)
    episode_ids=np.unique(data['episode_index'][positive]);fraction=positive.mean()
    horizon[event]={'known':int(valid.sum()),'positive':int(positive.sum()),'unknown':int((g&~valid).sum()),'positive_episodes':len(episode_ids),'positive_scenes':len(set(records[e]['scene_id'] for e in episode_ids)),'expected_positive_draws':float(draws*fraction),'expected_positive_per_batch':float(batch*fraction),'probability_batch_contains_no_such_positive':float((1-fraction)**batch)}
   item['horizons'][str(sec)]=horizon
  out['groups'][name]=item
 for family in sorted(set(e.get('family',e['scene_id'].split('_')[1]) for e in records)):
  selected=np.array([records[e].get('family',records[e]['scene_id'].split('_')[1])==family for e in data['episode_index']])
  out['per_family'][family]={'windows':int(selected.sum()),'anchor0':int((selected&groups['anchor0']).sum()),'pre_first_any_failure':int((selected&groups['pre_first_any_failure']).sum()),'future_positive_before_first_failure_at_final_horizon':{event:int((selected&groups['pre_first_any_failure']&(mask[:,-1]>0)&(label[:,-1]>0)).sum()) for event,(label,mask) in arrays.items()}}
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--old-pack',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 code=a.campaign/'snapshots/reporting_v1';sys.path[:0]=[str(code/'src'),str(code/'scripts')]
 report={'script_sha256':sha(__file__),'scope':'All train/validation rows, exact observed-prefix event onsets; test excluded. Expected exposure from declared uniform sampling, not asserted actual random-draw counts.','packs':{}}
 for kind,pack,draws,batch in [('narrow',a.old_pack,5000*64,64),('diverse',a.campaign/'packs/full_pair_v1/h60',5000*32,32)]:
  m=read(pack/'manifest.json');entry={'manifest_sha256':sha(pack/'manifest.json'),'splits':{}}
  for split in ['train','val']:
   records=read(pack/(split+'_episodes.json'))
   with np.load(pack/(split+'.npz')) as f:data={k:f[k] for k in ['events','event_mask','bounded_motion','bounded_motion_mask','anchor','episode_index']}
   if kind=='narrow':
    onsets=[{'contact':e['first_contact_frame'],'bounded':e['first_observed_bounded_motion_frame'],'sustained':e['first_observed_sustained_stall_frame'],'rollover':None} for e in records]
   else:
    with ProcessPoolExecutor(max_workers=8) as pool:onsets=list(pool.map(onset,records))
   entry['splits'][split]=summarize(data,records,onsets,draws if split=='train' else 0,batch)
   print(json.dumps({'pack':kind,'split':split,'windows':len(data['anchor'])}),flush=True)
  report['packs'][kind]=entry
 report['old_onset_caveat']='Narrow onset metadata reproduces its original report convention; rollover onset was not separately stored and is absent from narrow prefix stratification. Its contact target is asset-only; diverse includes chassis and solver-rate attitude.'
 (a.out/'exposure.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
