#!/usr/bin/env python3
"""Replay CPU sample RNG and audit label exposure; never create an optimizer."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
C=Path('/work1/dannegrut/harry/experiments/fdm_diverse_v1_20260909');B=Path('/work1/dannegrut/harry/experiments/fdm_failure_investigation_20260909/perception_support');E=json.loads((B/'exposure_v1/exposure.json').read_text());report={'scope':'Exact CPU sampler replay, verified against completed training draw digest. No optimizer/model forward. Counts are repeated training presentations, not independent trials.','torch_version':torch.__version__,'packs':{}}
for kind,pack,run,batch in [('narrow',Path('/work1/dannegrut/harry/experiments/traverse_mppi_20260908/data/fdm_rgbd_focused_pack_v2'),Path('/work1/dannegrut/harry/experiments/traverse_mppi_20260908/runs/rgbd_focused_pilot_v2/patch_rgbd_s11'),64),('diverse',C/'packs/full_pair_v1/h60',C/'runs/full_learning_h60_v3/patch_rgbd_s11',32)]:
 with np.load(pack/'train.npz') as z:d={k:z[k] for k in ['anchor','episode_index','events','event_mask','bounded_motion','bounded_motion_mask']}
 n=len(d['anchor']);onsets=E['packs'][kind]['splits']['train']['episode_onsets'];prior={key:np.array([onsets[e][key] is not None and a>=onsets[e][key] for e,a in zip(d['episode_index'],d['anchor'])]) for key in ['contact','rollover','bounded','sustained']};anyprior=np.logical_or.reduce(list(prior.values()));groups={'all':np.ones(n,bool),'anchor0':d['anchor']==0,'anchor_lt2s':d['anchor']<40,'pre_first_any_failure':~anyprior,'prior_any_failure':anyprior}
 rng=torch.Generator(device='cpu').manual_seed(11+1729);digest='';draws=[]
 for step in range(5000):
  ix=torch.randint(n,(batch,),generator=rng).numpy();digest=hashlib.sha256(digest.encode()+ix.tobytes()).hexdigest();draws.append(ix)
 draws=np.stack(draws);status=json.loads((run/'status.json').read_text());assert digest==status['draw_digest'],(kind,digest,status['draw_digest']);freq=np.bincount(draws.ravel(),minlength=n);norm=json.loads((run/'normalization.json').read_text());events={'contact':(d['events'][:,:,0],d['event_mask'][:,:,0],norm['event_pos_weight'][0]),'rollover':(d['events'][:,:,1],d['event_mask'][:,:,1],norm['event_pos_weight'][1]),'bounded':(d['bounded_motion'][:,:,0],d['bounded_motion_mask'][:,:,0],norm['event_pos_weight'][2])}
 result={'draw_digest':digest,'completed_training_draw_digest_matches':True,'total_draws':int(freq.sum()),'unique_sampled_windows':int((freq>0).sum()),'never_sampled_windows':int((freq==0).sum()),'draw_frequency_quantiles':np.quantile(freq,[0,.25,.5,.75,.95,1]).tolist(),'groups':{}}
 for name,g in groups.items():
  record={'actual_row_draws':int(freq[g].sum()),'positive_exposure':{}}
  for event,(labels,masks,pw) in events.items():
   positive=(labels>0)&(masks>0);valid=masks>0;last=g&positive[:,-1];anyh=g&positive.any(1)
   record['positive_exposure'][event]={'at_last_horizon_row_draws':int(freq[last].sum()),'at_any_horizon_row_draws':int(freq[anyh].sum()),'positive_horizon_element_draws':int((freq[g,None]*positive[g]).sum()),'valid_horizon_element_draws':int((freq[g,None]*valid[g]).sum()),'batches_with_last_horizon_positive':int(np.any(last[draws],axis=1).sum()),'unique_positive_windows_drawn':int(((freq>0)&last).sum()),'event_pos_weight':pw}
  result['groups'][name]=record
 report['packs'][kind]=result
 print(json.dumps({'kind':kind,'digest_verified':True,'anchor0':result['groups']['anchor0'],'pre_failure':result['groups']['pre_first_any_failure']}),flush=True)
out=B/'replayed_draws_v1';out.mkdir(exist_ok=False);(out/'replayed_draws.json').write_text(json.dumps(report,indent=2)+'\n')
