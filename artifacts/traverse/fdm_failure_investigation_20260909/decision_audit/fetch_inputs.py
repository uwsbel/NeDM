from pathlib import Path
import json,hashlib,sys,tarfile,io
base=Path('/work1/dannegrut/harry/experiments/fdm_diverse_v1_20260909')
tasks=json.loads((base/'online_tasks/protected_test_online_v1.json').read_text())['tasks']
scenes=['diverse_v1_test_'+name+'_00' for name in ('rolling_hills','rough_mosaic','cross_slopes')]
files={}
for task in tasks:
 if task['scene_id'] not in scenes or task['arm'] not in ('selected_rgbd_time','selected_rgbd_energy'):continue
 root=base/'online_protected_test_v1'/task['id']
 for name in ('trajectory.npz','rich_intervals.npz','rich_telemetry.npz','outcome.json','online_protocol.json','simulation_provenance.json','anchor_equality.json','anchor_state.npz','decisions/decision_00000.json'):
  files['raw/'+task['id']+'/'+name]=root/name
 obs=Path(task['arguments']['scene-observation'])
 for name in ('observation.npz','observation.json','simulation_provenance.json'):files['observations/'+task['scene_id']+'/'+name]=obs.parent/name
for scene in scenes:
 for route in sorted((base/'protected_test_cohort_v2/raw'/scene).glob('family_*')):
  for name in ('outcome.json','simulation_provenance.json','anchor_state.npz'):
   p=route/name
   if p.exists():files['fixed_references/'+scene+'/'+route.name+'/'+name]=p
  rp=base/'snapshots/campaign_v2/artifacts/traverse/fdm_diverse_v1_20260909/cases/routes'/scene/(route.name+'.json')
  if rp.exists():files['fixed_references/'+scene+'/'+route.name+'/reference.json']=rp
for name in ('online_tasks/protected_test_online_v1.json','campaign/protected_test_freeze_v1.json','snapshots/online_v9/source_manifest.json'):
 p=base/name
 if p.exists():files['declarations/'+p.name]=p
manifest={'scope':'POST-HOC DIAGNOSTIC; original test evaluation remains frozen','files':[]}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as t:
 for name,p in files.items():
  b=p.read_bytes();manifest['files'].append({'path':name,'remote_source':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)})
  info=tarfile.TarInfo(name);info.size=len(b);t.addfile(info,io.BytesIO(b))
 b=(json.dumps(manifest,indent=2)+'\n').encode();info=tarfile.TarInfo('download_manifest.json');info.size=len(b);t.addfile(info,io.BytesIO(b))
