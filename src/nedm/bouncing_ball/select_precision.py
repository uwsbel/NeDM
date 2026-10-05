"""Select on the common 100 Hz validation grid; fresh test remains sealed."""
import argparse
import hashlib
import json
import os
from datetime import datetime,timezone
from pathlib import Path
import torch
from nedm.bouncing_ball.collection import atomic_json
from nedm.bouncing_ball.model import load_model
from nedm.bouncing_ball.precision import prepare
from nedm.bouncing_ball.training import evaluate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('AMD GPU required')
    torch.set_num_threads(4)
    output=args.root/'selection.json'
    if output.exists():raise FileExistsError(output)
    index,states,lengths,splits,contacts=prepare(args.data,'cuda')
    ids=torch.nonzero(splits==1).flatten()
    limits=dict(position_rmse_p95_m=.003,endpoint_error_p95_m=.006,velocity_mae_p95_mps=.02,
                spin_mae_p95_radps=.1,contact_time_error_p95_s=.02,contact_order_fraction=1,max_penetration_p95_m=.005)
    candidates=[]
    for complete in sorted((args.root/'runs').glob('precision*/complete.json')):
        path=complete.parent/'best.pt'
        model,meta=load_model(path,'cuda')
        native_dt=model.dt
        model.dt=index['model_dt_s']
        measured=evaluate(model,states,lengths,contacts,ids,limits)
        compact={k:v for k,v in measured.items() if k!='per_episode'}
        eligible=(measured['finite_fraction']==1 and measured['contact_order_fraction']==1
                  and measured['position_rmse_m']['max']<.01 and measured['endpoint_error_m']['max']<.02)
        score=measured['position_rmse_m']['p95']+.5*measured['endpoint_error_m']['p95']
        entry=dict(name=complete.parent.name,checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   native_dt_s=native_dt,common_dt_s=model.dt,eligible=eligible,score=score,validation=compact,
                   config=meta['model_config'])
        candidates.append(entry)
        print(json.dumps({'run':entry['name'],'trajectory_p95_mm':measured['position_rmse_m']['p95']*1000,
                          'endpoint_p95_mm':measured['endpoint_error_m']['p95']*1000,'eligible':eligible}),flush=True)
    selected=min((c for c in candidates if c['eligible']),key=lambda c:c['score'])
    result=dict(selection_split='validation',fresh_test_read=False,selection_time_utc=datetime.now(timezone.utc).isoformat(),
                selection_score='position RMSE p95 + 0.5 endpoint distance p95, on matched 100 Hz validation grid',
                checkpoint=selected['checkpoint'],checkpoint_sha256=selected['checkpoint_sha256'],selected_name=selected['name'],
                candidates=candidates,job_id=os.environ.get('SLURM_JOB_ID'),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(output,result)
    print(json.dumps({'selected':result['selected_name'],'checkpoint_sha256':result['checkpoint_sha256']}),flush=True)


if __name__=='__main__':main()
