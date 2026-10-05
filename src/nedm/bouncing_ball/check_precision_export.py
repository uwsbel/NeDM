"""Exercise the public CPU model-loading path against saved GPU predictions."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch
from nedm.bouncing_ball.model import load_model
from nedm.bouncing_ball.collection import atomic_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    model,_=load_model(args.root/'runs/certified_v2/best.pt','cpu')
    launch=torch.tensor([[5.5,-9.75]],requires_grad=True)
    initial=torch.cat((torch.tensor([[0.,1.]]),launch,torch.zeros(1,1)),dim=-1)
    trajectory=model.rollout(initial,round(1.7/model.dt))
    gradient,=torch.autograd.grad(trajectory[:,-1,:2].sum(),launch)
    with np.load(args.root/'runs/launch_precision_v1/trajectories.npz') as packet:
        difference=float(np.abs(packet['before'][0]-trajectory[0].detach().numpy()).max())
    assert difference<1e-9 and torch.isfinite(gradient).all()
    atomic_json(args.root/'reports/public_loading_check.json',{'passed':True,'device':'cpu','input_dtype':'float32',
        'model_dtype':str(next(model.parameters()).dtype),'trajectory_shape':list(trajectory.shape),
        'cpu_gpu_max_state_difference':difference,'finite_launch_gradient':True,'gradient':gradient.detach().tolist(),
        'checkpoint_sha256':hashlib.sha256((args.root/'runs/certified_v2/best.pt').read_bytes()).hexdigest()})
    print('public CPU loading/gradient check passed; CPU/GPU max difference',difference,flush=True)


if __name__=='__main__':main()
