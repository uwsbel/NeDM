"""AMD precision campaign: matched diagnostics and learned event-response NRD.

Geometry/gravity are known; impulse laws come only from training data. The
external state is unchanged. Continuous or physics-grid event timing is an
explicit ablation. No simulator, material constants, launch identity or target
enters inference. Selection uses validation only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from nedm.bouncing_ball.collection import atomic_json
from nedm.bouncing_ball.model import load_model
from nedm.bouncing_ball.training import evaluate, quantiles


class EventNRD(nn.Module):
    def __init__(self, config, normalization):
        super().__init__()
        self.config = config
        self.dt = float(config['dt_s'])
        self.physics_dt = float(config['physics_step_s'])
        self.gravity = float(config['gravity_mps2'])
        self.floor_center_z = float(config['floor_center_z_m'])
        self.wall_center_x = float(config['wall_center_x_m'])
        self.register_buffer('state_mean', torch.tensor(normalization['mean'], dtype=torch.float64))
        self.register_buffer('state_std', torch.tensor(normalization['std'], dtype=torch.float64))
        self.feature_dim = config.get('feature_dim', 3)
        self.register_buffer('feature_mean', torch.zeros(2, self.feature_dim, dtype=torch.float64))
        self.register_buffer('feature_std', torch.ones(2, self.feature_dim, dtype=torch.float64))
        self.register_buffer('response_scale', torch.ones(2, 3, dtype=torch.float64))
        width = config.get('hidden', 64)
        activation = nn.ReLU if config.get('activation', 'relu') == 'relu' else nn.Tanh
        self.linear = nn.ModuleList([nn.Linear(self.feature_dim, 3) for _ in range(2)])
        self.residual = nn.ModuleList([nn.Sequential(nn.Linear(self.feature_dim, width), activation(), nn.Linear(width, width),
                                                    activation(), nn.Linear(width, 3)) for _ in range(2)])
        for net in self.residual:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
        self.double()

    def flight(self, state, duration):
        t = torch.as_tensor(duration, device=state.device, dtype=state.dtype)
        x = state[..., 0]+state[..., 2]*t
        z = state[..., 1]+state[..., 3]*t-.5*self.gravity*t*(t+self.physics_dt)
        broadcast = torch.zeros_like(t)
        return torch.stack((x, z, state[..., 2]+broadcast, state[..., 3]-self.gravity*t, state[..., 4]+broadcast), -1)

    def impact_time(self, state, kind):
        if kind == 0:
            v = state[..., 3]-.5*self.gravity*self.physics_dt
            discriminant = v.square()+2*self.gravity*(state[..., 1]-self.floor_center_z)
            tau = (v+discriminant.clamp_min(1e-12).sqrt())/self.gravity
        else:
            denominator = torch.where(state[..., 2].abs()>1e-9, state[..., 2], torch.ones_like(state[..., 2]))
            tau = (self.wall_center_x-state[..., 0])/denominator
        if self.config.get('timing') == 'quantized':
            # Collision is detected after a penetrating free-flight sample.
            # The impulse affects position from this grid boundary onward.
            tau = torch.ceil(tau/self.physics_dt-1e-9)*self.physics_dt
        else:
            tau = tau+self.config.get('delay_steps', .5)*self.physics_dt
        return tau

    def impulse(self, velocity, kind):
        velocity = velocity.to(dtype=self.feature_mean.dtype)
        feature = (velocity-self.feature_mean[kind])/self.feature_std[kind]
        affine = self.linear[kind](feature)
        if self.config.get('affine_normal') and kind == 0:
            return affine*self.response_scale[kind]
        residual = self.residual[kind](feature)
        if self.config.get('affine_normal'):
            residual = torch.cat((torch.zeros_like(residual[..., :1]), residual[..., 1:]), -1)
        return (affine+residual)*self.response_scale[kind]

    def forward(self, state):
        state = state.to(dtype=self.state_mean.dtype)
        # One interval can contain both events. Remaining duration is carried
        # through the sequential event solve, rather than dropping a collision.
        remaining = torch.full_like(state[..., 0], self.dt)
        current = state
        for kind in (0, 1):
            tau = self.impact_time(current, kind)
            closing = current[..., 3] < 0 if kind == 0 else current[..., 2] > 0
            active = closing & (tau >= -1e-9) & (tau <= remaining+1e-10)
            bounded = tau.clamp_min(0).minimum(remaining)
            at_impact = self.flight(current, bounded)
            dv = self.impulse(at_impact if self.feature_dim == 5 else at_impact[..., 2:], kind)
            changed = torch.cat((at_impact[..., :2], at_impact[..., 2:]+dv), -1)
            current = torch.where(active[..., None], changed, current)
            remaining = torch.where(active, remaining-bounded, remaining)
        return self.flight(current, remaining)

    def rollout(self, initial, steps):
        initial = initial.to(dtype=self.state_mean.dtype)
        state, trajectory = initial, [initial]
        for _ in range(steps):
            state = self(state)
            trajectory.append(state)
        return torch.stack(trajectory, -2)


def prepare(data_root, device, stride=1):
    index = json.loads((data_root/'campaign_index.json').read_text())
    path = data_root/'model_data.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == index['model_data_sha256']
    with np.load(path) as packet:
        states = torch.tensor(packet['states'][:, ::stride], dtype=torch.float64, device=device)
        lengths = torch.tensor((packet['lengths']-1)//stride+1, device=device)
        splits = torch.tensor(packet['splits'], device=device)
        labels = packet['contacts']
        contacts = np.zeros((len(labels), states.shape[1]-1, 2))
        for k in range(contacts.shape[1]):
            contacts[:, k] = labels[:, k*stride:(k+1)*stride].max(1)
        contacts = torch.tensor(contacts, dtype=torch.float64, device=device)
    return index, states, lengths, splits, contacts


def pairs_from_truth(states, contacts, indices, dt, gravity, physics_dt):
    result = []
    for kind in (0, 1):
        pairs = torch.nonzero(contacts[indices, :, kind] > 0)
        s = states[indices[pairs[:, 0]], pairs[:, 1]]
        y = states[indices[pairs[:, 0]], pairs[:, 1]+1]
        free = s.clone()
        free[:, 0] += s[:, 2]*dt
        free[:, 1] += s[:, 3]*dt-.5*gravity*dt*(dt+physics_dt)
        free[:, 3] -= gravity*dt
        dv = y[:, 2:]-free[:, 2:]
        dp = y[:, :2]-free[:, :2]
        remaining = (dp*dv[:, :2]).sum(-1)/dv[:, :2].square().sum(-1).clamp_min(1e-12)
        tau = dt-remaining
        at = s[:, 2:].clone()
        at[:, 1] -= gravity*tau
        at_state = s.clone()
        at_state[:, 0] += s[:, 2]*tau
        at_state[:, 1] += s[:, 3]*tau-.5*gravity*tau*(tau+physics_dt)
        at_state[:, 2:] = at
        residual = (dp-dv[:, :2]*remaining[:, None]).norm(dim=-1)
        result.append({'state':s,'next':y,'velocity':at,'impact_state':at_state,'impulse':dv,'tau':tau,'residual':residual})
    return result


def rollout_loss(prediction, truth, lengths, offsets):
    """Exclude padded or post-horizon targets from each training window."""
    valid = (offsets[:, None]+torch.arange(truth.shape[1],device=truth.device)[None,:]) < lengths[:, None]
    position = ((prediction[..., :2]-truth[..., :2])/.001).square().mean(-1)
    velocity = ((prediction[..., 2:]-truth[..., 2:])/prediction.new_tensor([.01,.01,.1])).square().mean(-1)
    return ((position+.05*velocity)*valid).sum()/valid.sum().clamp_min(1)


@torch.no_grad()
def diagnostic(args, device):
    index, states, lengths, splits, contacts = prepare(args.data, device)
    model, _ = load_model(args.checkpoint, device)
    model.double()
    indices = torch.nonzero(splits == 1).flatten()
    result = evaluate(model, states, lengths, contacts, indices, dict(position_rmse_p95_m=1, endpoint_error_p95_m=1,
        velocity_mae_p95_mps=1, spin_mae_p95_radps=10, contact_time_error_p95_s=1, contact_order_fraction=.99))
    compact = {k:v for k,v in result.items() if k != 'per_episode'}
    events = pairs_from_truth(states, contacts, indices, model.dt, model.gravity, model.physics_dt)
    event_result = []
    for kind, e in enumerate(events):
        predicted = model(e['state'])
        impulse_error = (predicted[:, 2:]-e['next'][:, 2:]).abs()
        v = e['state'][:, 3]-.5*model.gravity*model.physics_dt
        root = (v+(v.square()+2*model.gravity*(e['state'][:,1]-model.floor_center_z)).sqrt())/model.gravity if kind == 0 else (model.wall_center_x-e['state'][:,0])/e['state'][:,2]
        event_result.append({'kind': ['ground','wall'][kind], 'impulse_velocity_mae_mps':quantiles(impulse_error[:,:2].mean(-1).cpu()),
                             'impulse_spin_error_radps':quantiles(impulse_error[:,2].cpu()),
                             'effective_delay_steps':quantiles(((e['tau']-root)/model.physics_dt).cpu()),
                             'single_impulse_position_residual_m':quantiles(e['residual'].cpu()),
                             'quantized_timing_error_s':quantiles((e['tau']-torch.ceil(root/model.physics_dt-1e-9)*model.physics_dt).abs().cpu())})
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output/'diagnostic.json', {'validation_only':True,'baseline':compact,'events':event_result,
                'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()})
    print(json.dumps({'baseline':compact,'events':event_result},indent=2),flush=True)


def train(args, device):
    config = json.loads(args.config.read_text())
    torch.manual_seed(config['seed'])
    stride = config.get('stride', 1)
    index, states, lengths, splits, contacts = prepare(args.data, device, stride)
    train_ids, val_ids = [torch.nonzero(splits == k).flatten() for k in (0,1)]
    physics_dt = index['config']['simulation']['step_s']
    g = index['config']['simulation']['gravity_mps2']
    scene = index['config']['scene']
    dt = index['model_dt_s']*stride
    normalization = {'mean':states[train_ids, 0].mean(0).cpu().tolist(),'std':states[train_ids, 0].std(0).clamp_min(.01).cpu().tolist()}
    model_config = {**config,'architecture':'event_nrd_v3','dt_s':dt,'physics_step_s':physics_dt,'gravity_mps2':g,
                    'floor_center_z_m':scene['ground_z_m']+scene['radius_m'],'wall_center_x_m':scene['wall_front_x_m']-scene['radius_m']}
    model = EventNRD(model_config, normalization).to(device)
    events = pairs_from_truth(states, contacts, train_ids, dt, g, physics_dt)
    val_events = pairs_from_truth(states, contacts, val_ids, dt, g, physics_dt)
    for e in events+val_events:
        e['features'] = e['impact_state'] if model.feature_dim == 5 else e['velocity']
    with torch.no_grad():
        for kind, e in enumerate(events):
            model.feature_mean[kind] = e['features'].mean(0)
            model.feature_std[kind] = e['features'].std(0).clamp_min(1e-6)
            model.response_scale[kind] = e['impulse'].square().mean(0).sqrt().clamp_min(.01)
            if config.get('output_scale'):
                model.response_scale[kind] = torch.tensor(config['output_scale'],device=device,dtype=torch.float64)
            x = (e['features']-model.feature_mean[kind])/model.feature_std[kind]
            x = torch.cat((x, torch.ones(len(x),1,device=device,dtype=x.dtype)), -1)
            target = e['impulse']/model.response_scale[kind]
            beta = torch.linalg.solve(x.T@x+1e-8*torch.eye(model.feature_dim+1,device=device,dtype=x.dtype), x.T@target)
            model.linear[kind].weight.copy_(beta[:-1].T)
            model.linear[kind].bias.copy_(beta[-1])
    if config.get('freeze_affine'):
        for parameter in model.linear.parameters():
            parameter.requires_grad_(False)
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = {'config':config,'model_config':model_config,'data_sha256':index['model_data_sha256'],'job_id':os.environ.get('SLURM_JOB_ID'),
               'host':platform.node(),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'selection':'validation only; no test evaluation','state_dim':5}
    atomic_json(args.output/'run_config.json',runtime)
    limits = dict(position_rmse_p95_m=.003,endpoint_error_p95_m=.006,velocity_mae_p95_mps=.02,spin_mae_p95_radps=.1,
                  contact_time_error_p95_s=.02,contact_order_fraction=1.,max_penetration_p95_m=.002)
    best = float('inf')
    start = time.perf_counter()
    def save_check(update, stage, loss):
        nonlocal best
        model.eval()
        result = evaluate(model, states, lengths, contacts, val_ids[:64] if args.smoke else val_ids, limits)
        with torch.no_grad():
            result['contact_fit'] = [{name:quantiles((model.impulse(e['features'],kind)-e['impulse'])[:,j].abs().cpu())
                                     for j,name in enumerate(('vx_error_mps','vz_error_mps','spin_error_radps'))}
                                    for kind,e in enumerate(val_events)]
        score = result['position_rmse_m']['p95']+result['endpoint_error_m']['p95']*.5+(1-result['contact_order_fraction'])
        checkpoint = {'model_config':model_config,'normalization':normalization,'model_state_dict':model.state_dict(),'runtime':runtime,
                      'update':update,'validation':result}
        if score < best:
            best = score
            torch.save(checkpoint,args.output/'best.pt')
            atomic_json(args.output/'validation.json',result)
        torch.save(checkpoint,args.output/'last.pt')
        log={'update':update,'stage':stage,'loss':float(loss),'position_p95_m':result['position_rmse_m']['p95'],
             'endpoint_p95_m':result['endpoint_error_m']['p95'],'max_endpoint_m':result['endpoint_error_m']['max'],
             'contact_order':result['contact_order_fraction'],'elapsed_s':time.perf_counter()-start}
        log['contact_velocity_p95_mps'] = [[v[k]['p95'] for k in ('vx_error_mps','vz_error_mps')] for v in result['contact_fit']]
        with (args.output/'train_log.jsonl').open('a') as stream: stream.write(json.dumps(log)+'\n')
        print(json.dumps(log),flush=True)
        model.train()
    save_check(0,'linear_initialization',0)
    optimizer = torch.optim.Adam(model.parameters(),lr=config.get('learning_rate',.0003))
    updates = 100 if args.smoke else config.get('updates',10000)
    for update in range(1,updates+1):
        if config.get('cosine_decay'):
            ratio = .5*(1+np.cos(np.pi*update/updates))
            rate = config.get('final_lr',1e-6)+(config.get('learning_rate',.0003)-config.get('final_lr',1e-6))*ratio
            for group in optimizer.param_groups:
                group['lr'] = rate
        optimizer.zero_grad(set_to_none=True)
        loss = torch.zeros((),dtype=torch.float64,device=device)
        for kind, e in enumerate(events):
            selected = torch.randint(len(e['velocity']), (config.get('batch',1024),),device=device)
            predicted = model.impulse(e['features'][selected],kind)
            # Position error after 0.5 s emphasizes the persistent impulse error.
            dv_error = predicted-e['impulse'][selected]
            scale = dv_error.new_tensor([.002,.002,.02])
            loss = loss+(dv_error/scale).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),100.)
        optimizer.step()
        if update%config.get('evaluate_every',1000)==0 or update==updates:
            save_check(update,'contact_half_second_loss',loss.item())
    if config.get('lbfgs_updates',0) and not args.smoke:
        model,_=load_model(args.output/'best.pt',device)
        if config.get('freeze_affine'):
            for parameter in model.linear.parameters(): parameter.requires_grad_(False)
        optimizer=torch.optim.LBFGS([p for p in model.parameters() if p.requires_grad],lr=.5,max_iter=10,
                                    tolerance_grad=1e-10,tolerance_change=1e-14,line_search_fn='strong_wolfe')
        for update in range(1,config['lbfgs_updates']+1):
            def closure():
                optimizer.zero_grad(set_to_none=True)
                loss=torch.zeros((),dtype=torch.float64,device=device)
                for kind,e in enumerate(events):
                    if config.get('affine_normal') and kind==0: continue
                    error=model.impulse(e['features'],kind)-e['impulse']
                    loss=loss+(error/error.new_tensor([.002,.002,.02])).square().mean()
                loss.backward()
                return loss
            loss=optimizer.step(closure)
            if update%config.get('lbfgs_evaluate_every',50)==0 or update==config['lbfgs_updates']:
                save_check(update,'full_batch_contact_lbfgs',float(loss.detach()))
    # Optional free-running half-second / full episode refinement is separate
    # from contact-aligned training and must earn its complexity on validation.
    if config.get('rollout_updates',0) and not args.smoke:
        model,_=load_model(args.output/'best.pt',device)
        optimizer=torch.optim.Adam(model.parameters(),lr=config.get('rollout_lr',1e-5))
        for update in range(1,config['rollout_updates']+1):
            chosen=train_ids[torch.randint(len(train_ids),(config.get('rollout_batch',64),),device=device)]
            horizon=round(config.get('rollout_horizon_s',.5)/dt)
            if config.get('rollout_horizon_s',.5)>=1.7:
                offsets=torch.zeros_like(chosen)
                horizon=round(1.7/dt)
            else:
                # Alternate launch windows and windows spanning wall impact.
                wall=contacts[chosen,:,1].argmax(-1)
                offsets=(wall-2).clamp_min(0) if update%2 else torch.zeros_like(chosen)
            truth=states[chosen[:,None], offsets[:,None]+torch.arange(horizon+1,device=device)[None,:]]
            prediction=model.rollout(truth[:,0],horizon)
            loss=rollout_loss(prediction,truth,lengths[chosen],offsets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            optimizer.step()
            if update%config.get('rollout_evaluate_every',100)==0 or update==config['rollout_updates']:
                save_check(update,'recursive_rollout_loss',loss.item())
    atomic_json(args.output/'complete.json',{'complete':True,'validation_only':True,'elapsed_s':time.perf_counter()-start,'checkpoint_sha256':hashlib.sha256((args.output/'best.pt').read_bytes()).hexdigest()})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--diagnostic',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError('AMD compute GPU required')
    diagnostic(args,'cuda') if args.diagnostic else train(args,'cuda')


if __name__=='__main__': main()
