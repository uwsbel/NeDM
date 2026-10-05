"""Staged, validation-only training of causal Transformer/contact models."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from nedm.bouncing_ball.collection import atomic_json
from nedm.bouncing_ball.transformer_temporal import TemporalContactNRD,load_temporal,truth_history
from nedm.bouncing_ball.transformer_contact_eval import data_packet,free_metrics,teacher_metrics,launch_gradient_check
from nedm.bouncing_ball.training import quantiles


def main():
    parser=argparse.ArgumentParser()
    for name in ('data','config','output-dir'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('AMD compute nodes only')
    cfg=json.loads(args.config.read_text());torch.set_num_threads(1);torch.manual_seed(cfg['seed'])
    packet=data_packet(args.data,args.device,cfg.get('stride',1))
    states,lengths,splits,valid,contacts=[packet[k] for k in ('states','lengths','splits','valid','contacts')]
    train_mask=valid&(splits==0)[:,None]
    train_pairs=torch.nonzero(train_mask)
    contact_pairs=torch.nonzero(train_mask&(contacts.sum(-1)>.5))
    free_pairs=torch.nonzero(train_mask&(contacts.sum(-1)<.5))
    neighboring=torch.zeros_like(valid)
    for offset in (-3,-2,-1,1,2,3):
        neighboring[contact_pairs[:,0],(contact_pairs[:,1]+offset).clamp(0,valid.shape[1]-1)]=True
    near_pairs=torch.nonzero(neighboring&train_mask&(contacts.sum(-1)<.5))
    val=torch.nonzero(splits==1).flatten()
    reference=states[train_pairs[:,0],train_pairs[:,1]]
    free_delta=states[free_pairs[:,0],free_pairs[:,1]+1]-states[free_pairs[:,0],free_pairs[:,1]]
    norm=dict(mean=reference.mean(0).cpu().tolist(),std=reference.std(0).clamp_min(.01).cpu().tolist(),delta_mean=free_delta.mean(0).cpu().tolist())
    mc={**cfg['model'],'architecture':'transformer_temporal_contact_v2','dt_s':packet['dt']}
    model=TemporalContactNRD(mc,norm).to(args.device)
    rng=torch.Generator(device=args.device).manual_seed(cfg['seed']+1000)
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=False)
    runtime=dict(config=cfg,model_config=mc,training_episodes=int((splits==0).sum()),validation_episodes=len(val),
        test_used_for_selection=False,host=platform.node(),job_id=os.environ['SLURM_JOB_ID'],torch=torch.__version__,
        device=args.device,parameters=sum(p.numel() for p in model.parameters()),data_sha256=packet['index']['model_data_sha256'],
        inference_input='predicted current state and causal predicted-state history only',
        model_source_sha256=hashlib.sha256(Path(__file__).with_name('transformer_temporal.py').read_bytes()).hexdigest(),
        trainer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(output/'run_config.json',runtime)
    start=time.perf_counter();best=float('inf')
    batch=64 if args.smoke else cfg.get('batch',512)
    free_scale=model.state_mean.new_tensor(cfg.get('free_scale',[.0001,.0001,.0002,.0002,.002]))
    response_scale=model.state_mean.new_tensor(cfg.get('response_loss_scale',[.001,.001,.01,.01,.05]))
    rollout_scale=model.state_mean.new_tensor(cfg.get('rollout_scale',[.003,.003,.02,.02,.1]))
    def sample(pool,n):return pool[torch.randint(len(pool),(n,),device=args.device,generator=rng)]
    def balanced(n):return torch.cat((sample(contact_pairs,n//3),sample(near_pairs,n//3),sample(free_pairs,n-2*(n//3))))
    def truth(pairs):return states[pairs[:,0],pairs[:,1]],states[pairs[:,0],pairs[:,1]+1],contacts[pairs[:,0],pairs[:,1]]
    def history(pairs):return truth_history(states,pairs,model.context)
    def labels(c):return (c.sum(-1,keepdim=True)>.5).to(c.dtype) if model.modes==1 else c
    def gate_labels(c):return (c.sum(-1,keepdim=True)>.5).to(c.dtype) if model.gate_modes==1 else c
    def save(path,stage,update,validation=None):
        torch.save(dict(model_config=mc,normalization=norm,model_state_dict=model.state_dict(),runtime=runtime,
                        stage=stage,update=update,validation=validation),path)
    def event(stage,**extra):
        row=dict(stage=stage,elapsed_s=time.perf_counter()-start);row.update(extra)
        with (output/'train_log.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    def check(stage,update,loss):
        nonlocal best
        model.eval();report=free_metrics(model,packet,val[:24] if args.smoke else val)
        if report['selection_score']<best:
            best=report['selection_score'];save(output/'best.pt',stage,update,report)
        save(output/'last.pt',stage,update,report)
        event(stage,update=update,loss=float(loss),p95_trajectory_mm=1000*report['position_rmse_m']['p95'],
              p95_endpoint_mm=1000*report['endpoint_error_m']['p95'],order=report['contact_order_fraction'])
        model.train()
    core_params=list(model.backbone.parameters())+list(model.core_head.parameters())
    if cfg.get('core_checkpoint'):
        source=Path(cfg['core_checkpoint']);pretrained=torch.load(source,map_location=args.device,weights_only=False)
        for key in norm:
            if not torch.allclose(torch.tensor(norm[key]),torch.tensor(pretrained['normalization'][key]),atol=1e-10,rtol=1e-10):
                raise ValueError('Core reuse requires matching training normalization')
        if pretrained['runtime']['data_sha256']!=runtime['data_sha256']:raise ValueError('Core reuse data mismatch')
        for prefix,module in (('backbone.',model.backbone),('core_head.',model.core_head)):
            module.load_state_dict({key[len(prefix):]:value for key,value in pretrained['model_state_dict'].items() if key.startswith(prefix)})
        runtime['reused_core_checkpoint']=str(source)
        runtime['reused_core_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
        atomic_json(output/'run_config.json',runtime);event('reused_trained_transformer_core',sha256=runtime['reused_core_sha256'])
    if model.affine_preserving and not cfg.get('core_checkpoint'):
        # Fit the Transformer readout on observed free-transition deltas. The
        # initialized residual blocks preserve its token input; no flight law.
        pairs=sample(free_pairs,8192 if not args.smoke else 128);s,y,_=truth(pairs)
        with torch.no_grad():
            h=model.backbone((history(pairs)-model.state_mean)/model.state_std)[:,-1]
            design=torch.cat((h,torch.ones_like(h[:,:1])),-1)
            target=(y-s-model.delta_mean)/model.delta_scale
            beta=torch.linalg.lstsq(design.cpu(),target.cpu(),rcond=1e-10,driver='gelsd').solution.to(args.device)
            model.core_head.weight.copy_(beta[:-1].T);model.core_head.bias.copy_(beta[-1])
        event('data_fitted_transformer_readout')
    opt=torch.optim.Adam(core_params,lr=cfg.get('core_lr',3e-4))
    updates=(0 if model.affine_preserving else 5) if args.smoke else cfg.get('core_warmup',15000)
    for update in range(1,updates+1):
        lr=cfg.get('core_lr',3e-4)*(.001+.999*.5*(1+math.cos(math.pi*update/updates)))
        for group in opt.param_groups:group['lr']=lr
        pairs=sample(free_pairs,batch);s,y,_=truth(pairs)
        loss=((s+model.core_delta(s,history(pairs))-y)/free_scale).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(core_params,10.);opt.step()
    if cfg.get('core_lbfgs',0) and not args.smoke:
        pairs=sample(free_pairs,cfg.get('lbfgs_batch',8192));s,y,_=truth(pairs);h=history(pairs)
        opt=torch.optim.LBFGS(core_params,lr=1.,max_iter=15,history_size=30,line_search_fn='strong_wolfe',tolerance_grad=1e-12,tolerance_change=1e-15)
        def closure():
            opt.zero_grad();loss=((s+model.core_delta(s,h)-y)/free_scale).square().mean();loss.backward();return loss
        for _ in range(cfg['core_lbfgs']):opt.step(closure)
        loss=closure()
    event('free_core',updates=updates,loss=float(loss.detach()) if updates or cfg.get('core_lbfgs',0) else None)
    save(output/'free_core.pt','free_core',updates)
    with torch.no_grad():
        pairs=torch.nonzero(valid&(splits==1)[:,None]&(contacts.sum(-1)<.5));errors=[]
        for chosen in pairs.split(2048):
            s,y,_=truth(chosen);errors.append(s+model.core_delta(s,history(chosen))-y)
        error=torch.cat(errors)
        core_report=dict(position_error_m=quantiles(error[:,:2].norm(dim=-1).cpu()),velocity_error_mps=quantiles(error[:,2:4].norm(dim=-1).cpu()))
    atomic_json(output/'free_core_validation.json',core_report)
    event('free_core_validation',p95_position_mm=1000*core_report['position_error_m']['p95'],p95_velocity_mps=core_report['velocity_error_mps']['p95'])
    freeze=cfg.get('freeze_core',True)
    if freeze:
        for parameter in core_params:parameter.requires_grad_(False)

    if model.contact_enabled:
        # A data-learned linear classifier is an initialization/explicit gate
        # ablation. The primary variant retains the learned MLP latent term.
        gp=balanced(128 if args.smoke else cfg.get('gate_fit_batch',60000))
        gs,_,gc=truth(gp);gy=gate_labels(gc)
        with torch.no_grad():
            model.contact_head.weight.zero_();model.contact_head.bias.zero_()
        opt=torch.optim.LBFGS(model.contact_linear.parameters(),lr=1.,max_iter=15,history_size=30,line_search_fn='strong_wolfe',tolerance_grad=1e-10,tolerance_change=1e-14)
        def gate_closure():
            opt.zero_grad();logits=model.contact_linear(model.contact_features(gs))
            loss=F.binary_cross_entropy_with_logits(logits,gy)+cfg.get('gate_l2',1e-10)*model.contact_linear.weight.square().mean()
            loss.backward();return loss
        for _ in range(2 if args.smoke else cfg.get('gate_linear_lbfgs',100)):opt.step(gate_closure)
        gate_params=list(model.contact_encoder.parameters())+list(model.contact_head.parameters())+list(model.contact_linear.parameters())
        opt=torch.optim.Adam(gate_params,lr=cfg.get('gate_lr',1e-4))
        count=5 if args.smoke else cfg.get('gate_warmup',3000)
        for update in range(count):
            s,_,c=truth(balanced(batch));loss=F.binary_cross_entropy_with_logits(model.raw_gate_logits(s),gate_labels(c))
            opt.zero_grad();loss.backward();opt.step()
        if cfg.get('gate_mlp_lbfgs',0) and not args.smoke:
            opt=torch.optim.LBFGS(gate_params,lr=1.,max_iter=15,history_size=30,line_search_fn='strong_wolfe',tolerance_grad=1e-10,tolerance_change=1e-14)
            def mlp_gate_closure():
                opt.zero_grad();loss=F.binary_cross_entropy_with_logits(model.raw_gate_logits(gs),gy)
                loss.backward();return loss
            for _ in range(cfg['gate_mlp_lbfgs']):opt.step(mlp_gate_closure)
        event('gate_fit',loss=float(F.binary_cross_entropy_with_logits(model.raw_gate_logits(gs),gy).detach()),mlp_updates=count)
        for parameter in gate_params:parameter.requires_grad_(False)
        # Separate per-mode contact normalization supplies numerical resolution
        # around contact; it supplies no timing rule or contact at inference.
        with torch.no_grad():
            for kind in range(model.modes):
                pairs=contact_pairs if model.modes==1 else torch.nonzero(train_mask&(contacts[:,:,kind]>.5))
                s,y,_=truth(pairs)
                raw=s[:,2:4] if model.bounce_input_dim==2 else s[:,2:] if model.bounce_input_dim==3 else s
                model.bounce_mean[kind]=raw.mean(0);model.bounce_std[kind]=raw.std(0).clamp_min(.01)
                target=y-s-model.core_delta(s,history(pairs))
                model.response_mean[kind]=target.mean(0);model.response_std[kind]=target.std(0).clamp_min(.001)
                design=torch.cat((model.bounce_features(s,kind),torch.ones_like(s[:,:1])),-1)
                beta=torch.linalg.lstsq(design.cpu(),((target-model.response_mean[kind])/model.response_std[kind]).cpu(),rcond=1e-10,driver='gelsd').solution.to(args.device)
                model.bounce_linear[kind].weight.copy_(beta[:-1].T);model.bounce_linear[kind].bias.copy_(beta[-1])
        bounce_params=list(model.bounce.parameters())+list(model.bounce_linear.parameters())
        opt=torch.optim.Adam(bounce_params,lr=cfg.get('bounce_lr',1e-3))
        count=5 if args.smoke else cfg.get('bounce_warmup',10000)
        for update in range(1,count+1):
            lr=cfg.get('bounce_lr',1e-3)*(.001+.999*.5*(1+math.cos(math.pi*update/count)))
            for group in opt.param_groups:group['lr']=lr
            pairs=sample(contact_pairs,batch);s,y,c=truth(pairs)
            with torch.no_grad():core=model.core_delta(s,history(pairs))
            prediction=s+core+(model.responses(s)*labels(c)[...,None]).sum(-2)
            loss=((prediction-y)/response_scale).square().mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(bounce_params,10.);opt.step()
            if update%cfg.get('evaluate_every',5000)==0 or update==count:check('bounce_adam',update,loss.detach())
        if cfg.get('bounce_lbfgs',0) and not args.smoke:
            pairs=contact_pairs;s,y,c=truth(pairs);h=history(pairs)
            with torch.no_grad():core=model.core_delta(s,h)
            opt=torch.optim.LBFGS(bounce_params,lr=1.,max_iter=15,history_size=40,line_search_fn='strong_wolfe',tolerance_grad=1e-10,tolerance_change=1e-14)
            def bounce_closure():
                opt.zero_grad();prediction=s+core+(model.responses(s)*labels(c)[...,None]).sum(-2)
                loss=((prediction-y)/response_scale).square().mean();loss.backward();return loss
            for update in range(1,cfg['bounce_lbfgs']+1):
                opt.step(bounce_closure)
                if update%25==0 or update==cfg['bounce_lbfgs']:check('bounce_lbfgs',update,bounce_closure().detach())
    else:
        # True Transformer-only control: no learned or analytical switch.
        for p in core_params:p.requires_grad_(True)
        opt=torch.optim.Adam(core_params,lr=cfg.get('bounce_lr',1e-4))
        count=5 if args.smoke else cfg.get('bounce_warmup',10000)
        for update in range(1,count+1):
            pairs=balanced(batch);s,y,_=truth(pairs);prediction=model(s,history(pairs))
            loss=((prediction-y)/response_scale).square().mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(core_params,10.);opt.step()
            if update%cfg.get('evaluate_every',5000)==0 or update==count:check('transformer_only',update,loss.detach())

    model,metadata=load_temporal(output/'best.pt',args.device)
    for p in model.parameters():p.requires_grad_(False)
    trainable=list(model.bounce.parameters())+list(model.bounce_linear.parameters()) if model.contact_enabled else list(model.backbone.parameters())+list(model.core_head.parameters())
    if not freeze and model.contact_enabled:trainable+=list(model.backbone.parameters())+list(model.core_head.parameters())
    for p in trainable:p.requires_grad_(True)
    opt=torch.optim.Adam(trainable,lr=cfg.get('rollout_lr',1e-5))
    count=2 if args.smoke else cfg.get('rollout_updates',200)
    horizon=4 if args.smoke else round(cfg.get('rollout_horizon_s',.5)/model.dt)
    rb=4 if args.smoke else cfg.get('rollout_batch',32)
    for update in range(1,count+1):
        near=sample(contact_pairs,rb//2).clone()
        near[:,1]=(near[:,1]-torch.randint(1,horizon,(len(near),),device=args.device,generator=rng)).clamp_min(0)
        pairs=torch.cat((near,sample(train_pairs,rb-len(near))))
        current=states[pairs[:,0],pairs[:,1]];hist=history(pairs);loss=current.sum()*0.;den=0
        for step in range(horizon):
            j=pairs[:,1]+step+1;active=j<lengths[pairs[:,0]]
            y=states[pairs[:,0],j.clamp_max(states.shape[1]-1)]
            prediction,info=model.details(current,hist)
            loss=loss+((((prediction-y)/rollout_scale).square().mean(-1))*active).sum()
            den+=int(active.sum());current=torch.where(active[:,None],prediction,current)
            hist=model.advance_history(hist,current)
        loss=loss/max(1,den)
        # Retain free-flight fit in every rollout update when unfreezing core.
        if not freeze or not model.contact_enabled:
            chosen=sample(free_pairs,batch);s,y,_=truth(chosen)
            loss=loss+cfg.get('retention_weight',1.)*((s+model.core_delta(s,history(chosen))-y)/free_scale).square().mean()
        opt.zero_grad();loss.backward();gradient=torch.nn.utils.clip_grad_norm_(trainable,1.)
        if not torch.isfinite(gradient):raise RuntimeError('nonfinite rollout gradient')
        opt.step()
        if update%(2 if args.smoke else cfg.get('rollout_evaluate_every',50))==0 or update==count:check('half_second',update,loss.detach())
    model,metadata=load_temporal(output/'best.pt',args.device)
    report=dict(free_rollout=free_metrics(model,packet,val[:24] if args.smoke else val),teacher_forced=teacher_metrics(model,packet,val[:24] if args.smoke else val))
    atomic_json(output/'validation.json',report)
    atomic_json(output/'gradient_check.json',launch_gradient_check(model,[[4.3,-10.2],[5.5,-9.75],[6.8,-9.2]]))
    done=dict(complete=True,smoke=args.smoke,test_used_for_selection=False,elapsed_s=time.perf_counter()-start,
        checkpoint_sha256=hashlib.sha256((output/'best.pt').read_bytes()).hexdigest(),best_stage=metadata['stage'],best_update=metadata['update'],
        validation_score=report['free_rollout']['selection_score'])
    atomic_json(output/'complete.json',done);event('complete',**done)


if __name__=='__main__':main()
