"""Train faithful Transformer contact residuals on existing Chrono transitions."""
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
from nedm.bouncing_ball.transformer_contact import TransformerContactNRD,load_transformer_contact
from nedm.bouncing_ball.transformer_contact_eval import data_packet,free_metrics,teacher_metrics,launch_gradient_check
from nedm.bouncing_ball.training import quantiles


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True);p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):raise RuntimeError("Run all Torch work on AMD compute nodes")
    cfg=json.loads(args.config.read_text());torch.set_num_threads(cfg.get("threads",1))
    torch.manual_seed(cfg["seed"])
    data=data_packet(args.data,args.device,cfg.get("stride",1))
    states,lengths,splits,valid,contacts=[data[k] for k in ("states","lengths","splits","valid","contacts")]
    train=torch.nonzero(splits==0).flatten();val=torch.nonzero(splits==1).flatten()
    train_mask=valid&(splits==0)[:,None]
    all_pairs=torch.nonzero(train_mask)
    contact_pairs=torch.nonzero(train_mask&(contacts.sum(-1)>.5))
    free_pairs=torch.nonzero(train_mask&(contacts.sum(-1)<.5))
    neighboring=torch.zeros_like(valid)
    for delta in (-3,-2,-1,1,2,3):
        j=(contact_pairs[:,1]+delta).clamp(0,valid.shape[1]-1)
        neighboring[contact_pairs[:,0],j]=True
    neighbor_pairs=torch.nonzero(neighboring&train_mask&(contacts.sum(-1)<.5))
    reference=states[all_pairs[:,0],all_pairs[:,1]]
    ds=states[free_pairs[:,0],free_pairs[:,1]+1]-states[free_pairs[:,0],free_pairs[:,1]]
    norm=dict(mean=reference.mean(0).cpu().tolist(),std=reference.std(0).clamp_min(.01).cpu().tolist(),delta_mean=ds.mean(0).cpu().tolist())
    model_cfg={**cfg["model"],"architecture":"transformer_contact_residual_v1","dt_s":data["dt"]}
    model=TransformerContactNRD(model_cfg,norm).to(args.device)
    rng=torch.Generator(device=args.device).manual_seed(cfg["seed"]+1000)
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=False)
    runtime=dict(host=platform.node(),job_id=os.environ["SLURM_JOB_ID"],device=args.device,torch=torch.__version__,
        gpu=torch.cuda.get_device_name() if args.device=="cuda" else None,config=cfg,model_config=model_cfg,
        training_episodes=len(train),validation_episodes=len(val),test_used_for_selection=False,
        data_sha256=data["index"]["model_data_sha256"],parameters=sum(x.numel() for x in model.parameters()),
        model_source_sha256=hashlib.sha256(Path(__file__).with_name("transformer_contact.py").read_bytes()).hexdigest(),
        trainer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        label_definition="any contact during the next native interval" if model.modes==1 else "ground and wall contact during the next native interval",
        inference_input="current predicted state only; no gravity law, geometry, time, true contact or analytical phase")
    atomic_json(output/"run_config.json",runtime)
    start=time.perf_counter();best=float("inf");best_path=output/"best.pt"
    loss_scale=torch.tensor(cfg.get("loss_scale",[.01,.01,.2,.2,.5]),device=args.device,dtype=model.state_mean.dtype)
    free_scale=torch.tensor(cfg.get("free_scale",[.01,.01,.02,.02,.05]),device=args.device,dtype=model.state_mean.dtype)
    batch=128 if args.smoke else cfg.get("batch",512)

    def sample(pool,n):return pool[torch.randint(len(pool),(n,),device=args.device,generator=rng)]
    def balanced(n):
        return torch.cat((sample(contact_pairs,n//3),sample(neighbor_pairs,n//3),sample(free_pairs,n-2*(n//3))))
    def truth(pairs):return states[pairs[:,0],pairs[:,1]],states[pairs[:,0],pairs[:,1]+1],contacts[pairs[:,0],pairs[:,1]]
    def labels(c):return (c.sum(-1,keepdim=True)>.5).to(c.dtype) if model.modes==1 else c
    def optimizer(parameters,lr):return torch.optim.Adam(parameters,lr=lr)
    def save(path,stage,update,validation):
        torch.save(dict(model_config=model_cfg,normalization=norm,model_state_dict=model.state_dict(),runtime=runtime,
            stage=stage,update=update,validation=validation),path)
    def check(stage,update,loss,select=True):
        nonlocal best
        model.eval()
        report=free_metrics(model,data,val[:32] if args.smoke else val)
        if select and report["selection_score"]<best:
            best=report["selection_score"];save(best_path,stage,update,report)
        save(output/"last.pt",stage,update,report)
        event=dict(stage=stage,update=update,loss=float(loss),elapsed_s=time.perf_counter()-start,
            p95_trajectory_mm=report["position_rmse_m"]["p95"]*1000,p95_endpoint_mm=report["endpoint_error_m"]["p95"]*1000,
            order_fraction=report["contact_order_fraction"],selection_score=report["selection_score"])
        with (output/"train_log.jsonl").open("a") as handle:handle.write(json.dumps(event)+"\n")
        print(json.dumps(event),flush=True);model.train()
        return report

    core_params=list(model.backbone.parameters())+list(model.core_head.parameters())
    opt=optimizer(core_params,cfg.get("core_lr",3e-4))
    warmup=8 if args.smoke else cfg.get("core_warmup",3000)
    for update in range(1,warmup+1):
        if cfg.get("core_cosine",False):
            lr=cfg.get("core_lr",3e-4)*(.003+.997*.5*(1+math.cos(math.pi*update/warmup)))
            for group in opt.param_groups:group["lr"]=lr
        s,y,_=truth(sample(free_pairs,batch))
        loss=((s+model.core_delta(s)-y)/free_scale).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(core_params,10.);opt.step()
    print(json.dumps(dict(stage="free_core",updates=warmup,loss=float(loss.detach()),elapsed_s=time.perf_counter()-start)),flush=True)

    if cfg.get("core_lbfgs",0) and not args.smoke:
        pairs=sample(free_pairs,cfg.get("core_lbfgs_batch",4096));s,y,_=truth(pairs)
        opt=torch.optim.LBFGS(core_params,lr=1.,max_iter=15,history_size=50,line_search_fn="strong_wolfe",tolerance_grad=1e-12,tolerance_change=1e-15)
        def closure():
            opt.zero_grad(set_to_none=True)
            loss=((s+model.core_delta(s)-y)/free_scale).square().mean()
            loss.backward();return loss
        for update in range(cfg["core_lbfgs"]):opt.step(closure)
        print(json.dumps(dict(stage="free_core_lbfgs",outer_updates=cfg["core_lbfgs"],loss=float(closure().detach()),elapsed_s=time.perf_counter()-start)),flush=True)
    save(output/"free_core.pt","free_core",warmup,None)
    with torch.no_grad():
        pairs=torch.nonzero(valid&(splits==1)[:,None]&(contacts.sum(-1)<.5))
        errors=[]
        for chosen in pairs.split(4096):
            s,y,_=truth(chosen);errors.append(s+model.core_delta(s)-y)
        error=torch.cat(errors)
        core_report=dict(position_error_m=quantiles(error[:,:2].norm(dim=-1).cpu()),
            velocity_error_mps=quantiles(error[:,2:4].norm(dim=-1).cpu()),spin_error_radps=quantiles(error[:,4].abs().cpu()))
    atomic_json(output/"free_core_validation.json",core_report)
    print(json.dumps(dict(stage="free_core_validation",position_p95_mm=core_report["position_error_m"]["p95"]*1000,velocity_p95_mps=core_report["velocity_error_mps"]["p95"])),flush=True)

    if model.contact_enabled:
        gate_params=list(model.contact_encoder.parameters())+list(model.contact_head.parameters())
        opt=optimizer(gate_params,cfg.get("gate_lr",1e-3))
        updates=8 if args.smoke else cfg.get("gate_warmup",3000)
        for update in range(1,updates+1):
            s,_,c=truth(balanced(batch))
            logits=model.contact_head(model.contact_encoder(model.contact_features(s)))
            loss=F.binary_cross_entropy_with_logits(logits,labels(c))
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        print(json.dumps(dict(stage="contact_fit",updates=updates,loss=float(loss.detach()),elapsed_s=time.perf_counter()-start)),flush=True)
        bounce_params=list(model.bounce.parameters());opt=optimizer(bounce_params,cfg.get("bounce_lr",1e-3))
        updates=8 if args.smoke else cfg.get("bounce_warmup",3000)
        for update in range(1,updates+1):
            s,y,c=truth(sample(contact_pairs,batch))
            with torch.no_grad():core=model.core_delta(s)
            response=torch.stack([net(model.bounce_features(s))*model.delta_scale for net in model.bounce],-2)
            # True contacts route only this training warm-up loss, never rollout.
            predicted=s+core+(response*labels(c)[...,None]).sum(-2)
            loss=((predicted-y)/loss_scale).square().mean()
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(bounce_params,10.);opt.step()
        check("bounce_fit",updates,loss.detach())

        # Train on the actual predicted gate, including outgoing/noncontact
        # states that the position-only classifier may also switch on.
        updates=0 if args.smoke else cfg.get("predicted_bounce_updates",0)
        opt=optimizer(bounce_params,cfg.get("predicted_bounce_lr",1e-4))
        for update in range(1,updates+1):
            s,y,c=truth(balanced(batch))
            with torch.no_grad():
                core=model.core_delta(s)
                logits=model.contact_head(model.contact_encoder(model.contact_features(s)))
                threshold=float(model.config.get("gate_threshold",.5))
                gates=(logits>=math.log(threshold/(1-threshold))).to(s.dtype) if model.config.get("gate","hard")=="hard" else logits.sigmoid()
            response=torch.stack([net(model.bounce_features(s))*model.delta_scale for net in model.bounce],-2)
            predicted=s+core+(response*gates[...,None]).sum(-2)
            loss=((predicted-y)/loss_scale).square().mean()
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(bounce_params,10.);opt.step()
            if update%cfg.get("evaluate_every",2000)==0 or update==updates:check("predicted_bounce",update,loss.detach())

    # The matched control spends the same total update budget on its Transformer.
    updates=12 if args.smoke else cfg.get("joint_updates",10000)+(cfg.get("gate_warmup",3000)+cfg.get("bounce_warmup",3000)+cfg.get("predicted_bounce_updates",0) if not model.contact_enabled else 0)
    rng.manual_seed(cfg["seed"]+2000)
    opt=optimizer(model.parameters(),cfg.get("joint_lr",1e-4))
    every=12 if args.smoke else cfg.get("evaluate_every",2000)
    for update in range(1,updates+1):
        lr=cfg.get("joint_lr",1e-4)*(.03+.97*.5*(1+math.cos(math.pi*update/updates)))
        for group in opt.param_groups:group["lr"]=lr
        s,y,c=truth(balanced(batch));prediction,info=model.details(s)
        loss=((prediction-y)/loss_scale).square().mean()
        if model.contact_enabled:
            loss=loss+cfg.get("contact_loss_weight",.5)*F.binary_cross_entropy_with_logits(info["logits"],labels(c))
        if model.contact_enabled or cfg.get("matched_free_core_loss",False):
            # Preserve the branch interpretation while allowing core phase correction.
            free=c.sum(-1)<.5
            loss=loss+cfg.get("free_core_weight",.1)*((s[free]+info["delta1"][free]-y[free])/free_scale).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
        if update%every==0 or update==updates:check("joint",update,loss.detach())

    model,_=load_transformer_contact(best_path,args.device)
    opt=optimizer(model.parameters(),cfg.get("rollout_lr",1e-5))
    count=2 if args.smoke else cfg.get("rollout_updates",400)
    horizon=min(8,round(.5/model.dt)) if args.smoke else round(cfg.get("rollout_horizon_s",.5)/model.dt)
    rollout_batch=4 if args.smoke else cfg.get("rollout_batch",32)
    for update in range(1,count+1):
        # Half start near contact, half random; all horizons mask padded tails.
        near=sample(contact_pairs,rollout_batch//2)
        near=near.clone();near[:,1]=(near[:,1]-torch.randint(1,horizon,(len(near),),device=args.device)).clamp_min(0)
        pairs=torch.cat((near,sample(all_pairs,rollout_batch-len(near))))
        current=states[pairs[:,0],pairs[:,1]];total=current.sum()*0.;denominator=0
        for step in range(horizon):
            j=pairs[:,1]+step+1;active=j<lengths[pairs[:,0]]
            y=states[pairs[:,0],j.clamp_max(states.shape[1]-1)]
            predicted,info=model.details(current)
            total=total+((((predicted-y)/loss_scale).square().mean(-1))*active).sum()
            if model.contact_enabled:
                c=contacts[pairs[:,0],(j-1).clamp_max(contacts.shape[1]-1)]
                total=total+cfg.get("rollout_contact_weight",.1)*(F.binary_cross_entropy_with_logits(info["logits"],labels(c),reduction="none").mean(-1)*active).sum()
            denominator+=int(active.sum());current=torch.where(active[:,None],predicted,current)
        loss=total/max(1,denominator)
        opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(grad):raise RuntimeError("nonfinite rollout training gradient")
        opt.step()
        if update%(2 if args.smoke else cfg.get("rollout_evaluate_every",100))==0 or update==count:check("half_second",update,loss.detach())

    model,metadata=load_transformer_contact(best_path,args.device)
    validation=free_metrics(model,data,val[:32] if args.smoke else val)
    teacher=teacher_metrics(model,data,val[:32] if args.smoke else val)
    atomic_json(output/"validation.json",dict(free_rollout=validation,teacher_forced=teacher))
    gradient=launch_gradient_check(model,[[4.3,-10.2],[5.5,-9.75],[6.8,-9.2]])
    atomic_json(output/"gradient_check.json",gradient)
    final=dict(complete=True,smoke=args.smoke,test_used_for_selection=False,checkpoint=str(best_path),
        checkpoint_sha256=hashlib.sha256(best_path.read_bytes()).hexdigest(),elapsed_s=time.perf_counter()-start,
        best_stage=metadata["stage"],best_update=metadata["update"],validation_score=validation["selection_score"])
    atomic_json(output/"complete.json",final);print(json.dumps(final),flush=True)


if __name__=="__main__":main()
