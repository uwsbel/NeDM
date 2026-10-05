"""AMD-only staged contact-latent experiment; validation chooses checkpoints."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from nedm.bouncing_ball.collection import atomic_json
from nedm.bouncing_ball.model import load_model
from nedm.bouncing_ball.latent_model import LatentContactNRD, checkpoint, load_latent
from nedm.bouncing_ball.latent_evaluation import transition_data, teacher_metrics, free_metrics, launch_gradient_check


def pools(packet):
    mask = packet["valid"]&(packet["splits"]==0)[:,None]
    contact = packet["contacts"].sum(-1)>0
    positive = torch.nonzero(mask&contact)
    negative = torch.nonzero(mask&~contact)
    near = torch.zeros_like(contact)
    for shift in (-2,-1,1,2):
        near |= torch.roll(contact,shift,1)
    adjacent = torch.nonzero(mask&~contact&near)
    return positive, negative, adjacent


def sampled(pool, n):
    return pool[torch.randint(len(pool),(n,),device=pool.device)]


def batch_loss(model, packet, pairs, config):
    s = packet["states"][pairs[:,0],pairs[:,1]]
    y = packet["states"][pairs[:,0],pairs[:,1]+1]
    labels = packet["contacts"][pairs[:,0],pairs[:,1]]
    phase = packet["phases"][pairs[:,0],pairs[:,1]]
    if config["stage"] == "response":
        latent, logits, raw_phase = model.contact(s)
        details = {"logits":logits,"raw_phase":raw_phase}
        prediction = None
    else:
        prediction, details = model.details(s)
    classification = F.binary_cross_entropy_with_logits(details["logits"], labels)
    timing_error = (details["raw_phase"]*model.dt-phase)/config.get("timing_loss_scale_s",.001)
    phase_loss = (timing_error.square()*labels).sum()/labels.sum().clamp_min(1)
    if config["stage"] == "response":
        loss = classification+phase_loss*config.get("phase_loss_weight",1.)
    else:
        scale = s.new_tensor(config.get("one_step_scale",[.02,.02,.2,.2,1.]))
        motion = ((prediction-y)/scale).square().mean()
        loss = motion+config.get("contact_loss_weight",100.)*classification
        if config.get("phase_loss_weight",0.):
            loss = loss+config["phase_loss_weight"]*phase_loss
    return loss, {"classification":float(classification.detach()),"phase_loss":float(phase_loss.detach())}


def train(args):
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("Use an AMD MI350 compute allocation")
    if not os.environ.get("SLURM_JOB_ID"): raise RuntimeError("Training must run in an AMD compute allocation")
    device = args.device
    config = json.loads(args.config.read_text())
    torch.set_num_threads(config.get("threads",4))
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    packet = transition_data(args.data,device,config.get("stride",5))
    train_ids, val_ids = [torch.nonzero(packet["splits"]==k).flatten() for k in (0,1)]
    valid = packet["valid"]&(packet["splits"]==0)[:,None]
    ref = packet["states"][:,:-1][valid]
    normalization = {"mean":ref.mean(0).cpu().tolist(),"std":ref.std(0).clamp_min(.01).cpu().tolist()}
    model_config = {**config,"architecture":"learned_contact_latent_v1","dt_s":packet["dt"],
        "gravity_mps2":packet["gravity"],"physics_step_s":packet["physics_dt"]}
    if config["stage"] == "response":
        response, response_metadata = load_model(args.response,device)
        model_config["response_hidden"] = response.config["hidden"]
    model = LatentContactNRD(model_config,normalization).to(device)
    if config["stage"] == "response":
        model.responses.initialize(response)
    if args.contact_init:
        initializer,_=load_latent(args.contact_init,device)
        model.contact.load_state_dict(initializer.contact.state_dict())
        if config.get("freeze_contact",False):
            for parameter in model.contact.parameters(): parameter.requires_grad_(False)
    if config["stage"] == "unified_gated":
        assert args.contact_init is not None
        with torch.no_grad():
            training_pairs=torch.nonzero(packet["valid"]&(packet["splits"]==0)[:,None]&(packet["contacts"].sum(-1)>0))
            current=packet["states"][training_pairs[:,0],training_pairs[:,1]]
            target=packet["states"][training_pairs[:,0],training_pairs[:,1]+1]
            latent, logits, _=model.contact(current)
            features=torch.cat(((current-model.contact.mean)/model.contact.std,latent,logits.sigmoid()),-1)
            design=torch.cat((features,torch.ones(len(features),1,device=device,dtype=features.dtype)),-1)
            normalized=(target-model.flight(current))/model.correction_scale
            beta=torch.linalg.solve(design.T@design+1e-8*torch.eye(design.shape[1],device=device,dtype=design.dtype),design.T@normalized)
            model.linear_core.weight.copy_(beta[:-1].T)
            model.linear_core.bias.copy_(beta[-1])
    args.output.mkdir(parents=True,exist_ok=False)
    runtime = {"config":config,"model_config":model_config,"training_data_sha256":packet["index"]["model_data_sha256"],
        "training_episodes":len(train_ids),"validation_episodes":len(val_ids),"test_used_for_selection":False,
        "response_checkpoint_sha256":hashlib.sha256(args.response.read_bytes()).hexdigest() if config["stage"]=="response" else None,
        "contact_initializer_sha256":hashlib.sha256(args.contact_init.read_bytes()).hexdigest() if args.contact_init else None,
        "host":platform.node(),"job_id":os.environ.get("SLURM_JOB_ID"),"torch":torch.__version__,
        "device":device,"gpu":torch.cuda.get_device_name() if device=="cuda" else None,"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_source_sha256":hashlib.sha256(Path(__file__).with_name("latent_model.py").read_bytes()).hexdigest(),
        "label_definition":"ground/wall impact occurs during the next model interval; phase derived from training current/next states only",
        "inference_input":"predicted state only, no geometric thresholds, distance, analytical roots, contact labels, future states or launch identity",
        "phase_label_max_position_residual_m":packet["max_position_label_residual_m"]}
    atomic_json(args.output/"run_config.json",runtime)
    positive, negative, near = pools(packet)
    start = time.perf_counter()
    best = float("inf")
    best_fit = float("inf")
    validation_ids = val_ids[:64] if args.smoke else val_ids

    def evaluate_check(update, stage, loss, parts):
        nonlocal best,best_fit
        model.eval()
        teacher = teacher_metrics(model,packet,validation_ids)
        free = free_metrics(model,packet,validation_ids)
        result={"teacher_forced":teacher,"free_rollout":free}
        score = free["selection_score"]
        fit_score = sum(1-c["f1"] for c in teacher["contact"])+sum(v["p95"] for v in teacher["phase_error_s"])*10
        if score < best:
            best = score
            torch.save(checkpoint(model,runtime,update,result),args.output/"best.pt")
            atomic_json(args.output/"validation.json",result)
        if fit_score < best_fit:
            best_fit = fit_score
            torch.save(checkpoint(model,runtime,update,result),args.output/"best_contact_fit.pt")
            atomic_json(args.output/"contact_fit_validation.json",result)
        torch.save(checkpoint(model,runtime,update,result),args.output/"last.pt")
        record={"update":update,"stage":stage,"loss":float(loss),"parts":parts,"elapsed_s":time.perf_counter()-start,
            "trajectory_p95_mm":free["position_rmse_m"]["p95"]*1000,"endpoint_p95_mm":free["endpoint_error_m"]["p95"]*1000,
            "contact_order":free["contact_order_fraction"],"contact_f1":[c["f1"] for c in teacher["contact"]],
            "phase_p95_us":[p["p95"]*1e6 for p in teacher["phase_error_s"]]}
        with (args.output/"train_log.jsonl").open("a") as stream: stream.write(json.dumps(record)+"\n")
        print(json.dumps(record),flush=True)
        model.train()
        return result

    evaluate_check(0,"initialization",0,{})
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=config.get("learning_rate",.001))
    updates = 100 if args.smoke else config.get("updates",20000)
    batch = 256 if args.smoke else config.get("batch",2048)
    for update in range(1,updates+1):
        pairs = torch.cat((sampled(positive,batch//2),sampled(near,batch//4),sampled(negative,batch-batch//2-batch//4)))
        loss_config = ({**config,"phase_loss_weight":0.} if update <= config.get("classification_warmup_updates",0) else config)
        loss, parts = batch_loss(model,packet,pairs,loss_config)
        if config.get("cosine_decay",True):
            rate = config.get("final_lr",1e-6)+(config.get("learning_rate",.001)-config.get("final_lr",1e-6))*.5*(1+math.cos(math.pi*update/updates))
            for group in optimizer.param_groups: group["lr"]=rate
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),100.)
        if not torch.isfinite(norm): raise RuntimeError("Nonfinite training gradient")
        optimizer.step()
        if update%config.get("evaluate_every",2000)==0 or update==updates:
            evaluate_check(update,"supervised_contact" if config["stage"]=="response" else "unified_one_step",loss,parts)
    if not args.smoke and config.get("lbfgs_updates",0):
        # A deterministic training-only batch: contacts plus neighbouring and
        # uniformly sampled free transitions. Validation still selects.
        chosen_negative = torch.cat((near, sampled(negative,min(len(negative),8192))))
        chosen = torch.cat((positive,chosen_negative))
        optimizer=torch.optim.LBFGS([p for p in model.parameters() if p.requires_grad],lr=.5,max_iter=10,
            tolerance_grad=1e-10,tolerance_change=1e-14,line_search_fn="strong_wolfe")
        for update in range(1,config["lbfgs_updates"]+1):
            def closure():
                optimizer.zero_grad(set_to_none=True)
                total, _ = batch_loss(model,packet,chosen,config)
                total.backward()
                return total
            loss=optimizer.step(closure)
            if update%config.get("lbfgs_evaluate_every",50)==0 or update==config["lbfgs_updates"]:
                evaluate_check(update,"lbfgs",loss,{})
    if config.get("rollout_updates",0):
        model,_ = load_latent(args.output/"best.pt",device)
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=config.get("rollout_lr",1e-5))
        steps=min(4,round(config.get("rollout_horizon_s",.5)/model.dt)) if args.smoke else round(config.get("rollout_horizon_s",.5)/model.dt)
        scale=packet["states"].new_tensor([.01,.01,.1,.1,.5])
        rollout_updates=2 if args.smoke else config["rollout_updates"]
        for update in range(1,rollout_updates+1):
            ids=train_ids[torch.randint(len(train_ids),(8 if args.smoke else config.get("rollout_batch",64),),device=device)]
            if config.get("rollout_horizon_s",.5)>1.:
                offsets=torch.zeros_like(ids)
            else:
                kind=torch.randint(2,(len(ids),),device=device)
                label=packet["contacts"][ids, :, :].gather(2,kind[:,None,None].expand(-1,packet["contacts"].shape[1],1))[...,0]
                offsets=(label.argmax(1)-torch.randint(1,4,(len(ids),),device=device)).clamp_min(0)
            trajectory=model.rollout(packet["states"][ids,offsets],steps)
            gather=offsets[:,None]+torch.arange(steps+1,device=device)[None,:]
            valid=gather<packet["lengths"][ids,None]
            truth=packet["states"][ids[:,None],gather.clamp_max(packet["states"].shape[1]-1)]
            error=((trajectory-truth)/scale).square().mean(-1)
            motion=(error*valid).sum()/valid.sum().clamp_min(1)
            pairs=torch.cat((sampled(positive,512),sampled(near,256),sampled(negative,256)))
            auxiliary,_=batch_loss(model,packet,pairs,config)
            loss=motion+config.get("rollout_aux_weight",.01)*auxiliary
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            if not torch.isfinite(norm): raise RuntimeError("Nonfinite recursive rollout gradient")
            optimizer.step()
            if update%config.get("rollout_evaluate_every",100)==0 or update==rollout_updates:
                evaluate_check(update,"predicted_state_rollout",loss,{})
    model, chosen = load_latent(args.output/"best.pt",device)
    gradient = launch_gradient_check(model,[[4.3,-10.2],[5.5,-9.75],[6.8,-9.2]])
    atomic_json(args.output/"gradient_check.json",gradient)
    complete={"complete":True,"smoke":args.smoke,"selection_split":"validation","test_used":False,
        "checkpoint_sha256":hashlib.sha256((args.output/"best.pt").read_bytes()).hexdigest(),
        "elapsed_s":time.perf_counter()-start,"validation":chosen["validation"],"runtime":runtime}
    atomic_json(args.output/"complete.json",complete)
    print(json.dumps({k:complete[k] for k in ("complete","elapsed_s","checkpoint_sha256")}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True)
    p.add_argument("--response",type=Path,required=True);p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--smoke",action="store_true")
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--contact-init",type=Path)
    train(p.parse_args())


if __name__=="__main__": main()
