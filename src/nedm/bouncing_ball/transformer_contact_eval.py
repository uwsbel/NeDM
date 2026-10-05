"""Matched native-grid evaluation; scene geometry is used for metrics only."""
from __future__ import annotations
import math
import numpy as np
import torch
from nedm.bouncing_ball.precision import prepare
from nedm.bouncing_ball.training import quantiles


def data_packet(root,device,stride=1):
    index,states,lengths,splits,contacts=prepare(root,device,stride)
    valid=torch.arange(states.shape[1]-1,device=device)[None,:]<(lengths-1)[:,None]
    return dict(index=index,states=states,lengths=lengths,splits=splits,contacts=contacts,
                valid=valid,dt=index["model_dt_s"]*stride)


def contact_scores(logits,labels,threshold=.5):
    predicted=logits>=math.log(threshold/(1-threshold))
    if logits.shape[-1]==1: labels=(labels.sum(-1,keepdim=True)>.5)
    else: labels=labels>.5
    output=[]
    for k in range(logits.shape[-1]):
        p,t=predicted[...,k],labels[...,k]
        tp,fp,fn,tn=[int(v) for v in ((p&t).sum(),(p&~t).sum(),(~p&t).sum(),(~p&~t).sum())]
        output.append(dict(kind="any" if logits.shape[-1]==1 else ("ground","wall")[k],
            tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
            f1=2*tp/max(1,2*tp+fp+fn),threshold=threshold))
    return output


@torch.no_grad()
def free_metrics(model,packet,indices,save_traces=None):
    if abs(model.dt-packet['dt'])>1e-12:
        raise ValueError(f"Model step {model.dt} differs from evaluation data step {packet['dt']}")
    entries=[]
    scene=packet["index"]["config"]["scene"]
    floor=scene["ground_z_m"]+scene["radius_m"]
    wall=scene["wall_front_x_m"]-scene["radius_m"]
    traces=[];rollout_logits=[];rollout_labels=[]
    common_steps=int(packet["lengths"][indices].max())-1
    for chosen in indices.split(256):
        truth=packet["states"][chosen]
        trajectory=model.rollout(truth[:,0],common_steps)
        if getattr(model,"contact_enabled",False):
            flat=trajectory[:,:-1].reshape(-1,5)
            gate_logits=model.gate_logits(flat) if hasattr(model,'gate_logits') else model.contact_head(model.contact_encoder(model.contact_features(flat)))
            gate_logits=gate_logits.reshape(len(chosen),common_steps,-1)
            active=torch.arange(common_steps,device=chosen.device)[None,:]<(packet["lengths"][chosen]-1)[:,None]
            rollout_logits.append(gate_logits[active].cpu())
            rollout_labels.append(packet["contacts"][chosen,:common_steps][active].cpu())
        pred=trajectory.detach().cpu().numpy().astype(np.float64)
        actual=truth.cpu().numpy().astype(np.float64)
        for row,idx in enumerate(chosen.tolist()):
            n=int(packet["lengths"][idx]);s=pred[row,:n];t=actual[row,:n]
            finite=bool(np.isfinite(s).all())
            item=dict(index=idx,finite=finite)
            if finite:
                distance=np.linalg.norm(s[:,:2]-t[:,:2],axis=-1)
                # Geometry disambiguates signatures in this metric only. A
                # wall tangential impulse can also reverse vertical velocity.
                g=packet["index"]["config"]["simulation"]["gravity_mps2"]
                h=packet["index"]["config"]["simulation"]["step_s"]
                free_z=s[:-1,1]+s[:-1,3]*model.dt-.5*g*model.dt*(model.dt+h)
                free_x=s[:-1,0]+s[:-1,2]*model.dt
                ground=np.flatnonzero((s[:-1,3]<0)&(s[1:,3]>0)&(free_z<=floor+.005))+1
                walls=np.flatnonzero((s[:-1,2]>0)&(s[1:,2]<0)&(free_x>=wall-.005))+1
                order=len(ground)==1 and len(walls)==1 and ground[0]<walls[0]
                expected=[int(torch.nonzero(packet["contacts"][idx,:n-1,k]>.5)[0])+1 for k in (0,1)]
                terr=max(abs(int(ground[0])-expected[0]),abs(int(walls[0])-expected[1]))*model.dt if order else n*model.dt
                mask=np.ones(n,dtype=bool)
                for step in expected:mask[max(0,step-2):min(n,step+3)]=False
                item.update(position_rmse_m=float(np.sqrt(np.mean(distance**2))),endpoint_error_m=float(distance[-1]),
                    maximum_position_error_m=float(distance.max()),contact_order=bool(order),contact_time_error_s=float(terr),
                    velocity_mae_mps=float(np.abs(s[mask,2:4]-t[mask,2:4]).mean()),
                    spin_mae_radps=float(np.abs(s[mask,4]-t[mask,4]).mean()),
                    max_penetration_m=float(max(0.,floor-s[:,1].min(),s[:,0].max()-wall)),
                    ground_events=(ground*model.dt).tolist(),wall_events=(walls*model.dt).tolist())
            entries.append(item)
        if save_traces is not None:traces.append(pred)
    finite=[r for r in entries if r["finite"]]
    result=dict(episodes=len(entries),finite_fraction=len(finite)/len(entries),
        contact_order_fraction=sum(r.get("contact_order",False) for r in entries)/len(entries),
        failure_count=sum(not r.get("contact_order",False) for r in entries),per_episode=entries)
    for key in ("position_rmse_m","endpoint_error_m","maximum_position_error_m","velocity_mae_mps","spin_mae_radps","contact_time_error_s","max_penetration_m"):
        result[key]=quantiles([r[key] for r in finite]) if finite else dict(mean=1e6,median=1e6,p95=1e6,max=1e6)
    result["selection_score"]=result["position_rmse_m"]["p95"]+.5*result["endpoint_error_m"]["p95"]+10*(1-result["contact_order_fraction"])+100*(1-result["finite_fraction"])
    result["contact_from_predicted_states"]=contact_scores(torch.cat(rollout_logits),torch.cat(rollout_labels),model.config.get("gate_threshold",.5)) if rollout_logits else None
    if save_traces is not None:
        np.savez_compressed(save_traces,predicted=np.concatenate(traces),indices=indices.cpu().numpy(),
            lengths=packet["lengths"][indices].cpu().numpy(),dt_s=model.dt)
    return result


@torch.no_grad()
def teacher_metrics(model,packet,indices):
    included=torch.zeros(len(packet["splits"]),device=packet["states"].device,dtype=torch.bool)
    included[indices]=True
    pairs=torch.nonzero(packet["valid"]&included[:,None])
    predictions=[];logits=[];core=[];bounce=[]
    for batch in pairs.split(4096):
        state=packet["states"][batch[:,0],batch[:,1]]
        if hasattr(model,'context'):
            from nedm.bouncing_ball.transformer_temporal import truth_history
            prediction,info=model.details(state,truth_history(packet['states'],batch,model.context))
        else:
            prediction,info=model.details(state)
        predictions.append(prediction);logits.append(info["logits"])
        core.append(info["delta1"]);bounce.append(prediction-state-info["delta1"])
    y=torch.cat(predictions)
    target=packet["states"][pairs[:,0],pairs[:,1]+1]
    labels=packet["contacts"][pairs[:,0],pairs[:,1]]
    result=dict(transitions=len(pairs),contact_classification=contact_scores(torch.cat(logits),labels,model.config.get("gate_threshold",.5)) if model.contact_enabled else None)
    for name,mask in (("contact",labels.sum(-1)>.5),("noncontact",labels.sum(-1)<.5)):
        result[name]=dict(count=int(mask.sum()),position_error_m=quantiles((y-target)[:,:2].norm(dim=-1)[mask].cpu()),
            velocity_error_mps=quantiles((y-target)[:,2:4].norm(dim=-1)[mask].cpu()),
            core_velocity_delta_norm_mps=quantiles(torch.cat(core)[:,2:4].norm(dim=-1)[mask].cpu()),
            bounce_velocity_delta_norm_mps=quantiles(torch.cat(bounce)[:,2:4].norm(dim=-1)[mask].cpu()))
    return result


def launch_gradient_check(model,launches):
    for parameter in model.parameters():parameter.requires_grad_(False)
    action=torch.tensor(launches,dtype=model.state_mean.dtype,device=model.state_mean.device,requires_grad=True)
    def endpoint(a):
        s=torch.cat((torch.zeros_like(a[:,:1]),torch.ones_like(a[:,:1]),a,torch.zeros_like(a[:,:1])),-1)
        return model.rollout(s,round(1.7/model.dt))[:,-1,:2]
    y=endpoint(action)
    jacobian=torch.stack([torch.autograd.grad(y[:,k].sum(),action,retain_graph=True)[0] for k in (0,1)],1)
    rows=[]
    with torch.no_grad():
        for epsilon in (1e-5,1e-4,1e-3):
            columns=[]
            for k in (0,1):
                offset=torch.zeros_like(action);offset[:,k]=epsilon
                columns.append((endpoint(action+offset)-endpoint(action-offset))/(2*epsilon))
            finite=torch.stack(columns,-1)
            diff=(finite-jacobian).abs()
            rows.append(dict(epsilon=epsilon,max_abs_error=float(diff.max()),p95_abs_error=float(torch.quantile(diff.flatten(),.95)),finite=bool(torch.isfinite(finite).all())))
    return dict(finite_autograd=bool(torch.isfinite(jacobian).all()),jacobian=jacobian.detach().cpu().tolist(),
                launches=launches,finite_differences=rows,gate=model.config.get("gate","hard"))
