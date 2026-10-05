"""Teacher-forced contact fit and independent free predicted-state rollouts."""
from __future__ import annotations
import json
import numpy as np
import torch
from nedm.bouncing_ball.precision import prepare
from nedm.bouncing_ball.training import quantiles


def transition_data(root, device, stride=5):
    index, states, lengths, splits, contacts = prepare(root, device, stride)
    dt = index["model_dt_s"]*stride
    g = index["config"]["simulation"]["gravity_mps2"]
    h = index["config"]["simulation"]["step_s"]
    valid = torch.arange(states.shape[1]-1, device=device)[None, :] < (lengths-1)[:, None]
    assert not bool((contacts.sum(-1)>1).any()), "Multiple contacts per interval need a sequential model"
    phases = torch.zeros_like(contacts)
    residuals = []
    for kind in (0, 1):
        pairs = torch.nonzero((contacts[..., kind]>0)&valid)
        s = states[pairs[:, 0], pairs[:, 1]]
        y = states[pairs[:, 0], pairs[:, 1]+1]
        free = s.clone()
        free[:, 0] += s[:, 2]*dt
        free[:, 1] += s[:, 3]*dt-.5*g*dt*(dt+h)
        free[:, 3] -= g*dt
        dv = y[:, 2:]-free[:, 2:]
        dp = y[:, :2]-free[:, :2]
        remaining = (dp*dv[:, :2]).sum(-1)/dv[:, :2].square().sum(-1).clamp_min(1e-12)
        phases[pairs[:, 0], pairs[:, 1], kind] = dt-remaining
        residuals.append(float((dp-dv[:, :2]*remaining[:, None]).norm(dim=-1).max()))
    return {"index":index,"states":states,"lengths":lengths,"splits":splits,"contacts":contacts,
            "phases":phases,"valid":valid,"dt":dt,"gravity":g,"physics_dt":h,
            "max_position_label_residual_m":residuals}


def classification(logits, labels):
    decision = logits >= 0
    true = labels > .5
    output = []
    for k, name in enumerate(("ground", "wall")):
        p, t = decision[..., k].flatten(), true[..., k].flatten()
        tp, fp, fn, tn = [int(x) for x in ((p&t).sum(), (p&~t).sum(), (~p&t).sum(), (~p&~t).sum())]
        output.append({"kind":name,"tp":tp,"fp":fp,"fn":fn,"tn":tn,
            "precision":tp/max(1,tp+fp),"recall":tp/max(1,tp+fn),"f1":2*tp/max(1,2*tp+fp+fn),
            "accuracy":(tp+tn)/max(1,tp+fp+fn+tn)})
    return output


@torch.no_grad()
def teacher_metrics(model, packet, indices):
    pairs = torch.nonzero(packet["valid"]&(torch.zeros_like(packet["splits"],dtype=torch.bool).scatter(0,indices,True))[:,None])
    predictions, logits, phases = [], [], []
    current = packet["states"][pairs[:, 0], pairs[:, 1]]
    target = packet["states"][pairs[:, 0], pairs[:, 1]+1]
    labels = packet["contacts"][pairs[:, 0], pairs[:, 1]]
    timing = packet["phases"][pairs[:, 0], pairs[:, 1]]
    for batch in current.split(4096):
        y, details = model.details(batch)
        predictions.append(y); logits.append(details["logits"]); phases.append(details["phase_s"])
    prediction, logits, phases = map(torch.cat,(predictions,logits,phases))
    metrics = {"transitions":len(pairs),"contact":classification(logits,labels),
        "one_step_position_error_m":quantiles((prediction[:,:2]-target[:,:2]).norm(dim=-1).cpu()),
        "one_step_velocity_error_mps":quantiles((prediction[:,2:4]-target[:,2:4]).norm(dim=-1).cpu()),
        "phase_error_s":[quantiles((phases[:,k]-timing[:,k]).abs()[labels[:,k]>.5].cpu()) for k in (0,1)]}
    metrics["explicit_phase_head_used_for_motion"] = model.config["stage"]=="response" and model.config.get("learned_phase",True)
    flight=model.flight(current)
    dv=prediction[:,2:4]-flight[:,2:4]
    dp=prediction[:,:2]-flight[:,:2]
    remaining=(dp*dv).sum(-1)/dv.square().sum(-1).clamp_min(1e-12)
    implicit_phase=model.dt-remaining
    residual=(dp-dv*remaining[:,None]).norm(dim=-1)
    metrics["effective_impulse_phase_error_s"]=[quantiles((implicit_phase-timing[:,k]).abs()[labels[:,k]>.5].cpu()) for k in (0,1)]
    metrics["impulse_position_consistency_residual_m"]=[quantiles(residual[labels[:,k]>.5].cpu()) for k in (0,1)]
    for name,mask in (("contact_transitions",labels.sum(-1)>.5),("no_contact_transitions",labels.sum(-1)<.5)):
        metrics[name]={"count":int(mask.sum()),"position_error_m":quantiles((prediction[:,:2]-target[:,:2]).norm(dim=-1)[mask].cpu()),
            "velocity_error_mps":quantiles((prediction[:,2:4]-target[:,2:4]).norm(dim=-1)[mask].cpu())}
    return metrics


@torch.no_grad()
def free_metrics(model, packet, indices):
    entries, logit_entries, true_entries, phase_entries = [], [], [], []
    states, lengths, labels = packet["states"], packet["lengths"], packet["contacts"]
    scene = packet["index"]["config"]["scene"]
    floor = scene["ground_z_m"]+scene["radius_m"]
    wall = scene["wall_front_x_m"]-scene["radius_m"]
    for batch in indices.split(256):
        current, predictions, all_logits, all_phases = states[batch,0], [states[batch,0]], [], []
        nsteps = int(lengths[batch].max())-1
        for step in range(nsteps):
            current, details = model.details(current)
            predictions.append(current)
            all_logits.append(details["logits"])
            all_phases.append(details["phase_s"])
        predicted = torch.stack(predictions,1).cpu().numpy()
        logits = torch.stack(all_logits,1)
        phases = torch.stack(all_phases,1)
        for j, idx in enumerate(batch.tolist()):
            n = int(lengths[idx])
            truth = states[idx,:n].cpu().numpy()
            pred = predicted[j,:n]
            lab = labels[idx,:n-1]
            expected_steps = [int(torch.nonzero(lab[:,k]>.5)[0])+1 for k in (0,1)]
            finite = bool(np.isfinite(pred).all())
            entry = {"index":idx,"finite":finite}
            if finite:
                distance = np.linalg.norm(pred[:,:2]-truth[:,:2],axis=-1)
                # Kinematic bounce signatures only; evaluation geometry cannot
                # feed into state transitions or choose event routing.
                ground = np.flatnonzero((pred[:-1,3]<0)&(pred[1:,3]>0))+1
                walls = np.flatnonzero((pred[:-1,2]>0)&(pred[1:,2]<0))+1
                order = len(ground)==1 and len(walls)==1 and ground[0]<walls[0]
                time_error = max(abs(int(ground[0])-expected_steps[0]),abs(int(walls[0])-expected_steps[1]))*model.dt if order else n*model.dt
                mask = np.ones(n,dtype=bool)
                for impact in expected_steps: mask[max(0,impact-1):min(n,impact+2)]=False
                entry.update(position_rmse_m=float(np.sqrt(np.mean(distance**2))),
                    endpoint_error_m=float(distance[-1]), maximum_position_error_m=float(distance.max()),
                    velocity_mae_mps=float(np.abs(pred[mask,2:4]-truth[mask,2:4]).mean()),
                    spin_mae_radps=float(np.abs(pred[mask,4]-truth[mask,4]).mean()),contact_order=bool(order),
                    contact_time_error_s=float(time_error),
                    ground_event_times_s=(ground*model.dt).tolist(),wall_event_times_s=(walls*model.dt).tolist(),
                    max_penetration_m=float(max(0.,floor-pred[:,1].min(),pred[:,0].max()-wall)))
            entries.append(entry)
            logit_entries.append(logits[j,:n-1]); true_entries.append(lab)
            phase_entries.append(phases[j,:n-1])
    finite = [e for e in entries if e["finite"]]
    report = {"episodes":len(entries),"finite_fraction":len(finite)/len(entries),
        "contact_order_fraction":sum(e["contact_order"] for e in finite)/len(entries),
        "failure_count":sum(not e["finite"] or not e.get("contact_order",False) for e in entries),
        "trajectory_accuracy_failure_count_3mm":sum(not e["finite"] or e.get("position_rmse_m",1e6)>.003 for e in entries),
        "endpoint_accuracy_failure_count_6mm":sum(not e["finite"] or e.get("endpoint_error_m",1e6)>.006 for e in entries),
        "penetration_failure_count_5mm":sum(not e["finite"] or e.get("max_penetration_m",1e6)>.005 for e in entries),
        "contact_from_predicted_states":classification(torch.cat(logit_entries),torch.cat(true_entries)),"per_episode":entries}
    for field in ("position_rmse_m","endpoint_error_m","maximum_position_error_m","velocity_mae_mps",
                  "spin_mae_radps","contact_time_error_s","max_penetration_m"):
        report[field] = quantiles([e[field] for e in finite]) if finite else dict(mean=1e6,median=1e6,p95=1e6,max=1e6)
    report["selection_score"] = report["position_rmse_m"]["p95"]+.5*report["endpoint_error_m"]["p95"]+10*(1-report["contact_order_fraction"])
    return report


def launch_gradient_check(model, launches, t=1.7):
    """Exact autograd vs centered finite differences, including gate changes."""
    parameter_flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters(): p.requires_grad_(False)
    action = torch.tensor(launches,device=model.contact.mean.device,dtype=torch.float64,requires_grad=True)
    def endpoint(a):
        state = torch.cat((torch.zeros_like(a[:,:1]), torch.ones_like(a[:,:1]), a, torch.zeros_like(a[:,:1])), -1)
        return model.rollout(state,round(t/model.dt))[:,-1,:2]
    y = endpoint(action)
    rows = [torch.autograd.grad(y[:,j].sum(),action,retain_graph=True)[0] for j in (0,1)]
    analytic = torch.stack(rows,1).detach()
    checks = []
    with torch.no_grad():
        for epsilon in (1e-5,1e-4,1e-3):
            columns=[]
            for k in (0,1):
                offset=torch.zeros_like(action); offset[:,k]=epsilon
                columns.append((endpoint(action+offset)-endpoint(action-offset))/(2*epsilon))
            numerical=torch.stack(columns,-1)
            checks.append({"epsilon":epsilon,"max_abs_error":float((analytic-numerical).abs().max()),
                "p95_abs_error":float(torch.quantile((analytic-numerical).abs().flatten(),.95)),
                "finite":bool(torch.isfinite(numerical).all())})
    for p, flag in zip(model.parameters(),parameter_flags): p.requires_grad_(flag)
    return {"finite_autograd":bool(torch.isfinite(analytic).all()),"launches":launches,
            "jacobian":analytic.cpu().tolist(),"finite_differences":checks,"contact_switch":"learned logits >= 0"}


@torch.no_grad()
def learned_dense_trajectory(model, initial, steps, samples_per_step=100):
    """Dense sampling of a stage1 interval using its learned gate and phase.

    Network predictions are held within their trained 50 ms interval. No
    analytical event roots, simulator labels or true times are consulted.
    Dense states at native boundaries exactly match recursive forward.
    """
    if model.config["stage"]!="response": raise ValueError("Unified five-output decoders do not define within-step motion")
    current=initial.to(dtype=model.contact.mean.dtype)
    times=torch.arange(samples_per_step,device=current.device,dtype=current.dtype)*model.dt/samples_per_step
    trajectory=[]
    for step in range(steps):
        following,details=model.details(current)
        dense=model.flight(current[:,None,:],times[None,:])
        for kind in (0,1):
            tau=details["phase_s"][:,kind]
            velocity=current[:,2:]+torch.stack((torch.zeros_like(tau),-model.gravity*tau,torch.zeros_like(tau)),-1)
            impulse=model.responses(velocity,kind)
            active=(times[None,:]>=tau[:,None])*details["gates"][:,kind,None]
            remaining=(times[None,:]-tau[:,None]).clamp_min(0.)
            change=torch.cat((impulse[:,None,:2]*remaining[:,:,None],impulse[:,None,:].expand(-1,samples_per_step,-1)),-1)
            dense=dense+active[:,:,None]*change
        trajectory.append(dense)
        current=following
    return torch.cat((*trajectory,current[:,None,:]),1)
