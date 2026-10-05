"""Train and evaluate the ball NRD on AMD, with whole-episode held-out splits."""
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
import torch.nn.functional as F

from nedm.bouncing_ball.collection import STATE_FIELDS, atomic_json
from nedm.bouncing_ball.model import BouncingBallNRD, load_model


def quantiles(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "median": float(np.median(values)), "p95": float(np.quantile(values, 0.95)), "max": float(values.max())}


def infer_contact_events(predicted, model):
    """A wall's tangential impulse may reverse vz far above the floor."""
    free_z = predicted[:-1,1] + predicted[:-1,3]*model.dt - .5*model.gravity*model.dt*(model.dt+model.physics_dt)
    free_x = predicted[:-1,0] + predicted[:-1,2]*model.dt
    ground = np.flatnonzero((predicted[:-1,3]<0) & (predicted[1:,3]>=0) & (free_z<=model.floor_center_z+.005))+1
    wall = np.flatnonzero((predicted[:-1,2]>0) & (predicted[1:,2]<=0) & (free_x>=model.wall_center_x-.005))+1
    return ground, wall


@torch.no_grad()
def evaluate(model, states, lengths, contacts, indices, limits):
    entries = []
    for batch in indices.split(256):
        truth = states[batch]
        prediction = model.rollout(truth[:, 0], int(lengths[batch].max())-1)
        for j, index in enumerate(batch.tolist()):
            n = int(lengths[index])
            expected = truth[j, :n].cpu().numpy()
            predicted = prediction[j, :n].cpu().numpy()
            labels = contacts[index, :n-1].cpu().numpy()
            if not np.isfinite(predicted).all():
                entries.append({"index": index, "finite": False})
                continue
            position_error = np.linalg.norm(predicted[:, :2]-expected[:, :2], axis=-1)
            expected_events = [int(np.flatnonzero(labels[:, k])[0]+1) for k in range(2)]
            ground_events, wall_events = infer_contact_events(predicted,model)
            order = len(ground_events)==1 and len(wall_events)==1 and ground_events[0]<wall_events[0]
            contact_errors = ([abs(int(ground_events[0])-expected_events[0])*model.dt,
                               abs(int(wall_events[0])-expected_events[1])*model.dt] if order else [model.dt*n]*2)
            mask = np.ones(n, dtype=bool)
            for event in expected_events:
                mask[max(0,event-2):min(n,event+3)] = False
            entries.append({"index": index, "finite": True, "position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
                            "endpoint_error_m": float(position_error[-1]),
                            "velocity_mae_mps": float(np.abs(predicted[mask, 2:4]-expected[mask, 2:4]).mean()),
                            "spin_mae_radps": float(np.abs(predicted[mask, 4]-expected[mask, 4]).mean()),
                            "contact_time_error_s": max(contact_errors), "contact_order": bool(order),
                            "ground_event_times_s": (ground_events*model.dt).tolist(), "wall_event_times_s": (wall_events*model.dt).tolist(),
                            "max_penetration_m": float(max(0,model.floor_center_z-predicted[:,1].min(),predicted[:,0].max()-model.wall_center_x))})
    finite = [entry for entry in entries if entry["finite"]]
    result = {"episodes": len(entries), "finite_fraction": len(finite)/len(entries), "per_episode": entries}
    for key in ["position_rmse_m", "endpoint_error_m", "velocity_mae_mps", "spin_mae_radps", "contact_time_error_s", "max_penetration_m"]:
        result[key] = quantiles([e[key] for e in finite]) if finite else {"p95": 1e6, "mean": 1e6, "median": 1e6, "max": 1e6}
    result["contact_order_fraction"] = sum(entry["contact_order"] for entry in finite)/len(entries)
    result["passed"] = (result["finite_fraction"] == 1 and result["contact_order_fraction"] >= limits["contact_order_fraction"]
        and result["position_rmse_m"]["p95"] <= limits["position_rmse_p95_m"]
        and result["endpoint_error_m"]["p95"] <= limits["endpoint_error_p95_m"]
        and result["velocity_mae_mps"]["p95"] <= limits["velocity_mae_p95_mps"]
        and result["spin_mae_radps"]["p95"] <= limits["spin_mae_p95_radps"]
        and result["contact_time_error_s"]["p95"] <= limits["contact_time_error_p95_s"]
        and result["max_penetration_m"]["p95"] <= limits.get("max_penetration_p95_m",float("inf")))
    result["selection_score"] = result["position_rmse_m"]["p95"] + 0.1*result["velocity_mae_mps"]["p95"] + 0.02*result["spin_mae_radps"]["p95"] + (1-result["contact_order_fraction"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/bouncing_ball/nrd_v2_train.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validation-only", action="store_true", help="keep test episodes sealed during model refinement")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate-only", type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU training/evaluation must run in an AMD compute allocation")
    device = torch.device("cuda")
    config = json.loads(args.config.read_text())
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not (args.resume or args.evaluate_only):
        raise FileExistsError(f"refusing to overwrite a training run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    index = json.loads((args.data / "campaign_index.json").read_text())
    data = np.load(args.data / "model_data.npz")
    if not index["complete"] or hashlib.sha256((args.data / "model_data.npz").read_bytes()).hexdigest() != index["model_data_sha256"]:
        raise RuntimeError("campaign is incomplete or packed data hash differs")
    states = torch.as_tensor(data["states"], device=device)
    lengths = torch.as_tensor(data["lengths"], device=device)
    contacts = torch.as_tensor(data["contacts"], device=device)
    splits = torch.as_tensor(data["splits"], device=device)
    train, val, test = [torch.nonzero(splits==k).flatten() for k in range(3)]
    all_train = states[train]
    valid_states = torch.arange(states.shape[1],device=device)[None,:] < lengths[train,None]
    reference = all_train[valid_states]
    normalization = {"mean": reference.mean(0).cpu().tolist(), "std": reference.std(0).clamp_min(0.01).cpu().tolist()}
    del reference, all_train
    scene = index["config"]["scene"]
    model_config = {**config["model"], "dt_s": index["model_dt_s"], "physics_step_s": index["config"]["simulation"]["step_s"],
                    "gravity_mps2": index["config"]["simulation"]["gravity_mps2"],
                    "floor_center_z_m": scene["ground_z_m"]+scene["radius_m"],
                    "wall_center_x_m": scene["wall_front_x_m"]-scene["radius_m"]}
    model = BouncingBallNRD(model_config, normalization).to(device)
    runtime = {"host": platform.node(), "job_id": os.environ.get("SLURM_JOB_ID"), "torch_version": torch.__version__,
               "hip_version": torch.version.hip, "gpu": torch.cuda.get_device_name(), "state_fields": STATE_FIELDS,
               "data_hash": index["model_data_sha256"], "model_source_hash": hashlib.sha256(Path(__file__).with_name("model.py").read_bytes()).hexdigest(),
               "trainer_source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "config": config}
    atomic_json(output / "run_config.json", runtime)
    if args.evaluate_only:
        model, _ = load_model(args.evaluate_only, device)
    elif args.resume:
        model, _ = load_model(args.resume, device)
    model_config = model.config
    normalization = {"mean": model.state_mean.cpu().tolist(), "std": model.state_std.cpu().tolist()}
    runtime["model_config"] = model_config
    runtime["loaded_checkpoint"] = str(args.resume or args.evaluate_only) if (args.resume or args.evaluate_only) else None
    atomic_json(output / "run_config.json",runtime)
    valid_transitions = torch.arange(states.shape[1]-1,device=device)[None,:] < (lengths-1)[:,None]
    train_mask = valid_transitions & (splits==0)[:,None]
    floor_pairs = torch.nonzero(train_mask & (contacts[...,0]>0))
    wall_pairs = torch.nonzero(train_mask & (contacts[...,1]>0))
    free_pairs = torch.nonzero(train_mask & (contacts.sum(-1)==0))
    optimizer = torch.optim.AdamW(model.parameters(),lr=config["training"]["learning_rate"],weight_decay=config["training"]["weight_decay"])
    scale = torch.tensor([0.05,0.05,0.5,1,2],device=device)
    rollout_scale = torch.tensor([0.05,0.05,0.15,0.15,0.5],device=device)
    best = float("inf")
    best_passed = False
    start = time.perf_counter()

    def checkpoint(path, update, validation):
        torch.save({"model_state_dict": model.state_dict(), "model_config": model_config,
                    "normalization": normalization, "runtime": runtime, "update": update,
                    "validation": validation}, path)

    def validation_check(update, stage, loss):
        nonlocal best, best_passed
        model.eval()
        validation = evaluate(model, states, lengths, contacts, val[:64] if args.smoke else val, config["evaluation"])
        if ((validation["passed"] and not best_passed) or
            (validation["passed"] == best_passed and validation["selection_score"] < best)):
            best = validation["selection_score"]
            best_passed = validation["passed"]
            checkpoint(output / "best.pt", update, validation)
        checkpoint(output / "last.pt", update, validation)
        atomic_json(output / "validation.json", validation)
        event = {"stage": stage, "update": update, "loss": float(loss), "elapsed_s": time.perf_counter()-start,
                 "validation_score": validation["selection_score"], "validation_passed": validation["passed"],
                 "position_p95_m": validation["position_rmse_m"]["p95"], "velocity_p95_mps": validation["velocity_mae_mps"]["p95"],
                 "spin_p95_radps": validation["spin_mae_radps"]["p95"], "contact_order_fraction": validation["contact_order_fraction"]}
        with (output / "train_log.jsonl").open("a") as log:log.write(json.dumps(event)+"\n")
        print(json.dumps(event),flush=True)
        model.train()
        return validation

    if not args.evaluate_only:
        converged = False
        updates = 100 if args.smoke else config["training"]["one_step_updates"]
        batch_size = min(config["training"]["batch_size"],256) if args.smoke else config["training"]["batch_size"]
        for update in range(1,updates+1):
            pairs = torch.cat([pool[torch.randint(len(pool),(count,),device=device)]
                               for pool,count in [(floor_pairs,batch_size*2//5),(wall_pairs,batch_size*2//5),(free_pairs,batch_size-2*(batch_size*2//5))]])
            current, target = states[pairs[:,0],pairs[:,1]], states[pairs[:,0],pairs[:,1]+1]
            prediction, logits = model.step_with_logits(current)
            loss = ((prediction-target)/scale).square().mean() + 0.05*F.binary_cross_entropy_with_logits(logits,contacts[pairs[:,0],pairs[:,1]])
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            if update % config["training"]["evaluate_every"] == 0 or update == updates:
                measured = validation_check(update,"one_step",loss.item())
                if measured["passed"] and config["training"].get("stop_when_passed",False) and not args.smoke:
                    converged = True
                    break
        # Starting from the best held-out full rollout avoids losing an already
        # useful model while trying to refine its impact timing.
        model, _ = load_model(output / "best.pt",device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=config["training"]["rollout_learning_rate"],weight_decay=0)
        rollout_updates = 2 if args.smoke else (0 if converged else config["training"]["rollout_updates"])
        for update in range(1,rollout_updates+1):
            chosen = train[torch.randint(len(train),(8 if args.smoke else config["training"]["rollout_batch_size"],),device=device)]
            current = states[chosen,0]
            total = torch.zeros((),device=device)
            n_steps = min(16,int(lengths[chosen].max())-1) if args.smoke else int(lengths[chosen].max())-1
            for step in range(n_steps):
                predicted = model(current)
                active = lengths[chosen]>step+1
                total = total + ((((predicted-states[chosen,step+1])/rollout_scale).square().mean(-1))*active).mean()
                current = torch.where(active[:,None],predicted,current)
            loss = total/n_steps
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("nonfinite rollout-training gradient")
            optimizer.step()
            if update % config["training"]["rollout_evaluate_every"] == 0 or update == rollout_updates:
                measured = validation_check(update,"full_rollout",loss.item())
                if measured["passed"] and config["training"].get("stop_when_passed",False) and not args.smoke:
                    break
    model, checkpoint_data = load_model(args.evaluate_only or output / "best.pt",device)
    model.eval()
    validation = evaluate(model,states,lengths,contacts,val[:64] if args.smoke else val,config["evaluation"])
    # Test is sealed until model selection is finished. It never selects a checkpoint.
    inspect_test = not (args.smoke or args.validation_only) and validation["passed"]
    testing = evaluate(model,states,lengths,contacts,test,config["evaluation"]) if inspect_test else None
    gradient_indices = test[:8] if inspect_test else val[:8]
    initial = states[gradient_indices,0].clone().requires_grad_(True)
    final = model.rollout(initial,170)[:,-1,:2]
    gradient = torch.autograd.grad(final.sum(),initial)[0]
    differentiability = {"finite": bool(torch.isfinite(gradient).all()), "initial_velocity_gradient_abs_max": float(gradient[:,2:4].abs().max())}
    result = {"complete": True, "smoke": args.smoke, "validation_only": args.validation_only,
              "test_evaluated": inspect_test, "validation": validation, "test": testing,
              "differentiability": differentiability, "runtime": runtime,
              "passed": validation["passed"] and (testing is None or testing["passed"]) and differentiability["finite"],
              "checkpoint": str((args.evaluate_only or output / "best.pt").resolve()),
              "checkpoint_sha256": hashlib.sha256((args.evaluate_only or output / "best.pt").read_bytes()).hexdigest(),
              "contact_calibration": checkpoint_data.get("contact_calibration"),
              "elapsed_s": time.perf_counter()-start}
    atomic_json(output / "evaluation.json",result)
    print(json.dumps({k:result[k] for k in ["complete","smoke","passed","elapsed_s","differentiability","checkpoint"]}),flush=True)
    if not args.smoke and not result["passed"]:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
