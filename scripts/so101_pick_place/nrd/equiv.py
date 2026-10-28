"""Rollouts and task metrics of one checkpoint with one code tree, for the original-vs-released check.

Per data folder (all its episodes): (1) a 100-step rollout from the first recorded state, (2) a 100-step rollout
from the recorded 4-state history at model step 100 (the grasp / carry part of the episode), both with the recorded
commands and with the contact switches; (3) data.evaluate(by_family=True), the task metrics of training and test.
Every tensor goes to OUT.pt, the evaluate reports to OUT.json. The code tree's own load_model, load_data and
step_average_rates are used.

    python -I equiv.py CODE_SRC_DIR CKPT DEVICE OUT_PREFIX NAME=DATA_DIR [NAME=DATA_DIR ...]

Release check of the SO101 pick-and-place study (so101_pick_place/README.md, "The NRD"; reproduce.sh steps
nrd_val, nrd_strip).
"""
import json
import sys
import time


def main():
    src, ckpt, device, out = sys.argv[1:5]
    sys.path.insert(0, src)
    import torch
    torch.set_num_threads(8)
    from nedm.contact_nrd import data as D, model as M
    model, packet = M.load_model(ckpt, device)
    kinds = set(model.config["integrate"]["kinds"])
    tensors, reports, info = {}, {}, dict(code=M.__file__, ckpt=ckpt, device=device, torch=torch.__version__)
    for item in sys.argv[5:]:
        name, root = item.split("=", 1)
        t0 = time.time()
        data = D.load_data(root, device, model_step_s=model.dt)
        data = D.step_average_rates(data, kinds)
        s = data["stride"]
        st, act = data["states"][:, ::s], data["actions"][:, ::s]
        with torch.no_grad():
            traj, gates = model.rollout(st[:, 0], act[:, :100], return_gates=True)
            k0 = 100
            hist = st[:, k0 - model.k + 1:k0 + 1]
            ok = torch.ones(hist.shape[:2], dtype=torch.bool, device=hist.device)
            traj2, gates2 = model.rollout_from(hist, ok, act[:, k0:k0 + 100], return_gates=True)
            rep = D.evaluate(model, data, torch.arange(len(data["lengths"]), device=device), by_family=True)
        tensors[name] = dict(first_state=traj.cpu(), first_state_gates=gates.cpu(), mid=traj2.cpu(), mid_gates=gates2.cpu())
        reports[name] = rep
        info[name] = dict(episodes=int(len(data["lengths"])), seconds=round(time.time() - t0, 1),
                          selection_score=rep.get("selection_score"))
        print(name, json.dumps(info[name]), flush=True)
    torch.save(tensors, out + ".pt")
    with open(out + ".json", "w") as f:
        json.dump(dict(info=info, reports=reports), f, default=float)


if __name__ == "__main__":
    main()
