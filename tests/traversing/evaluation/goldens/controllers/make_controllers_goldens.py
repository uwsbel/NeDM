#!/usr/bin/env python3
"""Goldens for tests/traversing/evaluation/test_controllers.py (A9 actor, A10 controller units), from the ORIGINAL code at
901d6c9 (/home/harry/NeDM-traverse_mppi/scripts/gc_control.py, read-only):
  actor64       gc_control.NumpyActor.act on the 64 observations of pr3 check_actors.py (rng 0: normal * sqrt(var) + mean),
                one observation per call (as a controller calls it) and as one batch
  hold_chain    gc_control.hold_clip over 2,000 uniform(-2, 2) commands, each clipped against the previous result
  flows         the collectors' per-frame controller logic replayed on a synthetic 120-frame drive along a released
                tracker route: pid_held and policy, rigid (gen_collect_ext.GenExt.command, gen_collect_ext.py:243-266)
                and soil (crm_collect_ext.py:217-296): held triples and the policy observations
Inputs of the flows: pose (T, 3) f64 near the route, pre (T, 17) f32 = the state captured before Synchronize,
rec (T, 17) f32 = the recorded state (pre with the non-observable columns 7-10 and 16 changed), shadow (T, 3) f64.
Writes controllers.npz and controllers.json next to this file (or into the folder given as the only argument)
byte-reproducibly; not a test module (the experiment checkout at 901d6c9 is read-only input):
  /home/harry/miniconda3/envs/nedm/bin/python tests/traversing/evaluation/goldens/controllers/make_controllers_goldens.py
"""
import hashlib, io, json, platform, sys, zipfile
from pathlib import Path

import numpy as np

REPO = Path('/home/harry/NeDM-traverse_mppi')
sys.dont_write_bytecode = True
sys.path[:0] = [str(REPO / 'scripts'), str(REPO / 'src')]
import gc_control as G  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
ACTOR = REPO / 'artifacts/traverse/generalist_20260921/B_tracker/ppo_v2/actor.npz'
ROUTE = 'artifacts/traverse/fdm_f104_50h_20260909/cases_night2/cases/routes/f104_v2_group_0048/route_03.json'
T = 120


def cpu():
    model = next((ln.split(':', 1)[1].strip() for ln in open('/proc/cpuinfo') if ln.startswith('model name')), '?')
    blas = np.show_config(mode='dicts')['Build Dependencies']['blas']
    return dict(cpu=model, numpy=np.__version__, blas=f"{blas.get('name')} {blas.get('version')}",
                python=platform.python_version())


def flows(actor, route, pose, pre, rec, shadow):
    out = {}
    for ground in ('rigid', 'soil'):
        for mode in ('pid_held', 'policy'):
            held, obs = [], []
            po = G.PolicyObs.from_meta(route, actor.meta) if mode == 'policy' else None
            last_cmd, prev = None, 0.0                                      # rigid: GenExt.last_cmd, previous_steer
            last_action = np.asarray(G.SETTLE_ACTION, np.float64)          # soil: crm_collect_ext last_action
            for k in range(T):
                if ground == 'rigid':
                    if mode == 'pid_held':
                        cmd = G.hold_clip(tuple(shadow[k]), prev)
                    else:
                        last = last_cmd if last_cmd is not None else G.SETTLE_ACTION
                        if k == 0:
                            po.reset(pre[k])
                        o = po.observe(pose[k], pre[k], last)
                        obs.append(o)
                        cmd = G.hold_clip(actor.act(o), prev)
                        po.push(pre[k], cmd)
                    last_cmd, prev = cmd, float(cmd[0])
                else:
                    if mode == 'pid_held':
                        raw = tuple(shadow[k])
                    else:
                        if k == 0:
                            po.reset(pre[k])
                        o = po.observe(pose[k], pre[k], last_action)
                        obs.append(o)
                        raw = actor.act(o)
                    cmd = G.hold_clip(raw, float(last_action[0]))
                    action = np.array(cmd, np.float32)                     # the recorded action[k]
                    last_action = action.astype(np.float64)
                    if po is not None:
                        po.push(rec[k], action)
                held.append(cmd)
            out[f'{ground}_{mode}_held'] = np.asarray(held, np.float64)
            if obs:
                out[f'{ground}_{mode}_obs'] = np.asarray(obs, np.float64)
    return out


def main():
    actor = G.NumpyActor.from_npz(ACTOR)
    rng = np.random.default_rng(0)                                         # check_actors.py
    obs64 = rng.normal(size=(64, actor.num_obs)) * np.sqrt(actor.obs_var) + actor.obs_mean
    arrays = dict(obs64=obs64, act64=np.stack([actor.act(o) for o in obs64]), act64_batch=actor.act(obs64))
    rng = np.random.default_rng(0)                                         # gc_control self-test (1)
    cmds, prev, held = rng.uniform(-2, 2, (2000, 3)), 0.0, []
    for c in cmds:
        held.append(G.hold_clip(c, prev))
        prev = held[-1][0]
    arrays.update(hold_cmds=cmds, hold_held=np.asarray(held))
    route = json.loads((REPO / ROUTE).read_text())
    route = {k: np.asarray(route[k], float) for k in ('waypoints', 'speeds', 'stations', 'headings')}
    rng = np.random.default_rng(7)
    xy, hd = route['waypoints'], route['headings']
    i = np.minimum((np.arange(T) * 0.8).astype(int), len(xy) - 1)
    pose = np.stack([xy[i, 0] + rng.normal(0, .4, T), xy[i, 1] + rng.normal(0, .4, T), hd[i] + rng.normal(0, .15, T)], 1)
    pre = (rng.normal(0, 1, (T, 17)) * [2, .5, .1, .1, .2, .2, .3, 3e3, 3e3, 3e3, 3e3, 8, 8, 8, 8, 150, 300]
           + [2, 0, 0, 0, 0, 0, 0, 6e3, 6e3, 6e3, 6e3, 5, 5, 5, 5, 200, 100]).astype(np.float32)
    rec = pre.copy()
    rec[:, [7, 8, 9, 10, 16]] += rng.normal(0, 50, (T, 5)).astype(np.float32)   # tyre loads and torque differ after sync
    shadow = np.c_[rng.uniform(-.8, .8, T), rng.uniform(0, 1, T), rng.uniform(0, 1, T) * (rng.uniform(size=T) < .2)]
    arrays.update(pose=pose, pre=pre, rec=rec, shadow=shadow, **{f'route_{k}': v for k, v in route.items()},
                  **flows(actor, route, pose, pre, rec, shadow))
    po = G.PolicyObs.from_meta(route, actor.meta)
    meta = dict(source='gc_control.py at 901d6c9', gc_control_sha256=hashlib.sha256((REPO / 'scripts/gc_control.py')
                .read_bytes()).hexdigest(), actor=str(ACTOR.relative_to(REPO)), actor_sha256=hashlib.sha256(
                ACTOR.read_bytes()).hexdigest(), route=ROUTE, frames=T, machine=cpu(), layout=po.layout(),
                keys=sorted(arrays))
    OUT.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:              # fixed timestamps: byte-reproducible
        for k in sorted(arrays):
            b = io.BytesIO()
            np.lib.format.write_array(b, np.ascontiguousarray(arrays[k]), allow_pickle=False)
            z.writestr(zipfile.ZipInfo(f'{k}.npy', (1980, 1, 1, 0, 0, 0)), b.getvalue())
    (OUT / 'controllers.npz').write_bytes(buf.getvalue())
    (OUT / 'controllers.json').write_text(json.dumps(meta, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: list(v.shape) for k, v in arrays.items()}), (OUT / 'controllers.npz').stat().st_size)


if __name__ == '__main__':
    main()
