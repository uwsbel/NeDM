"""NRD data round 2 of the SO101 policy study (campaign expert_v2): tasks, closed-loop MuJoCo rollouts of the expert,
the collector's labels, the data directories, checks, and the NRD evaluation on them. Module docstring of
``nedm.so101_policy.data_round`` for the design; report ``~/so101_policy/docs/build3_round2.md``.

    PYTHONPATH=src python -P scripts/so101_policy/collect_round.py tasks --set campaign --n 13000 --out tasks.npz \\
        [--shard K/M] [--workers W]                     # task table (starts, goals, options); shards concatenate
    PYTHONPATH=src python -P scripts/so101_policy/collect_round.py run --tasks tasks.npz --rows A:B --workers 10 \\
        --out DIR [--python WORKER_PY] [--batch 1200]    # pass 1: closed loop in CPU MuJoCo -> DIR/ro_*.npz
    PYTHONPATH=src python -P scripts/so101_policy/collect_round.py label --rollouts DIR/ro_*.npz --workers 32
                                                          # pass 2: collector labels -> lab_*.pkl next to each file
    PYTHONPATH=src python -P scripts/so101_policy/collect_round.py write --labelled 'DIR/lab_*.pkl' --out DATA \\
        [--gate2 'DIR2/lab_*.pkl'] [--n-val 1000]        # DATA (training, chunked), DATA/val_full, DATA/gate2
    PYTHONPATH=src python -P scripts/so101_policy/collect_round.py verify --data DIR --master 20261021 --rows ... \\
        --out JSON [--python WORKER_PY]                 # stored collector episodes through this writer path, bit-exact
    PYTHONPATH=src python -P scripts/so101_policy/collect_round.py eval --data DATA/val_full \\
        --model s61=CKPT s62=CKPT --out JSON [--carry]  # NRD accuracy per family (lift, drop, 1 s, whole episode,
                                                          # --carry: cube error while MuJoCo holds the cube)

Round 3 (campaign policy_v3: MuJoCo rollouts of PPO policies; data_round.py section "round 3", report
~/so101_policy/docs/build6_round3.md):

    ... collect_round.py ptasks --set campaign --n 13000 --out tasks.npz [--workers W]   # also --set pgate --n 512
    ... collect_round.py prun --tasks tasks.npz [--expand-pgate] [--policies w5a kCa ...] --workers 10 \\
        --python WORKER_PY --out DIR [--batch 1000] [--policy-dir D] [--geom G --kin-points K]  # pass 1 -> DIR/ro_*.npz
    ... collect_round.py label --rollouts 'DIR/ro_*.npz' --workers 32                  # pass 2 (unchanged)
    ... collect_round.py write --campaign policy_v3 --labelled 'DIR/lab_*.pkl' --out DATA \\
        --gate2 'DIR2/lab_*.pkl' --gate-name pgate [--n-val 1000]                       # DATA, DATA/val_full, DATA/pgate

``tasks`` needs torch and mujoco (+ the overlap cache or scipy); ``run`` torch in the main process and numpy + mujoco
in the workers; ``label`` and ``write`` only numpy + mujoco (use an interpreter with scipy, e.g. conda env nrdsim or
the AMD venv simsrc_mj: the collector's visual-overlap values then match bit for bit); ``eval`` torch and a GPU.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import re
import time
from pathlib import Path

import numpy as np

from nedm.so101_policy import data_round as R


# ---- tasks ----------------------------------------------------------------------------------------------------------
def _task_shard(args):
    n, set_name, idx = args
    return R.make_tasks(n, set_name, idx=idx, log=lambda *a, **k: None)


def cmd_tasks(a):
    n = a.n
    idx = np.arange(n)
    if a.shard:
        k, m = (int(x) for x in a.shard.split("/"))
        idx = np.array_split(idx, m)[k]
    t0 = time.time()
    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        parts = np.array_split(idx, a.workers * 4)
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
            tables = list(pool.map(_task_shard, [(n, a.set, p) for p in parts if len(p)]))
        out = R.concat_tasks(tables)
    else:
        out = R.make_tasks(n, a.set, idx=idx)
    np.savez(a.out, **out)
    fam = out["family"]
    d = np.linalg.norm(out["goal_xy"] - out["state_cube_pos"][:, :2], axis=1)
    print(f"{len(idx)} tasks of set {a.set} (n {n}; start seeds {out['seeds'].min()}-{out['seeds'].max()}) -> {a.out} "
          f"in {time.time() - t0:.0f} s; families {np.bincount(fam, minlength=len(R.FAMILIES)).tolist()}; redraws "
          f"{int((out['tries'] > 1).sum())}; goal distance p50 {np.median(d) * 1e3:.0f} mm ({d.min() * 1e3:.0f}-"
          f"{d.max() * 1e3:.0f}); goal2 missing {int(np.isnan(out['goal2_xy']).any(1).sum())}", flush=True)


# ---- pass 1 ---------------------------------------------------------------------------------------------------------
def cmd_run(a):
    import torch
    torch.set_num_threads(a.threads)
    t = dict(np.load(a.tasks))
    lo, hi = (int(x) for x in a.rows.split(":")) if a.rows else (0, len(t["seeds"]))
    rows = np.arange(lo, min(hi, len(t["seeds"])))
    if a.stride:                                   # K/M: every M-th task from the K-th (a mix of all families)
        k, m = (int(x) for x in a.stride.split("/"))
        rows, lo = rows[k::m], k
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dec_cfg, ex_cfg = json.loads(a.decoder_cfg), json.loads(a.expert_cfg)
    speed = np.asarray(t["speed"])[rows]
    for sp in np.unique(speed):
        sel = rows[speed == sp]
        for b0 in range(0, len(sel), a.batch):
            part = sel[b0: b0 + a.batch]
            path = out / f"ro_{a.tag}{lo:06d}_s{sp}_{b0 // a.batch:02d}.npz"
            if path.exists():
                print(f"{path} exists: skipped", flush=True)
                continue
            tk = R.subset(t, part)
            print(f"run rows {part[0]}..{part[-1]} ({len(part)} tasks, speed class {sp}) -> {path}", flush=True)
            ro = R.run_rollouts(tk, workers=a.workers, python=a.python, speed=int(sp), decoder_overrides=dec_cfg,
                                expert_overrides=ex_cfg, log=print)
            ro["task_rows"] = part
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, **ro)
            tmp.replace(path)
            fam = ro["family"]
            print(f"  {len(part)} episodes, {ro['n_steps'].sum()} steps in {float(ro['wall_s']):.0f} s; expert success "
                  f"{ro['success'].mean() * 100:.1f} %, lifted {ro['lifted'].mean() * 100:.1f} %, env ends "
                  f"{np.bincount(ro['env_end'], minlength=8).tolist()}, cube moves {(ro['dist_step'] >= 0).sum()} of "
                  f"{(ro['dist_r'] > 0).sum()}; per family success "
                  f"{[round(float(ro['success'][fam == f].mean()) * 100, 1) if (fam == f).any() else None for f in range(len(R.FAMILIES))]}",
                  flush=True)


# ---- round 3: policy task tables and pass 1 ----------------------------------------------------------------------------
def _ptask_shard(args):
    n, set_name, idx = args
    return R.make_policy_tasks(n, set_name, idx=idx, log=lambda *a, **k: None)


def cmd_ptasks(a):
    n, idx = a.n, np.arange(a.n)
    t0 = time.time()
    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        parts = np.array_split(idx, a.workers * 4)
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
            out = R.concat_tasks(list(pool.map(_ptask_shard, [(n, a.set, p) for p in parts if len(p)])))
    else:
        out = R.make_policy_tasks(n, a.set, idx=idx)
    np.savez(a.out, **out)
    d = np.linalg.norm(out["goal_xy"] - out["state_cube_pos"][:, :2], axis=1)
    cells = {f"{R.POLICY_IDS[p]}/{R.NOISE_LEVELS[l]:g}": int(((out["policy"] == p) & (out["noise_level"] == l)).sum())
             for p in np.unique(out["policy"]) if p >= 0 for l in range(len(R.NOISE_LEVELS))}
    print(f"{n} tasks of set {a.set} (start seeds {out['seeds'].min()}-{out['seeds'].max()}) -> {a.out} in "
          f"{time.time() - t0:.0f} s; redraws {int((out['tries'] > 1).sum())}; spread 60 deg "
          f"{int((out['spread_deg'] == 60).sum())}; goal distance p50 {np.median(d) * 1e3:.0f} mm ({d.min() * 1e3:.0f}-"
          f"{d.max() * 1e3:.0f}); cells {cells}", flush=True)


def cmd_prun(a):
    import torch
    torch.set_num_threads(a.threads)
    t = dict(np.load(a.tasks))
    if a.expand_pgate:
        t = R.expand_pgate(t)
    lo, hi = (int(x) for x in a.rows.split(":")) if a.rows else (0, len(t["seeds"]))
    rows = np.arange(lo, min(hi, len(t["seeds"])))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = dict(geom_path=a.geom, kin_points=a.kin_points)
    pol_rows = np.asarray(t["policy"])[rows]
    if (pol_rows < 0).any():
        raise SystemExit("rows without a policy (a pgate task table needs --expand-pgate)")
    want = [R.POLICY_IDS.index(p) for p in a.policies] if a.policies else np.unique(pol_rows).tolist()
    for p in want:
        sel = rows[pol_rows == p]
        path = R.policy_path(p, a.policy_dir)
        for b0 in range(0, len(sel), a.batch):
            part = sel[b0: b0 + a.batch]
            f = out / f"ro_{a.tag}{R.POLICY_IDS[p]}_{lo:06d}_{b0 // a.batch:02d}.npz"
            if f.exists():
                print(f"{f} exists: skipped", flush=True)
                continue
            tk = R.subset(t, part)
            print(f"prun {R.POLICY_IDS[p]} ({path.name}) rows {part[0]}..{part[-1]} ({len(part)} tasks) -> {f}", flush=True)
            ro = R.run_policy_rollouts(tk, path, workers=a.workers, python=a.python, files=files, horizon=a.horizon,
                                       log=print)
            ro["task_rows"] = part
            tmp = f.with_suffix(".tmp.npz")
            np.savez(tmp, **ro)
            tmp.replace(f)
            sc = ro["noise_scale"]
            print(f"  {len(part)} episodes, {ro['n_steps'].sum()} steps, {int(ro['decisions'])} decisions in "
                  f"{float(ro['wall_s']):.0f} s; success per noise scale "
                  f"{ {float(s): round(float(ro['success'][sc == s].mean()) * 100, 1) for s in np.unique(sc)} } %, lifted "
                  f"{ro['lifted'].mean() * 100:.1f} %, env ends {np.bincount(ro['env_end'], minlength=8).tolist()}, task "
                  f"ends {np.bincount(ro['task_end'].astype(int), minlength=18).tolist()}", flush=True)


# ---- pass 2 ---------------------------------------------------------------------------------------------------------
META_RO = ("family", "index", "seeds", "tries", "task_end", "success", "success10", "lifted", "end_xy", "end_yaw",
           "n_steps", "env_end", "env_end_step", "noise", "sigma", "rt1", "rt2", "rt_step1", "rt_step2", "dist_step",
           "dist_r", "speed", "recoveries", "noisy_n", "pause_s", "hold_s", "lift_h", "spread_deg", "horizon",
           "harness_grasp", "harness_place", "task_rows", "decoder_overrides", "expert_overrides",
           # round 3 (policy rollouts)
           "policy", "noise_scale", "noise_level", "task", "tail", "task_steps", "ret", "length", "grasp_started",
           "place_started", "aborted", "gate_blocked", "policy_path", "policy_sha256", "geometry_files", "campaign",
           "decisions", "noise_salt", "task_end_names")


def cmd_label(a):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    files = sorted(set(sum((glob.glob(p) for p in a.rollouts), [])))
    pool = ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) if a.workers > 1 else None
    try:
        _label_files(files, pool)
    finally:
        if pool is not None:
            pool.shutdown()


def _label_files(files, pool):
    for f in files:
        out = Path(f).with_name(Path(f).name.replace("ro_", "lab_", 1)).with_suffix(".pkl")
        if out.exists():
            print(f"{out} exists: skipped", flush=True)
            continue
        t0 = time.time()
        segs = R.label_rollouts(f, pool=pool, log=print)
        with np.load(f) as z:
            meta = {k: z[k] for k in META_RO if k in z.files}
            wall1 = float(z["wall_s"])
        chk = [s.get("check") for ep in segs for s in ep if "check" in s]
        bad = [k for k, ep in enumerate(segs) if any(("rejected" in s) or not all(s["check"].values()) for s in ep)]
        rec = dict(source=str(f), meta=meta, segments=segs, bad=bad, wall_pass1_s=wall1, wall_pass2_s=time.time() - t0)
        tmp = out.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(rec, fh, protocol=4)
        tmp.replace(out)
        print(f"{out}: {len(segs)} episodes, {len(chk)} segments; check states {sum(c['states'] for c in chk)}, "
              f"contacts {sum(c['contacts'] for c in chk)}, end {sum(c['end'] for c in chk)} of {len(chk)}; episodes "
              f"dropped {len(bad)}; {time.time() - t0:.0f} s", flush=True)


# ---- write ----------------------------------------------------------------------------------------------------------
def _load_labelled(patterns):
    files = sorted(set(sum((glob.glob(p) for p in patterns), [])))
    if not files:
        raise SystemExit(f"no labelled files for {patterns}")
    recs = []
    for f in files:
        with open(f, "rb") as fh:
            recs.append(pickle.load(fh))
    return files, recs


def _parents(recs):
    """[(record index, episode k, parent index in the set, family)] of the episodes that passed every check."""
    out, dropped = [], []
    for i, r in enumerate(recs):
        bad = set(r["bad"])
        for k in range(len(r["segments"])):
            p = (i, k, int(r["meta"]["index"][k]), int(r["meta"]["family"][k]))
            (dropped if k in bad else out).append(p)
    out.sort(key=lambda x: x[2])
    return out, dropped


def _build(recs, parents, offset, chunked, records, campaign=R.CAMPAIGN):
    rows_of = R.policy_rows_of if campaign == R.P_CAMPAIGN else R.rows_of
    rows = []
    for i, k, pid, _ in parents:
        rows += rows_of(recs[i]["meta"], k, recs[i]["segments"][k], offset, pid, chunked=chunked, records=records)
    return rows


def _manifest(set_name, files, recs, parents, dropped, extra=None, campaign=R.CAMPAIGN):
    meta = {k: np.concatenate([r["meta"][k] for r in recs]) for k in ("family", "success", "lifted", "env_end",
                                                                       "task_end", "n_steps", "dist_step", "dist_r")}
    fams = meta["family"]
    chk = [s["check"] for r in recs for ep in r["segments"] for s in ep if "check" in s]
    pol = campaign == R.P_CAMPAIGN
    names = R.P_FAMILY_NAMES if pol else R.FAMILY_NAMES
    out = dict(
        collector="nedm.so101_policy.data_round (scripts/so101_policy/collect_round.py)", campaign=campaign,
        set=set_name, master_seed=R.P_MASTER_SEED if pol else R.MASTER_SEED,
        start_seeds=dict(campaign=[R.P_START0, R.P_START0 + 99_999], pgate=[R.PGATE_START0, R.PGATE_START0 + 9_999],
                         rule="start seed of task i = start0 + i (mujoco_env.sample_starts)") if pol else
        dict(campaign=[R.START0, R.START0 + 99_999], gate2=[R.GATE2_START0, R.GATE2_START0 + 99_999],
             rule="start seed of parent i = start0 + i (mujoco_env.sample_starts)"),
        families=[dict(name=f["name"], code=f["code"]) for f in R.POLICY_FAMILIES] if pol else
        [dict(name=f["name"], code=f["code"], share=f["share"], horizon_steps=f["horizon"]) for f in R.FAMILIES],
        parents=len(parents) + len(dropped), parents_kept=len(parents),
        dropped=[dict(file=Path(files[i]).name, episode=k, parent=pid, family=names[f]) for i, k, pid, f in dropped],
        **{"policy_mujoco" if pol else "expert_mujoco": dict(
            success15={names[f]: float(meta["success"][fams == f].mean()) for f in np.unique(fams)},
            lifted={names[f]: float(meta["lifted"][fams == f].mean()) for f in np.unique(fams)},
            env_ends=np.bincount(meta["env_end"].astype(int), minlength=8).tolist(),
            cube_moved=int((meta["dist_step"] >= 0).sum()), cube_move_planned=int((meta["dist_r"] > 0).sum()),
            steps=int(meta["n_steps"].sum()))},
        pass1_vs_pass2=dict(segments=len(chk), states_equal=sum(c["states"] for c in chk),
                            contacts_equal=sum(c["contacts"] for c in chk), end_equal=sum(c["end"] for c in chk)),
        wall_s=dict(pass1=float(sum(r["wall_pass1_s"] for r in recs)), pass2=float(sum(r["wall_pass2_s"] for r in recs))),
        sources=dict(labelled=[Path(f).name for f in files], code_sha256=R.source_hashes()),
        decoder_overrides=sorted({str(r["meta"].get("decoder_overrides", "{}")) for r in recs}),
        **(extra or {}))
    if pol:   # per policy and noise level: episodes, task success, lifted, task ends (MuJoCo, evaluation rules)
        ENDS = json.loads(str(recs[0]["meta"]["task_end_names"]))          # task.ENDS (no torch import here)
        pm = {k: np.concatenate([r["meta"][k] for r in recs]) for k in ("policy", "noise_scale", "task_end", "success",
                                                                          "lifted", "n_steps")}
        out["policies"] = [dict(id=p["id"], file=p["file"], source=p["source"], family=R.P_FAMILY_NAMES[p["family"]])
                           for p in R.POLICIES]
        out["policy_sha256"] = sorted({str(r["meta"].get("policy_sha256", "")) for r in recs})
        cells = {}
        for p in np.unique(pm["policy"]):
            for s_ in np.unique(pm["noise_scale"]):
                sel = (pm["policy"] == p) & (pm["noise_scale"] == s_)
                if sel.any():
                    ends = pm["task_end"][sel].astype(int)
                    cells[f"{R.POLICY_IDS[int(p)]}/{s_:g}"] = dict(
                        episodes=int(sel.sum()), success15=float(pm["success"][sel].mean()),
                        lifted=float(pm["lifted"][sel].mean()), steps=int(pm["n_steps"][sel].sum()),
                        task_ends={ENDS[e]: int((ends == e).sum()) for e in np.unique(ends)})
        out["policy_cells"] = cells
    return out


def _validate(d):
    from nedm.contact_nrd.collect import validate
    rep = validate.validate(d)
    (Path(d) / "validate.json").write_text(json.dumps(rep, indent=1, default=float))
    print(f"validate {d}: ok {rep['ok']}, fails {rep['fails']}, warns {len(rep['warns'])}", flush=True)
    if not rep["ok"]:
        raise SystemExit(f"{d} failed validation: {rep['fails']}")
    return rep


def cmd_write(a):
    camp = a.campaign
    pol = camp == R.P_CAMPAIGN
    system = R.round_system(campaign=camp)
    off = R.family_offset(system, camp)
    meta_keys = R.P_META_KEYS if pol else R.META_KEYS
    tail_name = R.POLICY_TAIL["name"] if pol else R.TAIL["name"]
    files, recs = _load_labelled(a.labelled)
    parents, dropped = _parents(recs)
    fam = np.array([p[3] for p in parents])
    pid = np.array([p[2] for p in parents])
    psplit = R.parent_splits(fam, pid, a.n_val, seed=R.P_MASTER_SEED if pol else R.MASTER_SEED)
    split_of = dict(zip(pid.tolist(), psplit.tolist()))
    out = Path(a.out)
    # training folder: chunks of <= 601 records, splits of the parents
    rows = _build(recs, parents, off, True, R.RECORDS, camp)
    sp = np.array([split_of[r["meta"]["parent"]] for r in rows], np.int8)
    man = _manifest("campaign", files, recs, parents, dropped, dict(
        layout="training: segments cut into chunks of <= 601 records at records 0, 492, 984 (data_round.chunk_starts); "
               f"later chunks in family {tail_name}; episodes.npz = per-row parent, segment, chunk, task outcome",
        validation_parents=int((psplit == 1).sum()), training_parents=int((psplit == 0).sum())), campaign=camp)
    info = R.write_round(out, rows, sp, system, R.RECORDS, man, meta_keys)
    print(f"{out}: {info['episodes']} rows, splits {info['split_counts']}", flush=True)
    _validate(out)
    # whole validation segments (no chunks) for whole-episode evaluation
    val = [p for p in parents if split_of[p[2]] == 1]
    rows = _build(recs, val, off, False, None, camp)
    rec_n = max(len(r["action"]) for r in rows)
    man = _manifest("campaign_val_full", files, recs, val, [], dict(
        layout="the validation parents of the training folder (split 1), one row per segment at full length "
               f"(records_per_episode {rec_n}); NOT for training (merge only the parent folder)"), campaign=camp)
    info = R.write_round(out / "val_full", rows, np.ones(len(rows), np.int8), system, rec_n, man, meta_keys)
    print(f"{out / 'val_full'}: {info['episodes']} rows (records {rec_n})", flush=True)
    _validate(out / "val_full")
    if a.gate2:
        name = a.gate_name or ("pgate" if pol else "gate2")
        files2, recs2 = _load_labelled(a.gate2)
        par2, drop2 = _parents(recs2)
        rows = _build(recs2, par2, off, False, None, camp)
        rec_n = max(len(r["action"]) for r in rows)
        start0 = R.PGATE_START0 if pol else R.GATE2_START0
        man = _manifest(name, files2, recs2, par2, drop2, dict(
            layout=f"{name}: a held-out set (start seeds {start0}+), all split 1, one row per segment at full "
                   f"length (records_per_episode {rec_n}); tasks file tasks_{name}.npz (gate_expert.py format)"
                   + ("; every policy on every task, deterministic and at 1.0 x its std (data_round.expand_pgate: "
                      "parent = (policy * 2 + level) * n_tasks + task)" if pol else "")), campaign=camp)
        info = R.write_round(out / name, rows, np.ones(len(rows), np.int8), system, rec_n, man, meta_keys)
        print(f"{out / name}: {info['episodes']} rows (records {rec_n})", flush=True)
        _validate(out / name)


# ---- verify: stored collector episodes through this writer path -----------------------------------------------------
def _regenerate(job):
    """(master, family, family_index, seed, arm0, cube0) -> collector plan start of a stored episode (bench_mujoco_env
    regenerate, any master seed)."""
    from nedm.contact_nrd.collect import export
    from nedm.contact_nrd.collect import so101_grasp as g
    master, fam, index, seed, arm0, cube0 = job
    s = R.scene()
    cfg_fam = s.cfg["families"][fam]
    if g.episode_seed(master, fam, index) != seed:
        raise ValueError(f"seed of family {fam} index {index} is not from master {master}")
    rng = np.random.default_rng(seed)
    n_rec = int(cfg_fam["records"])
    n_ctrl = (n_rec - 1) // s.per_ctrl + 1
    for tries in range(1, 201):
        try:
            plan = g.FAMILIES[cfg_fam["name"]](s, rng, n_ctrl)
        except g.PlanError:
            continue
        c = plan["cube"]
        s.reset(plan["q0"], c["pos"], c["quat"], c.get("vel"), c.get("omega"), c["spawn"])
        arm, cube = s.state()
        cube[3:7] = export.quat_continuous(cube[None, 3:7])[0]
        if np.array_equal(arm.astype(np.float32), arm0) and np.array_equal(cube.astype(np.float32), cube0):
            if c["spawn"]:
                raise ValueError("spawned (probe) episodes are not task episodes")
            return dict(q=np.asarray(plan["q0"], float), pos=np.asarray(c["pos"], float),
                        quat=np.asarray(c["quat"], float), tries=tries)
        if not isinstance(g.run_episode(s, plan, n_rec), str):
            raise RuntimeError(f"family {fam} index {index}: the kept plan does not give the stored first record")
    raise RuntimeError(f"family {fam} index {index}: no plan")


def cmd_verify(a):
    """Stored collector episodes (task families: warm-up starts, no spawn) through this writer path: pass 1 = their
    commands in MujocoWorlds (records logged as in run_rollouts), pass 2 = label_episode, rows_of + pack_episodes;
    every contract array compared bit for bit with the stored one."""
    import torch

    from nedm.contact_nrd.collect import export
    from nedm.contact_nrd.collect import so101_grasp as g
    from nedm.contact_nrd.rollouts import npz_members
    from nedm.so101_policy.mujoco_env import MujocoWorlds
    torch.set_num_threads(a.threads)
    root = Path(a.data)
    st = npz_members(root / "unified_data.npz")
    system = json.loads((root / "system.json").read_text())
    fam_all, split = np.asarray(st["family"]), np.asarray(st["splits"])
    if (split == 2).any():
        raise SystemExit("verify reads training / validation data only")
    names = [f["name"] for f in system["families"]]
    keep = [names.index(f) for f in a.families]
    rows = []
    for f in keep:
        cand = np.flatnonzero(fam_all == f)
        rows += cand[np.linspace(0, len(cand) - 1, min(len(cand), a.per_family)).round().astype(int)].tolist()
    term = np.asarray(st["terminal_reason"])
    if a.terminal:
        cand = np.flatnonzero(np.isin(fam_all, keep) & (term != 0))
        rows += cand[: a.terminal].tolist()
    rows = np.array(sorted(set(rows)))
    stored = {k: np.asarray(st[k])[rows] for k in R.STORED_KEYS + R.STORED_SCALARS +
              ("lengths", "family", "family_index", "seed")}
    jobs = [(a.master, int(stored["family"][j]), int(stored["family_index"][j]), int(stored["seed"][j]),
             stored["arm"][j, 0], stored["cube"][j, 0]) for j in range(len(rows))]
    t0 = time.time()
    starts = [_regenerate(j) for j in jobs]
    print(f"{len(rows)} stored episodes (rows {rows[:6].tolist()}...), starts regenerated in {time.time() - t0:.0f} s",
          flush=True)
    E = len(rows)
    L = stored["lengths"].astype(int)
    S = (L - 1) // 2                                         # control steps covered by the stored records
    cmd = np.zeros((E, S.max() + 1, 6), np.float32)
    phs = np.zeros((E, S.max() + 1), np.int8)
    for j in range(E):
        c = stored["action"][j, : L[j]][::2]
        cmd[j, : len(c)] = c
        phs[j, : len(c)] = stored["phase"][j, : L[j]][::2]
        if len(c) < S[j] + 1:                                # odd stored length: the last command repeats
            cmd[j, len(c):], phs[j, len(c):] = c[-1], phs[j, len(c) - 1]
    world = MujocoWorlds(E, workers=a.workers, python=a.python)
    try:
        world.reset(None, dict(q=np.stack([s["q"] for s in starts]), cube_pos=np.stack([s["pos"] for s in starts]),
                               cube_quat=np.stack([s["quat"] for s in starts])), warmup=True)
        T = int(S.max()) + 1
        rec = np.zeros((E, 2 * T + 1, 25), np.float32)
        rcon = np.zeros((E, 2 * T, 3), bool)
        rec[:, 0] = world.raw_state().numpy().astype(np.float32)
        for t in range(T):
            world.step(torch.as_tensor(cmd[:, min(t, cmd.shape[1] - 1)], dtype=torch.float64))
            r = world.records.numpy().astype(np.float32)
            rec[:, 2 * t + 1], rec[:, 2 * t + 2] = r[:, 0], r[:, 1]
            rcon[:, 2 * t: 2 * t + 2] = world.record_contact.numpy()
        es, er = world.end_step.numpy(), world.end_reason.numpy()
    finally:
        world.close()
    s = R.scene()
    results, eps = [], []
    for j in range(E):
        n_steps = int(S[j]) + (1 if term[rows[j]] != 0 else 0)  # a terminal episode: the fatal step is after the records
        n_steps = min(n_steps, T)
        env_end = int(er[j]) if 1 <= es[j] <= n_steps else 0
        ep = dict(q0=starts[j]["q"], cube_pos=starts[j]["pos"], cube_quat=starts[j]["quat"],
                  commands=cmd[j, : n_steps + 1] if n_steps + 1 <= cmd.shape[1] else
                  np.concatenate((cmd[j], cmd[j, -1:].repeat(n_steps + 1 - cmd.shape[1], 0))),
                  phases=phs[j, : n_steps + 1] if n_steps + 1 <= phs.shape[1] else
                  np.concatenate((phs[j], phs[j, -1:].repeat(n_steps + 1 - phs.shape[1], 0))), n_steps=n_steps, events=[])
        segs = R.label_episode(s, ep, dict(records=rec[j, : 2 * n_steps + 1], record_contact=rcon[j, : 2 * n_steps],
                                           jump_raw={}, env_end=env_end))
        if len(segs) != 1 or "rejected" in segs[0]:
            results.append(dict(row=int(rows[j]), ok=False, error=str(segs)))
            continue
        r = segs[0]
        r["outcome"] = g.outcome(r, s.half, system["families"][int(stored["family"][j])]["group"])
        ro = dict(family=[0], index=[int(stored["family_index"][j])], seeds=[int(stored["seed"][j])], tries=[1],
                  task_end=[0], success=[False], success10=[False], lifted=[False], end_xy=[0.0], end_yaw=[0.0],
                  n_steps=[n_steps], env_end=[env_end], noise=[0], sigma=[0.0], rt1=[-1], rt2=[-1], rt_step1=[-1],
                  rt_step2=[-1], dist_step=[-1], speed=[1], recoveries=[0], noisy_n=[0], pause_s=[0.0], hold_s=[0.0])
        row = R.rows_of(ro, 0, [r], 0, j, chunked=False, records=None)[0]
        eps.append(row)
        packed = export.pack_episodes([row], dict(system, records_per_episode=len(row["action"])))
        packed["lengths"] = np.asarray(packed["lengths"])
        cmp = R.compare_stored(stored, packed, j, 0)
        p1 = dict(states=bool(np.array_equal(rec[j, : L[j], :12], stored["arm"][j, : L[j]])
                              and np.array_equal(rec[j, : L[j], 12:], stored["cube"][j, : L[j]])),
                  contacts=bool(np.array_equal(rcon[j, : L[j] - 1], stored["contacts"][j, : L[j] - 1].astype(bool))))
        results.append(dict(row=int(rows[j]), family=names[int(stored["family"][j])], length=int(L[j]),
                            terminal=int(term[rows[j]]), ok=all(cmp.values()), arrays=cmp, check=r["check"],
                            pass1_records_equal_stored=p1))
    ok = sum(r["ok"] for r in results)
    summary = dict(data=str(root), master=a.master, episodes=len(results), all_arrays_bit_equal=ok,
                   per_array={k: sum(r.get("arrays", {}).get(k, False) for r in results) for k in
                              ("lengths",) + R.STORED_KEYS + R.STORED_SCALARS},
                   pass1_records_equal_stored=sum(r.get("pass1_records_equal_stored", {}).get("states", False)
                                                  for r in results),
                   pass1_vs_pass2_check=sum(all(r.get("check", {}).values()) and bool(r.get("check")) for r in results),
                   by_family={f: [sum(r["ok"] for r in results if r.get("family") == f),
                                  sum(1 for r in results if r.get("family") == f)] for f in a.families},
                   terminal_episodes=[sum(r["ok"] for r in results if r.get("terminal")),
                                      sum(1 for r in results if r.get("terminal"))],
                   wall_s=time.time() - t0)
    Path(a.out).write_text(json.dumps(dict(summary=summary, episodes=results), indent=1))
    print(json.dumps(summary, indent=1))


# ---- eval: NRD checkpoints on an expert_v2 split ------------------------------------------------------------------
def cmd_eval(a):
    """Per family (and noise phase / variant group) of one split: lift and drop flag agreement (free rollout from the
    first record vs the record; evaluation block flags: lifted = cube z > 25 mm with both jaw switches on and the table
    off, dropped = cube z > 25 mm with both jaw switches off), 1 s window cube position p95 (recorded 4-state histories
    every 0.5 s), free-rollout cube position error at 4 s and at the last record (whole episode: p50 / p90 / p95,
    share > 20 mm), by data.evaluate (the data's evaluation block)."""
    import importlib.util

    import torch

    from nedm.contact_nrd.data import (add_derived, add_relative, evaluate, file_rows, file_sha256, load_data,
                                       step_average_rates)
    from nedm.contact_nrd.model import load_model
    spec = importlib.util.spec_from_file_location("eval_nrd_noisy", str(R.REPO_ROOT / "scripts/so101_policy/eval_nrd_noisy.py"))
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    if a.split not in (0, 1):
        raise SystemExit("split 0 or 1 only")
    root = Path(a.data).expanduser()
    models = {}
    for item in a.model:
        name, path = item.split("=", 1)
        path = Path(path).expanduser()
        path = path / "best.pt" if path.is_dir() else path
        models[name] = (path, load_model(path, a.device)[0])
    first = next(iter(models.values()))[1]
    data = load_data(root, a.device, model_step_s=first.dt, core_only="drop")
    if a.batch:
        data["system"]["evaluation"]["batch"] = a.batch
    step_average_rates(data, set(first.config["integrate"]["kinds"]))
    add_derived(data, first.config.get("derived"))
    add_relative(data, first.config.get("relative"))
    names = [f["name"] for f in data["system"]["families"]]
    chosen = a.families or [n for n in names if n in R.FAMILY_NAMES + R.P_FAMILY_NAMES]   # no tail chunks
    keep = torch.tensor([n in chosen for n in names], device=a.device)
    idx = torch.nonzero((data["splits"] == a.split) & keep[data["family"]]).flatten()
    rows = data["file_index"][idx].cpu().numpy() if "file_index" in data else idx.cpu().numpy()
    with np.load(root / "unified_data.npz") as f:
        variant = f["variant"][rows]
    famrow = data["family"][idx].cpu().numpy()
    groups = {"all": np.ones(len(idx), bool)}
    groups.update({names[k]: famrow == k for k in np.unique(famrow)})
    for tag in ("appr", "align", "carry", "place"):
        sel = np.array([str(v).startswith(f"N:{tag}") for v in variant])
        if sel.any():
            groups[f"noise_{tag}"] = sel
    for tag in ("pre", "post"):
        sel = np.array([str(v).startswith(f"D:{tag}") for v in variant])
        if sel.any():
            groups[f"disturb_{tag}"] = sel
    for tag in ("rt1", "rt2"):
        sel = np.array([str(v).startswith(f"R:{tag}") for v in variant])
        if sel.any():
            groups[f"retarget_{tag}"] = sel
    pv = [re.match(r"^[WKQ]:(\w+)\.([dhf])", str(v)) for v in variant]      # policy_v3 rows: policy id, noise level
    if any(m is not None for m in pv):
        pid = np.array([m.group(1) if m else "" for m in pv])
        lev = np.array([m.group(2) if m else "" for m in pv])
        for x in [p for p in R.POLICY_IDS if (pid == p).any()]:
            groups[f"pol_{x}"] = pid == x
            for c in "dhf":
                if ((pid == x) & (lev == c)).any():
                    groups[f"pol_{x}_{c}"] = (pid == x) & (lev == c)
        for c in "dhf":
            if (lev == c).any():
                groups[f"noise_{c}"] = lev == c
    print(f"{root}: split {a.split}, {len(idx)} episodes, groups {[(k, int(v.sum())) for k, v in groups.items()]}",
          flush=True)
    out = dict(data=str(root), data_sha256=file_sha256(root / "unified_data.npz"), split=a.split, families=chosen,
               episodes=len(idx), groups={k: int(v.sum()) for k, v in groups.items()},
               group_rows={k: rows[v].tolist() for k, v in groups.items() if k != "all"}, models={})
    flags = {}
    for name, (path, model) in models.items():
        t = time.perf_counter()
        rep = file_rows(evaluate(model, data, idx, by_family=True), data)
        per = rep["per_episode"]
        rec = np.array([e["free"]["flags"]["lifted"]["recorded"] for e in per])
        pred = [e["free"]["flags"]["lifted"]["predicted"] for e in per]
        flags[name] = (rec, pred)
        drec = np.array([bool(E.pick(e, "free", "flags", "dropped", "recorded")) for e in per])
        dpred = [E.pick(e, "free", "flags", "dropped", "predicted") for e in per]
        dwrong = np.array([p is None or bool(p) != bool(r) for r, p in zip(drec, dpred)])
        end_mm = np.array([np.nan if E.pick(e, "free", "final", "cube", "pos_m") is None
                           else 1e3 * e["free"]["final"]["cube"]["pos_m"] for e in per])
        summary = {"all": E.brief(rep), **{k: E.brief(v) for k, v in rep.get("families", {}).items()}}
        for g, sel in groups.items():   # whole-episode error and lift per group from the per-episode values
            wrong = np.array([p is None or bool(p) != bool(r) for r, p in zip(rec, pred)])
            e = end_mm[sel]
            summary.setdefault(g, {})["group"] = dict(
                episodes=int(sel.sum()), lift_recorded=int(rec[sel].sum()), lift_wrong=int(wrong[sel].sum()),
                lift_agreement=float(1 - wrong[sel].mean()) if sel.any() else None,
                drop_recorded=int(drec[sel].sum()), drop_predicted=int(sum(bool(p) for p, s_ in zip(dpred, sel) if s_ and p)),
                drop_agreement=float(1 - dwrong[sel].mean()) if sel.any() else None,
                end_cube_mm_p90=float(np.nanpercentile(e, 90)) if np.isfinite(e).any() else None,
                end_cube_over_20mm=float(np.mean(e[np.isfinite(e)] > 20)) if np.isfinite(e).any() else None,
                end_cube_mm_p50=float(np.nanmedian(e)) if np.isfinite(e).any() else None,
                end_cube_mm_p95=float(np.nanpercentile(e, 95)) if np.isfinite(e).any() else None)
        if a.carry:   # cube error while MuJoCo holds the cube (carry_metrics), per group
            cm = carry_metrics(model, data, idx)
            for g, sel in groups.items():
                summary.setdefault(g, {})["carry"] = carry_summary(cm, sel)
            carry_rows = {k: [None if not np.isfinite(x) else round(float(x), 3) for x in v] if v.dtype.kind == "f"
                          else v.tolist() for k, v in cm.items()}
        out["models"][name] = dict(checkpoint=str(path), checkpoint_sha256=file_sha256(path), summary=summary,
                                   seconds=round(time.perf_counter() - t, 1),
                                   per_episode=[dict(row=int(e["index"]), family=e.get("family"),
                                                     lifted_recorded=bool(e["free"]["flags"]["lifted"]["recorded"]),
                                                     lifted_predicted=e["free"]["flags"]["lifted"]["predicted"],
                                                     dropped_recorded=bool(r_), dropped_predicted=p_,
                                                     end_cube_mm=None if not np.isfinite(m) else round(float(m), 3))
                                                for e, m, r_, p_ in zip(per, end_mm, drec, dpred)])
        if a.carry:
            out["models"][name]["carry_per_episode"] = carry_rows
        print(f"{name}: {time.perf_counter() - t:.0f} s", flush=True)
        for g, s in summary.items():
            gr = s.get("group", {})
            if "carry" in s:
                c = s["carry"]
                print(f"  {name} {g:22s} carry: rows {c['rows_with_carry']}, max err p50/p95 {c['max_mm_p50']}/"
                      f"{c['max_mm_p95']} mm, end err p50/p95 {c['end_mm_p50']}/{c['end_mm_p95']} mm, > 20 mm "
                      f"{c['max_over_20mm']}, NRD holds {c['nrd_held_frac_mean']} of the carry steps", flush=True)
            print(f"  {name} {g:22s} n {gr.get('episodes')}  lift agree {gr.get('lift_agreement')}  drop agree "
                  f"{gr.get('drop_agreement')} (rec {gr.get('drop_recorded')}, pred {gr.get('drop_predicted')})  1 s cube p50/p95 "
                  f"{s.get('win1s_cube_mm')} mm  4 s {s.get('free_4s_cube_mm')} mm  end p50/p95 "
                  f"{gr.get('end_cube_mm_p50')}/{gr.get('end_cube_mm_p95')} mm", flush=True)
    out["ensemble"] = E.ensemble(flags, groups)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))


def carry_metrics(model, data, indices, batch=512, half=0.015, up=0.010):
    """Cube error while MuJoCo holds the cube (round 3 'before' numbers), per episode. The free rollout of
    data.evaluate (from the first record, recorded commands; 20 ms model grid); a MuJoCo carry step j (state j -> j+1)
    = both jaw-cube contacts on and the cube-table contact off in the recorded interval labels of the step
    (data.physical_labels) and the recorded cube bottom more than ``up`` above the table at state j+1. Per episode: the
    number of carry steps, the cube position error |NRD - MuJoCo| (mm) at state j+1 over the carry steps (max, mean,
    at the last carry step), the share of carry steps in which the NRD also holds it (both jaw switches on in the
    model's convention and its cube bottom above ``up``), finite (the rollout is finite up to the last carry step)."""
    import torch

    from nedm.contact_nrd.data import channel_index, physical_labels
    system = data["system"]
    stride = data["stride"]
    tslot = model.moving.index(system["target_body"])
    tpos = channel_index(system, system["bodies"][system["target_body"]]["name"], "position")
    pn = [m.get("name") for m in system["pair_metadata"]]
    fj, mj, ct = pn.index("fixed_jaw_cube"), pn.index("moving_jaw_cube"), pn.index("cube_table")
    n_all = ((data["lengths"][indices] - 1) // stride + 1)
    e = len(indices)
    out = dict(steps=np.zeros(e, np.int64), max_mm=np.full(e, np.nan), mean_mm=np.full(e, np.nan),
               end_mm=np.full(e, np.nan), nrd_held_frac=np.full(e, np.nan), finite=np.ones(e, bool))
    with torch.no_grad():
        for b0 in range(0, e, batch):
            chosen = indices[b0: b0 + batch]
            sl = slice(b0, b0 + len(chosen))
            truth = data["states"][chosen][:, ::stride]
            acts = data["actions"][chosen][:, ::stride]
            steps = truth.shape[1] - 1
            pred, gates = model.rollout(truth[:, 0], acts[:, :steps], return_gates=True)
            lab = physical_labels(data["contacts"][chosen], stride)[:, :steps].bool()
            tp, pp = truth[:, 1:, tslot][..., tpos], pred[:, 1:, tslot][..., tpos]
            n = n_all[b0: b0 + len(chosen)].to(truth.device)
            valid = torch.arange(steps, device=truth.device)[None] < (n - 1)[:, None]
            held = lab[..., fj] & lab[..., mj] & ~lab[..., ct] & (tp[..., 2] - half > up) & valid
            err = (pp - tp).norm(dim=-1) * 1e3
            g = gates[:, :steps] > 0.5
            nrd_held = g[..., fj] & g[..., mj] & (pp[..., 2] - half > up)
            fin = torch.isfinite(pred[:, 1:].flatten(2)).all(-1)
            for r in range(len(chosen)):
                h = torch.nonzero(held[r]).flatten()
                k = int(b0 + r)
                out["steps"][k] = len(h)
                if not len(h):
                    continue
                last = int(h[-1])
                out["finite"][k] = bool(fin[r, : last + 1].all())
                if not out["finite"][k]:
                    continue
                er = err[r, h]
                out["max_mm"][k], out["mean_mm"][k] = float(er.max()), float(er.mean())
                out["end_mm"][k] = float(err[r, last])
                out["nrd_held_frac"][k] = float(nrd_held[r, h].float().mean())
    return out


def carry_summary(cm, sel):
    """Group summary of carry_metrics over the rows sel (bool [rows]); non-finite rollouts count as errors > 20 mm."""
    s = sel & (cm["steps"] > 0)
    fin = s & cm["finite"]

    def q(x, p):
        x = x[fin]
        return None if not len(x) else round(float(np.percentile(x, p)), 2)
    over = (cm["max_mm"][fin] > 20).sum() + (s & ~cm["finite"]).sum()
    return dict(rows=int(sel.sum()), rows_with_carry=int(s.sum()), nonfinite=int((s & ~cm["finite"]).sum()),
                carry_steps_p50=float(np.median(cm["steps"][s])) if s.any() else None,
                max_mm_p50=q(cm["max_mm"], 50), max_mm_p95=q(cm["max_mm"], 95), mean_mm_p50=q(cm["mean_mm"], 50),
                end_mm_p50=q(cm["end_mm"], 50), end_mm_p95=q(cm["end_mm"], 95),
                max_over_20mm=None if not s.any() else round(float(over / s.sum()), 4),
                nrd_held_frac_mean=None if not fin.any() else round(float(np.mean(cm["nrd_held_frac"][fin])), 4),
                nrd_held_90pct=None if not fin.any() else round(float(np.mean(cm["nrd_held_frac"][fin] >= 0.9)), 4))


# ---- merge with repeated rows, and the sampling share of each family ------------------------------------------------
def _folder_arrays(root, keys):
    with np.load(Path(root) / "unified_data.npz") as f:
        return {k: f[k] for k in keys if k in f.files}


def _parse_repeat(items):
    out = {}
    for it in items or []:
        path, k = it.rsplit("=", 1)
        out[str(Path(path).expanduser())] = float(k)
    return out


def repeat_rows(splits, family, factor, seed=R.MASTER_SEED):
    """Row order of a folder with its split-0 rows repeated ``factor`` times (integer part: every row; fractional
    part: a fixed random subset of the rows, seeded); split 1 / 2 rows once. -> int64 row indices (file order first)."""
    rows = np.arange(len(splits))
    train = rows[splits == 0]
    whole, frac = int(math.floor(factor)), factor - math.floor(factor)
    if whole < 1:
        raise ValueError("repeat factor must be >= 1")
    extra = [train] * (whole - 1)
    if frac > 1e-9:
        rng = np.random.default_rng([int(seed), 7])
        extra.append(np.sort(rng.choice(train, int(round(frac * len(train))), replace=False)))
    return np.concatenate([rows] + extra)


def sampling_shares(folders, repeat=None, model_step_s=0.02, free_phase_step=5, tallies=None):
    """Expected share of each family in the training draws of train.py (split 0), for folders merged in order (the
    first gives system.json) with split-0 rows repeated per ``repeat`` {folder: factor}:
    - contact batches (contact_sampling sqrt_groups: groups = the set of pairs on in the window, each group drawn in
      proportion to the square root of its window count, uniform inside it) and the contact-near half of the rollout
      starts (the same groups);
    - the held-cube group alone (both jaws on, table off: the carries);
    - any training window (the other half of the rollout starts, the collision 'any' third);
    - core windows per body (contact-free windows every free_phase_step records; train.py also bins them by the size
      of the rate change, sqrt weights per bin: not modelled here, so these shares are approximate).
    Counts use data.window labels as validate.training_windows does. tallies: family_tallies(folders) computed once
    (pick_repeat). -> dict."""
    repeat = {str(Path(k).expanduser()): v for k, v in (repeat or {}).items()}
    names, moving, per = tallies or family_tallies(folders, model_step_s, free_phase_step)
    tally = {}
    for root, fams in per.items():
        k = repeat.get(str(Path(root).expanduser()), 1.0)
        for f, src in fams.items():
            t = tally.setdefault(f, dict(windows=0.0, hits={}, core={b: 0.0 for b in moving}))
            t["windows"] += k * src["windows"]
            for c, v in src["hits"].items():
                t["hits"][c] = t["hits"].get(c, 0.0) + k * v
            for b in moving:
                t["core"][b] += k * src["core"][b]
    fams = [n for n in names if n in tally]
    codes = sorted({c for t in tally.values() for c in t["hits"]})
    gsize = {c: sum(tally[f]["hits"].get(c, 0.0) for f in fams) for c in codes}
    gw = {c: math.sqrt(n) for c, n in gsize.items()}
    tot = sum(gw.values())
    contact = {f: sum(gw[c] / tot * tally[f]["hits"].get(c, 0.0) / gsize[c] for c in codes) for f in fams}
    held = 3                                                           # fixed jaw + moving jaw, no table
    anyw = sum(tally[f]["windows"] for f in fams)
    out = dict(groups={str(c): dict(windows=gsize[c], weight=gw[c] / tot) for c in codes},
               families={f: dict(contact=contact[f],
                                 held=tally[f]["hits"].get(held, 0.0) / gsize[held] if gsize.get(held) else 0.0,
                                 any=tally[f]["windows"] / anyw,
                                 rollout=0.5 * contact[f] + 0.5 * tally[f]["windows"] / anyw,
                                 **{f"core_{b}": tally[f]["core"][b] / sum(tally[g]["core"][b] for g in fams)
                                    for b in moving},
                                 train_windows=tally[f]["windows"]) for f in fams},
               repeat={str(k): v for k, v in repeat.items()})
    return out


def family_tallies(folders, model_step_s=0.02, free_phase_step=5):
    """Per folder and family: training windows, windows per contact code, core windows per body (sampling_shares)
    -> (family names in merge order, moving bodies, {folder: {family: tally}})."""
    from nedm.contact_nrd.collect import validate
    system = json.loads((Path(folders[0]) / "system.json").read_text())
    system = dict(system, model_step_s=model_step_s)
    names = [f["name"] for f in system["families"]]
    moving = [b["name"] for b in system["bodies"] if b["moving"]]
    per = {}
    for root in folders:
        other = json.loads((Path(root) / "system.json").read_text())
        theirs = [f["name"] for f in other["families"]]
        for n in theirs[len(names):]:
            names.append(n)
        a = _folder_arrays(root, ("contacts", "lengths", "splits", "family"))
        lab, involved, valid, stride = validate.training_windows(dict(system, families=[]), a["contacts"], a["lengths"],
                                                                  a["splits"])
        train = valid & (a["splits"] == 0)[:, None]
        code = (lab.astype(np.int64) * (2 ** np.arange(lab.shape[-1]))).sum(-1)
        grid = (np.arange(train.shape[1]) % free_phase_step == 0)[None, :]
        mine = per.setdefault(str(Path(root).expanduser()), {})
        for f in np.unique(a["family"]):
            sel = (a["family"] == f)[:, None] & train
            t = mine.setdefault(theirs[int(f)], dict(windows=0.0, hits={}, core={b: 0.0 for b in moving}))
            t["windows"] += float(sel.sum())
            for c in np.unique(code[sel & lab.any(-1)]):
                t["hits"][int(c)] = t["hits"].get(int(c), 0.0) + float((sel & (code == c)).sum())
            for j, b in enumerate(moving):
                t["core"][b] += float((sel & grid & ~involved[..., j]).sum())
    return names, moving, per


def cmd_pick_repeat(a):
    """The smallest repeat factor of one folder (--vary) whose families get at least --held-min of the held-cube draws
    (sampling_shares, the merge order of --folders); writes every candidate's shares (--out) and prints the choice."""
    folders = [str(Path(p).expanduser()) for p in a.folders]
    vary = str(Path(a.vary).expanduser())
    tal = family_tallies(folders)
    mine = set(tal[2][vary])
    groups = _share_groups(folders)
    res, choice = {}, None
    for k in a.factors:
        sh = sampling_shares(folders, {vary: k}, tallies=tal)
        agg = {g: {key: sum(v[key] for f, v in sh["families"].items() if f in members) for key in
                   ("contact", "held", "any", "rollout", "core_arm", "core_cube")} for g, members in groups.items()}
        held = sum(v["held"] for f, v in sh["families"].items() if f in mine)
        res[str(k)] = dict(shares=sh, groups=agg, vary_held=held)
        print(f"repeat {k}: {Path(vary).name} held {100 * held:.1f} %; " + "; ".join(
            f"{g} contact {100 * v['contact']:.1f} / held {100 * v['held']:.1f} / any {100 * v['any']:.1f} %"
            for g, v in agg.items()), flush=True)
        if choice is None and held >= a.held_min:
            choice = k
    choice = a.factors[-1] if choice is None else choice
    Path(a.out).write_text(json.dumps(dict(folders=folders, vary=vary, held_min=a.held_min, choice=choice,
                                           candidates=res), indent=1))
    print(f"CHOICE {choice}", flush=True)


def _share_groups(folders):
    """Family groups of the share tables: production (the first folder's families), then each later folder's own."""
    out, seen = {}, []
    for k, root in enumerate(folders):
        fam = [f["name"] for f in json.loads((Path(root) / "system.json").read_text())["families"]]
        out["production" if k == 0 else Path(root).name] = [f for f in fam if f not in seen]
        seen += [f for f in fam if f not in seen]
    return out


def _share_table(sh, groups=None):
    groups = groups or {}
    fams = sh["families"]
    agg = {}
    for f, v in fams.items():
        g = next((k for k, members in groups.items() if f in members), f)
        a = agg.setdefault(g, {k: 0.0 for k in v})
        for k, x in v.items():
            a[k] += x
    keys = [k for k in next(iter(agg.values())) if k != "train_windows"]
    lines = [f"{'family':22s} " + " ".join(f"{k:>10s}" for k in keys)]
    for g, v in agg.items():
        lines.append(f"{g:22s} " + " ".join(f"{100 * v[k]:9.1f}%" for k in keys))
    return "\n".join(lines)


def cmd_shares(a):
    sh = sampling_shares([str(Path(p).expanduser()) for p in a.folders], _parse_repeat(a.repeat))
    groups = {"production (12+step)": [f["name"] for f in json.loads((Path(a.folders[0]).expanduser() / "system.json")
                                                                    .read_text())["families"]],
              "noisy_v1": ["noisy_nominal", "noisy_off_centre", "noisy_carry", "noisy_offset"],
              "expert_v2": list(R.FAMILY_NAMES) + [R.TAIL["name"]],
              "policy_v3": list(R.P_FAMILY_NAMES) + [R.POLICY_TAIL["name"]]}
    print(_share_table(sh))
    print()
    print(_share_table(sh, groups))
    if a.out:
        Path(a.out).write_text(json.dumps(sh, indent=1))


def cmd_merge(a):
    """scripts/contact_nrd/merge_data.py's rules (BASE's system.json; the same bodies, pairs, phases; families extend
    BASE's in order; episode_id / group_id offset) plus --repeat DIR=k: that folder's split-0 rows repeated k times
    (repeat_rows), so train.py draws its windows k times as often (train.py of this branch has no family mix)."""
    from nedm.contact_nrd.collect import export
    out, base, extras = Path(a.out), Path(a.base).expanduser(), [Path(p).expanduser() for p in a.extra]
    repeat = _parse_repeat(a.repeat)
    system = json.loads((base / "system.json").read_text())
    names = [f["name"] for f in system["families"]]
    t0 = time.time()
    parts, counts = [], []
    for k, root in enumerate([base] + extras):
        other = json.loads((root / "system.json").read_text())
        if k:
            for key in ("bodies", "pairs", "phases"):
                if other.get(key) != system.get(key):
                    raise SystemExit(f"{root}: {key} differ from {base}")
            their = [f["name"] for f in other["families"]]
            if their[: len(names)] != names[: len(their)]:
                raise SystemExit(f"{root}: families must extend {base}'s list in order")
            for f in other["families"][len(names):]:
                system["families"].append(f)
                names.append(f["name"])
        with np.load(root / "unified_data.npz") as f:
            arrays = {key: f[key] for key in f.files}
        if parts and set(arrays) != set(parts[0][1]):
            raise SystemExit(f"{root}: arrays differ: {sorted(set(arrays) ^ set(parts[0][1]))}")
        factor = repeat.get(str(root), 1.0)
        rows = repeat_rows(arrays["splits"], arrays["family"], factor) if factor != 1.0 else np.arange(len(arrays["lengths"]))
        parts.append((root, arrays, rows))
        counts.append(dict(folder=str(root), rows=int(len(arrays["lengths"])), repeat=factor, rows_out=int(len(rows)),
                           split0=int((arrays["splits"] == 0).sum())))
        print(f"{root}: {len(arrays['lengths'])} rows, repeat {factor} -> {len(rows)}", flush=True)
    merged, n = {}, 0
    for key in parts[0][1]:
        xs = []
        n = 0
        for root, arrays, rows in parts:
            x = arrays[key][rows]
            if key in ("episode_id", "group_id"):
                x = x + n
            n += len(arrays["lengths"])
            xs.append(x)
        merged[key] = np.concatenate(xs, 0)
    if out.exists():
        raise SystemExit(f"{out} exists")
    tmp = Path(str(out) + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    np.savez(tmp / "unified_data.npz", **merged)
    (tmp / "system.json").write_text(json.dumps(system, indent=1))
    (tmp / "manifest.json").write_text(json.dumps(dict(
        merged_from=[str(base)] + [str(e) for e in extras], repeat={k: v for k, v in repeat.items()}, parts=counts,
        episodes=int(len(merged["lengths"])), splits={str(s): int((merged["splits"] == s).sum()) for s in (0, 1, 2)},
        note="collect_round.py merge: merge_data.py rules + split-0 rows of --repeat folders repeated (episode_id and "
             "group_id of a repeated row equal those of its original)"), indent=1))
    tmp.replace(out)
    print(f"{out}: {len(merged['lengths'])} rows in {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tasks")
    p.add_argument("--set", choices=tuple(R.SET_IDS), default="campaign"), p.add_argument("--n", type=int, required=True)
    p.add_argument("--out", required=True), p.add_argument("--shard"), p.add_argument("--workers", type=int, default=1)
    p = sub.add_parser("run")
    p.add_argument("--tasks", required=True), p.add_argument("--out", required=True), p.add_argument("--rows")
    p.add_argument("--workers", type=int, default=0), p.add_argument("--python"), p.add_argument("--batch", type=int, default=1200)
    p.add_argument("--threads", type=int, default=2), p.add_argument("--tag", default="")
    p.add_argument("--stride", help="K/M: every M-th task from the K-th (after --rows)")
    p.add_argument("--decoder-cfg", default="{}", help="DecoderV2Config overrides (JSON); default: decoder v2 as is")
    p.add_argument("--expert-cfg", default="{}", help="ExpertConfig overrides (JSON)")
    p = sub.add_parser("label")
    p.add_argument("--rollouts", nargs="+", required=True), p.add_argument("--workers", type=int, default=1)
    p = sub.add_parser("write")
    p.add_argument("--labelled", nargs="+", required=True), p.add_argument("--out", required=True)
    p.add_argument("--gate2", nargs="+"), p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--campaign", choices=(R.CAMPAIGN, R.P_CAMPAIGN), default=R.CAMPAIGN)
    p.add_argument("--gate-name", help="held-out folder name (default gate2; policy_v3: pgate)")
    p = sub.add_parser("ptasks")
    p.add_argument("--set", choices=tuple(R.P_SET_IDS), default="campaign"), p.add_argument("--n", type=int, required=True)
    p.add_argument("--out", required=True), p.add_argument("--workers", type=int, default=1)
    p = sub.add_parser("prun")
    p.add_argument("--tasks", required=True), p.add_argument("--out", required=True), p.add_argument("--rows")
    p.add_argument("--expand-pgate", action="store_true", help="the tasks file is a pgate task table: run every policy "
                   "on every task, deterministic and at 1.0 x std (data_round.expand_pgate)")
    p.add_argument("--policies", nargs="+", help="policy ids (default all of data_round.POLICIES in the rows)")
    p.add_argument("--policy-dir", help="checkpoint folder (default data_round.POLICY_DIR)")
    p.add_argument("--geom", help="task_geom_v2.npz where the checkpoint's own path does not exist")
    p.add_argument("--kin-points", help="kin_points_v1.npz where the checkpoint's own path does not exist")
    p.add_argument("--workers", type=int, default=0), p.add_argument("--python"), p.add_argument("--batch", type=int, default=1000)
    p.add_argument("--threads", type=int, default=2), p.add_argument("--tag", default="")
    p.add_argument("--horizon", type=int, help="task horizon override (tests only)")
    p = sub.add_parser("verify")
    p.add_argument("--data", required=True), p.add_argument("--master", type=int, required=True)
    p.add_argument("--families", nargs="+", default=["nominal", "off_centre", "one_sided", "near_miss", "weak_closure",
                                                       "drop", "push"])
    p.add_argument("--per-family", type=int, default=4), p.add_argument("--terminal", type=int, default=8)
    p.add_argument("--workers", type=int, default=0), p.add_argument("--python"), p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", required=True)
    p = sub.add_parser("shares")
    p.add_argument("--folders", nargs="+", required=True), p.add_argument("--repeat", nargs="*")
    p.add_argument("--out")
    p = sub.add_parser("pick_repeat")
    p.add_argument("--folders", nargs="+", required=True), p.add_argument("--vary", required=True)
    p.add_argument("--factors", nargs="+", type=float, default=[1.0, 1.5, 2.0, 2.5, 3.0])
    p.add_argument("--held-min", type=float, default=0.40), p.add_argument("--out", required=True)
    p = sub.add_parser("merge")
    p.add_argument("--out", required=True), p.add_argument("--base", required=True)
    p.add_argument("--extra", nargs="+", required=True), p.add_argument("--repeat", nargs="*")
    p = sub.add_parser("eval")
    p.add_argument("--data", required=True), p.add_argument("--model", nargs="+", required=True)
    p.add_argument("--out", required=True), p.add_argument("--split", type=int, default=1)
    p.add_argument("--families", nargs="+"), p.add_argument("--batch", type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--carry", action="store_true", help="also the cube error while MuJoCo holds the cube (carry_metrics)")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
