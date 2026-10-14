"""Evaluate a footstep-target policy (FS, trained inside the G1 contact NRD) and the H-FS baseline in CPU MuJoCo
(ground truth) on footstep courses, optionally also inside the NRD from the same hand-over snapshots. Runner, baseline,
statistics and rendering: nedm.g1_footstep_rl.footsteps_mujoco (its docstring has the runner contract (1)-(13) and the
decisions M1-M7; "spec N" refers to the internal design spec of the study, not released).

Needs torch, rsl_rl, mujoco 3.15.0 and onnxruntime in one interpreter. Videos: set MUJOCO_GL (egl, or osmesa where
EGL does not work; osmesa takes about 1 s per camera frame).

Policies (one per eval directory):
  --ckpt RUN/x.pt          FS policy (env_cfg.json / train_cfg.json next to it); --zero-residual = correction zeroed
  --hfs JSON               H-FS with these parameters (footstep_baselines.py tune picks them), e.g. '{"g_v": 1.0}'

Cells: --levels from F1 F2 F3 MP MN ES (new levels, random heading) L1 L2 L3 L4 (old layouts + planner) and the
showcase names (rhythm ramp lane_change mixed_wide mixed_narrow_edge step_change). --seeds dev (new 1000-1063,
old 100-163), tune (0-15), or ranges "a-b". The alias test names the sealed final-test seeds (footsteps_task.SEEDS);
no released result, README step or reproduce.sh step uses them.

Subcommands (outputs in --out, default artifacts/g1_footstep_rl/eval/<name>):
  eval       MuJoCo episodes -> meta.json, rows.json (per episode), steps.json (per scored target), snapshots.npz
             (FS: hand-over snapshots), summary.json, SUMMARY.md; --plots N per-course PNGs, --video N mp4s
             (chase + top, markers, HUD); --nrd also runs the NRD part; --baseline NAME=DIR adds paired McNemar
  nrd        The NRD part for an existing FS eval directory: the same policy from the same hand-over snapshots inside
             G1FootstepsEnv (reset_from_snapshots), --nrd-members -> nrd_rows.json, nrd_steps.json
  report     Rebuild summary.json and SUMMARY.md (--baseline NAME=DIR, repeatable). Old-layout cells: L1 / L2 hold
             every layout with a plan (V3 "lunge" plans included, course_for); when V3 plans are present the report
             adds the spec 12.3 cell "L1[V1-V2]" / "L2[V1-V2]" (V1-V2 plans only; paired tests use the same subset)
  video      Re-run (level, seed) episodes of one or more policies with states kept and render each (chase + top,
             1920 x 544); --compose writes them side by side / as a grid (ffmpeg) with policy labels
  scatter    NRD-predicted vs MuJoCo completion per (eval directory, level) cell over several eval directories:
             Spearman with a bootstrap 95 % interval, per-episode agreement and phi
  parity     Observation parity: one recorded MuJoCo episode; at >= 18 steps G1FootstepsEnv.observe() after
             reset_from_snapshots of the runner's inputs must equal the runner's observation (<= 1e-6)

    cd <repository>; PYTHONPATH=src python -B scripts/g1_footstep_rl/eval_footsteps_mujoco.py \
        eval --ckpt artifacts/g1_footstep_rl/policy/fs_cmd_s1/policy.pt --levels F1 F2 MP MN ES --seeds dev \
        --workers 12 --nrd --nrd-members 0 1
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/g1_footstep_rl/eval"
NEW_CELLS = ("F1", "F2", "F3", "MP", "MN", "ES")
OLD_CELLS = ("L1", "L2", "L3", "L4")


def jdump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=_default) + "\n")


def _default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return str(o)


def _clean(o):
    """NaN -> None for JSON."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, np.generic):
        return _clean(o.item())
    return o


def seeds_for(level, items):
    """Seeds of a cell from items: "dev", "test", "tune", "a-b" or integers; showcases have no seed (-1)."""
    from nedm.g1_footstep_rl import footsteps_task as FT
    if level in FT.SHOWCASES:
        return [-1]
    old = level in OLD_CELLS
    out = []
    for it in items:
        it = str(it)
        if it in ("dev", "test", "tune"):
            key = {"dev": "dev_old" if old else "dev", "test": "test_old" if old else "test", "tune": "hfs_tune"}[it]
            a, b = FT.SEEDS[key]
            out += list(range(a, b))
        elif "-" in it:
            a, b = it.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(it))
    return out


def policy_spec(a):
    if getattr(a, "ckpt", None):
        spec = dict(kind="footsteps", path=str(Path(a.ckpt).resolve()), zero_residual=bool(a.zero_residual))
    elif getattr(a, "hfs", None) is not None:
        spec = dict(kind="hfs", params=json.loads(a.hfs) if isinstance(a.hfs, str) else dict(a.hfs))
    else:
        raise SystemExit("give a policy: --ckpt or --hfs")
    if getattr(a, "score", None):
        spec["score"] = json.loads(a.score)
    return spec


def make_jobs(levels, seed_items, out, plots=0, video=0, video_opts=None):
    jobs = []
    for lv in levels:
        for s in seeds_for(lv, seed_items):
            tag = f"{lv}_s{s:05d}" if s >= 0 else lv
            j = dict(level=lv, seed=s, episode_id=len(jobs))
            if len(jobs) < plots:
                j["plot"] = str(Path(out) / "plots" / f"{tag}.png")
            if len(jobs) < video:
                j["video"] = str(Path(out) / "videos" / f"{tag}.mp4")
                j["video_opts"] = dict(video_opts or {})
            jobs.append(j)
    return jobs


# ---- snapshots -----------------------------------------------------------------------------------------------------
SNAP_KEYS = ("step", "hist", "teacher_history", "teacher_last_action", "prev_cmd", "prev_u", "gate", "obs")


def save_snapshots(path, rows, snaps):
    ok = [(r, s) for r, s in zip(rows, snaps) if s is not None]
    arrays = {k: np.stack([np.asarray(s[k]) for _, s in ok]) for k in SNAP_KEYS} if ok else {}
    side = [dict(level=r["level"], seed=r["seed"], course=s["course"], progress=s["progress"]) for r, s in ok]
    np.savez(path, side=np.asarray(json.dumps(_clean(side))), **arrays)


def load_snapshots(path):
    with np.load(path) as f:
        side = json.loads(str(f["side"]))
        return [dict({k: f[k][i] for k in SNAP_KEYS}, **side[i]) for i in range(len(side))]


# ---- report ----------------------------------------------------------------------------------------------------------
def load_dir(d):
    d = Path(d)
    rows = json.loads((d / "rows.json").read_text())
    steps = json.loads((d / "steps.json").read_text()) if (d / "steps.json").exists() else []
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    return meta, rows, steps


def parse_baselines(items):
    out = {}
    for it in items or []:
        name, _, d = it.partition("=")
        out[name] = d
    return out


def nrd_compare(rows, steps, nrd_rows, nrd_steps):
    """Per member and level: agreement and phi of completion and clean completion (MuJoCo vs NRD, same snapshots),
    NRD vs MuJoCo per-target error (targets scored in both: median |difference|, Pearson r)."""
    from nedm.g1_footstep_rl.mujoco_policy import phi
    by = {(r["level"], r["seed"]): r for r in rows}
    st = {(s["level"], s["seed"], s["target"]): s for s in steps}
    out = {}
    for m in sorted({r["member"] for r in nrd_rows}):
        for lv in sorted({r["level"] for r in nrd_rows}) + ["all"]:
            nr = [r for r in nrd_rows if r["member"] == m and (lv == "all" or r["level"] == lv)
                  and (r["level"], r["seed"]) in by]
            if not nr:
                continue
            mj = [by[(r["level"], r["seed"])] for r in nr]
            d = dict(n=len(nr))
            for key in ("completed", "clean_completion", "success5"):
                x, y = [bool(r.get(key)) for r in mj], [bool(r.get(key)) for r in nr]
                d[key] = dict(mujoco=int(sum(x)), nrd=int(sum(y)), both=int(sum(a and b for a, b in zip(x, y))),
                              mujoco_only=int(sum(a and not b for a, b in zip(x, y))),
                              nrd_only=int(sum(b and not a for a, b in zip(x, y))),
                              agreement=float(np.mean([a == b for a, b in zip(x, y)])), phi=phi(x, y))
            pairs = [(st[(s["level"], s["seed"], s["target"])]["e"], s["e"]) for s in nrd_steps
                     if s["member"] == m and (lv == "all" or s["level"] == lv)
                     and (s["level"], s["seed"], s["target"]) in st]
            if pairs:
                a = np.asarray(pairs)
                d["target_error"] = dict(n=len(a), mujoco_p95=float(np.percentile(a[:, 0], 95)),
                                         nrd_p95=float(np.percentile(a[:, 1], 95)),
                                         abs_diff_p50=float(np.median(np.abs(a[:, 0] - a[:, 1]))),
                                         pearson=float(np.corrcoef(a.T)[0, 1]) if len(a) > 2 else None)
            ends = {}
            for r in nr:
                ends[r["outcome"]] = ends.get(r["outcome"], 0) + 1
            d["nrd_outcomes"] = ends
            d["first_obs_max_abs_diff"] = float(max(r.get("first_obs_max_abs_diff", 0.0) for r in nr))
            out.setdefault(f"member {m}", {})[lv] = d
    return out


SPEC_CELL = "[V1-V2]"           # spec 12.3: the old-layout cells L1, L2 use planner variants V1-V2 only


def cell_rows(rows, cell):
    """Rows of a report cell: a level name, or "<L1|L2>[V1-V2]" = the spec 12.3 old-layout cell (rows whose course
    plan is variant V1 or V2; the plain L1 / L2 cell also holds the V3 "lunge" plans of course_for)."""
    if cell.endswith(SPEC_CELL):
        lv = cell[:-len(SPEC_CELL)]
        return [r for r in rows if r["level"] == lv and r.get("variant") in ("V1", "V2")]
    return [r for r in rows if r["level"] == cell]


def summarise(rows, steps, levels=None):
    from nedm.g1_footstep_rl import footsteps_mujoco as FM
    levels = levels or sorted({r["level"] for r in rows}, key=lambda x: (x not in NEW_CELLS, x))
    cells = {lv: FM.cell_summary(cell_rows(rows, lv), steps) for lv in levels}
    for lv in ("L1", "L2"):                      # add the spec cell when the level holds V3 plans
        if lv in cells and any(r["level"] == lv and r.get("variant") == "V3" for r in rows):
            cells[lv + SPEC_CELL] = FM.cell_summary(cell_rows(rows, lv + SPEC_CELL), steps)
    return cells


def report(out_dir, baselines=None):
    from nedm.g1_footstep_rl import footsteps_mujoco as FM
    out_dir = Path(out_dir)
    meta, rows, steps = load_dir(out_dir)
    baselines = dict(meta.get("baselines") or {}, **(baselines or {}))
    cells = summarise(rows, steps)
    sm = dict(meta=meta, cells=cells, paired={})
    for name, d in baselines.items():
        try:
            _, brows, _ = load_dir(d)
        except FileNotFoundError:
            continue
        for lv in cells:
            rs = cell_rows(rows, lv)
            bs = cell_rows(brows, lv)
            if bs:
                sm["paired"].setdefault(lv, {})[name] = {k: FM.paired_mcnemar(rs, bs, k) for k in
                                                        ("completed", "clean_completion", "success8", "success5")}
    if (out_dir / "nrd_rows.json").exists():
        nrd_rows = json.loads((out_dir / "nrd_rows.json").read_text())
        nrd_steps = json.loads((out_dir / "nrd_steps.json").read_text()) if (out_dir / "nrd_steps.json").exists() else []
        sm["nrd"] = nrd_compare(rows, steps, nrd_rows, nrd_steps)
        sm["nrd_cells"] = {f"member {m}": summarise([r for r in nrd_rows if r["member"] == m],
                                                    [s for s in nrd_steps if s["member"] == m])
                           for m in sorted({r["member"] for r in nrd_rows})}
    jdump(out_dir / "summary.json", _clean(sm))
    lines = markdown(sm)
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return sm


def _pct(d):
    if not d or d.get("rate") is None or (isinstance(d.get("rate"), float) and not math.isfinite(d["rate"])):
        return "n/a"
    lo, hi = d["ci"]
    return f"{100 * d['rate']:.1f} [{100 * lo:.0f}, {100 * hi:.0f}]"


def _cm(d):
    if not d or d.get("value") is None or not math.isfinite(d["value"]):
        return "n/a"
    lo, hi = d["ci"]
    return f"{100 * d['value']:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def markdown(sm):
    meta = sm["meta"]
    pol = meta.get("policy", {})
    L = [f"# Footstep courses in MuJoCo: {pol.get('name', meta.get('name', ''))}", "",
         f"Policy: `{json.dumps(meta.get('spec'))}`. Host {meta.get('host')}. Seeds {meta.get('seed_items')}. "
         "Standing start, 1 s stand, 12 s policy (spec 12). Brackets: Wilson 95 % for episode rates, episode-cluster "
         "bootstrap 95 % for per-step rates and errors (cm).", "",
         "| Cell | n | Completion % | Clean completion % | Success@5 % | Hit@3 % (step) | Hit@5 % (step) | Hit@8 % (step) | "
         "Hit@5 % of all targets | Error p50 cm | Error p95 cm | Falls | Self-contacts | Max l p95 m | Max heading p95 deg |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for lv, c in sm["cells"].items():
        if not c.get("n"):
            continue
        dl = c.get("max_abs_l") or {}
        dh = c.get("max_abs_dpsi_deg") or {}
        L.append(f"| {lv} | {c['n']} | {_pct(c['completed'])} | {_pct(c['clean_completion'])} | {_pct(c['success5'])} | "
                 f"{_pct(c['hit3'])} | {_pct(c['hit5'])} | {_pct(c['hit8'])} | {_pct(c['hit5_all'])} | "
                 f"{_cm(c['err_p50'])} | {_cm(c['err_p95'])} | {c['fell']['k']} | {c['self_contact']['k']} | "
                 f"{dl.get('p95', float('nan')):.2f} | {dh.get('p95', float('nan')):.0f} |")
    L += ["", "| Cell | Outcomes | Re-steps (mean) | Wrong foot (mean) | Flickers (mean) | Toe gap at the cliff, completed (m) | "
          "NRD rules fired (legs / support) | Residual mean rad / saturation |", "|---|---|---|---|---|---|---|---|"]
    f2 = lambda v: "n/a" if v is None else f"{v:.2f}"
    for lv, c in sm["cells"].items():
        if not c.get("n"):
            continue
        tg = c.get("toe_gap_min_completed")
        tgs = f"p50 {tg['p50']:.3f} ({tg['min']:.3f}-{tg['max']:.3f}, n {tg['n']})" if tg else "-"
        res = (f"{c['residual_abs_mean_rad']:.4f} / {100 * c['residual_saturation']:.1f} %"
               if c.get("residual_abs_mean_rad") is not None else "-")
        oc = ', '.join(f'{k} {v}' for k, v in c['outcomes'].items())
        if c.get("planner_variants"):
            oc += " (planner " + ", ".join(f"{k} {v}" for k, v in sorted(c["planner_variants"].items())) + ")"
        L.append(f"| {lv} | {oc} | {f2(c['resteps_mean'])} | "
                 f"{f2(c['wrong_foot_mean'])} | {f2(c['flickers_mean'])} | {tgs} | {c['rule_legs_fired']} / "
                 f"{c['rule_support_fired']} | {res} |")
    if sm.get("paired"):
        L += ["", "Paired exact McNemar against baselines on the same courses (policy only / baseline only / p).", "",
              "| Cell | Baseline | n | Completion policy / baseline | p | Clean completion | p | Success@8 | p |",
              "|---|---|---|---|---|---|---|---|---|"]
        for lv, per in sm["paired"].items():
            for name, d in per.items():
                c, cc, s8 = d["completed"], d["clean_completion"], d["success8"]
                if not c:
                    continue
                L.append(f"| {lv} | {name} | {c['n']} | {c['policy']} / {c['baseline']} ({c['policy_only']} / "
                         f"{c['baseline_only']}) | {c['mcnemar_p']:.3g} | {cc['policy']} / {cc['baseline']} | "
                         f"{cc['mcnemar_p']:.3g} | {s8['policy']} / {s8['baseline']} | {s8['mcnemar_p']:.3g} |")
    if sm.get("nrd"):
        L += ["", "Same policy from the same hand-over snapshots inside the NRD (G1FootstepsEnv, 600 steps).", "",
              "| Member | Cell | n | Completion MuJoCo / NRD | Agreement | phi | Clean MuJoCo / NRD | Agreement | phi | "
              "Target error p95 MuJoCo / NRD cm | Median abs diff cm | First obs diff |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        f3 = lambda v: "n/a" if v is None else f"{v:.2f}"
        for m, per in sm["nrd"].items():
            for lv, d in per.items():
                c, cc = d["completed"], d["clean_completion"]
                te = d.get("target_error")
                tes = (f"{100 * te['mujoco_p95']:.1f} / {100 * te['nrd_p95']:.1f} | {100 * te['abs_diff_p50']:.1f}"
                       if te else "n/a | n/a")
                L.append(f"| {m} | {lv} | {d['n']} | {c['mujoco']} / {c['nrd']} | {100 * c['agreement']:.1f} % | "
                         f"{f3(c['phi'])} | {cc['mujoco']} / {cc['nrd']} | {100 * cc['agreement']:.1f} % | "
                         f"{f3(cc['phi'])} | {tes} | {d['first_obs_max_abs_diff']:.1e} |")
    return L


# ---- eval ----------------------------------------------------------------------------------------------------------
def run_eval(spec, levels, seed_items, out, workers=8, plots=0, video=0, video_opts=None, gl=None, extra_meta=None):
    """MuJoCo part of an evaluation directory (also used by footstep_baselines.py)."""
    from nedm.g1_footstep_rl import footsteps_mujoco as FM
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = make_jobs(levels, seed_items, out, plots, video, video_opts)
    meta = dict(spec=spec, name=out.name, levels=list(levels), seed_items=list(seed_items), host=socket.gethostname(),
                workers=workers, argv=sys.argv, contract="footsteps_mujoco module docstring (1)-(13)",
                episode_s=FM.EPISODE_S, **(extra_meta or {}))
    if spec["kind"] == "footsteps":
        pol = FM.load_footstep_policy(spec["path"], zero_residual=bool(spec.get("zero_residual")))
        meta["policy"] = dict(name=pol.name, **pol.meta)
        meta["env_cfg"] = {k: pol.env_cfg.get(k) for k in ("action", "obs", "score", "rules", "checkpoints")}
    else:
        meta["policy"] = dict(name=FM.spec_name(spec), **spec)
    jdump(out / "meta.json", meta)
    t0 = time.time()
    rows, snaps, steps = FM.run_pool(spec, jobs, workers, 1, gl)
    meta["mujoco_wall_s"] = round(time.time() - t0, 1)
    print(f"[mujoco] {out.name}: {len(rows)} episodes in {meta['mujoco_wall_s']} s", flush=True)
    jdump(out / "rows.json", _clean(rows))
    jdump(out / "steps.json", _clean(steps))
    if any(s is not None for s in snaps) and spec["kind"] == "footsteps":
        save_snapshots(out / "snapshots.npz", rows, snaps)
    jdump(out / "meta.json", meta)
    return rows, snaps, steps


def cmd_eval(a):
    spec = policy_spec(a)
    from nedm.g1_footstep_rl import footsteps_mujoco as FM
    name = a.name or FM.spec_name(spec)
    out = Path(a.out) if a.out else OUT / name
    if a.video:
        os.environ["MUJOCO_GL"] = a.gl
    vo = dict(fps=a.fps, size=tuple(a.size))
    run_eval(spec, a.levels, a.seeds, out, a.workers, a.plots, a.video, vo, a.gl if a.video else None,
             dict(baselines=parse_baselines(a.baseline)))
    if a.nrd:
        run_nrd(out, a.nrd_members, a.nrd_threads, a.nrd_batch, a.nrd_states)
    report(out)


# ---- NRD -------------------------------------------------------------------------------------------------------------
def run_nrd(out, members, threads=8, batch=128, keep_states=0):
    import torch
    from nedm.g1_footstep_rl import footsteps_mujoco as FM
    torch.set_num_threads(int(threads))
    out = Path(out)
    meta = json.loads((out / "meta.json").read_text())
    spec = meta["spec"]
    if spec["kind"] != "footsteps":
        raise SystemExit("--nrd needs a footstep policy")
    pol = FM.load_footstep_policy(spec["path"], zero_residual=bool(spec.get("zero_residual")))
    snaps = load_snapshots(out / "snapshots.npz")
    t0 = time.time()
    rows, steps, states = [], [], {}
    for i in range(0, len(snaps), batch):
        r, s, st = FM.nrd_rollout_courses(pol, snaps[i:i + batch], members=members, keep_states=keep_states > 0)
        rows += r
        steps += s
        if keep_states:
            for (lv, sd, m), x in st.items():
                if len({k[:2] for k in states}) < keep_states or (lv, sd) in {k[:2] for k in states}:
                    states[(lv, sd, m)] = x
    print(f"[nrd] {len(rows)} episodes in {time.time() - t0:.1f} s; first-observation max diff "
          f"{max(r['first_obs_max_abs_diff'] for r in rows):.2e}", flush=True)
    jdump(out / "nrd_rows.json", _clean(rows))
    jdump(out / "nrd_steps.json", _clean(steps))
    if states:
        arr = {}
        for (lv, sd, m), x in states.items():
            arr[f"{lv}_s{sd:05d}_m{m}_states"], arr[f"{lv}_s{sd:05d}_m{m}_gates"] = x["states"], x["gates"]
        np.savez_compressed(out / "nrd_states.npz", **arr)
    meta.update(nrd_members=list(members), nrd_wall_s=round(time.time() - t0, 1), nrd_threads=threads,
                nrd_missing_snapshots=int(len(json.loads((out / "rows.json").read_text())) - len(snaps)))
    jdump(out / "meta.json", meta)


def cmd_nrd(a):
    run_nrd(a.dir, a.nrd_members, a.nrd_threads, a.nrd_batch, a.nrd_states)
    report(a.dir)


# ---- videos ----------------------------------------------------------------------------------------------------------
def render_one(spec, level, seed, out, fps=25, size=(960, 544), t0_s=0.0, t1_s=None, plot=True, ghost=None):
    """Re-run one episode with states kept and render it (chase + top). Returns (row, mp4 path)."""
    from nedm.g1_footstep_rl import footsteps_mujoco as FM
    res = FM.run_spec_episode(spec, FM.load_spec(spec), level, seed, keep_states=True)
    if res is None:
        raise SystemExit(f"{level} seed {seed}: the old layout has no plan")
    row, ep, env, ctrl = res
    if plot:
        FM.plot_course(ep, row, Path(out).with_suffix(".png"), title=f"{ctrl.name} | {level} seed {seed}")
    FM.render_episode(env.model, row["states"], ep, row, out, policy_name=ctrl.name,
                      descriptor=getattr(ctrl, "descriptor", ""), fps=fps, size=size, t0_s=t0_s, t1_s=t1_s,
                      ghost=ghost)
    row.pop("states")
    return row, str(out)


def _render_job(job):
    os.environ["MUJOCO_GL"] = job["gl"]
    import torch
    torch.set_num_threads(1)
    return render_one(job["spec"], job["level"], job["seed"], job["out"], job["fps"], tuple(job["size"]),
                      job["t0_s"], job["t1_s"])


def cmd_video(a):
    """--policy NAME=SPEC_JSON (repeatable; SPEC_JSON as policy_spec) or the single policy flags."""
    import multiprocessing as mp
    from nedm.g1_footstep_rl import footsteps_mujoco as FM
    os.environ["MUJOCO_GL"] = a.gl
    specs = []
    for it in a.policy or []:
        name, _, js = it.partition("=")
        specs.append((name, json.loads(js)))
    if not specs:
        specs = [(a.name or None, policy_spec(a))]
    out = Path(a.out or OUT / "videos")
    jobs = []
    for lv in a.levels:
        for s in seeds_for(lv, a.seeds):
            tag = f"{lv}_s{s:05d}" if s >= 0 else lv
            for name, spec in specs:
                nm = name or FM.spec_name(spec)
                jobs.append(dict(spec=spec, level=lv, seed=s, out=str(out / f"{tag}__{nm}.mp4"), fps=a.fps,
                                 size=list(a.size), t0_s=a.t0, t1_s=a.t1, gl=a.gl, name=nm, tag=tag))
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(a.workers, len(jobs))) as pool:
        res = pool.map(_render_job, jobs, chunksize=1)
    print(f"[video] {len(jobs)} videos in {time.time() - t0:.1f} s", flush=True)
    rows = []
    for j, (row, path) in zip(jobs, res):
        rows.append(dict(policy=j["name"], level=j["level"], seed=j["seed"], video=path, outcome=row["outcome"],
                         completed=row["completed"], scored=row.get("scored"), hit5=row.get("hit5")))
        print(f"  {path}: {row['outcome']}, scored {row.get('scored')} / {row['n_targets']}", flush=True)
    if a.compose and len(specs) > 1:
        for tag in sorted({j["tag"] for j in jobs}):
            vids = [(j["name"], j["out"]) for j in jobs if j["tag"] == tag]
            FM.compose_videos([v for _, v in vids], out / f"{tag}__compare.mp4", cols=a.cols, width=a.compose_width,
                              labels=[n for n, _ in vids])
            print(f"  composed {out / f'{tag}__compare.mp4'}")
    jdump(out / "videos.json", _clean(rows))


# ---- NRD vs MuJoCo scatter -------------------------------------------------------------------------------------------
def _ranks(x):
    x = np.asarray(x, np.float64)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x), dtype=np.float64)
    for v in np.unique(x):                       # average ranks of ties
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(x, y):
    """Spearman rank correlation (average ranks for ties; nan if a side is constant or n < 3)."""
    if len(x) < 3:
        return float("nan")
    rx, ry = _ranks(x), _ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def cmd_scatter(a):
    """NRD vs MuJoCo per (eval directory, level) over several eval directories with nrd_rows.json (Spearman over the
    cells, per-episode agreement and phi)."""
    cells, eps = [], []
    for d in a.dirs:
        d = Path(d)
        _, rows, _ = load_dir(d)
        if not (d / "nrd_rows.json").exists():
            print(f"skip {d}: no nrd_rows.json")
            continue
        nrd = json.loads((d / "nrd_rows.json").read_text())
        by = {(r["level"], r["seed"]): r for r in rows}
        for lv in sorted({r["level"] for r in nrd}):
            nr = [r for r in nrd if r["level"] == lv and (r["level"], r["seed"]) in by]
            mj = [by[(r["level"], r["seed"])] for r in nr]
            for key in ("completed", "clean_completion"):
                cells.append(dict(dir=d.name, level=lv, key=key, n=len(nr),
                                  mujoco=float(np.mean([bool(r.get(key)) for r in mj])),
                                  nrd=float(np.mean([bool(r.get(key)) for r in nr]))))
            eps += [dict(dir=d.name, level=lv, mujoco=bool(m.get("completed")), nrd=bool(n.get("completed")))
                    for m, n in zip(mj, nr)]
    from nedm.g1_footstep_rl.mujoco_policy import phi
    res = {}
    rng = np.random.default_rng(0)
    for key in ("completed", "clean_completion"):
        c = [x for x in cells if x["key"] == key]
        if len(c) < 3:
            continue
        x, y = np.array([v["mujoco"] for v in c]), np.array([v["nrd"] for v in c])
        boot = [spearman(x[i], y[i]) for i in (rng.integers(0, len(c), len(c)) for _ in range(1000))]
        boot = [b for b in boot if math.isfinite(b)]
        res[key] = dict(cells=len(c), spearman=spearman(x, y),
                        ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if boot else None)
    res["episodes"] = dict(n=len(eps), agreement=float(np.mean([e["mujoco"] == e["nrd"] for e in eps])) if eps else None,
                           phi=phi([e["mujoco"] for e in eps], [e["nrd"] for e in eps]) if eps else None)
    res["cells"] = cells
    out = Path(a.out or OUT / "scatter")
    out.mkdir(parents=True, exist_ok=True)
    jdump(out / "scatter.json", _clean(res))
    print(json.dumps({k: v for k, v in _clean(res).items() if k != "cells"}, indent=1))


# ---- parity ----------------------------------------------------------------------------------------------------------
def cmd_parity(a):
    import torch
    from nedm.g1_footstep_rl import footsteps_mujoco as FM
    torch.set_num_threads(1)
    spec = policy_spec(a)
    res = FM.observation_parity(spec["path"], a.levels[0], seeds_for(a.levels[0], a.seeds)[0], n_steps=a.steps)
    out = Path(a.out or OUT / "parity")
    out.mkdir(parents=True, exist_ok=True)
    jdump(out / "parity.json", _clean(res))
    print(json.dumps({k: v for k, v in _clean(res).items() if k != "max_abs_diff_per_step"}, indent=1))
    return 0 if res["passed"] else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def pol_args(p, cells=True):
        p.add_argument("--ckpt", help="FS checkpoint (env_cfg.json, train_cfg.json next to it)")
        p.add_argument("--zero-residual", action="store_true", help="FS ablation: leg corrections set to 0")
        p.add_argument("--hfs", default=None, help="H-FS parameters as JSON")
        p.add_argument("--name", default=None)
        p.add_argument("--out", default=None)
        p.add_argument("--score", default=None, help="score overrides as JSON, e.g. '{\"strict\": false}' (S0-S1 "
                       "checkpoints); default: the policy's env config (FS) or FT.SCORE_DEFAULTS (baselines)")
        if cells:
            p.add_argument("--levels", nargs="+", default=list(NEW_CELLS))
            p.add_argument("--seeds", nargs="+", default=["dev"])

    def vid_args(p):
        p.add_argument("--gl", default=os.environ.get("MUJOCO_GL", "osmesa"))
        p.add_argument("--fps", type=int, default=25)
        p.add_argument("--size", type=int, nargs=2, default=[960, 544], help="per camera (chase + top side by side)")

    p = sub.add_parser("eval")
    pol_args(p)
    vid_args(p)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--plots", type=int, default=0, help="per-course PNGs of the first N episodes")
    p.add_argument("--video", type=int, default=0, help="mp4s of the first N episodes")
    p.add_argument("--nrd", action="store_true")
    p.add_argument("--nrd-members", type=int, nargs="+", default=[0, 1])
    p.add_argument("--nrd-threads", type=int, default=4, help="torch threads of the NRD part (keep low on a loaded host)")
    p.add_argument("--nrd-batch", type=int, default=128)
    p.add_argument("--nrd-states", type=int, default=0, help="keep the NRD states of the first N episodes (npz)")
    p.add_argument("--baseline", action="append", default=[], help="NAME=DIR for paired McNemar (repeatable)")
    p = sub.add_parser("nrd")
    p.add_argument("dir")
    p.add_argument("--nrd-members", type=int, nargs="+", default=[0, 1])
    p.add_argument("--nrd-threads", type=int, default=4, help="torch threads of the NRD part (keep low on a loaded host)")
    p.add_argument("--nrd-batch", type=int, default=128)
    p.add_argument("--nrd-states", type=int, default=0)
    p = sub.add_parser("report")
    p.add_argument("dir")
    p.add_argument("--baseline", action="append", default=[])
    p = sub.add_parser("video")
    pol_args(p)
    vid_args(p)
    p.add_argument("--policy", action="append", default=[], help="NAME=SPEC_JSON (repeatable), e.g. "
                   "'H-FS={\"kind\": \"hfs\", \"params\": {}}'")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--t0", type=float, default=0.0, help="start of the clip (s)")
    p.add_argument("--t1", type=float, default=None, help="end of the clip (s)")
    p.add_argument("--compose", action="store_true", help="side by side / grid of the policies per course")
    p.add_argument("--cols", type=int, default=1, help="grid columns of --compose (1: stacked rows)")
    p.add_argument("--compose-width", type=int, default=None)
    p = sub.add_parser("scatter")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--out", default=None)
    p = sub.add_parser("parity")
    pol_args(p)
    p.add_argument("--steps", type=int, default=24)
    a = ap.parse_args()
    fn = {"eval": cmd_eval, "nrd": cmd_nrd, "report": lambda a: report(a.dir, parse_baselines(a.baseline)),
          "video": cmd_video, "scatter": cmd_scatter, "parity": cmd_parity}[a.cmd]
    rc = fn(a)
    sys.exit(rc if isinstance(rc, int) else 0)


if __name__ == "__main__":
    main()
