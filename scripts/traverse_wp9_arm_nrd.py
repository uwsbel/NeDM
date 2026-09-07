#!/usr/bin/env python
"""A1 arm: the NRD imagination's per-candidate predicted TIME and predicted POSITIVE shaft WORK.

One row per (layout, candidate) of the A1 truth table (``traverse_wp9_truth.py``), for every dynamics
model we have, so the assembly stage can score the frozen and the fine-tuned imagination separately.

WHY THE WORK IS RECONSTRUCTED AND NOT READ
------------------------------------------
``traverse_wp8_head.py dump`` saves ``energy=env.energy_kj`` AFTER the rollout loop. The env runs with
``auto_reset=False`` and finished envs keep being stepped until the slowest one in the batch is done, so
that array keeps integrating past a route's own termination: over the 5745 frozen-model rows it is 2.95x
the reconstructed value on average, 2.42x at the median and up to 134x at worst, and correlates with it at
only 0.59 (measured, ``validation.defect_energy_array``). It is never read here.

Instead the work is rebuilt from the imagined PHYSICAL state that the dump does store per 0.2 s in
``seq``: the 17-D state carries ``engine_motor_speed_radps`` at index 15 and
``engine_motorshaft_torque_nm`` at index 16, and their product is exactly the quantity the Chrono runner
integrates (verified on the cache: max |z1[15]*z1[16]/1000 - power| = 9.2e-7 kW over 20 f105 episodes).
Masked by the dump's own ``active`` flag and integrated at dt = 0.2 s this gives

    pred_w_pos_kj = integral of max(P, 0) dt      (PRIMARY, mirrors truth.w_pos_kj)
    pred_w_neg_kj = integral of min(P, 0) dt      (reported separately, never netted off)
    pred_w_signed_kj = pos + neg                  (legacy secondary)

It is mechanical shaft WORK, not fuel.

IMAGINED FEASIBILITY, UNIFIED
-----------------------------
The two historical imagination runners disagreed: ``traverse_wp7_imagine_cache.py`` terminated at 34.4 deg
roll / 22.9 deg pitch and required completed & not failed & not collided, while ``traverse_wp8_head.py``
terminated at 60/60 and recorded only ``completed``. This uses the wp8 dumps (60/60, matching Chrono's own
``ROLL_PITCH_ABORT_RAD``) and adds the missing geometric check offline: the same three-disc footprint
(offsets -1.9/0/+1.9 m, half width 1.3 m) Chrono itself uses for ``min_clearance_m``, swept along the
IMAGINED pose track against the layout's true obstacle discs. So

    imagined_feasible = imagined_completed and imagined_min_clearance_m >= MIN_CLEARANCE_M (0.3, mission)

The obstacle discs are privileged geometry (as in every previous imagination run); the start pose is the
camera estimate and the pose track is dead-reckoned by the model, so the clearance is the planner's own
estimate, not the truth. ``imagined_min_clearance_m`` is emitted too, so a different threshold can be
applied downstream without a re-run. Deadline compliance is deliberately NOT applied: predicting the
deadline is part of each arm's job, so only ``pred_time_s`` is emitted.

  PYTHONPATH=src python scripts/traverse_wp9_arm_nrd.py --out artifacts/traverse/wp9_energy/arm_nrd.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from nedm.traverse.layout import EpisodeLayout  # noqa: E402

IMG_DT_S = 0.2      # traverse_wp8_head.SUB (4) x CTRL_DT_S (0.05): the dump's sample period
CTRL_DT_S = 0.05    # step period, for t_end -> seconds
SPD_I, TRQ_I = 15, 16                  # engine_motor_speed_radps, engine_motorshaft_torque_nm
POSE_SL = slice(17, 20)                # x, y, yaw written by traverse_wp8_head.cmd_dump
FOOTPRINT_DISCS = (-1.9, 0.0, 1.9)     # traverse_wp3_chrono_eval.py line ~373
FOOTPRINT_HALF_W = 1.3
MIN_CLEARANCE_M = 0.3                  # the A1 mission clearance requirement

CACHES = ["artifacts/traverse/wp7_cache_v1", "artifacts/traverse/wp7_cache_sealed", "artifacts/traverse/wp8_cache_sealed2"]
# model tag -> (checkpoint, [dump dirs]).  f106/f107 were missing from the wp8 head dumps and were filled
# in by this study with the same script and the same two checkpoints (see logs/dump_f106f107_*.log).
SOURCES = {
    # train_arenas / val_arena are read off each checkpoint's own config.json: they decide which of this arm's
    # rows are IN-SAMPLE for that model, which the assembly stage must respect.
    "frozen": {"checkpoint": "artifacts/traverse/wp2_mapv2_pt_dag_ro8_amd/ckpt_best.pt",
               "dirs": ["artifacts/traverse/wp8_head/img_frozen_trk", "artifacts/traverse/wp9_energy/img_frozen_trk_f106f107"],
               "train_data": "arena_v1 caches (wp2_z2_cache_v6 + wp2_z2_cache_dagger_v2)", "train_arenas": [], "val_arena": None,
               "role": "frozen arena_v1 dynamics -- never trained on any f-arena, so out of sample on all 11"},
    "moms1":  {"checkpoint": "artifacts/traverse/wp8e_mom_s1/ckpt_best.pt",
               "dirs": ["artifacts/traverse/wp8_head/img_moms1_trk", "artifacts/traverse/wp9_energy/img_moms1_trk_f106f107"],
               "train_data": "artifacts/traverse/wp7_cache_v1", "train_arenas": ["arena_f101", "arena_f102", "arena_f103", "arena_f104"],
               "val_arena": "arena_f105",
               "role": "momentum fine-tune (pre-registered primary of the previous sealed look) -- IN SAMPLE on f101-f104, "
                       "checkpoint-selected on f105"},
}
PRIMARY = "moms1"   # the model pre-registered as primary for the previous sealed look (traverse_wp8_sealed_look.sh)
EXPLORATORY_ARENAS = ("arena_f106", "arena_f107")  # spent sealed set; every re-analysis on them is exploratory
# wp7-format imagination that stores its own per-candidate energy, for the reconstruction check
WP7_REF = {"arena_f105": "artifacts/traverse/wp7_imagine_f105_frozen",
           "arena_f106": "artifacts/traverse/wp7_imagine_sealed_frozen",
           "arena_f107": "artifacts/traverse/wp7_imagine_sealed_frozen"}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def load_layout_obstacles(caches: list[str]) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, str, str]]]:
    """layout id -> (M, 3) obstacle discs, and episode key -> (layout, candidate, arena)."""
    obst: dict[str, np.ndarray] = {}
    meta: dict[str, tuple[str, str, str]] = {}
    for c in caches:
        cdir = Path(c)
        labels = json.loads((cdir / "labels.json").read_text())
        for k, lab in labels.items():
            meta[k] = (lab["layout"], lab["candidate"], lab["arena"])
        by_layout: dict[str, str] = {}
        for k, lab in labels.items():
            by_layout.setdefault(lab["layout"], k)
        for lay, k in by_layout.items():
            with np.load(cdir / f"{k}.npz", allow_pickle=True) as z:
                layout = EpisodeLayout.from_json(json.loads(str(z["layout_json"])))
            obst[lay] = np.asarray(layout.obstacles(), np.float64).reshape(-1, 3)
    return obst, meta


def min_footprint_clearance(pose: np.ndarray, active: np.ndarray, discs: np.ndarray) -> np.ndarray:
    """(N, T, 3) imagined pose track -> (N,) minimum three-disc footprint clearance over the ACTIVE window.

    Exactly the quantity traverse_wp3_chrono_eval.py records as ``min_clearance_m``, evaluated on the
    imagined track instead of the Chrono one. +inf when the layout has no discs (never happens here)."""
    n = len(pose)
    out = np.full(n, np.inf)
    if discs.size == 0:
        return out
    c, s = np.cos(pose[..., 2]), np.sin(pose[..., 2])
    best = np.full(pose.shape[:2], np.inf)
    for off in FOOTPRINT_DISCS:
        cx, cy = pose[..., 0] + off * c, pose[..., 1] + off * s
        d = np.hypot(discs[None, None, :, 0] - cx[..., None], discs[None, None, :, 1] - cy[..., None]) - discs[None, None, :, 2] - FOOTPRINT_HALF_W
        best = np.minimum(best, d.min(axis=-1))
    best = np.where(active, best, np.inf)
    return best.min(axis=1)


def read_dump(path: Path, obst: dict[str, np.ndarray], meta: dict[str, tuple[str, str, str]], min_clear_m: float = MIN_CLEARANCE_M) -> dict[str, dict]:
    """One arena_*.npz from traverse_wp8_head.py dump -> key -> prediction row."""
    with np.load(path) as z:
        keys = [str(k) for k in z["keys"]]
        seq = z["seq"].astype(np.float32)
        active = z["active"]
        completed = z["completed"]
        t_end = z["t_end"]
        bad_energy = z["energy"].astype(np.float64)   # the defective array; carried only for the audit
    p_kw = np.where(active, seq[..., SPD_I] * seq[..., TRQ_I] / 1000.0, 0.0)
    w_pos = np.maximum(p_kw, 0.0).sum(1) * IMG_DT_S
    w_neg = np.minimum(p_kw, 0.0).sum(1) * IMG_DT_S
    pose = seq[..., POSE_SL].astype(np.float64)
    clear = np.full(len(keys), np.inf)
    by_layout: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        by_layout[meta[k][0]].append(i)
    for lay, idx in by_layout.items():
        idx = np.asarray(idx)
        clear[idx] = min_footprint_clearance(pose[idx], active[idx], obst[lay])
    out = {}
    for i, k in enumerate(keys):
        lay, cand, arena = meta[k]
        out[k] = {"layout": lay, "candidate": cand, "arena": arena,
                  "pred_w_pos_kj": float(w_pos[i]), "pred_w_neg_kj": float(w_neg[i]),
                  "pred_w_signed_kj": float(w_pos[i] + w_neg[i]),
                  "pred_time_s": float(t_end[i]) * CTRL_DT_S,
                  "imagined_completed": bool(completed[i]),
                  "imagined_min_clearance_m": float(clear[i]) if np.isfinite(clear[i]) else None,
                  "imagined_feasible": bool(completed[i] and clear[i] >= min_clear_m),
                  "_bad_energy_kj": float(bad_energy[i])}
    return out


def reconstruction_check(preds: dict[str, dict]) -> dict:
    """The frozen model's reconstructed SIGNED work against the wp7-format runner's own stored
    ``img_energy`` on the arenas where both exist. wp7's ``img_energy`` is ``energy_pess`` =
    elementwise max(auxiliary power head, state-product integral); only the rows where it EQUALS the
    state term are the like-for-like comparison, so both subsets are reported."""
    out = {}
    for arena, ref_dir in WP7_REF.items():
        p = Path(ref_dir) / "rows.json"
        if not p.exists():
            continue
        ref = {r["key"]: r for r in json.loads(p.read_text()) if r["arena"] == arena}
        ks = [k for k in preds if k in ref]
        if not ks:
            continue
        mine_s = np.array([preds[k]["pred_w_signed_kj"] for k in ks])
        mine_t = np.array([preds[k]["pred_time_s"] for k in ks])
        e7 = np.array([ref[k]["img_energy"] for k in ks])
        e7h = np.array([ref[k]["img_energy_head"] for k in ks])
        t7 = np.array([ref[k]["img_time"] for k in ks])
        state_sel = ~np.isclose(e7, e7h, rtol=1e-6)
        rec = {"n": len(ks), "time_corr": float(np.corrcoef(mine_t, t7)[0, 1]), "time_mae_s": float(np.abs(mine_t - t7).mean())}
        for name, m in (("all_rows", np.ones(len(ks), bool)), ("state_selected_rows", state_sel)):
            if m.sum() < 3:
                continue
            d = mine_s[m] - e7[m]
            rec[name] = {"n": int(m.sum()), "corr": float(np.corrcoef(mine_s[m], e7[m])[0, 1]),
                         "mae_kj": float(np.abs(d).mean()), "median_abs_kj": float(np.median(np.abs(d))),
                         "bias_kj": float(d.mean()), "rel_mae": float(np.abs(d).mean() / e7[m].mean())}
        out[arena] = rec
    return out


def discretisation_control(cache: str, arena: str, n_max: int = 200) -> dict:
    """How much of the reconstruction residual is just the dump's 0.2 s sampling? Take real Chrono power
    series, integrate at 0.05 s (the truth) and at 0.2 s (the dump's rate), and report the gap."""
    files = sorted(Path(cache).glob(f"{arena}__*.npz"))[:n_max]
    fine, coarse = [], []
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            p = z["power"][:, 0].astype(np.float64)
            e = int(z["end_frame"])
        p = p[: e + 1 if e >= 0 else len(p)]
        if p.size < 8:
            continue
        fine.append(np.maximum(p, 0).sum() * CTRL_DT_S)
        coarse.append(np.maximum(p[::4], 0).sum() * IMG_DT_S)
    fine, coarse = np.array(fine), np.array(coarse)
    d = coarse - fine
    return {"arena": arena, "n": len(fine), "quantity": "w_pos_kj",
            "corr": float(np.corrcoef(fine, coarse)[0, 1]), "mae_kj": float(np.abs(d).mean()),
            "rel_mae": float(np.abs(d).mean() / fine.mean()), "bias_kj": float(d.mean()),
            "note": "0.2 s vs 0.05 s Riemann sum on the RECORDED Chrono power; the floor of any 0.2 s reconstruction"}


def clearance_identity_check(cache: str, arena: str, collect_globs: list[str], n_max: int = 250) -> dict:
    """Does the offline footprint sweep reproduce Chrono's own ``min_clearance_m``? Feed it the RECORDED
    pose track (truncated at the route end, exactly Chrono's accumulation window) and compare."""
    import glob as _glob
    ref = {}
    for pat in collect_globs:
        for d in sorted(_glob.glob(pat)):
            f = Path(d) / "rows.jsonl"
            if not f.exists():
                continue
            for line in f.read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("min_clearance_m") is not None:
                    ref[(r["key"], r["candidate"])] = float(r["min_clearance_m"])
    mine, chrono = [], []
    for f in sorted(Path(cache).glob(f"{arena}__*.npz"))[:n_max]:
        with np.load(f, allow_pickle=True) as z:
            pose = z["pose"].astype(np.float64); e = int(z["end_frame"])
            key, cand = str(z["layout"]), str(z["candidate"])
            discs = np.asarray(EpisodeLayout.from_json(json.loads(str(z["layout_json"]))).obstacles(), np.float64).reshape(-1, 3)
        r = ref.get((key, cand))
        if r is None:
            continue
        p_ = pose[: e + 1 if e >= 0 else len(pose)][None]
        mine.append(float(min_footprint_clearance(p_, np.ones(p_.shape[:2], bool), discs)[0])); chrono.append(r)
    mine, chrono = np.array(mine), np.array(chrono)
    if len(mine) < 3:
        return {"n": len(mine), "note": "no collect rows joined"}
    return {"n": len(mine), "max_abs_diff_m": float(np.abs(mine - chrono).max()),
            "mean_abs_diff_m": float(np.abs(mine - chrono).mean()), "corr": float(np.corrcoef(mine, chrono)[0, 1]),
            "note": "offline three-disc sweep on the RECORDED pose track vs the runner's own min_clearance_m; "
                    "confirms the geometry, so any imagined-clearance error is the model's pose track, not the code"}


def defect_audit(preds: dict[str, dict]) -> dict:
    """Evidence for not reading the dump's own ``energy`` array."""
    bad = np.array([r["_bad_energy_kj"] for r in preds.values()])
    good = np.array([r["pred_w_signed_kj"] for r in preds.values()])
    ok = good > 1.0
    ratio = bad[ok] / good[ok]
    return {"n": int(ok.sum()), "mean_ratio_bad_over_reconstructed": float(ratio.mean()),
            "median_ratio": float(np.median(ratio)), "max_ratio": float(ratio.max()),
            "corr": float(np.corrcoef(bad[ok], good[ok])[0, 1]),
            "cause": "traverse_wp8_head.py saves env.energy_kj after the loop; auto_reset=False, so envs that "
                     "terminated early keep being stepped and keep integrating until the batch's slowest route ends"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", default="artifacts/traverse/wp9_energy/truth.json")
    ap.add_argument("--caches", nargs="+", default=CACHES)
    ap.add_argument("--out", default="artifacts/traverse/wp9_energy/arm_nrd.json")
    ap.add_argument("--min-clearance-m", type=float, default=MIN_CLEARANCE_M)
    ap.add_argument("--collect-globs", nargs="*", default=["artifacts/traverse/wp7_collect_f10*", "artifacts/traverse/wp8_collect_f1*"],
                    help="collection dirs whose rows.jsonl carry Chrono's own min_clearance_m (validation only)")
    args = ap.parse_args()

    truth = json.loads(Path(args.truth).read_text())
    truth_by = {(r["key"], r["candidate"]): r for r in truth}
    print(f"truth: {len(truth)} runs, {len({r['key'] for r in truth})} layouts", flush=True)

    obst, meta = load_layout_obstacles(args.caches)
    print(f"layouts with obstacle discs: {len(obst)}; cache episodes: {len(meta)}", flush=True)

    predictions, coverage, comparison = {}, {}, {}
    for tag, src in SOURCES.items():
        preds: dict[str, dict] = {}
        cov: dict[str, dict] = {}
        for d in src["dirs"]:
            for f in sorted(Path(d).glob("arena_*.npz")):
                rows = read_dump(f, obst, meta, args.min_clearance_m)
                arena = f.stem
                preds.update(rows)
                in_train = arena in src["train_arenas"]
                cov[arena] = {"n_routes": len(rows), "model": tag, "checkpoint": src["checkpoint"],
                              "in_sample_for_this_model": in_train,
                              "checkpoint_selected_on_this_arena": arena == src["val_arena"],
                              "source_dump": str(f), "policy": "artifacts/traverse/wp3_tracker_v1 (WP3 tracker, from rest at the camera start pose)",
                              "status": "exploratory (spent sealed arena)" if arena in EXPLORATORY_ARENAS else "development",
                              "gap_filled_by_this_study": "wp9_energy" in str(f)}
                print(f"[{tag}] {arena}: {len(rows)} routes", flush=True)
        # ---- emit only rows the truth table also has, keyed exactly as the truth table keys them
        out_rows, missing = {}, 0
        for k, r in preds.items():
            t = truth_by.get((r["layout"], r["candidate"]))
            if t is None:
                missing += 1
                continue
            out_rows[f"{r['layout']}|{r['candidate']}"] = {
                "pred_w_pos_kj": r["pred_w_pos_kj"], "pred_time_s": r["pred_time_s"],
                "imagined_feasible": r["imagined_feasible"], "model": tag,
                "pred_w_neg_kj": r["pred_w_neg_kj"], "pred_w_signed_kj": r["pred_w_signed_kj"],
                "imagined_completed": r["imagined_completed"],
                "imagined_min_clearance_m": r["imagined_min_clearance_m"],
                "arena": r["arena"], "source_dump": cov[r["arena"]]["source_dump"]}
        predictions[tag] = out_rows
        coverage[tag] = {"arenas": cov, "n_predictions": len(out_rows), "n_dump_rows_not_in_truth": missing,
                         "n_truth_rows_without_prediction": len(truth) - len(out_rows)}
        print(f"[{tag}] {len(out_rows)} predictions, {missing} dump rows absent from truth, "
              f"{len(truth) - len(out_rows)} truth rows unpredicted", flush=True)

        # ---- honest comparison with the truth, per arena, on MISSION-COMPLIANT runs only
        per_arena = {}
        for arena in sorted({r["arena"] for r in preds.values()}):
            for subset, cond in (("compliant", lambda t: t["compliant"]), ("feasible", lambda t: t["feasible"]), ("all", lambda t: True)):
                pv, tv, pt, tt = [], [], [], []
                for k, r in preds.items():
                    if r["arena"] != arena:
                        continue
                    t = truth_by.get((r["layout"], r["candidate"]))
                    if t is None or not cond(t):
                        continue
                    pv.append(r["pred_w_pos_kj"]); tv.append(t["w_pos_kj"])
                    pt.append(r["pred_time_s"]); tt.append(t["time_s"])
                if len(pv) < 3:
                    continue
                pv, tv, pt, tt = map(np.array, (pv, tv, pt, tt))
                per_arena.setdefault(arena, {})[subset] = {
                    "n": len(pv),
                    "true_w_pos_mean_kj": float(tv.mean()), "pred_w_pos_mean_kj": float(pv.mean()),
                    "ratio_pred_over_true": float(pv.mean() / tv.mean()),
                    "median_ratio": float(np.median(pv / np.maximum(tv, 1e-6))),
                    "pearson_w_pos": float(np.corrcoef(pv, tv)[0, 1]), "spearman_w_pos": spearman(pv, tv),
                    "mae_w_pos_kj": float(np.abs(pv - tv).mean()),
                    "true_time_mean_s": float(tt.mean()), "pred_time_mean_s": float(pt.mean()),
                    "time_ratio_pred_over_true": float(pt.mean() / tt.mean()),
                    "pearson_time": float(np.corrcoef(pt, tt)[0, 1]), "spearman_time": spearman(pt, tt),
                    "mae_time_s": float(np.abs(pt - tt).mean())}
        # within-layout ranking of w_pos among that layout's compliant candidates: the quantity A1 needs
        rhos, n_lay = [], 0
        for lay in sorted({r["layout"] for r in preds.values()}):
            pv, tv = [], []
            for k, r in preds.items():
                if r["layout"] != lay:
                    continue
                t = truth_by.get((lay, r["candidate"]))
                if t is not None and t["compliant"]:
                    pv.append(r["pred_w_pos_kj"]); tv.append(t["w_pos_kj"])
            if len(pv) >= 3:
                n_lay += 1
                rho = spearman(np.array(pv), np.array(tv))
                if np.isfinite(rho):
                    rhos.append(rho)
        overall = {}
        for subset, cond in (("compliant", lambda t: t["compliant"]), ("feasible", lambda t: t["feasible"]), ("all", lambda t: True)):
            pv, tv = [], []
            for k, r in preds.items():
                t = truth_by.get((r["layout"], r["candidate"]))
                if t is not None and cond(t):
                    pv.append(r["pred_w_pos_kj"]); tv.append(t["w_pos_kj"])
            pv, tv = np.array(pv), np.array(tv)
            overall[subset] = {"n": len(pv), "ratio_pred_over_true": float(pv.mean() / tv.mean()),
                               "pearson_w_pos": float(np.corrcoef(pv, tv)[0, 1]), "spearman_w_pos": spearman(pv, tv),
                               "mae_w_pos_kj": float(np.abs(pv - tv).mean())}
        # imagined-feasibility confusion against Chrono feasibility (not the deadline: that stays predicted)
        conf = {}
        for rule, flag in (("completed_only_wp8_rule", "imagined_completed"), ("completed_and_clearance_unified", "imagined_feasible")):
            c = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
            for k, r in preds.items():
                t = truth_by.get((r["layout"], r["candidate"]))
                if t is None:
                    continue
                a, b = bool(r[flag]), bool(t["feasible"])
                c["tp" if (a and b) else "fp" if (a and not b) else "fn" if (b and not a) else "tn"] += 1
            c["accept_rate"] = (c["tp"] + c["fp"]) / max(sum(c[x] for x in ("tp", "fp", "tn", "fn")), 1)
            conf[rule] = c
        clr = np.array([r["imagined_min_clearance_m"] for r in preds.values() if r["imagined_min_clearance_m"] is not None])
        chr_clr, img_clr, done_c = [], [], []
        for k, r in preds.items():
            t = truth_by.get((r["layout"], r["candidate"]))
            if t is not None and t.get("min_clearance_m") is not None and r["imagined_min_clearance_m"] is not None:
                chr_clr.append(t["min_clearance_m"]); img_clr.append(r["imagined_min_clearance_m"]); done_c.append(t["status"] == "completed")
        chr_clr, img_clr, done_c = np.array(chr_clr), np.array(img_clr), np.array(done_c)
        cmp_ = {}
        for nm, m in (("all_runs", np.ones(len(chr_clr), bool)), ("chrono_completed_only", done_c)):
            if m.sum() > 2:
                cmp_[nm] = {"n": int(m.sum()), "corr": float(np.corrcoef(img_clr[m], chr_clr[m])[0, 1]),
                            "mae_m": float(np.abs(img_clr[m] - chr_clr[m]).mean())}
        conf["clearance"] = {
            "n_routes_completed_but_rejected_by_clearance": int(sum(1 for r in preds.values() if r["imagined_completed"] and not r["imagined_feasible"])),
            "imagined_min_clearance_m": {"median": float(np.median(clr)), "p05": float(np.quantile(clr, 0.05)),
                                         "frac_below_threshold": float((clr < args.min_clearance_m).mean()),
                                         "n_below_threshold": int((clr < args.min_clearance_m).sum())},
            "chrono_min_clearance_m": {"median": float(np.median(chr_clr)), "n_below_threshold": int((chr_clr < args.min_clearance_m).sum())},
            "vs_chrono": cmp_,
            "note": "the all_runs gap is dominated by Chrono runs that never completed (they stall far from any asset, "
                    "so their recorded window keeps a large clearance while imagination drives on)"}
        comparison[tag] = {"per_arena": per_arena, "overall": overall,
                           "within_layout_spearman_w_pos_compliant": {
                               "n_layouts": len(rhos), "n_layouts_ge3_compliant": n_lay,
                               "mean_rho": float(np.mean(rhos)) if rhos else None,
                               "median_rho": float(np.median(rhos)) if rhos else None,
                               "frac_positive": float(np.mean(np.array(rhos) > 0)) if rhos else None},
                           "imagined_feasible_vs_chrono_feasible": conf}
        if tag == "frozen":
            recon = reconstruction_check(preds)
            defect = defect_audit(preds)

    validation = {
        "reconstruction_vs_wp7_img_energy": recon,
        "reconstruction_note": ("wp7's img_energy = max(power-head integral, state-product integral); only the "
                                "state_selected_rows are like-for-like with the reconstruction. Compared on the "
                                "frozen model, the only one both runners share."),
        "power_identity_check": {"claim": "z1[15]*z1[16]/1000 equals the runner's recorded power series exactly",
                                 "max_abs_diff_kw": 9.15e-07, "n_episodes": 20, "cache": "wp7_cache_v1 arena_f105",
                                 "note": "engine.GetMotorSpeed() and transmission.GetOutputMotorshaftSpeed() agree in this vehicle configuration"},
        "sampling_floor": discretisation_control(CACHES[0], "arena_f105"),
        "clearance_identity_check": clearance_identity_check(CACHES[0], "arena_f105", args.collect_globs),
        "defect_energy_array": defect,
    }
    payload = {
        "arm": "nrd_imagination",
        "metric": "w_pos_kj = integral of max(engine_motor_speed * motorshaft_torque, 0) dt; mechanical shaft WORK, not fuel",
        "primary_model": PRIMARY,
        "predictions_format": "predictions[<model_tag>]['<layout>|<candidate>'] -> row; the same '<layout>|<candidate>' "
                              "key the other A1 arms use. Model tags: " + ", ".join(SOURCES),
        "models": {t: {k: s[k] for k in ("checkpoint", "dirs", "train_data", "train_arenas", "val_arena", "role")} for t, s in SOURCES.items()},
        "feasibility_rule": {"roll_pitch_limit_deg": 60.0, "matches": "traverse_wp3_chrono_eval.ROLL_PITCH_ABORT_RAD",
                             "rule": "imagined_completed and imagined_min_clearance_m >= %.2f" % args.min_clearance_m,
                             "footprint": {"disc_offsets_m": list(FOOTPRINT_DISCS), "half_width_m": FOOTPRINT_HALF_W},
                             "obstacles": "layout.obstacles() -- privileged true discs, house included, as in every previous imagination run",
                             "deadline": "NOT applied; pred_time_s is emitted so deadline prediction stays part of the arm's decision"},
        "horizon_s": 30.0, "imagination_dt_s": IMG_DT_S,
        "predictions": predictions, "coverage": coverage, "validation": validation, "comparison_to_truth": comparison,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    (out.parent / (out.stem + "_summary.json")).write_text(json.dumps(
        {k: v for k, v in payload.items() if k != "predictions"}, indent=1))
    print(f"\nwrote {out} ({sum(len(v) for v in predictions.values())} predictions over {len(predictions)} models)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
