"""Gate summary for one checkpoint against D on the same platform (SELECTION_v12.md section 2, Addenda A and B).

    python -B gates_summary.py --platform local --name <name> [--ref D] --out <json> --root DIR [--ref-root DIR]
Runs: <root>/runs/gates/<platform>/<name>/<run>.npz (run_gates.sh names). D's reference: <ref-root>/runs/gates/
<platform>/<ref>/ (default: the same root) with the "ref" runs (G2 repeats g2_b / g2_c, H68 block b).
Ported from the study's ~/so101_r2s_claude/v12/check/lib/gates_summary.py: --root is required (no default check
folder) and --validation names the check's validation result (was <check folder>/out/validation.json).

Addendum B:
- B2: G1, G2, G3 run in the student's own pinned look (check_eval.py arm "own"). The look each run recorded is checked
  against the arm of the name (visR / visRtbpl: the real-look parts; every other arm and D: the grey parts). D's
  reference runs are arm GR (= D's own look; own == GR bit-identical on D, 2026-10-27). A mismatch is an error (the
  checkpoint is then undecided). visR / visRtbpl also run g2_a_GR (grey-look G2): reported only.
- B1: G2 tolerance A1 = max(2.5 SD sqrt(4/3), 4); G3 = max(2.5 SD sqrt(2), 4) (one D run per plane).
- B3: the release veto from g4_h68_RL_a (block 613,000) and g4_h68_RL_b (block 613,300): grasp releases at >= 2
  distinct starts in each block. ctrl's veto is computed and reported (rank_selection.py exempts it).
- B6: every G2 / G3 result also gives the robot place rule (near_place: the jaw opens with the cube within 30 mm of
  the plate centre, or a sim place) next to the sim lenient place, per direction (lr = cube left L->R, rl = cube
  right R->L, from the task metadata).
- G4 (a)-(d) are reported only (B3 replaces their use; A3: the check is not usable as a gate).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_analyze as CA  # noqa: E402
import sel_rules as SR  # noqa: E402

_CACHE = {}
G1_RUNS = ("g1_dev100", "g1_homedev100")
G23_RUNS = ("g2_a", "g3_m3", "g3_p3")
METRICS = ("grasp", "place", "carry_stop")
DIRS = ("lr", "rl")


def run(root, plat, name, r, kind):
    f = Path(root) / f"runs/gates/{plat}/{name}/{r}.npz"
    key = str(f)
    if key in _CACHE:
        return _CACHE[key]
    if not f.exists():
        _CACHE[key] = None
        return None
    rows, info = CA.run_episodes(str(f))
    out = dict(file=str(f), info=info)
    if kind == "G1":
        out["n"], out["lenient"] = len(rows), sum(x["lenient"] for x in rows)
    elif kind == "J":
        for x in rows:
            x["row"], x["draw"] = x["ep"], 0
        st = CA.per_start(rows, len(rows))
        out["counts"] = CA.counts(st, rows)
        pl = info.get("plane")
        t = CA.j_task(str(f), tuple(pl) if pl else None, info)
        out["task"] = {k: v for k, v in t.items() if k != "per_episode"}
        out["per_episode"] = t["per_episode"]
    else:
        N = 68
        for x in rows:
            x["row"], x["draw"] = x["ep"] % N, x["ep"] // N
        st = CA.per_start(rows, N)
        out["counts"] = CA.counts(st, rows)
        out["plans"] = [s["plan"] for s in st]
    _CACHE[key] = out
    return out


def contrast(rl, gr):
    a = np.array([np.nan if v is None else v for v in rl["plans"]])
    b = np.array([np.nan if v is None else v for v in gr["plans"]])
    ok = np.isfinite(a) & np.isfinite(b)
    return float((a[ok] - b[ok]).mean()), int(ok.sum())


def collect(root, plat, name, is_ref, grey_g2):
    """-> (runs dict, required run names, report-only run names)."""
    req = list(G1_RUNS) + list(G23_RUNS) + ["g4_h68_RL_a", "g4_h68_RL_b"] + (["g2_b", "g2_c"] if is_ref else [])
    rep = ["g4_h68_GR_a", "g4_j224_RL"] + (["g4_h68_GR_b"] if is_ref else []) + (["g2_a_GR"] if grey_g2 else [])
    kind = {r: ("G1" if r.startswith("g1_") else "H" if "h68" in r else "J") for r in req + rep}
    return {r: run(root, plat, name, r, kind[r]) for r in req + rep}, req, rep


def run_look(info):
    """The look a run rendered: the recorded 'look' (check_eval.py since Addendum B), else from the arm (GR / RL)."""
    return info.get("look") or {"GR": "grey", "RL": "real"}.get(info.get("arm"))


def look_errors(R, arm, is_ref):
    """B2 checks: G1-G3 runs in the student's own look (candidates: arm 'own'); g2_a_GR in the grey look."""
    exp = SR.own_look(arm) if arm else ("grey" if is_ref else None)
    errs = []
    for r in G1_RUNS + G23_RUNS + (("g2_b", "g2_c") if is_ref else ()):
        x = R.get(r)
        if not x:
            continue
        lk, am = run_look(x["info"]), x["info"].get("arm")
        if exp and lk != exp:
            errs.append(f"{r}: look {lk} (arm {am}), not the own look {exp} of arm {arm or 'D'} (B2)")
        if not is_ref and am != "own":
            errs.append(f"{r}: arm {am}, not 'own' (B2: G1-G3 in the student's own pinned look)")
    if R.get("g2_a_GR") and run_look(R["g2_a_GR"]["info"]) != "grey":
        errs.append(f"g2_a_GR: look {run_look(R['g2_a_GR']['info'])}, not grey")
    for r in ("g4_h68_RL_a", "g4_h68_RL_b", "g4_j224_RL"):
        if R.get(r) and run_look(R[r]["info"]) != "real":
            errs.append(f"{r}: look {run_look(R[r]['info'])}, not the real look (arm RL)")
    for r in ("g4_h68_RL_a", "g4_h68_RL_b"):
        sb = {"g4_h68_RL_a": 613000, "g4_h68_RL_b": 613300}[r]
        if R.get(r) and (R[r]["info"].get("seed_base"), R[r]["info"].get("draws")) != (sb, 4):
            errs.append(f"{r}: seed base {R[r]['info'].get('seed_base')} / draws {R[r]['info'].get('draws')}, "
                        f"not {sb} / 4 (B3)")
    return errs


def by_direction(task):
    """B6: per direction the sim lenient place next to the robot place rule (near_place)."""
    return {d: dict(n=task[d]["n"], grasp=task[d]["grasp"], place_sim=task[d]["place"],
                    place_robot_rule=task[d]["near_place"], carry_stop=task[d]["carry_stop"]) for d in DIRS}


def by_direction_mean(tasks):
    return {d: {k: round(float(np.mean([by_direction(t)[d][k] for t in tasks])), 2)
                for k in ("n", "grasp", "place_sim", "place_robot_rule", "carry_stop")} for d in DIRS}


def judge(xr, ref_vals, tol, ref_dirs):
    out = {}
    for m in METRICS:
        c = xr["task"]["all"][m]
        ok = (c <= ref_vals[m] + tol[m]) if m == "carry_stop" else (c >= ref_vals[m] - tol[m])
        out[m] = dict(cand=c, D=round(ref_vals[m], 2), tol=round(tol[m], 3), pass_=bool(ok))
    out["pass"] = all(out[m]["pass_"] for m in METRICS)
    out["place_robot_rule_all"] = dict(cand=xr["task"]["all"]["near_place"])
    out["by_direction"] = by_direction(xr["task"])
    out["D_by_direction"] = ref_dirs
    return out


def summarize(plat, name, ref="D", root=HERE, ref_root=HERE, validation=None):
    is_ref = name == ref
    arm = SR.arm_of(name)
    run_name, it = SR.parse_ckpt(name)
    D, dreq, _ = collect(ref_root, plat, ref, True, False)
    X, xreq, xrep = (D, dreq, []) if is_ref and Path(root) == Path(ref_root) else \
        collect(root, plat, name, is_ref, arm in SR.REAL_LOOK_ARMS)
    miss = [f"{ref}:{k}" for k in dreq if D[k] is None] + ([] if is_ref else [f"{name}:{k}" for k in xreq if X[k] is None])
    miss_rep = [f"{name}:{k}" for k in xrep if X[k] is None]
    res = dict(platform=plat, name=name, ref=ref, arm=arm, run=run_name, it=it,
               seed=SR.parse_run(run_name)[1] if run_name else None,
               own_look=SR.own_look(arm) if arm else ("grey" if is_ref else None),
               missing=miss, missing_report_only=miss_rep, errors=[])
    if arm is None and not is_ref:
        res["errors"].append(f"{name}: not a v12 candidate name (v12_<arm>_s<seed>_it<1000|1500>)")
    res["errors"] += look_errors(X, arm, is_ref) + ([] if is_ref else [f"D reference: {e}" for e in look_errors(D, None, True)])
    shas = sorted({v["info"].get("ckpt_sha256") for v in X.values() if v and v["info"].get("ckpt_sha256")})
    ckpts = sorted({v["info"].get("ckpt") for v in X.values() if v and v["info"].get("ckpt")})
    if len(shas) > 1:
        res["errors"].append(f"the runs of {name} used {len(shas)} different checkpoints {[s[:12] for s in shas]}")
    res["ckpt"], res["ckpt_sha256"] = (ckpts[0] if len(ckpts) == 1 else ckpts), (shas[0] if len(shas) == 1 else shas)
    val = Path(validation) if validation else None
    res["check_usable_as_gate"] = json.loads(val.read_text()).get("verdict", {}).get("usable_as_gate") \
        if val and val.exists() else None
    G = {}
    # G1
    if all(X[r] and D[r] for r in G1_RUNS):
        def pc(R):
            return 100.0 * sum(R[r]["lenient"] for r in G1_RUNS) / sum(R[r]["n"] for r in G1_RUNS)
        G["G1"] = dict(cand=round(pc(X), 2), D=round(pc(D), 2), line=round(pc(D) - 3.0, 2),
                       cand_sets=[X[r]["lenient"] for r in G1_RUNS], D_sets=[D[r]["lenient"] for r in G1_RUNS])
        G["G1"]["pass"] = G["G1"]["cand"] >= G["G1"]["line"] - 1e-9
    # G2 / G3 (tolerances from D's three G2 runs on this platform; A1 and B1)
    if all(D[r] for r in ("g2_a", "g2_b", "g2_c")):
        tol2, tol3, dmean, tol_range, sd1 = {}, {}, {}, {}, {}
        for m in METRICS:
            v = [D[r]["task"]["all"][m] for r in ("g2_a", "g2_b", "g2_c")]
            tol_range[m] = max(1.5 * (max(v) - min(v)), 4.0)          # the first rule (22:12), replaced by A1
            M = np.array([[e[m] for e in D[r]["per_episode"]] for r in ("g2_a", "g2_b", "g2_c")], float)
            sd1[m] = float(np.sqrt(M.var(0, ddof=1).sum()))           # SD of one run's total (per-episode variances)
            tol2[m], tol3[m] = SR.tol_g2(sd1[m]), SR.tol_g3(sd1[m])
            dmean[m] = float(np.mean(v))
        res["tolerance"] = dict(G2=dict(rule="A1: max(2.5 x SD x sqrt(1 + 1/3), 4) (one run vs the mean of D's 3)",
                                        tol=tol2),
                                G3=dict(rule="B1: max(2.5 x SD x sqrt(2), 4) (one run vs ONE D run per plane)", tol=tol3),
                                sd_one_run=sd1, tol_first_rule_range=tol_range,
                                D_runs={m: [D[r]["task"]["all"][m] for r in ("g2_a", "g2_b", "g2_c")] for m in METRICS},
                                D_mean=dmean)
        d2dirs = by_direction_mean([D[r]["task"] for r in ("g2_a", "g2_b", "g2_c")])
        if X["g2_a"]:
            G["G2"] = judge(X["g2_a"], dmean, tol2, d2dirs)
            G["G2"]["look"] = run_look(X["g2_a"]["info"])
        if X.get("g2_a_GR"):                                          # B2: grey-look G2 of visR / visRtbpl, report only
            G["G2_grey_report"] = judge(X["g2_a_GR"], dmean, tol2, d2dirs)
            G["G2_grey_report"]["note"] = "report only (B2); D's reference = D in its own look (grey)"
        g3 = {}
        for r in ("g3_m3", "g3_p3"):
            if X[r] and D[r]:
                g3[r] = judge(X[r], {m: float(D[r]["task"]["all"][m]) for m in METRICS}, tol3, by_direction(D[r]["task"]))
        if len(g3) == 2:
            G["G3"] = dict(planes=g3, **{"pass": all(v["pass"] for v in g3.values())})
    # B3 release veto (H68, real look, blocks 613,000 / 613,300; distinct starts per block)
    def blocks(R):
        blk = {}
        for b, r in (("613000", "g4_h68_RL_a"), ("613300", "g4_h68_RL_b")):
            if R.get(r):
                c = R[r]["counts"]
                blk[b] = dict(grasp_releases=c["grasp_release"], episodes=c["episodes"], starts=c["starts_grasp_release"],
                              start_rows=c["grasp_release_starts"], loose_starts=c["loose"])
        return blk
    blk, dblk = blocks(X), blocks(D)
    sa, sb = (blk[b]["starts"] if b in blk else None for b in ("613000", "613300"))
    res["veto"] = dict(rule=f"B3: grasp releases at >= {SR.VETO_MIN_STARTS} distinct starts in block 613,000 AND in "
                            "block 613,300", blocks=blk, vetoed=SR.vetoed(sa, sb),
                       applies=SR.veto_applies(arm) if arm else None, D_blocks=dblk)
    # G4 (a)-(d): reported only
    need = ("g4_h68_RL_a", "g4_h68_GR_a", "g4_h68_RL_b", "g4_h68_GR_b", "g4_j224_RL")
    if all(D[r] for r in need) and all(X[r] for r in ("g4_h68_RL_a", "g4_h68_GR_a", "g4_j224_RL")):
        dl = max(D["g4_h68_RL_a"]["counts"]["loose"], D["g4_h68_RL_b"]["counts"]["loose"])
        dg = max(D["g4_h68_RL_a"]["counts"]["grasp_release"], D["g4_h68_RL_b"]["counts"]["grasp_release"])
        dj = D["g4_j224_RL"]["counts"]["grasp_release"]
        dc, _ = contrast(D["g4_h68_RL_a"], D["g4_h68_GR_a"])
        xc, nc = contrast(X["g4_h68_RL_a"], X["g4_h68_GR_a"])
        g4 = dict(a_loose=dict(cand=X["g4_h68_RL_a"]["counts"]["loose"], D=dl),
                  b_grasp_release=dict(cand=X["g4_h68_RL_a"]["counts"]["grasp_release"], D=dg),
                  c_j224_grasp_release=dict(cand=X["g4_j224_RL"]["counts"]["grasp_release"], D=dj, line=dj + 2),
                  d_contrast=dict(cand=round(xc, 3), D=round(dc, 3), n=nc),
                  plan_mean_RL=dict(cand=X["g4_h68_RL_a"]["counts"]["plan_mean"], D=D["g4_h68_RL_a"]["counts"]["plan_mean"]),
                  j224_RL_task=dict(place_sim_all=X["g4_j224_RL"]["task"]["all"]["place"],
                                    by_direction=by_direction(X["g4_j224_RL"]["task"])))
        g4["a_loose"]["pass_"] = g4["a_loose"]["cand"] <= dl
        g4["b_grasp_release"]["pass_"] = g4["b_grasp_release"]["cand"] <= dg
        g4["c_j224_grasp_release"]["pass_"] = g4["c_j224_grasp_release"]["cand"] <= dj + 2
        g4["d_contrast"]["pass_"] = xc < dc
        g4["pass"] = all(v["pass_"] for v in g4.values() if isinstance(v, dict) and "pass_" in v)
        g4["gates"] = False
        g4["note"] = "reported only (B3 replaces the use of G4 (a)-(d); A3: not usable as a gate)"
        G["G4_report"] = g4
    res["gates"] = G
    decided = not miss and not res["errors"]
    res["pass_G123"] = all(G.get(k, {}).get("pass", False) for k in ("G1", "G2", "G3")) if decided else None
    res["vetoed"] = res["veto"]["vetoed"] if decided else None
    res["eligible"] = SR.ckpt_test(res, arm) if arm else None      # B4 (1) test (ctrl: veto exempt)
    res["g2_places_own"] = X["g2_a"]["task"]["all"]["place"] if X.get("g2_a") else None   # B4 (2)
    res["gates_used"] = ["G1", "G2", "G3", "veto (B3; not ctrl)"]
    return res, D, X


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--platform", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--ref", default="D")
    p.add_argument("--root", required=True, help="folder with runs/gates/<platform>/<name>/")
    p.add_argument("--ref-root", default="", help="folder with D's reference runs (default: --root)")
    p.add_argument("--validation", default="", help="the check's validation result (verdict.usable_as_gate)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    res, D, X = summarize(a.platform, a.name, a.ref, a.root, a.ref_root or a.root, a.validation or None)
    if res["missing"]:
        print("missing runs:", res["missing"])
    if res["errors"]:
        print("errors:", res["errors"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    G = res["gates"]
    print(json.dumps({k: (v.get("pass") if isinstance(v, dict) else v) for k, v in G.items()}),
          "veto", res["veto"]["vetoed"], {b: v["starts"] for b, v in res["veto"]["blocks"].items()},
          "pass_G123", res["pass_G123"], "eligible", res["eligible"], "g2_places", res["g2_places_own"])


if __name__ == "__main__":
    main()
