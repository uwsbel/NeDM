#!/usr/bin/env python
"""A1 deliverable: the all-arena table and the time/work trade-off curves.

Every arm picks ONE candidate per layout out of the SAME frozen candidate bank, and is then scored by
what Chrono actually recorded for that candidate (``truth.json``). Nothing here re-runs a simulation and
nothing here refits a model: the three arm files are read as produced.

    PYTHONPATH=src python scripts/traverse_wp9_a1_table.py --dir artifacts/traverse/wp9_energy

DECISION RULES (frozen here, identical across the predictive arms so the comparison is like-for-like):

  fastest / slowest   model-free heuristics: highest / lowest commanded speed in the bank
                      (``v_cmd_max``, ties broken by shorter route then candidate name).
  rule                the hand-written 'slope_aware' profile; ABSTAINS on layouts where it is absent.
  analytic / cheap    among the candidates its own gate admits, the smallest PREDICTED positive work
  nrd / nrd_floor     among those it predicts will meet the deadline; if it predicts none on time, the
                      smallest predicted time (the arm's best attempt, never an abstention in disguise).
  oracle_best         the truly cheapest Chrono-compliant candidate. This is the regret denominator,
                      not a method: it reads the answer.

FEASIBILITY GATES (only in the full-bank regime): analytic has no feasibility model, so it is gated by
its predicted time alone; cheap by ``p_feasible >= 0.5``; nrd by ``imagined_feasible``. An arm whose gate
rejects the whole bank ABSTAINS, and the abstention is counted, never silently dropped.

REGIMES:
  (a) ranking isolation ``a_compliant``  -- bank restricted to the Chrono-MISSION-COMPLIANT candidates.
      A FULL ORACLE DIAGNOSTIC: feasibility AND the deadline are handed to every arm, so all that is left
      is cost ranking. Every pick is compliant by construction. It is not a planner result.
  (a') ``a_feasible`` -- the plan's literal wording: bank restricted to the Chrono-FEASIBLE candidates
      (oracle feasibility) while deadline prediction stays each arm's own job. Picks can be late.
  (b) ``b_full`` -- the whole bank. Feasibility and deadline prediction are part of the job.

WORK IS ONLY EVER AVERAGED OVER COMPLIANT PICKS, and every table carries its own n plus the failure and
abstention counts. Paired comparisons are computed on the layouts where BOTH arms picked a compliant
route, with a layout bootstrap and an arena-cluster bootstrap.

The metric is w_pos_kj: POSITIVE mechanical shaft work in kJ. It is work, not fuel.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

KINDS = {"x": "crossing", "s": "sequence", "f": "freeform"}
K_SWEEP = [0.8, 0.9, 1.0, 1.25, 1.5, 2.0]
P_FEASIBLE_THR = 0.5
BOOT = 2000
SEED = 12345


# ----------------------------------------------------------------------------- loading

def layout_kind(key: str) -> str:
    return KINDS.get(key.split("__")[1][0], "other")


def load(d: Path) -> dict:
    truth = json.loads((d / "truth.json").read_text())
    for r in truth:
        r["kind"] = layout_kind(r["key"])
        r["id"] = f"{r['key']}|{r['candidate']}"
        r["legacy_cost"] = r["time_s"] + r["w_signed_kj"] / 10.0
    arms: dict[str, dict] = {}
    missing: list[str] = []

    p = d / "arm_analytic.json"
    if p.exists():
        a = json.loads(p.read_text())
        arms["analytic"] = {
            "pred": {k: {"w": v["pred_w_pos_kj"], "t": v["pred_time_s"]} for k, v in a["predictions"].items()},
            "gate": "time_only",
            "floor": {k: v["floor_w_pos_kj"] for k, v in a["predictions"].items()},
            "source": str(p),
        }
    else:
        missing.append(str(p))

    p = d / "arm_cheap.json"
    if p.exists():
        c = json.loads(p.read_text())
        arms["cheap"] = {
            "pred": {k: {"w": v["pred_w_pos_kj"], "t": v["pred_time_s"], "pf": v["p_feasible"]} for k, v in c["predictions"].items()},
            "gate": "p_feasible",
            "source": str(p),
        }
    else:
        missing.append(str(p))

    p = d / "arm_nrd.json"
    if p.exists():
        n = json.loads(p.read_text())
        for tag, rows in n["predictions"].items():
            arms[f"nrd_{tag}"] = {
                "pred": {k: {"w": v["pred_w_pos_kj"], "t": v["pred_time_s"], "feas": bool(v["imagined_feasible"])} for k, v in rows.items()},
                "gate": "imagined_feasible",
                "source": f"{p}::{tag}",
            }
        nrd_models = n["models"]
        nrd_primary = n.get("primary_model")
    else:
        missing.append(str(p))
        nrd_models, nrd_primary = {}, None

    # nrd + refit geometry floor: pessimistic lower bound max(imagined work, floor)
    if "analytic" in arms:
        floor = arms["analytic"]["floor"]
        for tag in [a for a in list(arms) if a.startswith("nrd_")]:
            src = arms[tag]["pred"]
            arms[f"{tag}_floor"] = {
                "pred": {k: {"w": max(v["w"], floor.get(k, 0.0)), "t": v["t"], "feas": v["feas"]} for k, v in src.items()},
                "gate": "imagined_feasible",
                "source": arms[tag]["source"] + " + energy_floor_wpos refit",
                "floor_binds": {k: bool(floor.get(k, 0.0) > v["w"]) for k, v in src.items()},
            }
    return {"truth": truth, "arms": arms, "missing": missing, "nrd_models": nrd_models, "nrd_primary": nrd_primary}


# ----------------------------------------------------------------------------- selection

def is_compliant(r: dict, k: float) -> bool:
    return bool(r["feasible"] and r["time_s"] <= k * r["deadline_base_s"])


def bank_for(rows: list[dict], regime: str, k: float) -> list[dict]:
    if regime == "a_compliant":
        return [r for r in rows if is_compliant(r, k)]
    if regime == "a_feasible":
        return [r for r in rows if r["feasible"]]
    return list(rows)


def heuristic_pick(bank: list[dict], which: str) -> dict | None:
    if not bank:
        return None
    if which == "fastest":
        return min(bank, key=lambda r: (-r["v_cmd_max"], r["route_len_m"], r["candidate"]))
    if which == "slowest":
        return min(bank, key=lambda r: (r["v_cmd_max"], r["route_len_m"], r["candidate"]))
    if which == "rule":
        c = [r for r in bank if r["candidate"] == "slope_aware"]
        return c[0] if c else None
    if which == "oracle_best":
        return min(bank, key=lambda r: (r["w_pos_kj"], r["candidate"]))
    if which == "oracle_best_legacy":
        return min(bank, key=lambda r: (r["legacy_cost"], r["candidate"]))
    raise ValueError(which)


def model_pick(bank: list[dict], arm: dict, regime: str, k: float, pf_thr: float = P_FEASIBLE_THR) -> tuple[dict | None, str]:
    """The frozen predictive rule. Returns (row, note)."""
    pred = arm["pred"]
    cand = [r for r in bank if r["id"] in pred]
    if not cand:
        return None, "no_prediction"
    if regime == "b_full":                              # own feasibility gate
        if arm["gate"] == "p_feasible":
            g = [r for r in cand if pred[r["id"]]["pf"] >= pf_thr]
        elif arm["gate"] == "imagined_feasible":
            g = [r for r in cand if pred[r["id"]]["feas"]]
        else:
            g = cand                                    # analytic: predicted time is its only gate
        if not g:
            return None, "gate_rejected_bank"
        cand = g
    if regime == "a_compliant":                         # pure cost ranking, deadline handed over
        return min(cand, key=lambda r: (pred[r["id"]]["w"], r["candidate"])), "rank_only"
    on_time = [r for r in cand if pred[r["id"]]["t"] <= k * r["deadline_base_s"]]
    if on_time:
        return min(on_time, key=lambda r: (pred[r["id"]]["w"], r["candidate"])), "ok"
    return min(cand, key=lambda r: (pred[r["id"]]["t"], r["candidate"])), "none_predicted_on_time"


ORACLES = ("oracle_best", "oracle_best_legacy")
HEURISTICS = ("fastest", "slowest", "rule") + ORACLES


def select_all(by_layout: dict[str, list[dict]], arms: dict, regime: str, k: float, pf_thr: float = P_FEASIBLE_THR) -> dict:
    """arm -> {layout: {'row':row|None,'note':str}} over the layouts whose bank is non-empty."""
    out: dict[str, dict] = {}
    names = list(HEURISTICS) + list(arms)
    for name in names:
        sel = {}
        for key, rows in by_layout.items():
            # the oracles are the metric diagnostics / the regret denominator: they always read the truly
            # compliant subset, in every regime, so that a stalled cheap run can never become "the best".
            bank = bank_for(rows, "a_compliant" if name in ORACLES else regime, k)
            if not bank:
                continue
            if name in HEURISTICS:
                row = heuristic_pick(bank, name)
                note = "ok" if row is not None else ("no_slope_aware" if name == "rule" else "empty")
            else:
                row, note = model_pick(bank, arms[name], regime, k, pf_thr)
            sel[key] = {"row": row, "note": note}
        out[name] = sel
    return out


# ----------------------------------------------------------------------------- scoring

def fail_reason(r: dict, k: float) -> str:
    if not r["feasible"]:
        if r["status"] != "completed":
            return "did_not_complete"
        if r["max_contact_n"] > 1.0:
            return "asset_contact"
        return "clearance"
    return "late" if r["time_s"] > k * r["deadline_base_s"] else "ok"


def arm_metrics(name: str, sel: dict, arms: dict, by_layout: dict, k: float, keys: list[str] | None = None) -> dict:
    ks = list(sel) if keys is None else [x for x in keys if x in sel]
    picks = [(key, sel[key]["row"]) for key in ks if sel[key]["row"] is not None]
    abst = [(key, sel[key]["note"]) for key in ks if sel[key]["row"] is None]
    comp = [(key, r) for key, r in picks if is_compliant(r, k)]
    bad = [(key, r) for key, r in picks if not is_compliant(r, k)]
    reasons: dict[str, int] = defaultdict(int)
    for _, r in bad:
        reasons[fail_reason(r, k)] += 1
    ab_reasons: dict[str, int] = defaultdict(int)
    for _, n in abst:
        ab_reasons[n] += 1
    has_opt = sum(1 for key in ks if any(is_compliant(r, k) for r in by_layout[key]))
    m = {
        "n_layouts": len(ks),
        "n_layouts_with_a_compliant_option": has_opt,
        "n_picks": len(picks),
        "n_abstentions": len(abst),
        "abstention_reasons": dict(ab_reasons),
        "n_compliant_picks": len(comp),
        "n_noncompliant_picks": len(bad),
        "compliance_rate_of_picks": (len(comp) / len(picks)) if picks else None,
        "compliant_picks_over_layouts_with_an_option": (len(comp) / has_opt) if has_opt else None,
        "failure_reasons": dict(reasons),
    }
    if comp:
        w = np.array([r["w_pos_kj"] for _, r in comp])
        t = np.array([r["time_s"] for _, r in comp])
        lg = np.array([r["legacy_cost"] for _, r in comp])
        ws = np.array([r["w_signed_kj"] for _, r in comp])
        m.update({
            "mean_w_pos_kj_of_compliant_picks": float(w.mean()),
            "median_w_pos_kj_of_compliant_picks": float(np.median(w)),
            "total_w_pos_kj_of_compliant_picks": float(w.sum()),
            "mean_time_s_of_compliant_picks": float(t.mean()),
            "mean_w_signed_kj_of_compliant_picks": float(ws.mean()),
            "mean_legacy_cost_of_compliant_picks": float(lg.mean()),
        })
        # regret vs the oracle on exactly these layouts
        orc = {key: heuristic_pick(bank_for(by_layout[key], "a_compliant", k), "oracle_best") for key, _ in comp}
        rd, rr = [], []
        for key, r in comp:
            o = orc[key]
            if o is None:
                continue
            rd.append(r["w_pos_kj"] - o["w_pos_kj"])
            rr.append(r["w_pos_kj"] / max(o["w_pos_kj"], 1e-9))
        if rd:
            m.update({
                "n_regret": len(rd),
                "regret_mean_kj": float(np.mean(rd)),
                "regret_median_kj": float(np.median(rd)),
                "regret_mean_ratio": float(np.mean(rr)),
                "regret_pct": float(100.0 * (np.mean(rr) - 1.0)),
                "regret_total_ratio": float(np.sum([r["w_pos_kj"] for _, r in comp]) / np.sum([orc[key]["w_pos_kj"] for key, _ in comp])),
            })
        # prediction error at the picked candidate (the optimiser's curse)
        if name in arms:
            pred = arms[name]["pred"]
            pw = np.array([pred[r["id"]]["w"] for _, r in comp if r["id"] in pred])
            tw = np.array([r["w_pos_kj"] for _, r in comp if r["id"] in pred])
            pt = np.array([pred[r["id"]]["t"] for _, r in comp if r["id"] in pred])
            tt = np.array([r["time_s"] for _, r in comp if r["id"] in pred])
            if pw.size:
                m.update({
                    "pred_at_pick_n": int(pw.size),
                    "pred_at_pick_work_ratio": float(np.mean(pw / np.maximum(tw, 1e-9))),
                    "pred_at_pick_work_mape_pct": float(100.0 * np.mean(np.abs(pw - tw) / np.maximum(tw, 1e-9))),
                    "pred_at_pick_work_mae_kj": float(np.mean(np.abs(pw - tw))),
                    "pred_at_pick_time_ratio": float(np.mean(pt / np.maximum(tt, 1e-9))),
                    "pred_at_pick_time_mae_s": float(np.mean(np.abs(pt - tt))),
                })
            if "floor_binds" in arms[name]:
                fb = [arms[name]["floor_binds"].get(r["id"], False) for _, r in picks]
                m["floor_binds_at_pick_rate"] = float(np.mean(fb)) if fb else None
    return m


# ----------------------------------------------------------------------------- paired stats

def paired(sel_a: dict, sel_b: dict, k: float, rng: np.random.Generator, arena_of: dict) -> dict | None:
    common = []
    for key in sel_a:
        ra, rb = sel_a[key]["row"], sel_b.get(key, {}).get("row")
        if ra is None or rb is None:
            continue
        if not (is_compliant(ra, k) and is_compliant(rb, k)):
            continue
        common.append((key, ra["w_pos_kj"], rb["w_pos_kj"], ra["time_s"], rb["time_s"]))
    if len(common) < 3:
        return None
    keys = [c[0] for c in common]
    a = np.array([c[1] for c in common]); b = np.array([c[2] for c in common])
    ta = np.array([c[3] for c in common]); tb = np.array([c[4] for c in common])
    d = a - b
    pct = 100.0 * d / np.maximum(b, 1e-9)
    idx = rng.integers(0, len(d), size=(BOOT, len(d)))
    bs_d = d[idx].mean(axis=1)
    bs_pct = pct[idx].mean(axis=1)
    bs_tot = 100.0 * (a[idx].sum(axis=1) / b[idx].sum(axis=1) - 1.0)
    # arena-cluster bootstrap
    ar = np.array([arena_of[key] for key in keys])
    uar = sorted(set(ar.tolist()))
    per_arena = {}
    for x in uar:
        mm = ar == x
        per_arena[x] = {"n": int(mm.sum()), "mean_diff_kj": float(d[mm].mean()), "mean_pct": float(pct[mm].mean())}
    bs_ar = []
    for _ in range(BOOT):
        pick = rng.choice(len(uar), size=len(uar), replace=True)
        vals = np.concatenate([pct[ar == uar[i]] for i in pick])
        bs_ar.append(vals.mean())
    bs_ar = np.array(bs_ar)
    return {
        "n_layouts": len(d),
        "mean_diff_kj": float(d.mean()),
        "se_diff_kj": float(d.std(ddof=1) / math.sqrt(len(d))),
        "ci95_diff_kj": [float(np.quantile(bs_d, 0.025)), float(np.quantile(bs_d, 0.975))],
        "mean_pct": float(pct.mean()),
        "se_pct": float(pct.std(ddof=1) / math.sqrt(len(pct))),
        "ci95_pct_layout_bootstrap": [float(np.quantile(bs_pct, 0.025)), float(np.quantile(bs_pct, 0.975))],
        "ci95_pct_arena_cluster_bootstrap": [float(np.quantile(bs_ar, 0.025)), float(np.quantile(bs_ar, 0.975))],
        "total_work_ratio_pct": float(100.0 * (a.sum() / b.sum() - 1.0)),
        "ci95_total_ratio_pct": [float(np.quantile(bs_tot, 0.025)), float(np.quantile(bs_tot, 0.975))],
        "a_better": int((d < 0).sum()), "b_better": int((d > 0).sum()), "ties": int((d == 0).sum()),
        "mean_time_diff_s": float((ta - tb).mean()),
        "per_arena": per_arena,
        "arenas_favouring_a": int(sum(1 for v in per_arena.values() if v["mean_diff_kj"] < 0)),
        "n_arenas": len(uar),
    }


# ----------------------------------------------------------------------------- text rendering

def f(x, n=1, w=8):
    return ("-" * 0).rjust(w) if x is None else f"{x:{w}.{n}f}"


def render_arm_table(title: str, rows: dict, order: list[str]) -> list[str]:
    out = [title]
    out.append(f"{'arm':<20}{'lay':>5}{'pick':>6}{'abst':>6}{'comp':>6}{'bad':>5}{'comp%':>7}"
               f"{'W+ kJ':>9}{'time s':>8}{'legacy':>8}{'regret kJ':>10}{'regret%':>9}{'pred/true':>10}{'MAPE%':>7}")
    for name in order:
        m = rows.get(name)
        if m is None:
            continue
        cr = m["compliance_rate_of_picks"]
        out.append(
            f"{name:<20}{m['n_layouts']:>5}{m['n_picks']:>6}{m['n_abstentions']:>6}{m['n_compliant_picks']:>6}"
            f"{m['n_noncompliant_picks']:>5}{(100*cr if cr is not None else float('nan')):>7.1f}"
            f"{f(m.get('mean_w_pos_kj_of_compliant_picks'),1,9)}{f(m.get('mean_time_s_of_compliant_picks'),2,8)}"
            f"{f(m.get('mean_legacy_cost_of_compliant_picks'),1,8)}{f(m.get('regret_mean_kj'),2,10)}"
            f"{f(m.get('regret_pct'),2,9)}{f(m.get('pred_at_pick_work_ratio'),3,10)}{f(m.get('pred_at_pick_work_mape_pct'),1,7)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="artifacts/traverse/wp9_energy")
    ap.add_argument("--boot", type=int, default=BOOT)
    args = ap.parse_args()
    d = Path(args.dir)
    globals()["BOOT"] = args.boot

    D = load(d)
    truth, arms = D["truth"], D["arms"]
    by_layout: dict[str, list[dict]] = defaultdict(list)
    for r in truth:
        by_layout[r["key"]].append(r)
    arena_of = {key: rows[0]["arena"] for key, rows in by_layout.items()}
    kind_of = {key: rows[0]["kind"] for key, rows in by_layout.items()}
    arenas = sorted(set(arena_of.values()))
    rng = np.random.default_rng(SEED)

    arm_order = ["oracle_best", "oracle_best_legacy", "fastest", "slowest", "rule", "analytic", "cheap"] + \
                [a for a in arms if a.startswith("nrd_") and not a.endswith("_floor")] + \
                [a for a in arms if a.endswith("_floor")]
    arm_order = [a for a in arm_order if a in HEURISTICS or a in arms]

    result = {
        "metric": "w_pos_kj = positive mechanical shaft WORK in kJ (not fuel; the installed Chrono engine has no consumption model)",
        "mission": "compliant = completed AND max_contact <= 1 N AND min_clearance >= 0.3 m AND time_s <= K*(L_min/5 + 5/1.5); K=1.0 frozen in truth.json",
        "sources": {n: a["source"] for n, a in arms.items()},
        "missing_arm_files": D["missing"],
        "nrd_models": D["nrd_models"],
        "nrd_declared_primary": D["nrd_primary"],
        "decision_rules": {
            "fastest": "highest v_cmd_max in the bank; ties -> shorter route, then candidate name",
            "slowest": "lowest v_cmd_max; same tie-break",
            "rule": "the 'slope_aware' candidate; abstains where absent",
            "predictive": "min predicted w_pos among candidates it predicts on time; if none predicted on time, min predicted time",
            "gates_in_full_bank": {"analytic": "predicted time only (no feasibility model)",
                                   "cheap": f"p_feasible >= {P_FEASIBLE_THR}",
                                   "nrd_*": "imagined_feasible", "heuristics": "none"},
            "oracle_best": "min true w_pos among truly compliant candidates -- the regret denominator, NOT a method",
            "nrd_floor": "pred = max(imagined w_pos, refit WP5 geometry floor)",
        },
        "regimes": {}, "tradeoff": {}, "counts": {
            "n_runs": len(truth), "n_layouts": len(by_layout), "arenas": arenas,
            "n_compliant_runs": sum(1 for r in truth if r["compliant"]),
            "n_feasible_runs": sum(1 for r in truth if r["feasible"]),
            "layouts_with_a_compliant_candidate": sum(1 for rows in by_layout.values() if any(r["compliant"] for r in rows)),
        },
    }

    txt: list[str] = []
    txt.append("=" * 132)
    txt.append("A1 ALL-ARENA TABLE -- positive mechanical shaft WORK (w_pos_kj), identical candidate bank, one pick per layout")
    txt.append("=" * 132)
    txt.append(f"{len(truth)} Chrono runs, {len(by_layout)} layouts, {len(arenas)} arenas. "
               f"{result['counts']['n_feasible_runs']} feasible, {result['counts']['n_compliant_runs']} mission-compliant runs, "
               f"{result['counts']['layouts_with_a_compliant_candidate']} layouts have any compliant candidate.")
    txt.append("Work is averaged ONLY over compliant picks; n is printed in every row. A failed or late pick is a failure, never a saving.")
    if D["missing"]:
        txt.append("MISSING ARM FILES (skipped): " + ", ".join(D["missing"]))
    txt.append("ALL 11 ARENAS ARE DEVELOPMENT DATA under the governing plan. Nothing below is a fresh/sealed confirmation.")
    txt.append("")

    regimes = {
        "a_compliant": "(a) RANKING ISOLATION -- ORACLE DIAGNOSTIC, NOT A PLANNER. Bank = Chrono-compliant candidates only,\n"
                       "     so feasibility AND the deadline are handed to every arm and all that remains is cost ranking.",
        "a_feasible": "(a') ORACLE FEASIBILITY, PREDICTED DEADLINE -- the plan's literal wording. Bank = Chrono-feasible\n"
                      "     candidates; each arm still has to call the deadline itself, so picks can be late.",
        "b_full": "(b) FULL BANK -- every candidate available. Feasibility and deadline prediction are part of the job.",
    }
    K0 = 1.0
    sels = {}
    for reg, blurb in regimes.items():
        sel = select_all(by_layout, arms, reg, K0)
        sels[reg] = sel
        overall = {n: arm_metrics(n, sel[n], arms, by_layout, K0) for n in arm_order}
        per_arena = {a: {n: arm_metrics(n, sel[n], arms, by_layout, K0, [key for key in sel[n] if arena_of[key] == a]) for n in arm_order} for a in arenas}
        by_kind = {kk: {n: arm_metrics(n, sel[n], arms, by_layout, K0, [key for key in sel[n] if kind_of[key] == kk]) for n in arm_order} for kk in ("crossing", "sequence", "freeform")}
        # common set where every arm that made picks picked compliant
        def common_set(names):
            return [key for key in sel["oracle_best"]
                    if all(sel[n].get(key, {}).get("row") is not None and is_compliant(sel[n][key]["row"], K0) for n in names)]
        pair_names = [n for n in arm_order if n not in ORACLES]
        common = common_set(pair_names)
        common_tbl = {n: arm_metrics(n, sel[n], arms, by_layout, K0, common) for n in arm_order}
        pair_names_norule = [n for n in pair_names if n not in ("rule", "slowest")]
        common_nr = common_set(pair_names_norule)
        common_nr_tbl = {n: arm_metrics(n, sel[n], arms, by_layout, K0, common_nr) for n in arm_order}
        pairs = {}
        for a_name in arm_order:
            for b_name in ("analytic", "cheap", "fastest"):
                if a_name == b_name or b_name not in sel or a_name in ORACLES:
                    continue
                pr = paired(sel[a_name], sel[b_name], K0, np.random.default_rng(SEED), arena_of)
                if pr:
                    pairs[f"{a_name} - {b_name}"] = pr
        result["regimes"][reg] = {"description": blurb.replace("\n", " "),
                                  "selections": {n: {key: (v["row"]["candidate"] if v["row"] else None) for key, v in sel[n].items()} for n in arm_order},
                                  "overall": overall,
                                  "per_arena": per_arena, "by_kind": by_kind,
                                  "common_compliant_layouts": {"n": len(common), "arms": pair_names, "table": common_tbl},
                                  "common_compliant_layouts_core_arms": {"n": len(common_nr), "arms": pair_names_norule, "table": common_nr_tbl},
                                  "paired": pairs}

        txt.append("-" * 132)
        txt.append(blurb)
        txt.append("-" * 132)
        txt += render_arm_table("ALL ARENAS  (lay=layouts scored, pick=picks made, abst=abstentions, comp=compliant picks, bad=non-compliant picks;\n"
                                "             W+/time/legacy/regret are over the COMPLIANT PICKS of that row only; legacy = time + w_signed/10)",
                                overall, arm_order)
        txt.append("")
        txt.append(f"  same-layout control 1: mean W+ over the {len(common)} layouts where EVERY arm (incl. the abstaining 'rule') picked a compliant route")
        txt.append("    " + "  ".join(f"{n}={common_tbl[n]['mean_w_pos_kj_of_compliant_picks']:.1f}(n={common_tbl[n]['n_compliant_picks']})"
                                      for n in arm_order if common_tbl[n].get("mean_w_pos_kj_of_compliant_picks") is not None))
        txt.append(f"  same-layout control 2: mean W+ over the {len(common_nr)} layouts where every CORE arm (fastest/analytic/cheap/nrd*, i.e. not 'rule' or 'slowest') picked a compliant route")
        txt.append("    " + "  ".join(f"{n}={common_nr_tbl[n]['mean_w_pos_kj_of_compliant_picks']:.1f}(n={common_nr_tbl[n]['n_compliant_picks']})"
                                      for n in arm_order if common_nr_tbl[n].get("mean_w_pos_kj_of_compliant_picks") is not None))
        txt.append("    (n= is printed because a row whose n is below the set size is NOT on the same layouts -- 'rule' abstains and 'slowest' is almost always late.)")
        txt.append("")
        for a in arenas:
            txt += render_arm_table(f"  arena {a}", per_arena[a], arm_order)
            txt.append("")
        for kk in ("crossing", "sequence", "freeform"):
            txt += render_arm_table(f"  layout kind: {kk}", by_kind[kk], arm_order)
            txt.append("")
        txt.append("  PAIRED work difference on the layouts where BOTH arms picked a compliant route")
        txt.append(f"  {'A - B':<34}{'n':>5}{'dkJ':>9}{'SE':>7}{'CI95 kJ':>18}{'d%':>8}{'CI95% (layout)':>20}{'CI95% (arena)':>20}{'A<B':>6}{'B<A':>6}{'arenasA':>9}")
        for kname, pr in pairs.items():
            txt.append(f"  {kname:<34}{pr['n_layouts']:>5}{pr['mean_diff_kj']:>9.2f}{pr['se_diff_kj']:>7.2f}"
                       f"  [{pr['ci95_diff_kj'][0]:>6.2f},{pr['ci95_diff_kj'][1]:>7.2f}]{pr['mean_pct']:>8.2f}"
                       f"  [{pr['ci95_pct_layout_bootstrap'][0]:>7.2f},{pr['ci95_pct_layout_bootstrap'][1]:>8.2f}]"
                       f"  [{pr['ci95_pct_arena_cluster_bootstrap'][0]:>7.2f},{pr['ci95_pct_arena_cluster_bootstrap'][1]:>8.2f}]"
                       f"{pr['a_better']:>6}{pr['b_better']:>6}{pr['arenas_favouring_a']:>4}/{pr['n_arenas']:<4}")
        txt.append("")

    # ------------------------------------------------------------------ trade-off curve
    txt.append("=" * 132)
    txt.append("TIME/WORK TRADE-OFF: deadline slack K sweep (deadline = K*(L_min/5 + 5/1.5)). Everything is recomputed at each K:")
    txt.append("the compliance label, each arm's own deadline call, and (in regime a) the bank itself.")
    txt.append("Reads: whether an apparent work saving is only a slower route being allowed.")
    txt.append("Caveat: the arms are NOT refitted at each K. Only the compliance label and each arm's own on-time threshold move.")
    txt.append("The learned arm additionally takes the K=1.0 deadline as an INPUT FEATURE, so its predictions are frozen at K=1.")
    txt.append("=" * 132)
    for reg in regimes:
        result["tradeoff"][reg] = {}
        txt.append(f"-- regime {reg}")
        txt.append(f"  {'K':>5}  " + "".join(f"{n[:13]:>15}" for n in arm_order))
        rows_w, rows_n, rows_t = [], [], []
        for K in K_SWEEP:
            sel = select_all(by_layout, arms, reg, K)
            cell = {}
            for n in arm_order:
                m = arm_metrics(n, sel[n], arms, by_layout, K)
                cell[n] = {"n_compliant_picks": m["n_compliant_picks"], "n_picks": m["n_picks"],
                           "n_abstentions": m["n_abstentions"],
                           "compliance_rate_of_picks": m["compliance_rate_of_picks"],
                           "mean_w_pos_kj_of_compliant_picks": m.get("mean_w_pos_kj_of_compliant_picks"),
                           "mean_time_s_of_compliant_picks": m.get("mean_time_s_of_compliant_picks"),
                           "regret_mean_kj": m.get("regret_mean_kj")}
            result["tradeoff"][reg][str(K)] = cell
            rows_w.append(f"  {K:>5.2f}  " + "".join(f"{(cell[n]['mean_w_pos_kj_of_compliant_picks'] or float('nan')):>15.1f}" for n in arm_order))
            rows_n.append(f"  {K:>5.2f}  " + "".join(f"{cell[n]['n_compliant_picks']:>15d}" for n in arm_order))
            rows_t.append(f"  {K:>5.2f}  " + "".join(f"{(cell[n]['mean_time_s_of_compliant_picks'] or float('nan')):>15.2f}" for n in arm_order))
        txt.append("   mean W+ kJ of compliant picks")
        txt += rows_w
        txt.append("   n compliant picks")
        txt += rows_n
        txt.append("   mean time s of compliant picks")
        txt += rows_t
        txt.append("")

    # ------------------------------------------------------------------ the optimiser's curse, made explicit
    ref = {}
    comp_rows = [r for r in truth if r["compliant"]]
    for name, arm in arms.items():
        pr = arm["pred"]
        rr = [pr[r["id"]]["w"] / max(r["w_pos_kj"], 1e-9) for r in comp_rows if r["id"] in pr]
        tr = [pr[r["id"]]["t"] / max(r["time_s"], 1e-9) for r in comp_rows if r["id"] in pr]
        ref[name] = {"n_compliant_candidates": len(rr),
                     "mean_pred_over_true_work": float(np.mean(rr)),
                     "mean_pred_over_true_time": float(np.mean(tr))}
        for reg in regimes:
            at = result["regimes"][reg]["overall"][name].get("pred_at_pick_work_ratio")
            ref[name][f"at_pick_{reg}"] = at
            ref[name][f"curse_{reg}"] = (at - ref[name]["mean_pred_over_true_work"]) if at is not None else None
    result["optimisers_curse"] = ref
    txt.append("=" * 132)
    txt.append("THE OPTIMISER'S CURSE: each arm's predicted/true work ratio over ALL compliant candidates, and at the candidate it PICKED.")
    txt.append("Selecting the minimum predicted work preferentially selects candidates the arm under-predicts, so the ratio at the pick")
    txt.append("should sit below the arm's own average ratio; the gap is the curse.")
    txt.append("=" * 132)
    txt.append(f"  {'arm':<20}{'n cand':>8}{'all cand':>10}{'a_compliant':>13}{'a_feasible':>12}{'b_full':>9}{'curse (b_full)':>16}")
    for name, r in ref.items():
        txt.append(f"  {name:<20}{r['n_compliant_candidates']:>8}{r['mean_pred_over_true_work']:>10.3f}"
                   f"{f(r['at_pick_a_compliant'], 3, 13)}{f(r['at_pick_a_feasible'], 3, 12)}{f(r['at_pick_b_full'], 3, 9)}"
                   f"{f(r['curse_b_full'], 3, 16)}")
    txt.append("")

    # ------------------------------------------------------------------ gate sensitivity (full bank only)
    gs = {}
    txt.append("=" * 132)
    txt.append("GATE SENSITIVITY (full bank): the learned arm's feasibility threshold is a free choice, so it is swept.")
    txt.append("=" * 132)
    txt.append(f"  {'p_feasible >=':>14}{'picks':>8}{'abst':>7}{'comp':>7}{'comp%':>8}{'W+ kJ':>9}{'regret kJ':>11}")
    if "cheap" in arms:
        for thr in (0.1, 0.3, 0.5, 0.7, 0.9):
            sel = select_all(by_layout, arms, "b_full", K0, thr)
            m = arm_metrics("cheap", sel["cheap"], arms, by_layout, K0)
            gs[str(thr)] = m
            txt.append(f"  {thr:>14.2f}{m['n_picks']:>8}{m['n_abstentions']:>7}{m['n_compliant_picks']:>7}"
                       f"{100 * (m['compliance_rate_of_picks'] or 0):>8.1f}{f(m.get('mean_w_pos_kj_of_compliant_picks'), 1, 9)}"
                       f"{f(m.get('regret_mean_kj'), 2, 11)}")
    result["gate_sensitivity_cheap_full_bank"] = gs
    txt.append("")

    # ------------------------------------------------------------------ planning cost, as reported by each arm
    result["inference_cost_per_candidate"] = {
        "analytic": {"ms": 0.349, "source": "arm 'analytic' report: resample + all terms, 85-waypoint route, terrain load excluded"},
        "cheap": {"ms": 0.496, "source": "arm_cheap_diagnostics.json cost{} (0.452 ms CPU features + 0.044 ms forward at batch 16, RTX 5090)"},
        "nrd": {"ms_approx": 111.0, "source": "NOT a benchmark: 1077 gap-fill routes per model in about 2 min of wall clock on the 5090 "
                                              "(arm 'nrd' report). arm_nrd.json carries no measured per-candidate inference time."},
    }
    txt.append("PLANNING COST per candidate: analytic 0.35 ms; learned predictor 0.50 ms; NRD imagination about 111 ms "
               "(wall-clock estimate from the gap-fill run, not a benchmark) -- a factor of 200-300.")
    txt.append("")

    # ------------------------------------------------------------------ verdict
    txt.append("=" * 132)
    txt.append("CONTINUATION GATE (plan A2, fixed before the pilot): >= 5% paired work reduction against the strongest")
    txt.append("inexpensive baseline at comparable deadline/completion performance, paired interval supporting a reduction,")
    txt.append("benefit across multiple terrains.")
    txt.append("=" * 132)
    verdict = {}
    for reg in regimes:
        base_scores = {}
        for b in ("analytic", "cheap", "fastest"):
            m = result["regimes"][reg]["overall"].get(b)
            if m and m.get("regret_mean_kj") is not None:
                base_scores[b] = m["regret_mean_kj"]
        strongest = min(base_scores, key=base_scores.get) if base_scores else None
        vv = {"strongest_inexpensive_baseline": strongest, "baseline_mean_regret_kj": base_scores,
              "nrd_vs_baseline": {}, "pass": False}
        for base in [b for b in ("analytic", "cheap") if b in base_scores]:
            for a_name in [a for a in arm_order if a.startswith("nrd_")]:
                pr = result["regimes"][reg]["paired"].get(f"{a_name} - {base}")
                if pr is None:
                    continue
                ok_size = pr["mean_pct"] <= -5.0
                ok_ci = pr["ci95_pct_layout_bootstrap"][1] < 0.0
                ok_ar = pr["arenas_favouring_a"] >= max(2, (pr["n_arenas"] + 1) // 2)
                cm = result["regimes"][reg]["overall"][a_name]
                bm = result["regimes"][reg]["overall"][base]
                ok_comp = (cm["compliance_rate_of_picks"] or 0) >= (bm["compliance_rate_of_picks"] or 0) - 0.02
                ok = bool(ok_size and ok_ci and ok_ar and ok_comp)
                vv["nrd_vs_baseline"][f"{a_name} - {base}"] = {
                    "baseline_is_strongest_by_regret": base == strongest,
                    "mean_pct": pr["mean_pct"], "ci95_pct": pr["ci95_pct_layout_bootstrap"],
                    "ci95_pct_arena": pr["ci95_pct_arena_cluster_bootstrap"],
                    "total_work_ratio_pct": pr["total_work_ratio_pct"],
                    "mean_diff_kj": pr["mean_diff_kj"], "n_layouts": pr["n_layouts"],
                    "arenas_favouring_nrd": f"{pr['arenas_favouring_a']}/{pr['n_arenas']}",
                    "nrd_compliance_rate": cm["compliance_rate_of_picks"], "baseline_compliance_rate": bm["compliance_rate_of_picks"],
                    "criteria": {"reduction_ge_5pct": bool(ok_size), "interval_supports_a_reduction": bool(ok_ci),
                                 "majority_of_arenas": bool(ok_ar), "comparable_compliance": bool(ok_comp)},
                    "pass": ok,
                }
                if base == strongest:
                    vv["pass"] = vv["pass"] or ok
        verdict[reg] = vv
        txt.append(f"-- regime {reg}: strongest inexpensive baseline by mean regret = {strongest} "
                   f"({', '.join(f'{k} {v:.2f} kJ' for k, v in sorted(base_scores.items(), key=lambda x: x[1]))})")
        for a_name, r in vv["nrd_vs_baseline"].items():
            txt.append(f"   {a_name}: {r['mean_pct']:+.2f}% "
                       f"[{r['ci95_pct'][0]:+.2f},{r['ci95_pct'][1]:+.2f}] layout-bootstrap, "
                       f"[{r['ci95_pct_arena'][0]:+.2f},{r['ci95_pct_arena'][1]:+.2f}] arena-cluster, "
                       f"{r['mean_diff_kj']:+.2f} kJ ({r['total_work_ratio_pct']:+.2f}% of total work) on n={r['n_layouts']}; "
                       f"arenas favouring NRD {r['arenas_favouring_nrd']}; "
                       f"compliance {(r['nrd_compliance_rate'] or 0)*100:.1f}% vs {(r['baseline_compliance_rate'] or 0)*100:.1f}% "
                       f"-> {'PASS' if r['pass'] else 'FAIL'}"
                       f"{'' if r['baseline_is_strongest_by_regret'] else '   (baseline is not the strongest one)'}")
        txt.append(f"   REGIME VERDICT: {'PASS' if vv['pass'] else 'FAIL'}")
        txt.append("")
    result["caveats"] = [
        "ALL 11 arenas (f101-f111) are development data under the governing plan. This is not a fresh or sealed confirmation.",
        "Regime a_compliant is a full oracle diagnostic: feasibility AND the deadline are given, so it measures cost ranking only and cannot support a planner claim.",
        "The NRD arm's 'moms1' model was fine-tuned on f101-f104 and checkpoint-selected on f105, so its numbers on those five arenas are in-sample; 'frozen' (arena_v1 only) is the clean out-of-sample NRD model. Both are reported.",
        "The analytic arm's feature family was explored with f105-f111 held-out numbers visible, and the learned arm's hyperparameters were chosen on f102/f105/f108 pilots. Both baselines therefore carry development-selection bias in their own favour -- which makes the negative NRD result more, not less, robust.",
        "Work is averaged over compliant picks only, and n is printed on every row; rows whose n is below the block's layout count are not on the same layouts (the 'rule' arm abstains on 300 of 385 layouts in regime a).",
        "The trade-off sweep does not refit any arm; the learned arm even receives the K=1.0 deadline as an input feature.",
        "Every arm reads the privileged arena height field (NRD's crop projection, the analytic terms, the learned arm's terrain channels). The camera-decoded variants exist only for the learned arm (arm_cheap_camera.json) and are not scored here.",
        "The 5 truth rows with no collection join are treated as clearance-passing, inherited from traverse_wp9_truth.py.",
        "Uncertainty is a layout bootstrap plus an arena-cluster bootstrap; layouts inside an arena share terrain, so the arena-cluster interval is the conservative one.",
        "NRD inference cost is a wall-clock estimate from the gap-fill run, not a benchmark.",
    ]
    result["verdict"] = verdict
    overall_pass = any(v["pass"] for v in verdict.values())
    txt.append(f"A1 VERDICT: {'PASS' if overall_pass else 'FAIL'} -- the 5% paired work-reduction gate is "
               f"{'met' if overall_pass else 'NOT met'} in any regime. A negative result here is a valid outcome and is preserved.")
    result["overall_pass"] = bool(overall_pass)

    (d / "a1_table.json").write_text(json.dumps(result, indent=1))
    text = "\n".join(txt)
    (d / "a1_table.txt").write_text(text + "\n")
    print(text)
    print(f"\nwrote {d/'a1_table.json'} and {d/'a1_table.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
