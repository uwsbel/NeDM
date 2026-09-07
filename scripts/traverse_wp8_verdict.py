#!/usr/bin/env python
"""Pre-registered verdict for the control-input augmentation runs (notes §13.9). Endpoints on f105:
(a) un-jittered true-stop prediction / in-stall hold / launch hold >= 0.30 / 0.45 / 0.45 AND each falls < 0.10 under the
    tracker-matched AR(1) probe;
(b) the decision test with the tracker in the loop reaches AUC >= 0.80 for 'stuck within 8 s' with closed-loop feasible
    completion >= 0.90.
Writes wp8_eval/verdict.json with the branch: PASS (a and b for some run), B_FAIL (a holds, b fails) or A_FAIL.
  python scripts/traverse_wp8_verdict.py --runs wp8c_* wp8d_*   (names under artifacts/traverse/wp8_eval)
"""
from __future__ import annotations
import argparse, glob, json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "artifacts/traverse/wp8_eval"
PY = sys.executable
TH = {"stop": 0.30, "hold": 0.45, "launch": 0.45, "drop": 0.10, "auc": 0.80, "feasible_done": 0.90}


def parse_analyze(path: Path) -> dict:
    """per class: the fractions this verdict needs (absolute-speed harness)"""
    out, cls = {}, None
    for line in path.read_text().splitlines():
        m = re.match(r"--- (\w+) \(n=(\d+)\)", line)
        if m:
            cls = m.group(1); continue
        if cls and "|" in line and not line.startswith("model"):
            head = line.split("|")[0].split()
            try:
                rest_done = float(head[5])
                pre = line.split("|")[1].split(); pre_lt = float(pre[1])
                hold = re.search(r"in-stall \+4s (-?[\d.]+) \(<0.5: ([\d.]+)\)", line)
                ar = re.search(r"AR\(1\) tracker-matched: pre ([\d.]+|nan) in-stall ([\d.]+|nan) rest<1 ([\d.]+|nan)", line)
                out[cls] = {"pre_stop": pre_lt, "hold": float(hold.group(2)) if hold else None, "trk_done": rest_done,
                            "pre_stop_ar": float(ar.group(1)) if ar else None, "hold_ar": float(ar.group(2)) if ar else None}
            except (ValueError, IndexError):
                pass
    return out


def ensure_ar(run: str, ckpt: Path) -> Path:
    """re-score a run whose analyze table predates the AR(1) probe"""
    a = EVAL / run / "analyze_f105.txt"
    if a.exists() and "AR(1)" in a.read_text():
        return a
    subprocess.run([PY, str(ROOT / "scripts/traverse_wp7_stall_diagnosis.py"), "model", "--arenas", "arena_f105", "--classes", "launch", "stop",
                    "--dynamics-checkpoints", str(ckpt), "--tag", f"verdict_{run}", "--roll-limit-deg", "60", "--pitch-limit-deg", "60"],
                   check=True, capture_output=True, env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"}, cwd=ROOT)
    txt = subprocess.run([PY, str(ROOT / "scripts/traverse_wp7_stall_diagnosis.py"), "analyze", "--tag", f"verdict_{run}", "--examples", "0"],
                         check=True, capture_output=True, text=True, env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"}, cwd=ROOT).stdout
    (EVAL / run).mkdir(parents=True, exist_ok=True); a.write_text(txt)
    for f in (ROOT / "artifacts/traverse/wp7_stall_diag").glob(f"model_tests_verdict_{run}.json"):
        f.unlink()
    return a


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--baselines", nargs="*", default=["wp2_mapv2_pt_dag_ro8_amd", "wp8_p6_k80_prog"])
    ap.add_argument("--policy", default="artifacts/traverse/wp3_tracker_v1")
    ap.add_argument("--max-decision", type=int, default=4, help="candidates carried into the tracker-in-the-loop decision test")
    args = ap.parse_args()
    runs = sorted(set(r for pat in args.runs for r in [Path(p).name for p in glob.glob(str(EVAL / pat))]))
    rows = {}
    for run in runs + args.baselines:
        ckpt = ROOT / "artifacts/traverse" / run / "ckpt_best.pt"
        if not ckpt.exists():
            continue
        a = ensure_ar(run, ckpt)
        p = parse_analyze(a)
        if "stop" not in p or "launch" not in p:
            continue
        r = {"stop": p["stop"]["pre_stop"], "hold": p["stop"]["hold"], "launch": p["launch"]["pre_stop"], "stop_ar": p["stop"]["pre_stop_ar"],
             "hold_ar": p["stop"]["hold_ar"], "launch_ar": p["launch"]["pre_stop_ar"], "feasible_done": p["feasible"]["trk_done"] if "feasible" in p else None}
        drops = [r[k] - r[k + "_ar"] for k in ("stop", "hold", "launch") if r[k] is not None and r[k + "_ar"] is not None]
        r["a_levels"] = bool(r["stop"] >= TH["stop"] and (r["hold"] or 0) >= TH["hold"] and r["launch"] >= TH["launch"])
        r["a_robust"] = bool(drops and max(drops) < TH["drop"])
        r["a"] = r["a_levels"] and r["a_robust"]
        r["score"] = (r["stop_ar"] or 0) + (r["hold_ar"] or 0) + (r["launch_ar"] or 0)  # jitter-robust stall reproduction
        rows[run] = r
    cands = [k for k in runs if k in rows and rows[k]["a"]]
    pool = cands or sorted([k for k in runs if k in rows], key=lambda k: -rows[k]["score"])[: args.max_decision]
    pool = pool[: args.max_decision]
    # (b) decision test with the tracker in the loop
    ck = [str(ROOT / "artifacts/traverse" / k / "ckpt_best.pt") for k in pool + [b for b in args.baselines if b in rows]]
    out = subprocess.run([PY, "-u", str(ROOT / "scripts/traverse_wp7_stall_diagnosis.py"), "decision", "--arenas", "arena_f105", "--dynamics-checkpoints", *ck,
                          "--policy", args.policy, "--max-ladders", "0", "--tag", "verdict"], capture_output=True, text=True,
                         env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"}, cwd=ROOT).stdout
    (EVAL / "verdict_decision.txt").write_text(out)
    for line in out.splitlines():
        m = re.match(r"(\S+) \[tracker\]\s+([\d.]+)\s+([\d.]+)", line)
        if m and m.group(1) in rows:
            rows[m.group(1)]["auc_tracker"] = float(m.group(2)); rows[m.group(1)]["auc_infeasible_tracker"] = float(m.group(3))
        m = re.match(r"(\S+) \[recorded controls\]\s+([\d.]+)\s+([\d.]+)", line)
        if m and m.group(1) in rows:
            rows[m.group(1)]["auc_rec"] = float(m.group(2))
    for k, r in rows.items():
        r["b"] = bool(r.get("auc_tracker", 0) >= TH["auc"] and (r.get("feasible_done") or 0) >= TH["feasible_done"])
    passed = [k for k in pool if rows[k]["a"] and rows[k]["b"]]
    branch = "PASS" if passed else ("B_FAIL" if cands else "A_FAIL")
    best = (passed or cands or pool)[0] if pool else None
    if best and not passed:
        best = max(pool, key=lambda k: (rows[k]["a"], rows[k].get("auc_tracker", 0), rows[k]["score"]))
    verdict = {"branch": branch, "best": best, "passed": passed, "a_candidates": cands, "thresholds": TH, "rows": rows}
    (EVAL / "verdict.json").write_text(json.dumps(verdict, indent=1))
    hdr = f"{'run':22s} {'stop':>5s} {'ar':>5s} {'hold':>5s} {'ar':>5s} {'launch':>6s} {'ar':>5s} {'feasDone':>8s} {'aucRec':>6s} {'aucTrk':>6s}  a  b"
    lines = [hdr]
    for k, r in sorted(rows.items(), key=lambda kv: -kv[1]["score"]):
        f = lambda v: "  -  " if v is None else f"{v:5.2f}"
        lines.append(f"{k:22s} {f(r['stop'])} {f(r['stop_ar'])} {f(r['hold'])} {f(r['hold_ar'])} {f(r['launch']):>6s} {f(r['launch_ar'])} {f(r['feasible_done']):>8s} {f(r.get('auc_rec')):>6s} {f(r.get('auc_tracker')):>6s}  {int(r['a'])}  {int(r['b'])}")
    lines.append(f"\nBRANCH: {branch}   best: {best}   passed: {passed}   a-candidates: {cands}")
    (EVAL / "verdict.txt").write_text("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    main()
