#!/usr/bin/env python3
"""The declared Holm family of four (PLAN 7.3) for arena_gator_20260925: soil P1 / P2 from the soil results json
(scripts/ag_analyze.py with e6/analysis/spec_soil_v1.json) and rigid fixed 2 m/s P3 / P4 from the rigid results json
(E6b; RESULTS_rigid), if it exists; otherwise P3 / P4 are marked pending and the soil Holm-adjusted p-values are upper
bounds (a missing test enters as p = 1). Holm at alpha over the four one-sided cluster p-values with ag_analyze.holm, the
decision with ag_analyze.decide (improves / no meaningful difference / inconclusive; 'worse' is added in words when the
whole 90 % cluster interval lies above +margin).
  PYTHONPATH=src:scripts python scripts/ag_s2_family.py --soil <results_soil.json> [--rigid <results_rigid.json>] --out <family.json>
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ag_analyze as AA  # noqa: E402

NAMES = ['P1_soil_M3_vs_M1', 'P2_soil_A3_vs_M1', 'P3_rigid_fx2_M3_vs_M1', 'P4_rigid_fx2_A3_vs_M1']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def grab(res, name):
    for f in res.get('family', []):
        if f['name'] == name:
            return f
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--soil', required=True)
    ap.add_argument('--rigid')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    soil = json.load(open(a.soil))
    rigid = json.load(open(a.rigid)) if a.rigid and Path(a.rigid).exists() else None
    alpha = float(soil['spec'].get('alpha', 0.05)); margin = float(soil['spec'].get('margin_pts', 2.0))
    mg = int(soil['spec'].get('min_groups', 50))
    rows = []
    for n in NAMES:
        src = soil if 'soil' in n else rigid
        f = grab(src, n) if src is not None else None
        r = f['result'] if f is not None else None
        ok = r is not None and r.get('n')
        rows.append(dict(name=n, source=(a.soil if 'soil' in n else a.rigid) if f is not None else None, pending=not ok,
                         test=f.get('test') if f else None, ref=f.get('ref') if f else None, label=f.get('label') if f else None,
                         world=f.get('world') if f else ('crm' if 'soil' in n else 'rigid'),
                         result=r, p_one_sided_cluster=(r['cluster']['p_one_sided'] if ok else 1.0)))
    adj, rej = AA.holm([x['p_one_sided_cluster'] for x in rows], alpha)
    for x, p_, r_ in zip(rows, adj, rej):
        x['p_holm'] = p_; x['holm_reject'] = r_
        if x['pending']:
            x['decision'] = 'pending (no rigid results json yet)' if 'rigid' in x['name'] else 'no data'
            continue
        d = AA.decide(x['result'], p_, r_, margin, mg)
        lo, hi = x['result']['cluster']['ci90']
        if d == 'inconclusive' and lo > margin:
            d = 'inconclusive (worse: the whole 90 % interval lies above +2 points)'
        x['decision'] = d
    out = dict(schema='ag_s2_family_v1', tool='scripts/ag_s2_family.py', tool_sha256=sha(__file__), created=time.strftime('%Y-%m-%d %H:%M:%S'),
               soil=dict(file=a.soil, sha256=sha(a.soil)), rigid=dict(file=a.rigid, sha256=sha(a.rigid)) if rigid is not None else None,
               alpha=alpha, margin_pts=margin, complete=not any(x['pending'] for x in rows), tests=rows,
               note='Holm over the four one-sided cluster-bootstrap p-values; with a pending test (p = 1) the adjusted p of the others are upper bounds')
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    for x in rows:
        r = x['result'] or {}
        cl = r.get('cluster', {})
        print(f"{x['name']:26s} n={r.get('n')} diff={r.get('diff_pts')} ci90={cl.get('ci90')} p1={x['p_one_sided_cluster']:.4g} "
              f"holm={x['p_holm']:.4g} -> {x['decision']}")


if __name__ == '__main__':
    main()
