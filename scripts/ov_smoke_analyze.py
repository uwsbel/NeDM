#!/usr/bin/env python3
"""Smoke-test read-out and the frozen decision rule of offroad_vehicles_20260927 (module M3; PLAN sections 2.3, 2.4).
numpy only (runs locally on a synced copy, or on the cluster login node with python3 + numpy).

  python3 -B ov_smoke_analyze.py --tasks <smoke task file> --runs $G4/soil_v1/runs [--out-root $G4/soil_v1] \
      [--stored-gator $G3/soil_v1/runs:gator__] [--path-map CLUSTER_PREFIX=LOCAL_PREFIX ...] --json out.json [--text out.txt]

WHAT IS COMPARED
  Primary endpoint = goal not reached (outcome status != goal_reached) on sample A (144 routes, 24 groups), each
  vehicle arm paired route by route with the same-launch Gator re-drive 'gatorctl' (both drives validated).
  A vehicle is BETTER than the Gator if all hold (PLAN 2.4, declared before any drive):
    1. its validity gates pass (below);
    2. its failure rate is lower by >= 10 points (point estimate, on the paired routes);
    3. exact one-sided McNemar p <= its Holm level (alpha 0.05; family = the primary arms driven among polaris, m113,
       m113_g4; the p value is also printed against a family of 3);
    4. the 95 % bootstrap interval (10,000 resamples of the 24 groups, seed 20260927) of the difference
       (vehicle - Gator failure rate) lies below 0;
    5. its failure rate is below 80.6 % (116/144, the stored Gator with 0.08 m larger soil wheels).
  M113: the better (lower failure rate on A) of m113 / m113_g4 is the vehicle's result; both are in the Holm family.
  Consistency (polaris): its failure rate on sample B's straight 6 m/s routes must be below the stored Gator's 83.3 %;
  if A says better and B does not, the verdict is MIXED (do not collect without the user).
  The reference must hold: the gatorctl re-drives and the bit-identity rows (HMMWV and Gator through the new
  dispatcher) must have the same end state as the stored runs on >= 95 % of rows (pooled); otherwise the verdict is
  BLOCKED (the Gator path or the build changed). Identical arrays are reported, not gating (CRITIC item 3).
VALIDITY GATES per arm (PLAN 2.3; denominators = the arm's run:true rows of the task file)
  G1 >= 95 % of ids complete and passing crm_qa.check; G2 < 1 % crashed (failure record without a completed run and
  without a launch failure) / non-finite / explosion / unreadable / shape; G3 < 5 % ids with a failed launch check
  (collection_failure.json or the id's log naming the settled-launch check, or initial_state_validation passed=false);
  G4 belly flag (lowest hull point > 0.05 m under the undisturbed surface for > 1 s in a row, vehicle_extra.npz
  belly_clearance_min_m, 0.05 s frames, as ag_s1_gator_soil_qa.py) on <= 10 % of validated drives, belly data present
  on >= 95 % of them; G5 vehicle record: outcome.json 'vehicle' block whose name is the arm's vehicle on every
  validated drive.
REPORTED, NOT GATING
  Polaris sensitivity arms polaris_pc / polaris_4wd / polaris_w08 (paired with polaris; 'driveline-dependent' if
  polaris and polaris_pc or polaris_4wd differ by > 10 points), groups with any goal (of 24), the H_full-pick goal rate
  on B, end-state mix, simulated seconds per route, wall seconds per simulated second (loop: 1 / crm.rtf; with setup:
  outcome wall_s / elapsed_s) overall and on MI350X hosts, cost ratio against gatorctl, failure rate with belly-flagged
  drives counted as failures, array identity of the bit-identity and gatorctl rows. Ceiling rule: if polaris fails
  < 12 % of B's straight 6 m/s routes, the report says the straight route already nearly meets the bar.
TEST OPTIONS (never for the decision; the output is then labelled TEST): --arm-source ARM=DIR[:PREFIX] reads the arm's
  sample-A / bitid runs from DIR/PREFIX<collect_v1 id> (e.g. HMMWV collect_v1 runs as a fake 'polaris');
  --b-from-stored V=hmmwv|gator feeds the B rows of V with the stored runs named in the rows; --test-relax-gates
  belly,vehicle (only with --test-label).
"""
import argparse, json, math, os, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True     # never leave __pycache__ in a frozen source tree
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import crm_qa  # noqa: E402  (numpy only)

DT = 0.05
ALPHA = 0.05
MIN_GAIN = 0.10
WHEEL_REF = 116 / 144            # stored Gator + 0.08 m wheels on sample A (scratch/S3/sample_A.json fail.gatorR8)
B_GATOR_STRAIGHT6 = 0.8333       # stored Gator, sample B straight 6 m/s (scratch/S3/sample_B.json)
CEILING = 0.12
AGREE_MIN = 0.95
BOOT_N, BOOT_SEED = 10000, 20260927
PRIMARY = ('polaris', 'm113', 'm113_g4')
SENS = ('polaris_pc', 'polaris_4wd', 'polaris_w08')
DRIVELINE_ARMS = ('polaris_pc', 'polaris_4wd')
G3_RUNS = '/work1/dannegrut/harry/experiments/arena_gator_20260925/soil_v1/runs'
MI350X_HOST = 'k007-005'


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def longest_run(m):
    best = cur = 0
    for v in m:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def mcnemar_one_sided(b, c):
    """P(X >= b), X ~ Bin(b + c, 1/2): b = routes only the Gator fails, c = routes only the vehicle fails."""
    n = b + c
    if n == 0:
        return 1.0
    return float(sum(math.comb(n, k) for k in range(b, n + 1)) / 2 ** n)


def mcnemar_two_sided(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n))


def holm(pvals):
    """{name: p} -> {name: (level, rejected)} (step-down, alpha 0.05)."""
    order = sorted(pvals, key=lambda k: pvals[k])
    m, out, still = len(order), {}, True
    for i, k in enumerate(order):
        level = ALPHA / (m - i)
        rej = still and pvals[k] <= level
        still = rej
        out[k] = (level, rej)
    return out


def group_bootstrap(pairs, n=BOOT_N, seed=BOOT_SEED):
    """pairs: [(group, fail_v, fail_g)] -> 95 % percentile interval of mean(fail_v) - mean(fail_g), groups resampled."""
    by = defaultdict(lambda: [0, 0, 0])
    for g, fv, fg in pairs:
        b = by[g]; b[0] += fv; b[1] += fg; b[2] += 1
    gs = sorted(by)
    if not gs:
        return None
    v = np.array([by[g][0] for g in gs], float); w = np.array([by[g][1] for g in gs], float); c = np.array([by[g][2] for g in gs], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(gs), size=(n, len(gs)))
    d = (v[idx].sum(1) - w[idx].sum(1)) / c[idx].sum(1)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


class PathMap:
    def __init__(self, specs):
        self.m = [tuple(s.split('=', 1)) for s in specs or []]

    def __call__(self, p):
        for a, b in self.m:
            if p.startswith(a):
                return b + p[len(a):]
        return p


def npz_equal(d1, d2, names=('trajectory.npz', 'crm_extra.npz', 'anchor_state.npz', 'command_reference.npz')):
    """identical arrays in the listed npz files (both present): (all_equal, first difference or None)."""
    for n in names:
        p1, p2 = os.path.join(d1, n), os.path.join(d2, n)
        if not (os.path.exists(p1) and os.path.exists(p2)):
            continue
        try:
            a, b = np.load(p1), np.load(p2)
        except Exception as e:  # noqa: BLE001
            return False, f'{n}: unreadable ({type(e).__name__})'
        if sorted(a.files) != sorted(b.files):
            return False, f'{n}: different arrays'
        for k in a.files:
            x, y = a[k], b[k]
            if x.shape != y.shape:
                return False, f'{n}:{k} shape {x.shape} vs {y.shape}'
            if not np.array_equal(x, y, equal_nan=x.dtype.kind == 'f'):
                diff = float(np.nanmax(np.abs(x.astype(float) - y.astype(float)))) if x.dtype.kind in 'fiu' else None
                return False, f'{n}:{k} max |diff| {diff}'
    return True, None


def read_run(d, out_root, rid):
    """one drive -> record (complete, qa, launch, status, sim, wall, host, vehicle name, belly)."""
    x = dict(dir=d, complete=os.path.isfile(os.path.join(d, 'episode_complete.json')))
    fr = load(os.path.join(out_root, 'failed', f'{rid}.json')) if out_root else None
    x['attempts'] = int(fr.get('attempts', 0)) if fr else 0
    cf = load(os.path.join(d, 'collection_failure.json'))
    launch_fail = bool(cf and 'launch' in str(cf.get('error', '')).lower())
    iv = load(os.path.join(d, 'initial_state_validation.json'))
    if iv is not None and not iv.get('passed', False):
        launch_fail = True
    if out_root and (fr or cf):
        try:
            with open(os.path.join(out_root, 'logs', f'{rid}.log'), errors='replace') as f:
                launch_fail = launch_fail or 'Invalid settled launch' in f.read()
        except OSError:
            pass
    x['launch_fail'] = launch_fail
    x['failure_record'] = bool(fr or cf)
    if not x['complete']:
        return x
    q = crm_qa.check(d)
    x['validated'], x['qa_flag'] = bool(q['ok']), q.get('flag')
    o = load(os.path.join(d, 'outcome.json')) or {}
    x['status'] = o.get('status')
    x['fail'] = int(o.get('status') != 'goal_reached')
    x['sim_s'] = float(o.get('elapsed_s') or 0)
    x['wall_s'] = float(o.get('wall_s') or 0)
    rtf = (o.get('crm') or {}).get('rtf_sim_over_wall')
    x['loop_wall_per_sim'] = 1.0 / rtf if rtf else None
    vb = o.get('vehicle')
    x['vehicle'] = vb.get('name') if isinstance(vb, dict) else vb
    req = load(os.path.join(d, 'collection_request.json')) or {}
    x['host'] = req.get('host')
    try:
        b = np.load(os.path.join(d, 'vehicle_extra.npz'))['belly_clearance_min_m'].astype(float)
        deep = b < -0.05
        x['belly_min_m'] = float(np.nanmin(b)) if b.size else None
        x['belly_flag'] = longest_run(deep) * DT > 1.0
    except Exception as e:  # noqa: BLE001
        x['belly_error'] = type(e).__name__
    return x


def rate(xs):
    xs = list(xs)
    return (float(np.mean(xs)) if xs else None, len(xs))


def pct(v):
    return 'n/a' if v is None else f'{100 * v:.1f} %'


def gates(arm, rows, recs, vehicle, relax):
    n = len(rows)
    comp = [recs[r['id']] for r in rows if recs[r['id']]['complete']]
    val = [x for x in comp if x.get('validated')]
    bad_phys = [x for x in comp if x.get('qa_flag') in ('nonfinite', 'explosion', 'unreadable', 'shape')]
    crashed = [recs[r['id']] for r in rows if not recs[r['id']]['complete'] and recs[r['id']]['failure_record'] and not recs[r['id']]['launch_fail']]
    launch = [recs[r['id']] for r in rows if recs[r['id']]['launch_fail']]
    belly_ok = [x for x in val if 'belly_flag' in x]
    belly = rate(x['belly_flag'] for x in belly_ok)
    vnames = Counter(str(x.get('vehicle')) for x in val)
    g = dict(rows=n, complete=len(comp), validated=len(val), not_run_or_in_flight=n - len(comp) - len(crashed) - len([x for x in launch if not x['complete']]),
             qa_flags=dict(Counter(x.get('qa_flag') for x in comp if not x.get('validated'))),
             crashed=len(crashed), bad_physics=len(bad_phys), launch_failures=len(launch),
             belly_flag=belly[0], belly_data=len(belly_ok), vehicle_names=dict(vnames))
    g['G1_validated'] = len(val) / n >= 0.95 if n else False
    g['G2_crash'] = (len(crashed) + len(bad_phys)) / n < 0.01 if n else False
    g['G3_launch'] = len(launch) / n < 0.05 if n else False
    if 'belly' in relax:
        g['G4_belly'] = None
    else:
        g['G4_belly'] = bool(val) and len(belly_ok) >= 0.95 * len(val) and belly[0] is not None and belly[0] <= 0.10
    g['G5_vehicle'] = None if 'vehicle' in relax else (bool(val) and vnames == Counter({vehicle: len(val)}))
    g['pass'] = all(g[k] is not False for k in ('G1_validated', 'G2_crash', 'G3_launch', 'G4_belly', 'G5_vehicle'))
    return g


def cost(xs):
    xs = [x for x in xs if x.get('validated')]
    loop = [x['loop_wall_per_sim'] for x in xs if x.get('loop_wall_per_sim')]
    loop350 = [x['loop_wall_per_sim'] for x in xs if x.get('loop_wall_per_sim') and str(x.get('host', '')).startswith(MI350X_HOST)]
    wall = sum(x['wall_s'] for x in xs); sim = sum(x['sim_s'] for x in xs)
    return dict(n=len(xs), sim_h=round(sim / 3600, 3), mean_sim_s=round(sim / len(xs), 2) if xs else None,
                loop_wall_per_sim=round(float(np.mean(loop)), 3) if loop else None,
                loop_wall_per_sim_mi350x=round(float(np.mean(loop350)), 3) if loop350 else None, n_mi350x=len(loop350),
                wall_per_sim_with_setup=round(wall / sim, 3) if sim else None,
                hosts=dict(Counter(str(x.get('host', '?')).rsplit('-v', 1)[0] for x in xs)))


def paired(a_recs, b_recs, groups):
    """a = vehicle, b = reference; both dicts pair_id -> record. -> stats on validated pairs."""
    ids = sorted(p for p in a_recs if p in b_recs and a_recs[p].get('validated') and b_recs[p].get('validated'))
    fa = [a_recs[p]['fail'] for p in ids]; fb = [b_recs[p]['fail'] for p in ids]
    bb = sum(1 for x, y in zip(fa, fb) if y and not x)     # only the reference fails
    cc = sum(1 for x, y in zip(fa, fb) if x and not y)     # only the vehicle fails
    fail_a = float(np.mean(fa)) if ids else None; fail_b = float(np.mean(fb)) if ids else None
    ci = group_bootstrap([(groups[p], x, y) for p, x, y in zip(ids, fa, fb)])
    belly_as_fail = float(np.mean([int(a_recs[p]['fail'] or bool(a_recs[p].get('belly_flag'))) for p in ids])) if ids else None
    return dict(n_pairs=len(ids), n_groups=len({groups[p] for p in ids}), fail=fail_a, fail_ref=fail_b,
                diff=None if fail_a is None else fail_a - fail_b, only_ref_fails=bb, only_vehicle_fails=cc,
                p_one_sided=mcnemar_one_sided(bb, cc), p_two_sided=mcnemar_two_sided(bb, cc), ci95=ci,
                fail_belly_as_fail=belly_as_fail)


def describe(recs, groups):
    val = [(p, x) for p, x in recs.items() if x.get('validated')]
    by_g = defaultdict(list)
    for p, x in val:
        by_g[groups[p]].append(x['fail'])
    return dict(status=dict(Counter(x['status'] for _, x in val)), groups_with_goal=sum(1 for v in by_g.values() if min(v) == 0),
                groups=len(by_g), fail=rate(x['fail'] for _, x in val)[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tasks', required=True)
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out-root', default=None, help='the launch output folder (failed/, logs/); default: runs/..')
    ap.add_argument('--stored-gator', default=f'{G3_RUNS}:gator__', help='DIR[:PREFIX] of the stored Gator sample-A runs')
    ap.add_argument('--path-map', nargs='*', default=[], help='CLUSTER_PREFIX=LOCAL_PREFIX for paths named in the rows')
    ap.add_argument('--sample-a', default=str(HERE.parent / 'artifacts/traverse/offroad_vehicles_20260927/scratch/S3/sample_A.json'))
    ap.add_argument('--arm-source', nargs='*', default=[], help='TEST: ARM=DIR[:PREFIX]')
    ap.add_argument('--b-from-stored', nargs='*', default=[], help='TEST: V=hmmwv|gator')
    ap.add_argument('--test-label', default=None)
    ap.add_argument('--test-relax-gates', default='')
    ap.add_argument('--json', required=True)
    ap.add_argument('--text', default=None)
    a = ap.parse_args()
    relax = {s for s in a.test_relax_gates.split(',') if s}
    if (a.arm_source or a.b_from_stored or relax) and not a.test_label:
        raise SystemExit('--arm-source / --b-from-stored / --test-relax-gates are test options: give --test-label')
    pm = PathMap(a.path_map)
    out_root = a.out_root or str(Path(a.runs).parent)
    rows = [r for r in json.load(open(a.tasks)) if r.get('run', True)]
    SA = json.load(open(a.sample_a)) if os.path.exists(a.sample_a) else {'fail': {'gator': 135 / 144}}   # stored values only
    src = {}
    for s in a.arm_source:
        arm, rest = s.split('=', 1)
        d, _, pre = rest.partition(':')
        src[arm] = (d, pre)
    bstored = dict(s.split('=', 1) for s in a.b_from_stored)

    # ---------------------------------------------------------------- read every drive
    arms = defaultdict(list)
    for r in rows:
        arms[r.get('arm') or r['id'].split('__', 1)[0]].append(r)
    recs, groups = {}, {}
    A = defaultdict(dict)          # arm -> pair_id -> record
    Bv = defaultdict(lambda: defaultdict(dict))   # vehicle -> arm short name -> group -> record
    for arm, rs in arms.items():
        for r in rs:
            if r.get('sample') == 'B':
                v = r['vehicle']
                if v in bstored:
                    x = dict(complete=False, failure_record=False, launch_fail=False)
                    for short, sd in (r.get(f'stored_{bstored[v]}') or {}).items():
                        x = read_run(pm(sd), None, os.path.basename(sd))
                        Bv[v][short][r['group']] = x
                    recs[r['id']] = x
                else:
                    x = read_run(os.path.join(a.runs, r['id']), out_root, r['id'])
                    recs[r['id']] = x
                    for arm_name in r['arms']:
                        Bv[v][arm_name[:-len(v) - 1] if arm_name.endswith('_' + v) else arm_name][r['group']] = x
                continue
            if arm in src:
                d, pre = src[arm]
                x = read_run(os.path.join(d, pre + r['pair_id']), None, pre + r['pair_id'])
            else:
                x = read_run(os.path.join(a.runs, r['id']), out_root, r['id'])
            recs[r['id']] = x
            if r.get('sample') in ('A', 'bitid'):
                A[arm][r['pair_id']] = x
                groups[r['pair_id']] = r['group']
    # stored Gator runs for the gatorctl rows
    sd, _, spre = a.stored_gator.partition(':')
    stored = {p: read_run(os.path.join(pm(sd), spre + p), None, spre + p) for p in A.get('gatorctl', {})}

    # ---------------------------------------------------------------- reference checks (outcome level)
    ref = dict(gatorctl=None, bitid={})
    agree_n = agree_same = 0
    if A.get('gatorctl'):
        same, ident, n, disc = 0, 0, 0, []
        for p, x in A['gatorctl'].items():
            y = stored[p]
            if not (x.get('complete') and y.get('complete')):
                continue
            n += 1
            if x['status'] == y['status']:
                same += 1
            else:
                disc.append(dict(pair_id=p, redrive=x['status'], stored=y['status']))
            eq, _ = npz_equal(x['dir'], y['dir'])
            ident += int(eq)
        ref['gatorctl'] = dict(n=n, same_end_state=same, identical_arrays=ident, discordant=disc[:20],
                               fail_redrive=rate(x['fail'] for x in A['gatorctl'].values() if x.get('validated'))[0],
                               fail_stored_sample_a=SA['fail']['gator'])
        agree_n += n; agree_same += same
    for arm in ('bitid_hmmwv', 'bitid_gator'):
        for r in arms.get(arm, []):
            x = recs[r['id']]
            refd = pm(r['ref_run'])
            y = read_run(refd, None, os.path.basename(refd))
            if not (x.get('complete') and y.get('complete')):
                ref['bitid'][r['id']] = dict(complete=x.get('complete'), ref_complete=y.get('complete'))
                continue
            eq, why = npz_equal(x['dir'], y['dir'])
            ref['bitid'][r['id']] = dict(status=x['status'], ref_status=y['status'], same_end_state=x['status'] == y['status'],
                                         identical_arrays=eq, first_difference=why)
            agree_n += 1; agree_same += int(x['status'] == y['status'])
    ref['pooled_same_end_state'] = agree_same / agree_n if agree_n else None
    ref['pooled_n'] = agree_n
    ref['ok'] = bool(agree_n) and agree_same / agree_n >= AGREE_MIN

    # ---------------------------------------------------------------- per arm
    res = dict(schema='ov_smoke_analyze_v1', tool='scripts/ov_smoke_analyze.py', time=time.strftime('%F %T'),
               test_label=a.test_label, relaxed_gates=sorted(relax), tasks=os.path.abspath(a.tasks), runs=a.runs,
               stored_gator=a.stored_gator, arm_sources=src, b_from_stored=bstored, reference=ref, arms={}, vehicles={})
    vehicle_of = {arm: rs[0]['vehicle'] for arm, rs in arms.items()}
    gc = A.get('gatorctl', {})
    for arm in sorted(A):
        rs = [r for r in arms[arm] if r.get('sample') in ('A', 'bitid')]
        # belly clearance is defined only for vehicles with hull points (not the HMMWV bit-identity rows)
        g = gates(arm, rs, recs, vehicle_of[arm], (relax if arm not in ('gatorctl',) else set()) | ({'belly'} if vehicle_of[arm] == 'hmmwv' else set()))
        e = dict(gates=g, cost=cost(A[arm].values()), describe=describe(A[arm], groups))
        if arm not in ('gatorctl',) and not arm.startswith('bitid') and gc:
            e['vs_gatorctl'] = paired(A[arm], gc, groups)
        if arm in SENS and 'polaris' in A:
            e['vs_polaris'] = paired(A[arm], A['polaris'], groups)
        gcost = cost(gc.values()) if gc else {}
        if gcost.get('loop_wall_per_sim') and e['cost'].get('loop_wall_per_sim'):
            e['cost']['ratio_to_gatorctl'] = round(e['cost']['loop_wall_per_sim'] / gcost['loop_wall_per_sim'], 3)
        if gcost.get('loop_wall_per_sim_mi350x') and e['cost'].get('loop_wall_per_sim_mi350x'):
            e['cost']['ratio_to_gatorctl_mi350x'] = round(e['cost']['loop_wall_per_sim_mi350x'] / gcost['loop_wall_per_sim_mi350x'], 3)
        res['arms'][arm] = e
    # B
    res['sample_B'] = {}
    for v, per in Bv.items():
        res['sample_B'][v] = {short: dict(fail=rate(x['fail'] for x in d.values() if x.get('validated'))[0],
                                          n_validated=sum(1 for x in d.values() if x.get('validated')), n=len(d),
                                          status=dict(Counter(x.get('status') for x in d.values() if x.get('validated'))))
                              for short, d in per.items()}

    # ---------------------------------------------------------------- decision (PLAN 2.4)
    fam = {arm: res['arms'][arm]['vs_gatorctl']['p_one_sided'] for arm in PRIMARY if arm in res['arms'] and 'vs_gatorctl' in res['arms'][arm]}
    H = holm(fam) if fam else {}
    for arm in fam:
        e = res['arms'][arm]; s = e['vs_gatorctl']
        lvl, rej = H[arm]
        crit = dict(c1_gates=e['gates']['pass'], c2_gain_10pts=s['diff'] is not None and s['diff'] <= -MIN_GAIN,
                    c3_mcnemar=bool(rej), c4_bootstrap_below_0=s['ci95'] is not None and s['ci95'][1] < 0,
                    c5_below_wheel_ref=s['fail'] is not None and s['fail'] < WHEEL_REF)
        e['decision'] = dict(criteria=crit, holm_level=lvl, holm_family=sorted(fam), p_one_sided=s['p_one_sided'],
                             passes_at_family_of_3_bonferroni_first_step=s['p_one_sided'] <= ALPHA / 3,
                             better_on_A=all(crit.values()))
    for veh, cand in (('polaris', ['polaris']), ('m113', ['m113', 'm113_g4'])):
        cand = [c for c in cand if c in fam]
        if not cand:
            continue
        best = min(cand, key=lambda c: (res['arms'][c]['vs_gatorctl']['fail'] if res['arms'][c]['vs_gatorctl']['fail'] is not None else 9, c))
        d = res['arms'][best]['decision']
        verdict = 'better' if d['better_on_A'] else 'not better'
        cons = None
        if veh == 'polaris':
            b = res['sample_B'].get('polaris', {}).get('straight6', {})
            cons = dict(fail_straight6=b.get('fail'), n=b.get('n_validated'), gator_stored=B_GATOR_STRAIGHT6,
                        consistent=b.get('fail') is not None and b['fail'] < B_GATOR_STRAIGHT6)
            hf = res['sample_B'].get('polaris', {}).get('Hfull_free', {})
            cons['goal_rate_Hfull_pick'] = None if hf.get('fail') is None else 1 - hf['fail']
            cons['ceiling_note'] = b.get('fail') is not None and b['fail'] < CEILING
            if verdict == 'better' and not cons['consistent']:
                verdict = 'mixed'
            cons['complete'] = bool(b.get('n')) and b.get('n_validated') == b.get('n')
            if verdict in ('better', 'mixed') and not cons['complete']:
                verdict = f"better on A; sample B not complete ({b.get('n_validated', 0)}/{b.get('n', 96)} validated straight 6 m/s drives; B reading provisional: {verdict})"

        gcg = res['arms'].get('gatorctl', {}).get('gates', {})
        todo = {c: res['arms'][c]['gates']['not_run_or_in_flight'] for c in cand + ['gatorctl'] if c in res['arms']}
        if not ref['ok']:
            verdict = 'BLOCKED: Gator re-drive / bit-identity rows disagree with the stored runs (reference changed)'
        elif not all(gcg.get(k) for k in ('G1_validated', 'G2_crash', 'G3_launch')):
            verdict = 'BLOCKED: the Gator re-drive itself fails validity gates G1-G3'
        if any(todo.values()):
            verdict = f'INCOMPLETE: drives not finished {todo} (provisional reading: {verdict})'
        sens = {}
        if veh == 'polaris':
            for s_ in SENS:
                if s_ in res['arms'] and 'vs_polaris' in res['arms'][s_]:
                    sens[s_] = res['arms'][s_]['vs_polaris']
            dd = [abs(sens[s_]['diff']) for s_ in DRIVELINE_ARMS if s_ in sens and sens[s_]['diff'] is not None]
            sens['driveline_dependent'] = (max(dd) > MIN_GAIN) if dd else None
        res['vehicles'][veh] = dict(arm=best, candidates=cand, verdict=verdict, consistency_B=cons, sensitivity=sens,
                                    test_label=a.test_label)
    json.dump(res, open(a.json, 'w'), indent=1, default=str)
    txt = report(res)
    if a.text:
        open(a.text, 'w').write(txt)
    print(txt)


def report(res):
    L = []
    if res['test_label']:
        L.append(f"*** TEST RUN ({res['test_label']}): not a decision; relaxed gates {res['relaxed_gates'] or 'none'} ***")
    L.append(f"smoke read-out {res['time']}  tasks {res['tasks']}")
    ref = res['reference']
    g = ref.get('gatorctl')
    if g:
        L.append(f"Gator re-drive vs stored runs: same end state {g['same_end_state']}/{g['n']}, identical arrays {g['identical_arrays']}/{g['n']}; "
                 f"failure re-drive {pct(g['fail_redrive'])} vs stored {pct(g['fail_stored_sample_a'])} (sample_A.json)")
    for k, v in ref['bitid'].items():
        L.append(f"  {k}: {v}")
    L.append(f"reference holds (pooled same end state >= 95 %): {ref['ok']} ({pct(ref['pooled_same_end_state'])} of {ref['pooled_n']})")
    L.append('')
    L.append(f"{'arm':12s} {'valid':>9s} {'launch':>6s} {'crash':>5s} {'belly':>7s} {'gates':>5s} {'fail':>7s} {'Gator':>7s} {'diff':>7s} "
             f"{'only G':>6s} {'only V':>6s} {'p(1s)':>9s} {'95% CI (groups)':>18s} {'goal grp':>8s} {'sim s':>6s} {'wall/sim':>8s} {'x Gator':>7s}")
    for arm, e in res['arms'].items():
        gt, s, c, d = e['gates'], e.get('vs_gatorctl') or {}, e['cost'], e['describe']
        ci = s.get('ci95')
        L.append(f"{arm:12s} {gt['validated']:>4d}/{gt['rows']:<4d} {gt['launch_failures']:>6d} {gt['crashed'] + gt['bad_physics']:>5d} "
                 f"{pct(gt['belly_flag']):>7s} {'pass' if gt['pass'] else 'FAIL':>5s} {pct(s.get('fail', d['fail'])):>7s} {pct(s.get('fail_ref')):>7s} "
                 f"{('%+.1f' % (100 * s['diff'])) if s.get('diff') is not None else 'n/a':>7s} {s.get('only_ref_fails', ''):>6} {s.get('only_vehicle_fails', ''):>6} "
                 f"{('%.2e' % s['p_one_sided']) if 'p_one_sided' in s else '':>9s} "
                 f"{('[%+.1f, %+.1f]' % (100 * ci[0], 100 * ci[1])) if ci else '':>18s} {d['groups_with_goal']:>3d}/{d['groups']:<4d} "
                 f"{c['mean_sim_s'] if c['mean_sim_s'] is not None else 'n/a':>6} {c['loop_wall_per_sim'] if c['loop_wall_per_sim'] is not None else 'n/a':>8} "
                 f"{c.get('ratio_to_gatorctl', ''):>7}")
    L.append("(every arm paired with gatorctl on the routes both drove validly; 'only G' = only the Gator fails, 'only V' = only the arm fails;"
             " wall/sim = loop wall seconds per simulated second, x Gator = its ratio to gatorctl)")
    for arm, e in res['arms'].items():
        s = e.get('vs_polaris')
        if s and s.get('n_pairs'):
            ci = s['ci95']
            L.append(f"  sensitivity {arm} vs polaris: {pct(s['fail'])} vs {pct(s['fail_ref'])} on {s['n_pairs']} routes, diff {100 * s['diff']:+.1f} points, "
                     f"only polaris fails {s['only_ref_fails']}, only {arm} fails {s['only_vehicle_fails']}, two-sided p {s['p_two_sided']:.2e}, "
                     f"95 % CI [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]")
    for v, per in res['sample_B'].items():
        L.append(f"sample B {v}: " + '; '.join(f"{k} fail {pct(x['fail'])} ({x['n_validated']}/{x['n']})" for k, x in per.items()))
    L.append('')
    for veh, d in res['vehicles'].items():
        e = res['arms'][d['arm']]['decision']
        L.append(f"VEHICLE {veh} (arm {d['arm']} of {d['candidates']}): {d['verdict'].upper()}")
        L.append(f"  criteria {e['criteria']}; Holm level {e['holm_level']:.4f} over {e['holm_family']}; p {e['p_one_sided']:.2e}")
        if d['consistency_B']:
            cb = d['consistency_B']
            L.append(f"  sample B straight 6 m/s failure {pct(cb['fail_straight6'])} vs stored Gator {pct(cb['gator_stored'])}: consistent {cb['consistent']}; "
                     f"goal on the H_full picks {pct(cb['goal_rate_Hfull_pick'])}" + ("; CEILING: the straight route already nearly meets the bar" if cb['ceiling_note'] else ''))
        if d['sensitivity']:
            L.append(f"  driveline-dependent: {d['sensitivity'].get('driveline_dependent')}; "
                     + '; '.join(f"{k} {pct(v['fail'])} vs polaris {pct(v['fail_ref'])}" for k, v in d['sensitivity'].items() if isinstance(v, dict)))
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()
