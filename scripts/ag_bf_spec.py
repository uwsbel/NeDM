#!/usr/bin/env python3
"""Analysis spec of task B stage 2 ("Bfull", PLAN 7.10) for arena_gator_20260925: the Gator-trained (G_full) and the
HMMWV-trained (H_full) soil planners retrained on ALL collected tiers (0-12, the 15,235 Gator-validated collect_v1 ids),
driven on the 800-group f104 suite from a standing start (CEM 4 x 64, ag_picks.py --mode free), fail = goal not reached.
Written and frozen (sha256 in LOG.md) before any G_full / H_full pick or drive exists (the models were still training).
Disclosure: the stage-1 task B drives (tiers 0-6 models) were already complete and had been indexed for sizing when this
spec was written (G 34.7 %, H 51.9 %, straight 6 m/s 85.8 % goal not reached on the Gator); nothing about G_full / H_full
was known. Same tool, statistics and settings as e6/analysis/spec_soil_v1.json (margin 2 points, alpha 0.05, 4000
bootstrap draws, seed 0, clusters = nearest terrain feature to the start-goal midpoint, >= 50 groups).

Arm names = the ag_eval_tasks.py --arm names of e6/tasks/soil_eval_bf1.json (tier -9, Gator) and soil_eval_bf2.json
(tier -8, HMMWV); every stage-1 arm is re-listed there and resolved to its existing stage-1 drive by route content:
  Gator:  Gfull_free_gator  (G_full on the Gator, new)        Hfull_free_gator (H_full on the Gator, new)
          G_free_gator      (stage-1 G, tiers 0-6, reused)    H_free_gator     (stage-1 H = M1a, tiers 0-6, reused)
          straight6_gator   (straight route at 6 m/s, reused)
  HMMWV:  Hfull_free        (H_full on the HMMWV, new, secondary tier)
          M1a_free          (stage-1 H on the HMMWV, reused)  straight6        (reused)

Declared family (Holm at 0.05 over the four one-sided cluster-bootstrap p-values, as the tool's family block):
  F1 PRIMARY  G_full vs H_full on the Gator          F2  G_full vs straight 6 m/s on the Gator
  F3          H_full vs straight 6 m/s on the Gator  F4  G_full vs stage-1 G on the Gator (does more data help the Gator)
Decision per test: 'improves' if Holm rejects and the test arm fails less; else 'no meaningful difference' if the 90 %
cluster interval lies within +-2 points; else 'inconclusive'. The paired group bootstrap (95 / 90 % intervals, one-sided
p) and the exact McNemar test (two- and one-sided) are reported beside every contrast. Task B 'works' for the full-tier
models = (1) F2 improves; (2) G_full within 2 points of H_full or better (the non-inferiority bound of F1, one-sided upper
95 % cluster bound < 2 points); (3) offline within-group AUC >= 0.95 on held-out f104 groups (e5/offline, reported for
val and dev fold + val); (4) headroom closed on the Gator vs H_full on the HMMWV (reported, no threshold).
Secondary (unadjusted, reported): H_full vs stage-1 H on the Gator; G_full vs H_full on 'unsafe'; H_full vs stage-1 H on
the HMMWV; H_full vs straight 6 on the HMMWV; cross-vehicle G_full-on-Gator and H_full-on-Gator vs H_full-on-HMMWV.
H_full on the HMMWV sorts after the Gator rows (tier -8); if it is incomplete, its contrasts use the groups driven.
  python scripts/ag_bf_spec.py --out artifacts/traverse/arena_gator_20260925/e6/analysis/spec_soil_v1_Bfull.json
"""
import argparse, hashlib, json
from pathlib import Path

H, GA, ANY = 'hmmwv', 'gator', 'any'
S = 'f104_800'


def c(name, test, ref, label='fail', vehicle=GA, set_=S, **kw):
    return dict(name=name, world='crm', vehicle=vehicle, test=test, ref=ref, label=label, set=set_, **kw)


SPEC = dict(
    schema='ag_analyze_spec_v1', margin_pts=2.0, alpha=0.05, boot=4000, seed=0, cluster_key='cluster', min_groups=50,
    note=('Bfull (task B stage 2, PLAN 7.10) spec, frozen before any G_full / H_full pick or drive exists. Soil, speed free, CEM 4 x 64 from '
          'a standing start, 800-group f104 suite, fail = goal not reached. G_full = Gator-trained on all 15,235 validated Gator ids '
          '(tiers 0-12), H_full = HMMWV-trained on exactly those ids; stage-1 G / H (tiers 0-6) and straight 6 m/s are the existing '
          'stage-1 drives, reused by route content. Family (Holm at 0.05 over the one-sided cluster p): F1 PRIMARY G_full vs H_full on the '
          'Gator, F2 G_full vs straight 6 on the Gator, F3 H_full vs straight 6 on the Gator, F4 G_full vs stage-1 G on the Gator. '
          'Decision: improves if Holm rejects; else no meaningful difference if the 90 % cluster interval lies within +-2 points; else '
          'inconclusive. Group bootstrap and exact McNemar reported beside each. Works (full tiers) = (1) F2 improves, (2) G_full within 2 '
          'points of H_full or better (F1 one-sided upper 95 % cluster bound < 2), (3) offline AUC >= 0.95 (e5/offline), (4) headroom '
          'closed vs H_full on the HMMWV (reported). Disclosure: the stage-1 task B outcomes were known when this was written (G 34.7 %, '
          'H 51.9 %, straight 6 85.8 % fail on the Gator); nothing about G_full / H_full was. H_full on the HMMWV is a secondary, later '
          'tier (-8); its contrasts use the groups driven.'),
    family=[
        c('F1_PRIMARY_Bfull_G_full_vs_H_full_on_gator', 'Gfull_free_gator', 'Hfull_free_gator', declared_primary=True),
        c('F2_Bfull_works1_G_full_vs_straight6_on_gator', 'Gfull_free_gator', 'straight6_gator'),
        c('F3_Bfull_H_full_vs_straight6_on_gator', 'Hfull_free_gator', 'straight6_gator'),
        c('F4_Bfull_G_full_vs_G_stage1_on_gator', 'Gfull_free_gator', 'G_free_gator')],
    contrasts=[
        c('Bfull_H_full_vs_H_stage1_on_gator', 'Hfull_free_gator', 'H_free_gator'),
        c('Bfull_G_full_vs_H_full_on_gator_unsafe', 'Gfull_free_gator', 'Hfull_free_gator', 'unsafe'),
        c('Bfull_G_full_vs_H_stage1_on_gator', 'Gfull_free_gator', 'H_free_gator'),
        c('Bfull_stage1_G_vs_H_on_gator', 'G_free_gator', 'H_free_gator'),
        c('Bfull_H_full_vs_H_stage1_on_hmmwv', 'Hfull_free', 'M1a_free', vehicle=H),
        c('Bfull_H_full_vs_straight6_on_hmmwv', 'Hfull_free', 'straight6', vehicle=H),
        c('Bfull_G_full_gator_vs_H_full_hmmwv', 'Gfull_free_gator', 'Hfull_free', vehicle=ANY),
        c('Bfull_H_full_gator_vs_H_full_hmmwv', 'Hfull_free_gator', 'Hfull_free', vehicle=ANY)],
    headroom=[dict(world='crm', vehicle=GA, model='Gfull_free_gator', straight='straight6_gator', label='fail', set=S),
              dict(world='crm', vehicle=GA, model='Hfull_free_gator', straight='straight6_gator', label='fail', set=S),
              dict(world='crm', vehicle=GA, model='G_free_gator', straight='straight6_gator', label='fail', set=S),
              dict(world='crm', vehicle=GA, model='H_free_gator', straight='straight6_gator', label='fail', set=S),
              dict(world='crm', vehicle=H, model='Hfull_free', straight='straight6', label='fail', set=S),
              dict(world='crm', vehicle=H, model='M1a_free', straight='straight6', label='fail', set=S)],
    time_ratio=[dict(world='crm', vehicle=GA, test='Gfull_free_gator', ref='Hfull_free_gator', set=S),
                dict(world='crm', vehicle=GA, test='Gfull_free_gator', ref='G_free_gator', set=S),
                dict(world='crm', vehicle=H, test='Hfull_free', ref='M1a_free', set=S)],
    rates=dict(sets=[S, 'indist_f104']))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    assert not Path(a.out).exists(), f'{a.out} exists: a frozen spec is never rewritten'
    Path(a.out).write_text(json.dumps(SPEC, indent=1) + '\n')
    print(a.out, hashlib.sha256(Path(a.out).read_bytes()).hexdigest())
