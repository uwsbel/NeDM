#!/usr/bin/env python3
"""Soil evaluation analysis spec for arena_gator_20260925 (soil track step 2; PLAN 2.3, 3, 7.3, 7.4, 7.7, 7.10).
Written and frozen (sha256 in LOG.md) before any soil evaluation outcome is synced. Arm names = the ag_eval_tasks.py
--arm names of the soil builds e6/tasks/soil_eval_p{1..5}.json (see NOTES_S2.md):
  unseen arenas (8 x the declared 125 groups):  M1a_free M1b_free M3a_free M3b_free A3_free M2_free straight6   (HMMWV)
  f104 800-group suite:  G_free_gator H_free_gator straight6_gator (Gator); M1a_free (= H on the HMMWV) and straight6 (HMMWV)
  f104 in distribution (200 of the 800):  M1a_free (the H drive) M3a_free A3_free straight6
  g203 / g228 held-out (150 each):  M1a_free M3a_free A3_free straight6
Family (PLAN 7.3; identical definitions to e6/analysis/spec_rigid_v1.json): P1 soil M3 vs M1 and P2 soil A3 vs M1 on
fail (goal not reached), M1 / M3 = per-group mean of the a/b ensembles (REVIEW_R1 4), pooled over the 8 unseen arenas;
P3 / P4 = the rigid fixed 2 m/s tests, taken from the rigid results json (RESULTS_rigid) by scripts/ag_s2_family.py,
which applies Holm over all four; in this soil-only run they have no rows (p = 1 placeholders), so the Holm-adjusted soil
p-values printed by ag_analyze.py are upper bounds.
  python scripts/ag_s2_spec.py --out artifacts/traverse/arena_gator_20260925/e6/analysis/spec_soil_v1.json
"""
import argparse, hashlib, json
from pathlib import Path

H, GA, ANY = 'hmmwv', 'gator', 'any'


def c(name, test, ref, label='fail', set_='unseen', vehicle=H, **kw):
    return dict(name=name, world='crm', vehicle=vehicle, test=test, ref=ref, label=label, set=set_, **kw)


SPEC = dict(
    schema='ag_analyze_spec_v1', margin_pts=2.0, alpha=0.05, boot=4000, seed=0, cluster_key='cluster', min_groups=50,
    note=('S2 soil evaluation spec, frozen before any soil evaluation outcome is synced. Soil, speed free, CEM 4 x 64 from a standing '
          'start. fail = goal not reached. Family = PLAN 7.3 (soil P1/P2 here; rigid P3/P4 from the rigid results json, combined by '
          'scripts/ag_s2_family.py with Holm at 0.05 over the four one-sided cluster p-values). Decision: improves if Holm rejects; '
          'else no meaningful difference if the 90 % cluster interval lies within +-2 points; else inconclusive (a whole interval '
          'above +2 points is reported in words as worse). Declared evaluation order (PLAN 7.10): A primary (M1 = M1a+M1b, M3a, A3, '
          'straight 6) + B primary (G and H on the Gator) -> M3b, M2 -> anchors (straight 6 on the Gator, H and straight 6 on the '
          'HMMWV) -> in distribution; P1 needs M3b (tier 2): if M3b is incomplete, P1 is evaluated on the groups where both M3 '
          'ensembles were driven and the single-ensemble contrast M3a vs M1 is reported beside it. Task B primary: G vs H on the '
          'Gator, 800 groups, declared primary outside the Holm family; works = (1) G beats straight 6 on the Gator (lower bound '
          '> 0, i.e. one-sided p < 0.05), (2) G within 2 points of H or better, (3) offline AUC >= 0.95 (S1), (4) headroom closed '
          'vs the HMMWV (reported).'),
    combine=[dict(world='crm', vehicle=H, name='M1', members=['M1a_free', 'M1b_free']),
             dict(world='crm', vehicle=H, name='M3', members=['M3a_free', 'M3b_free'])],
    family=[c('P1_soil_M3_vs_M1', 'M3', 'M1'),
            c('P2_soil_A3_vs_M1', 'A3_free', 'M1'),
            dict(name='P3_rigid_fx2_M3_vs_M1', world='rigid', vehicle=H, test='M3_fx2', ref='M1_fx2', label='unsafe', set='unseen'),
            dict(name='P4_rigid_fx2_A3_vs_M1', world='rigid', vehicle=H, test='A3_fx2', ref='M1_fx2', label='unsafe', set='unseen')],
    contrasts=[
        # task A, unseen arenas
        c('A_soil_M3a_vs_M1', 'M3a_free', 'M1'), c('A_soil_M3a_vs_M1a', 'M3a_free', 'M1a_free'), c('A_soil_A3_vs_M1a', 'A3_free', 'M1a_free'),
        c('A_soil_M2_vs_M1', 'M2_free', 'M1'), c('A_soil_A3_vs_M3', 'A3_free', 'M3'), c('A_soil_M3_vs_M2', 'M3', 'M2_free'),
        c('A_soil_seed_M1a_vs_M1b', 'M1a_free', 'M1b_free'), c('A_soil_seed_M3a_vs_M3b', 'M3a_free', 'M3b_free'),
        c('A_soil_M1_vs_straight6', 'M1', 'straight6'), c('A_soil_A3_vs_straight6', 'A3_free', 'straight6'),
        c('A_soil_M3_vs_M1_unsafe', 'M3', 'M1', 'unsafe'), c('A_soil_A3_vs_M1_unsafe', 'A3_free', 'M1', 'unsafe'),
        c('A_soil_near_M3_vs_M1', 'M3', 'M1', set_='near'), c('A_soil_spread_M3_vs_M1', 'M3', 'M1', set_='spread'),
        c('A_soil_near_A3_vs_M1', 'A3_free', 'M1', set_='near'), c('A_soil_spread_A3_vs_M1', 'A3_free', 'M1', set_='spread'),
        # task A, in distribution (no harm on f104, 2-point non-inferiority margin; in-arena effect on the held-out groups)
        c('A_soil_noharm_f104_M3a_vs_M1a', 'M3a_free', 'M1a_free', set_='indist_f104'),
        c('A_soil_noharm_f104_A3_vs_M1a', 'A3_free', 'M1a_free', set_='indist_f104'),
        c('A_soil_inarena_M3a_vs_M1a', 'M3a_free', 'M1a_free', set_='heldout'),
        c('A_soil_inarena_A3_vs_M1a', 'A3_free', 'M1a_free', set_='heldout'),
        c('A_soil_inarena_g203_A3_vs_M1a', 'A3_free', 'M1a_free', set_='heldout_g203'),
        c('A_soil_inarena_g228_A3_vs_M1a', 'A3_free', 'M1a_free', set_='heldout_g228'),
        # task B (PLAN 3 / 7.7): declared primary G vs H on the Gator, then the 'works' criteria and anchors
        c('PRIMARY_B_soil_G_vs_H_on_gator', 'G_free_gator', 'H_free_gator', set_='f104_800', vehicle=GA, declared_primary=True),
        c('B_soil_G_vs_H_on_gator_unsafe', 'G_free_gator', 'H_free_gator', 'unsafe', set_='f104_800', vehicle=GA),
        c('B_works1_soil_G_vs_straight6_on_gator', 'G_free_gator', 'straight6_gator', set_='f104_800', vehicle=GA),
        c('B_soil_H_vs_straight6_on_gator', 'H_free_gator', 'straight6_gator', set_='f104_800', vehicle=GA),
        c('B_soil_H_gator_vs_H_hmmwv', 'H_free_gator', 'M1a_free', set_='f104_800', vehicle=ANY),
        c('B_soil_G_gator_vs_H_hmmwv', 'G_free_gator', 'M1a_free', set_='f104_800', vehicle=ANY),
        c('B_soil_straight6_gator_vs_hmmwv', 'straight6_gator', 'straight6', set_='f104_800', vehicle=ANY),
        c('B_soil_H_vs_straight6_on_hmmwv', 'M1a_free', 'straight6', set_='f104_800')],
    headroom=[dict(world='crm', vehicle=GA, model='G_free_gator', straight='straight6_gator', label='fail', set='f104_800'),
              dict(world='crm', vehicle=GA, model='H_free_gator', straight='straight6_gator', label='fail', set='f104_800'),
              dict(world='crm', vehicle=H, model='M1a_free', straight='straight6', label='fail', set='f104_800')],
    gaps=[dict(world='crm', vehicle=H, model=m, straight='straight6', labels=['fail', 'unsafe'], unseen='unseen', indist='indist_f104')
          for m in ('M1a_free', 'M3a_free', 'A3_free')],
    dose=[dict(world='crm', vehicle=H, arms=['M1', 'M2_free', 'M3'], label='fail', set='unseen'),
          dict(world='crm', vehicle=H, arms=['M1a_free', 'M2_free', 'M3a_free'], label='fail', set='unseen')],
    time_ratio=[dict(world='crm', vehicle=H, test='M3a_free', ref='M1a_free', set='unseen'),
                dict(world='crm', vehicle=H, test='A3_free', ref='M1a_free', set='unseen'),
                dict(world='crm', vehicle=H, test='M1a_free', ref='straight6', set='unseen'),
                dict(world='crm', vehicle=GA, test='G_free_gator', ref='H_free_gator', set='f104_800')],
    rates=dict(sets=['unseen', 'near', 'spread', 'indist_f104', 'heldout', 'f104_800', 'per_arena']))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    Path(a.out).write_text(json.dumps(SPEC, indent=1) + '\n')
    print(a.out, hashlib.sha256(Path(a.out).read_bytes()).hexdigest())
