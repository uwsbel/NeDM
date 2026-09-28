#!/usr/bin/env python3
"""Job list for scripts/ov_train.sbatch: the PLAN 1.7 soil planner recipe for one vehicle's training subset
(offroad_vehicles_20260927, module M4). Two lanes (one MI350X, 2 lanes per GPU as in arena_gator's jobs 439361/439362):
  lane 1  <tag>_deploy   deploy mode, 5 seeds x 30 epochs, --roundtrip-check     -> the planner's ensemble
  lane 2  <tag>_holdout  holdout mode (dev fold scored), same recipe               -> offline AUC
Recipe (identical to arena_gator e5/jobs/soil_bf_G.tsv): --arch gru --cond none --domain-filter crm --ctx geom
--split-eval val --bs 256 --epochs 30 --seeds 5 --seed0 0 (trainer defaults lr 2e-3, wd 1e-4; fresh normalisers).
--quick writes a 1-epoch, 1-seed smoke list instead (for a local or devel check of the job script).

  python scripts/ov_train_jobs.py --tag polaris_full_soil --ds $G4/e4/subsets/polaris_full_f104_polaris_soil.npz \
      --out $K4/e5/jobs/soil_polaris_full.tsv            # then copy to $G4/e5/jobs/ and submit ov_train.sbatch
"""
import argparse, re
from pathlib import Path

RECIPE = '--arch gru --cond none --domain-filter crm --ctx geom --split-eval val --bs 256'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tag', required=True, help='e.g. polaris_full_soil (lane tags <tag>_deploy / <tag>_holdout)')
    ap.add_argument('--ds', required=True, action='append', help='subset npz (cluster path); repeatable')
    ap.add_argument('--out-deploy', default=None, help='output folder relative to G4 (default e5/train/<tag>)')
    ap.add_argument('--out-holdout', default='e5/train/offline_soil')
    ap.add_argument('--seeds', type=int, default=5); ap.add_argument('--seed0', type=int, default=0); ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--quick', action='store_true', help='1 epoch, 1 seed (smoke of the job script)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    assert re.match(r'^[A-Za-z0-9_]+$', a.tag), a.tag
    ep, se = (1, 1) if a.quick else (a.epochs, a.seeds)
    ds = ' '.join(f'--ds {d}' for d in a.ds)
    common = f'{ds} {RECIPE} --epochs {ep} --seeds {se} --seed0 {a.seed0}'
    lines = ['# lane\ttag\tout (rel. G4)\tci_train arguments (offroad_vehicles_20260927 PLAN 1.7 recipe' + (', QUICK smoke' if a.quick else '') + ')',
             f"1\t{a.tag}_deploy\t{a.out_deploy or 'e5/train/' + a.tag}\t{common.replace(RECIPE, '--mode deploy ' + RECIPE)} --roundtrip-check",
             f"2\t{a.tag}_holdout\t{a.out_holdout}\t{common.replace(RECIPE, '--mode holdout ' + RECIPE)}"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
