"""Run traversing evaluation arms on a suite on this machine, or one block of a prepared run folder (what each SLURM
array task of slurm.py runs). The run folder layout, pinning and resume rules are runner.py's.

    PYTHONPATH=src python scripts/traversing/evaluation/run.py --out OUT \\
        --arms configs/traversing/evaluation/m2_shared_risk.toml --ground soil [--arm NAME ...] \\
        --suite f104_800 [--subset 'stratum=fresh,lowest_md5=10'] [--stage all|plan] [--block-size N] [--block I] \\
        [--workers N] [--gpus 0,1]
    PYTHONPATH=src python scripts/traversing/evaluation/run.py --out OUT --block I [--workers N] [--gpus 0] \\
        [--deadline EPOCH_S] [--accept-code-change]
Prints the statuses per arm; compare.py checks a run folder against the released record.
"""
import argparse
import json
from collections import Counter

from nedm.traversing.evaluation.config import load_arms
from nedm.traversing.evaluation.runner import TraversalEval, run_arms, run_block
from nedm.traversing.evaluation.suites import load_suite


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True, help='the run folder')
    p.add_argument('--arms', help='TOML arm file of a new run (prepare + run); else --block of a prepared folder')
    p.add_argument('--ground', choices=('rigid', 'soil'), help='keep the arms of one ground (required if names repeat)')
    p.add_argument('--arm', action='append', help='keep this arm only (repeatable)')
    p.add_argument('--suite', help='released suite name or custom:<dir>')
    p.add_argument('--subset', help="suites.select terms, e.g. 'stratum=fresh,lowest_md5=10'")
    p.add_argument('--stage', choices=('all', 'plan'), default='all', help='plan: stop once every pick.json is written')
    p.add_argument('--block-size', type=int, help='pairs per block (default: one block)')
    p.add_argument('--block', type=int, help='run this block only')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--gpus', default='', help='comma-separated GPU indices (soil: one drive process per GPU)')
    p.add_argument('--deadline', type=float, help='epoch s; no pair starts within arms x runner.GUARD_S of it (exit 1)')
    p.add_argument('--accept-code-change', action='store_true', help='resume a block pinned with other code')
    a = p.parse_args()
    gpus = [int(g) for g in a.gpus.split(',') if g]
    if a.arms:
        arms = [c for c in load_arms(a.arms, a.ground) if not a.arm or c.name in a.arm]
        if not a.suite or set(a.arm or ()) - {c.name for c in arms}:
            p.error(f'--suite is required; --arm must name arms of {a.arms} (ground {a.ground})')
        recs = run_arms([TraversalEval(c, a.out) for c in arms], load_suite(a.suite, a.subset), stage=a.stage,
                        block=a.block, block_size=a.block_size, workers=a.workers, gpus=gpus)
    elif a.block is None:
        p.error('give --arms and --suite (a new run) or --block (of a prepared run folder)')
    else:
        recs = run_block(a.out, a.block, stage=a.stage, workers=a.workers, gpus=gpus, deadline=a.deadline,
                         accept_code_change=a.accept_code_change)
    for arm in sorted({r.arm for r in recs}):
        print(json.dumps(dict(arm=arm, statuses=Counter(r.status for r in recs if r.arm == arm))), flush=True)


if __name__ == '__main__':
    main()
