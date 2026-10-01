"""Write a SLURM job for traversing evaluation arms on a suite, and submit it with --submit: the run
folder (runner.prepare: config, tasks, md5-sorted blocks of whole pairs), <out>/job/run.sbatch (one array task per
block = one node, running run.py --block) and <out>/job/code_sha.json. Run on the AMD HPC Fund login node:

    PYTHONPATH=src python scripts/traversing/evaluation/slurm.py --out OUT --arms ARMS.toml --ground soil \\
        --suite f104_800 [--subset 'stratum=fresh,lowest_md5=10'] [--arm NAME ...] [--block-size 5] \\
        [--time 03:50:00] [--submit] [--resume [--accept-code-change]]
The environment recipe of each ground is the one its recorded drives ran with (nrd env.sh):
  soil   crm_collect.sbatch: chrono-build-fsi; one drive process per GPU (rocminfo), threads as runner.process_env
  rigid  gen_array_g.sbatch: chrono-build + lavapipe; single-threaded drives, workers = CPUs - 2
NEDM_DATA and NEDM_RELEASE_CACHE come from this shell. Refused: an MI210 partition (torch 2.10 crashes there and CRM is
slower) unless --allow-slow-partition; a non-empty --out without --resume; a dirty or unknown git tree unless
--allow-dirty (stamped in code_sha.json; a copy without .git reads GIT_COMMIT = {"commit", "dirty"}, written when the
tree was copied); a --time that cannot fit one pair of every arm (runner.GUARD_S); at run time, an array task whose
code sha differs from code_sha.json.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from nedm.traversing.evaluation.config import REPO_ROOT, ConfigError, Env, ReleaseError, load_arms, write_atomic
from nedm.traversing.evaluation.runner import GUARD_S, TraversalEval, code_sha, prepare
from nedm.traversing.evaluation.suites import load_suite

NRD = '/work1/dannegrut/harry/nrd'
RECIPE = dict(soil='export CHRONO_BUILD=$NRD_ROOT/chrono-build-fsi\nnrd_pychrono\n'
                   'GPUS=$(rocminfo | grep -c -E "^\\s+Name:\\s+gfx"); W=$GPUS; G=$(seq -s, 0 $((GPUS - 1)))',
              rigid='nrd_pychrono\nnrd_use_lavapipe\nW=$(( ${SLURM_CPUS_PER_TASK:-16} - 2 )); G=')
SBATCH = '''#!/bin/bash
#SBATCH -A dannegrut
#SBATCH -N 1
#SBATCH -n 1
# written by: {argv}
set -eo pipefail
source {nrd}/env.sh
{recipe}
export PYTHONPATH={src}:$PYTHONPATH NEDM_CHRONO_DATA=$CHRONO_BUILD/data {data}
echo "host=$(hostname) job=${{SLURM_ARRAY_JOB_ID}}_$SLURM_ARRAY_TASK_ID part=$SLURM_JOB_PARTITION workers=$W gpus=$G" \\
     "start=$(date -Is)"
exec "$NRD_PYTHON" -P -u {run} --out {out} --block $SLURM_ARRAY_TASK_ID --workers $W --gpus "$G" \\
    --deadline $(( $(date +%s) + {wall} )){accept}
'''


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('--out', '--arms', '--suite'):
        p.add_argument(k, required=True)
    p.add_argument('--ground', choices=('rigid', 'soil'), required=True)
    p.add_argument('--arm', action='append', help='keep this arm only (repeatable)')
    p.add_argument('--subset')
    p.add_argument('--block-size', type=int, required=True, help='pairs per block (= per node); size it for --time')
    p.add_argument('--partition', default='mi3501x')
    p.add_argument('--time', default='03:50:00', help='HH:MM:SS per array task (the submit filter caps it at 4 h)')
    for k in ('--submit', '--resume', '--accept-code-change', '--allow-dirty', '--allow-slow-partition'):
        p.add_argument(k, action='store_true')
    a = p.parse_args()
    try:
        submit(a)
    except (ConfigError, ReleaseError) as e:
        sys.exit(f'refused: {e}')


def submit(a):
    out, env = Path(a.out).resolve(), Env.from_environ()
    git = [subprocess.run(['git', '-C', str(REPO_ROOT), *c], capture_output=True, text=True).stdout.strip()
           for c in (('rev-parse', 'HEAD'), ('status', '--porcelain'))] if (REPO_ROOT / '.git').exists() else \
        [(g := json.loads(gc.read_text()))['commit'], g['dirty']] if (gc := REPO_ROOT / 'GIT_COMMIT').exists() else None
    refusals = []
    if (a.partition.startswith('mi210') or a.partition == 'devel') and not a.allow_slow_partition:
        refusals.append(f'partition {a.partition}: MI210 (torch 2.10 crashes, CRM slower); --allow-slow-partition')
    if out.exists() and any(out.iterdir()) and not a.resume:
        refusals.append(f'{out} is not empty: --resume')
    if (git is None or bool(git[1])) and not a.allow_dirty:
        refusals.append(f'git tree dirty or unknown ({REPO_ROOT}): --allow-dirty')
    if refusals:
        sys.exit('refused: ' + '; '.join(refusals))
    every = load_arms(a.arms, a.ground)
    if set(a.arm or ()) - {c.name for c in every}:
        sys.exit(f'refused: --arm {a.arm}: the arms of {a.arms} on {a.ground} are {[c.name for c in every]}')
    arms = [c for c in every if not a.arm or c.name in a.arm]
    hms = re.fullmatch(r'(\d+):(\d\d):(\d\d)', a.time)
    wall = hms and 3600 * int(hms[1]) + 60 * int(hms[2]) + int(hms[3])
    if not wall or wall < len(arms) * GUARD_S[a.ground] + 120:     # run_block starts no pair within that of the end
        sys.exit(f'refused: --time {a.time}: HH:MM:SS of at least {len(arms)} arms x {GUARD_S[a.ground]:.0f} s (the '
                 'walltime guard of one pair, runner.GUARD_S) + 2 min start-up; size --block-size for the rest')
    n = len(prepare(out, [TraversalEval(c, out, env) for c in arms], load_suite(a.suite, a.subset, env), a.block_size))
    (job := out / 'job' / 'logs').mkdir(parents=True, exist_ok=True)
    data = ' '.join(f'{k}={v}' for k, v in (('NEDM_DATA', env.data), ('NEDM_RELEASE_CACHE', env.release_cache)) if v)
    write_atomic(out / 'job/run.sbatch', SBATCH.format(
        argv=' '.join(sys.argv), nrd=NRD, recipe=RECIPE[a.ground], src=REPO_ROOT / 'src', data=data, out=out,
        run=Path(__file__).resolve().with_name('run.py'), wall=wall,
        accept=' --accept-code-change' if a.accept_code_change else ''))
    write_atomic(out / 'job/code_sha.json', json.dumps(dict(
        code_sha=code_sha(), git_commit=git and git[0], git_dirty=git and bool(git[1]), allow_dirty=a.allow_dirty,
        written=time.strftime('%Y-%m-%dT%H:%M:%S'), argv=sys.argv), indent=1))
    cmd = ['sbatch', '--parsable', '-p', a.partition, '-c', '24', '-t', a.time, f'--array=0-{n - 1}',
           '-J', f'ev_{out.name}', '-o', f'{out}/job/logs/%x_%A_%a.out', f'{out}/job/run.sbatch']
    if not a.submit:
        print('written (submit with --submit):', ' '.join(cmd))
        return
    jid = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    with open(out / 'job/submissions.tsv', 'a') as f:
        f.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")}\t{jid}\t{a.partition}\t{n} blocks\t{os.getcwd()}\n')
    print(f'submitted job {jid}: {n} array tasks, {out}')


if __name__ == '__main__':
    main()
