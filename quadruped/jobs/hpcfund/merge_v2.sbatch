#!/bin/bash
# Merge all 24 v2 shards into the full corpus and pack it as one file.
#
# tar is told to tolerate "file changed as we read it". WekaFS updates directory
# metadata a beat after a copy, and the last merge's tar returned 1 on exactly that --
# a warning, not a failure: the archive was complete and extracted to all 1318 segments.
# Under set -e a warning is indistinguishable from an error, so it is named here instead.
#SBATCH -J kyle-merge-v2
#SBATCH -p devel
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -o /work1/dannegrut/kyle/logs/%x-%j.out
#SBATCH -e /work1/dannegrut/kyle/logs/%x-%j.err
set -uo pipefail
K=/work1/dannegrut/kyle
source $K/nedm/env-scoring.sh
cd $K/qrun/data
# SHARD 1 IS EXCLUDED. It died in its 33rd episode on a Chrono GPU fault
# (illegal memory access, SphBceManager.cu:543, array job 430005) and, since the
# manifest is written at the end, its 32 finished episodes have none and cannot be
# merged or traced. 23 shards = 1150 episodes.
SHARDS=$(for S in $(seq 0 23); do [ $S -eq 1 ] && continue; echo -n " go2_crm_v2_s$S"; done)
rm -rf go2_crm_v2
python $K/qrun/quadruped/merge_corpus.py --shards $SHARDS --out . --name go2_crm_v2 || exit 1
sync
tar --warning=no-file-changed -czf go2_crm_v2.tgz go2_crm_v2
rc=$?
[ $rc -le 1 ] || { echo "tar failed rc=$rc"; exit 1; }
ls -la go2_crm_v2.tgz
echo "MERGE_DONE"
