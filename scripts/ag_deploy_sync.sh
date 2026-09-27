#!/bin/bash
# arena_gator_20260925 E5a: copy one finished deploy ensemble (checkpoints + trainer summary + training logits, the latter needed by
# ag_offline_auc.py's fitted-group guard) from G3/e5/train/<subdir> to K3/e5/deploy/<model>/, write SHA256SUMS there and require
# the same sha256 on the cluster.
set -eo pipefail
K3=artifacts/traverse/arena_gator_20260925; G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
M=$1; SUB=$2; TAG=$3; D=$K3/e5/deploy/$M
mkdir -p $D
rsync -a amd:$G3/e5/train/$SUB/${TAG}_s*.pt amd:$G3/e5/train/$SUB/${TAG}.json amd:$G3/e5/train/$SUB/${TAG}_logits.npz $D/
( cd $D && sha256sum ${TAG}_s*.pt ${TAG}.json ${TAG}_logits.npz > SHA256SUMS )
ssh amd "cd $G3/e5/train/$SUB && sha256sum ${TAG}_s*.pt ${TAG}.json ${TAG}_logits.npz" > /tmp/remote_$M.sha
if diff -q $D/SHA256SUMS /tmp/remote_$M.sha >/dev/null; then echo "$M: $(ls $D/${TAG}_s*.pt | wc -l) members, sha256 equal on both sides"; else echo "$M: SHA MISMATCH"; exit 1; fi
