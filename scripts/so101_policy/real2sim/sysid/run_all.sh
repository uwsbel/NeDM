#!/bin/bash
# v8 real-data system identification: all parts, CPU only (~20 min, the servo grid takes most). Outputs (and one
# <part>.log each) in ~/so101_r2s_claude/v8/sysid/. appearance.py first: table_height.py reads its camera turn for
# each session. servo_check.py needs the MuJoCo so101.xml: SO101_XML=<path> if the repo's vendor copy
# (contact_nrd/so101_grasp/vendor, contact_nrd/so101_grasp/fetch_sources.py) is not fetched in this worktree.
set -eo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(cd ../../../.. && pwd)/src"
PY=${PY:-/home/harry/miniconda3/envs/so101_vision/bin/python}
OUT=/home/harry/so101_r2s_claude/v8/sysid
mkdir -p "$OUT"
for s in appearance table_height timing lag_fit roll servo_check make_json; do
  echo "== $s"; $PY $s.py 2>&1 | tee "$OUT/$s.log"
done
