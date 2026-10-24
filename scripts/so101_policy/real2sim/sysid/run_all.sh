#!/bin/bash
# v8 real-data system identification: all parts, CPU only (~15 min, the servo grid takes most). Outputs in
# ~/so101_r2s_claude/v8/sysid/. appearance.py first: table_height.py reads its camera turn for each session.
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(cd ../../../.. && pwd)/src"
PY=${PY:-/home/harry/miniconda3/envs/so101_vision/bin/python}
for s in appearance table_height timing lag_fit roll servo_check make_json; do
  echo "== $s"; $PY $s.py
done
