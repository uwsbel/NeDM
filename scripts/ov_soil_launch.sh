#!/bin/bash
# Sized soil (CRM) launch for offroad_vehicles_20260927 (module M3): a copy of ag_soil_launch.sh pointed at G4.
# Run ON THE CLUSTER LOGIN NODE, with NEDM_VEHICLE unset:
#   env -u NEDM_VEHICLE bash $G4/source/scripts/ov_soil_launch.sh <tasks.json> <out dir> <collector path> <partition>:<cpus>:<hours>:<n tasks> [...]
# e.g. env -u NEDM_VEHICLE bash $G4/source/scripts/ov_soil_launch.sh $G4/tasks/smoke_v1.json $G4/soil_v1 \
#        $G4/source/scripts/ov_crm_collect.py mi3501x:24:4:6
# Differences from ag_soil_launch.sh:
#   - root G4; job script G4/source/scripts/ov_soil.sbatch; records in G4/launch/submissions.tsv;
#   - refuses when NEDM_VEHICLE is set, when the task file or the collector is writable (both must be frozen, chmod a-w),
#     or when the collector is not under G4/source;
#   - queue guard: ALL of the user's queued + running array tasks (this study's and any other session's, squeue -r -u
#     $USER) plus this launch must stay <= cap - reserve (50 - 5 by default); the count by job name is printed;
#   - episode timeout: the largest per-row "timeout_s" among the file's run:true rows (default 2,400 s, the worker's own
#     default) is exported as CRM_EPISODE_TIMEOUT_S; the claim margin (stop claiming new drives this long before the
#     time limit) is 1,000 s as before, or the timeout itself (>= 3,000 s) when any row needs more than 2,400 s (M113).
#     OV_EPISODE_TIMEOUT_S / OV_CLAIM_MARGIN_S may raise, never lower, these values.
#   - OV_DRY_RUN=1 prints the sbatch commands and submits nothing; OV_ROOT=<other tree> (dry run only) checks a staged
#     copy such as G4/stage_dryrun.
set -uo pipefail
G4=/work1/dannegrut/harry/experiments/offroad_vehicles_20260927
ROOTD=${OV_ROOT:-$G4}
DRY=${OV_DRY_RUN:-0}
if [[ "$ROOTD" != "$G4" && "$DRY" != "1" ]]; then echo "OV_ROOT other than G4 is for dry runs only"; exit 2; fi
if [[ -n "${NEDM_VEHICLE+x}" ]]; then echo "NEDM_VEHICLE is set: refused (use env -u NEDM_VEHICLE)"; exit 2; fi
(( $# >= 4 )) || { echo "usage: $0 <tasks.json> <out dir> <collector> <part>:<cpus>:<hours>:<n> [...]"; exit 2; }
TASKS=$1; OUT=$2; COLLECTOR=$3; shift 3
CFG=configs/crm_main.json
S=$ROOTD/source/scripts/ov_soil.sbatch
CAP=${OV_QUEUE_CAP:-50}; RESERVE=${OV_QUEUE_RESERVE:-5}
test -f "$TASKS" && test -f "$COLLECTOR" && test -f "$ROOTD/$CFG" && test -f "$S" && test -f "$ROOTD/source/DISPATCHER.sha256" \
  || { echo "missing input (tasks, collector, $ROOTD/$CFG, $S or $ROOTD/source/DISPATCHER.sha256)"; exit 2; }
case "$COLLECTOR" in *crm_collect*) ;; *) echo "collector name must contain crm_collect"; exit 2;; esac
case "$(readlink -f "$COLLECTOR")" in "$ROOTD"/source/scripts/*) ;; *) echo "collector must be under $ROOTD/source/scripts"; exit 2;; esac
case "$(readlink -f "$OUT")" in "$G4"/*) ;; *) echo "output folder must be under $G4"; exit 2;; esac
if [[ -w "$TASKS" || -w "$COLLECTOR" ]]; then echo "task file and collector must be read-only (chmod a-w) before a launch"; exit 2; fi
( cd "$ROOTD/source" && sha256sum --quiet -c DISPATCHER.sha256 ) || { echo "dispatcher files differ from DISPATCHER.sha256"; exit 2; }
# per-row configs must exist; largest per-row timeout
read -r NEED CFGS < <(python3 - "$TASKS" <<'EOF'
import json, sys
rows = [r for r in json.load(open(sys.argv[1])) if r.get('run', True)]
need = max([int(r.get('timeout_s', 2400)) for r in rows] + [2400])
cfgs = sorted({r['config'] for r in rows if r.get('config')})
print(need, ','.join(cfgs) or '-')
EOF
)
[[ "$NEED" =~ ^[0-9]+$ ]] || { echo "could not read the task file"; exit 2; }
if [[ "$CFGS" != "-" ]]; then
  IFS=, read -r -a cl <<< "$CFGS"
  for c in "${cl[@]}"; do test -f "$ROOTD/$c" || { echo "per-row config $ROOTD/$c missing"; exit 2; }; done
fi
EPT=${OV_EPISODE_TIMEOUT_S:-$NEED}
(( EPT >= NEED )) || { echo "OV_EPISODE_TIMEOUT_S=$EPT below the rows' timeout_s $NEED: refused"; exit 2; }
if (( NEED > 2400 )); then MDEF=$(( EPT > 3000 ? EPT : 3000 )); else MDEF=1000; fi
MARGIN=${OV_CLAIM_MARGIN_S:-$MDEF}
(( MARGIN >= MDEF )) || { echo "OV_CLAIM_MARGIN_S=$MARGIN below the required $MDEF: refused"; exit 2; }
want=0; for spec in "$@"; do want=$(( want + ${spec##*:} )); done
have=$(squeue -h -r -u "$USER" | wc -l)
echo "queued/running array tasks of $USER now (all sessions): $have; this launch adds $want; cap $CAP, keep $RESERVE free"
squeue -h -r -u "$USER" -o "%j %T" | sort | uniq -c | sed 's/^/   /'
if (( have + want > CAP - RESERVE )); then echo "REFUSED: would exceed cap-reserve"; exit 3; fi
echo "episode timeout ${EPT} s (rows need ${NEED}); claim margin ${MARGIN} s; per-row configs: $CFGS"
[[ "$DRY" == "1" ]] || mkdir -p "$OUT/logs" "$G4/launch"
csha=$(sha256sum "$COLLECTOR" | cut -c1-16); tsha=$(sha256sum "$TASKS" | cut -c1-16); dsha=$(sha256sum "$ROOTD/source/DISPATCHER.sha256" | cut -c1-16)
rc=0
for spec in "$@"; do
  IFS=: read -r part cpus hrs n <<< "$spec"
  [[ "$part" && "$cpus" =~ ^[0-9]+$ && "$hrs" =~ ^[0-9]+$ && "$n" =~ ^[0-9]+$ ]] || { echo "bad spec $spec"; rc=1; continue; }
  if [[ "$part" == mi3501x && "$hrs" -gt 4 ]]; then echo "mi3501x is capped at 4 h: $spec refused"; rc=1; continue; fi
  budget=$(( hrs * 3600 - MARGIN ))
  (( budget > 600 )) || { echo "$spec: time limit too short for the claim margin"; rc=1; continue; }
  cmd=(sbatch --parsable -p "$part" -c "$cpus" -t "$(printf '%02d:00:00' "$hrs")" --array=0-$(( n - 1 )) \
       -J "ov_soil_$(basename "$OUT")" -o "$OUT/logs/%x_%A_%a.out" \
       --export=ALL,CRM_TASKS=$TASKS,CRM_OUT=$OUT,CRM_CONFIG=$CFG,CRM_COLLECTOR=$COLLECTOR,CRM_BUDGET_S=$budget,CRM_EPISODE_TIMEOUT_S=$EPT \
       "$S")
  if [[ "$DRY" == "1" ]]; then echo "DRY RUN: env -u NEDM_VEHICLE ${cmd[*]}"; continue; fi
  jid=$(env -u NEDM_VEHICLE "${cmd[@]}" 2>/tmp/ov_soil_launch_$$.err)
  if [[ "$jid" =~ ^[0-9]+$ ]]; then
    echo "$part array 0-$(( n - 1 )) -> job $jid"
    printf '%s\t%s\t0-%s\t%s\t%s\t%s\t%s\t%s\ttimeout=%s\tmargin=%s\tdispatcher=%s\n' "$(date +%F_%T)" "$part" "$(( n - 1 ))" "$jid" \
      "$TASKS:$tsha" "$OUT" "$COLLECTOR:$csha" "${hrs}h" "$EPT" "$MARGIN" "$dsha" >> "$G4/launch/submissions.tsv"
  else
    echo "SUBMIT FAILED on $part:"; grep -v '^sbatch: ' /tmp/ov_soil_launch_$$.err; tail -3 /tmp/ov_soil_launch_$$.err; rc=1
  fi
done
rm -f /tmp/ov_soil_launch_$$.err
exit $rc
