#!/bin/bash
# Stage the offroad_vehicles_20260927 cluster root (module M3; PLAN sections 1.6, 6; S3 section 3.2). Run LOCALLY from the
# worktree root (it calls ssh / rsync amd). Nothing in G3 or any older root is written; G3 files are only read (cp source).
#
#   bash scripts/ov_stage.sh --dry-run [--allow-missing]      -> DEST = G4/stage_dryrun (a throw-away copy, same layout)
#   bash scripts/ov_stage.sh --real                           -> DEST = G4 (refuses if DEST/source/DISPATCHER.sha256 exists)
#   bash scripts/ov_stage.sh --tasks <local task file> ... [--real|--dry-run]
#                                                             -> only copy task files (+ .meta.json) to DEST/tasks, read-only
#   bash scripts/ov_stage.sh --tools scripts/ov_collect_status.py ... [--real|--dry-run]
#                                                             -> copy / refresh login-node tools (not frozen files)
#   --with-m113 (with --dry-run / --real): also stage scripts/ov_m113.py and configs/crm_m113.json as frozen files
#   bash scripts/ov_stage.sh --add scripts/ov_m113.py configs/crm_m113_v2.json ... [--real|--dry-run]
#                                                             -> add NEW files to a staged tree (never replaces; appends
#                                                                their sha256 to DISPATCHER.sha256, freezes them)
#
# Layout written under DEST:
#   source/            src/, scripts/ of this worktree (no __pycache__ / *.pyc), assets/traverse/arena_f104_50h_v1 and
#                      assets/traverse/vehicles/ (the private vehicle-data folders of M1/M2), source_manifest.json (byte copy
#                      of G3/source's, sha256 c9e4ff01...), DISPATCHER.sha256 (paths relative to source/)
#   configs/           crm_main.json = byte copy of G3/configs/crm_main.json (sha256 90cd049e..., the file behind K3's soil
#                      jobs and collect_v1); crm_m113.json from this worktree if present (module M2)
#   tasks/ soil_v1/ runtime/ launch/submissions.tsv checks/
# Checks: the unchanged collector files equal G3's frozen copies (crm_collect.py cb6792be, crm_worker.py f856b998,
# ag_vehicle.py 072716ee, ag_crm_collect.py b52e1fa6, ag_gator_belly.json 0c25a57c, gen_collect.py b6ba0622,
# traverse_fdm_rgbd_diverse_chrono.py 2996c567); every file of DISPATCHER.sha256 has the same sha256 locally and on the
# cluster; the dispatcher files parse (python3 ast, no bytecode written). Then chmod a-w on the dispatcher files, the
# configs, every src/nedm/**/*.py and DISPATCHER.sha256. A record goes to DEST/runtime/stage_record.txt and to
# K4/stage_records/.
set -euo pipefail
K4=artifacts/traverse/offroad_vehicles_20260927
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
G4=/work1/dannegrut/harry/experiments/offroad_vehicles_20260927
MODE=""; ALLOW_MISSING=0; WITH_M113=0; TASKS=(); ADD=(); TOOLS=()
while (( $# )); do
  case "$1" in
    --dry-run) MODE=dry;; --real) MODE=real;; --allow-missing) ALLOW_MISSING=1;; --with-m113) WITH_M113=1;;
    --tools) shift; while (( $# )) && [[ "$1" != --* ]]; do TOOLS+=("$1"); shift; done; continue;;
    --tasks) shift; while (( $# )) && [[ "$1" != --* ]]; do TASKS+=("$1"); shift; done; continue;;
    --add) shift; while (( $# )) && [[ "$1" != --* ]]; do ADD+=("$1"); shift; done; continue;;
    *) echo "unknown argument $1"; exit 2;;
  esac; shift
done
[[ -n "$MODE" ]] || { echo "give --dry-run or --real"; exit 2; }
test -f scripts/ov_stage.sh || { echo "run from the worktree root"; exit 2; }
if [[ "$MODE" == dry ]]; then DEST=$G4/stage_dryrun; else DEST=$G4; fi
TS=$(date +%Y%m%d_%H%M%S)

# ---------------------------------------------------------------- task files only
if (( ${#TASKS[@]} )); then
  ssh amd "test -f $DEST/source/DISPATCHER.sha256" || { echo "$DEST is not staged yet"; exit 2; }
  for t in "${TASKS[@]}"; do
    b=$(basename "$t"); ls=$(sha256sum "$t" | cut -d' ' -f1)
    ssh amd "test ! -e $DEST/tasks/$b" || { echo "$DEST/tasks/$b exists: task files are never replaced (new name)"; exit 2; }
    rsync -a "$t" amd:"$DEST/tasks/$b"
    [[ -f "$t.meta.json" ]] && rsync -a "$t.meta.json" amd:"$DEST/tasks/$b.meta.json"
    rs=$(ssh amd "chmod a-w $DEST/tasks/$b $DEST/tasks/$b.meta.json 2>/dev/null; sha256sum $DEST/tasks/$b" | cut -d' ' -f1)
    [[ "$ls" == "$rs" ]] || { echo "sha256 mismatch for $b"; exit 1; }
    echo "staged $DEST/tasks/$b sha256 $rs (read-only)"
    printf '%s\ttask\t%s\t%s\n' "$TS" "$DEST/tasks/$b" "$rs" | ssh amd "cat >> $DEST/runtime/stage_record.txt"
  done
  exit 0
fi

# ---------------------------------------------------------------- login-node tools (replace allowed; never a frozen file)
if (( ${#TOOLS[@]} )); then
  ssh amd "test -f $DEST/source/DISPATCHER.sha256" || { echo "$DEST is not staged yet"; exit 2; }
  for f in "${TOOLS[@]}"; do
    [[ -f "$f" && "$f" == scripts/ov_* ]] || { echo "$f: only existing scripts/ov_* tool files"; exit 2; }
    ssh amd "! grep -q '  $f\$' $DEST/source/DISPATCHER.sha256" || { echo "$f is a frozen dispatcher file: refused"; exit 2; }
    rsync -a "$f" amd:"$DEST/source/$f"
    printf '%s\ttool\t%s\t%s\n' "$TS" "$f" "$(sha256sum "$f" | cut -d' ' -f1)" | ssh amd "cat >> $DEST/runtime/stage_record.txt"
    echo "tool $f -> $DEST/source/$f"
  done
  exit 0
fi

# ---------------------------------------------------------------- add NEW files to an already staged tree
# (e.g. scripts/ov_m113.py arriving after the Polaris launch, or configs/crm_m113_v2.json): never replaces a file; the
# new entries are appended to DISPATCHER.sha256 (old entries unchanged, so running jobs are unaffected), frozen, recorded.
if (( ${#ADD[@]} )); then
  ssh amd "test -f $DEST/source/DISPATCHER.sha256" || { echo "$DEST is not staged yet"; exit 2; }
  for f in "${ADD[@]}"; do
    [[ -f "$f" ]] || { echo "$f missing locally"; exit 2; }
    case "$f" in configs/*) dst="$DEST/$f"; rel="../$f";; scripts/*|src/*|assets/*) dst="$DEST/source/$f"; rel="$f";; *) echo "$f: only configs/, scripts/, src/, assets/"; exit 2;; esac
    ssh amd "test ! -e $dst" || { echo "$dst exists: never replaced (use a new name)"; exit 2; }
    ssh amd "mkdir -p $(dirname "$dst")"
    rsync -a "$f" amd:"$dst"
    h=$(sha256sum "$f" | cut -d' ' -f1)
    ssh amd bash -s -- "$DEST" "$dst" "$rel" "$h" <<'ADDEOF'
set -euo pipefail
DEST=$1; dst=$2; rel=$3; h=$4
[[ "$(sha256sum "$dst" | cut -d' ' -f1)" == "$h" ]] || { echo "sha256 mismatch after copy: $dst"; exit 1; }
chmod a-w "$dst"
cd "$DEST/source"; chmod u+w DISPATCHER.sha256; echo "$h  $rel" >> DISPATCHER.sha256; chmod a-w DISPATCHER.sha256
sha256sum --quiet -c DISPATCHER.sha256 && echo "added $rel ($h); DISPATCHER.sha256 re-checked"
ADDEOF
    printf '%s\tadd\t%s\t%s\n' "$TS" "$rel" "$h" | ssh amd "cat >> $DEST/runtime/stage_record.txt"
    mkdir -p "$K4/stage_records"; printf '%s\tadd\t%s\t%s\t%s\n' "$TS" "$MODE" "$rel" "$h" >> "$K4/stage_records/added_files.tsv"
  done
  exit 0
fi

# ---------------------------------------------------------------- local checks
declare -A FROZEN=( [scripts/crm_collect.py]=cb6792bebeb158bc1adf429b5b85a85c8b2cbbd4705a088cc8ff083f74eb9881
  [scripts/crm_worker.py]=f856b99848449466663b785a9d57c3cd56d283e3823735963786a156a6311fe8
  [scripts/ag_vehicle.py]=072716ee715469950409c334482bf990853dd4f70b3c85790eec99f04b331f9d
  [scripts/ag_crm_collect.py]=b52e1fa69afe5745839bc9c32c5425b7a0328acbed61abdf0e4963644e5e2346
  [scripts/ag_gator_belly.json]=0c25a57cc77011b71eb7bd31dd6a1b926581f52efacccba3bfd4d20f54d0ded0
  [scripts/gen_collect.py]=b6ba062260aa3633f1eb9093c575e9c3b0dee2b9e37035bb7c675a3c519d6389
  [scripts/traverse_fdm_rgbd_diverse_chrono.py]=2996c567f248126e7e25498a8a74124caf5c5d5fb01e62aaee3c631df2bdb920 )
for f in "${!FROZEN[@]}"; do
  h=$(sha256sum "$f" | cut -d' ' -f1)
  [[ "$h" == "${FROZEN[$f]}" ]] || { echo "$f differs from the frozen G3 copy ($h)"; exit 1; }
done
# the new dispatcher (M1), the M113 module (M2, lazy import), the job scripts (M3)
NEW=(scripts/ov_crm_collect.py scripts/ov_vehicle.py scripts/ov_soil.sbatch scripts/ov_soil_launch.sh)
# the M113 module and its 0.5 ms config are staged only with --with-m113 (else added later with --add once module M2
# has passed its checks: a frozen draft could not be replaced, and ov_crm_collect.py imports ov_m113 by name)
(( WITH_M113 )) && NEW+=(scripts/ov_m113.py)
# login-node tools of this study (copied, NOT frozen; later versions with --tools)
TOOLS_DEFAULT=(scripts/ov_collect_status.py scripts/ov_smoke_analyze.py)
[[ -f scripts/crm_collect_ext.py ]] && NEW_OPT=(scripts/crm_collect_ext.py) || NEW_OPT=()
# more files the soil loop executes or reads (e.g. a helper module of M1): OV_STAGE_EXTRA="scripts/a.py scripts/b.json"
EXTRA=(${OV_STAGE_EXTRA:-})
for f in "${EXTRA[@]}"; do [[ -f "$f" ]] || { echo "OV_STAGE_EXTRA file $f missing"; exit 2; }; done
MISSING=()
for f in "${NEW[@]}"; do [[ -f "$f" ]] || MISSING+=("$f"); done
[[ -d assets/traverse/vehicles ]] || MISSING+=(assets/traverse/vehicles/)
if (( ${#MISSING[@]} )); then
  echo "missing: ${MISSING[*]}"
  if [[ "$MODE" == real || "$ALLOW_MISSING" != 1 ]]; then
    # the M113 module is optional (PLAN 1.3 bounded path); everything else is required for a real stage
    echo "refused (use --dry-run --allow-missing to rehearse without them)"; exit 2
  fi
fi
if [[ "$MODE" == real ]]; then
  ssh amd "test ! -e $DEST/source/DISPATCHER.sha256" || { echo "$DEST/source is already staged and frozen: put new files next to it by name, never rsync over it"; exit 2; }
fi

# ---------------------------------------------------------------- local bundle
B=$(mktemp -d /tmp/ov_stage_XXXX)
trap 'rm -rf "$B"' EXIT
mkdir -p "$B/source/assets/traverse"
# scripts/: everything except this study's ov_* files (other modules' drafts must not be frozen or shadow later
# versions) and the other session's sp_* file; then the dispatcher set and the tools by name
rsync -a --exclude '__pycache__' --exclude '*.pyc' --exclude 'ov_*' --exclude 'sp_*' src scripts "$B/source/"
for f in "${NEW[@]}" "${EXTRA[@]}" "${TOOLS_DEFAULT[@]}"; do [[ -f "$f" ]] && cp -p "$f" "$B/source/$f"; done
rsync -a assets/traverse/arena_f104_50h_v1 "$B/source/assets/traverse/"
[[ -d assets/traverse/vehicles ]] && rsync -a --exclude '__pycache__' assets/traverse/vehicles "$B/source/assets/traverse/"
mkdir -p "$B/configs"
(( WITH_M113 )) && [[ -f configs/crm_m113.json ]] && cp -p configs/crm_m113.json "$B/configs/"
# DISPATCHER.sha256: every file the soil loop of this study executes or reads from the source tree + both configs
(
  cd "$B/source"
  {
    for f in "${!FROZEN[@]}" "${NEW[@]}" "${NEW_OPT[@]}" "${EXTRA[@]}"; do [[ -f "$f" ]] && echo "$f"; done
    find src/nedm -name '*.py' | sort
    [[ -d assets/traverse/vehicles ]] && find assets/traverse/vehicles -type f | sort
    find assets/traverse/arena_f104_50h_v1 -type f | sort
  } | sort -u > /tmp/ov_stage_files_$$
  xargs -a /tmp/ov_stage_files_$$ -d '\n' sha256sum > DISPATCHER.sha256.part
  rm -f /tmp/ov_stage_files_$$
)
cmain_local=artifacts/traverse/crm_f104_v1/configs/crm_main.json
[[ "$(sha256sum $cmain_local | cut -d' ' -f1)" == 90cd049ee6b3479a74237e7cdf2ff3567c26d580c117e2bab5ab8eff2218b248 ]] || { echo "local crm_main.json reference differs"; exit 1; }
( cd "$B/source"; echo "90cd049ee6b3479a74237e7cdf2ff3567c26d580c117e2bab5ab8eff2218b248  ../configs/crm_main.json" >> DISPATCHER.sha256.part
  [[ -f ../configs/crm_m113.json ]] && sha256sum ../configs/crm_m113.json >> DISPATCHER.sha256.part
  mv DISPATCHER.sha256.part DISPATCHER.sha256 )
python3 - "$B/source" <<'EOF'
import ast, sys, pathlib
root = pathlib.Path(sys.argv[1])
for line in open(root / 'DISPATCHER.sha256'):
    f = line.split(None, 1)[1].strip()
    if f.endswith('.py'):
        ast.parse((root / f).read_text(), filename=f)
print('dispatcher python files parse')
EOF
echo "bundle: $(du -sh "$B" | cut -f1), $(wc -l < "$B/source/DISPATCHER.sha256") files in DISPATCHER.sha256"

# ---------------------------------------------------------------- cluster
ssh amd "mkdir -p $DEST/source $DEST/configs $DEST/tasks $DEST/soil_v1 $DEST/runtime $DEST/launch $DEST/checks"
rsync -a "$B/source/" amd:"$DEST/source/"
rsync -a "$B/configs/" amd:"$DEST/configs/"
ssh amd bash -s -- "$DEST" "$G3" <<'EOF'
set -euo pipefail
DEST=$1; G3=$2
cp -p --remove-destination "$G3/configs/crm_main.json" "$DEST/configs/crm_main.json"
cp -p --remove-destination "$G3/source/source_manifest.json" "$DEST/source/source_manifest.json"
[[ "$(sha256sum $DEST/configs/crm_main.json | cut -d' ' -f1)" == 90cd049ee6b3479a74237e7cdf2ff3567c26d580c117e2bab5ab8eff2218b248 ]] || { echo "crm_main.json copy differs"; exit 1; }
[[ "$(sha256sum $DEST/source/source_manifest.json | cut -d' ' -f1)" == c9e4ff01f9299c404695ee4edf8225c087475c0cebfc7dbd234d44e9bce3fd1b ]] || { echo "source_manifest copy differs"; exit 1; }
cd "$DEST/source"
sha256sum --quiet -c DISPATCHER.sha256 && echo "cluster: every DISPATCHER.sha256 entry matches"
# freeze: the listed files (not directories), the configs, the record itself
awk '{print $2}' DISPATCHER.sha256 | xargs -d '\n' chmod a-w
chmod a-w DISPATCHER.sha256 source_manifest.json ../configs/*.json
[[ -f ../launch/submissions.tsv ]] || printf 'time\tpartition\tarray\tjob\ttasks:sha\tout\tcollector:sha\tlimit\ttimeout\tmargin\tdispatcher\n' > ../launch/submissions.tsv
n_w=$(awk '{print $2}' DISPATCHER.sha256 | { c=0; while read -r f; do if [[ -w "$f" ]]; then c=$((c+1)); fi; done; echo $c; })
echo "writable frozen files after chmod: $n_w"
n_pc=$(find . -name __pycache__ | wc -l); echo "__pycache__ folders in source: $n_pc"
EOF
REC=$(mktemp)
{
  echo "time $TS mode $MODE dest $DEST"
  echo "worktree $(pwd) branch $(git rev-parse --abbrev-ref HEAD) commit $(git rev-parse HEAD) dirty_files $(git status --porcelain | wc -l)"
  echo "missing_at_stage ${MISSING[*]:-none}"
  echo "DISPATCHER.sha256 sha256 $(sha256sum "$B/source/DISPATCHER.sha256" | cut -d' ' -f1)"
  grep -E 'scripts/(ov_|ag_vehicle|ag_crm|crm_collect|crm_worker)|configs/' "$B/source/DISPATCHER.sha256"
} > "$REC"
ssh amd "cat >> $DEST/runtime/stage_record.txt" < "$REC"
mkdir -p "$K4/stage_records"; cp "$REC" "$K4/stage_records/stage_${MODE}_${TS}.txt"; cp "$B/source/DISPATCHER.sha256" "$K4/stage_records/DISPATCHER_${MODE}_${TS}.sha256"
rm -f "$REC"
echo "staged $DEST ($MODE); record $K4/stage_records/stage_${MODE}_${TS}.txt"
