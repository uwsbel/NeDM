#!/usr/bin/env bash
# =====================================================================================================================
# so101_push_rl/reproduce.sh: train and check the SO101 push-T policy C1f (results: so101_push_rl/README.md).
#
# A 5-joint SO101 arm pushes a T block to a goal pose (x, y, yaw). PPO trains the policy inside a learned simulator
# (the NRD below); Project Chrono checks it.
#
# NRD (the learned simulator; every PPO run below steps this network, not Chrono)
#   Hub      harryzhang1018/NeDM (dataset), so101_push_rl/nrd/rot_sc8_s61/best.pt, at the revision pinned in
#            so101_push_rl/release_manifest.json
#   sha256   425d836a19ed4113a2f18ce1536fa18103999119ada75dd832bd421611f784d7 (10,248,031 bytes)
#   What     contact NRD "rot_sc8_s61" of the arm and the T: core, collision and contact Transformer networks,
#            2,420,190 parameters, history 4, 20 ms step, seed 61. Same design and file format as contact_nrd/; loaded
#            by nedm.contact_nrd.model.load_model. Only best.pt is read. run_config.json (training config, data
#            sha256) and contact_scale.json next to it are provenance.
#   Chain    rot_scratch_s61 (data merged_v1 = full_v1 + rot_v1; best checkpoint at 80,000 of 120,000 updates)
#            -> rot_sc2_s61 -> ... -> rot_sc8_s61: 7 fine-tune rounds. Each round starts from the round before and
#            trains 20,000 more updates (lr 1e-4); before each round, Chrono rollouts of earlier PPO policies were added
#            to the data (rollout sets r1-r8). Config dict, data sha256 and checkpoint sha256 of every round:
#            provenance.nrd.fine_tune_chain in the Hub copy of so101_push_rl/release_manifest.json.
#   Data     so101_push_rl/nrd_data/train/ = merged_r8_qonly, the data of the last round (15.8 GB, unified_data.npz
#            sha256 8aeb70cd9af52200b1585689dc14b557ff857e42ff2d9e6b028ee9b9501e2dc2): 144,112 train / 7,828
#            validation episodes = full_v1 51,200 / 1,000 (scripted pushes, as contact_nrd/so101_push_t/train),
#            rot_v1 39,320 / 2,500 (scripted walks and turns), rounds r1-r8 53,592 / 4,328 (Chrono rollouts of 10
#            checkpoints of 9 earlier PPO runs). This script does not need it (--with-nrd-data downloads it).
#
# PPO (rsl_rl 2.2.4 PPO, 16,384 NRD environments, one decision = 0.1 s, seed 1)
#   B1   new policy; curriculum so101_push_rl/configs/curriculum_fresh.json (13 stages); 400 updates
#        -> B1_rt_fresh/stage12_end.pt (released as so101_push_rl/policy/b1f_fresh_stage12/policy.pt, sha256 57fffa6f...)
#   C1   --init-from B1 stage12_end.pt (weights only); curriculum so101_push_rl/configs/curriculum_cont.json (2 stages);
#        250 updates -> C1_cont_sc8/stage1_end.pt = the selected policy C1f
#        (released as so101_push_rl/policy/c1f_selected/policy.pt, sha256 ec2eff8a...)
#   Both use so101_push_rl/configs/env.json: NRD rot_sc8_s61, train task table, train starts, rl_train bank, obs stats.
#   Arguments = those of the original runs (one MI350X each; B1 2 h 32 min, C1 1 h 34 min). The trainer stops after
#   3.6 h (--max-hours); this script then runs it again with --resume, as the original resume jobs did.
#   As in the original runs, the trainer starts Chrono checks in the background every 50 updates and at each stage end
#   (--chrono-every 50 --chrono-n 48, 16 workers). They do not feed back into training. They need NRD_PYTHON, also for
#   short runs (--updates, smoke); --no-train-chrono turns them off.
#
# Chrono check: eval_so101_planar_chrono.py on the rows of the released eval id list (val 928 rows, test 912 rows),
#   stage env {"decoder": {"ws_r_min": 0.10}}, deterministic policy, seed 0. Score: final T pose within 10 mm and 3 deg
#   at the end of the task, T at rest (last 0.1 s), no invalid event (score_eval.py).
#
# Files this script uses (paths from the repository root)
#   so101_push_rl/configs/env.json                       environment config; {data} / {code} are filled in below
#   so101_push_rl/configs/curriculum_fresh.json          curriculum of B1 (13 stages)
#   so101_push_rl/configs/curriculum_cont.json           curriculum of C1 (2 stages)
#   so101_push_rl/release_manifest.json                  Hub revision, bytes and sha256 of every released file
#   scripts/so101_push_rl/train_so101_planar_ppo.py      PPO trainer: curriculum, NRD validation, checkpoints
#   scripts/so101_push_rl/eval_so101_planar_chrono.py    Chrono evaluation of policies on task rows (worker pool)
#   scripts/so101_push_rl/score_eval.py                  final-pose score per goal type, paired McNemar test
#   src/nedm/contact_nrd/model.py                        the NRD network and load_model (from contact_nrd/)
#   src/nedm/so101_push_rl/ (PPO environment and Chrono controller):
#     so101_planar_env.py                                NRD environment for PPO: tasks, decoder, reward, validity
#     so101_planar_common.py                             policy action map, observation, validity rules, goal coverage
#     so101_planar_decoder.py                            action -> 20 ms joint commands (planar finger target, IK)
#     so101_planar_chrono.py                             the policy as a Chrono controller (same decoder and rules)
#     so101_direct_common.py                             finger-T distance geometry and observation pieces
#     so101_goal_common.py                               goal error, potential, "T settled inside the goal" test
#     so101_goal_chrono.py                               Chrono goal episode: recorded prefix, then the controller
#     so101_push_chrono_tracking_env.py                  client of the Chrono step server, measured state history
#     so101_push_common.py                               history length, state layout, obs normalisation, action map
#     so101_fk.py                                        batched forward kinematics, dense collision points (torch)
#     so101_kin_torch.py                                 batched arm kinematics and IK (torch)
#   src/nedm/so101_push/chrono_server.py                 Chrono step server, one process per episode ($NRD_PYTHON)
#   src/nedm/so101_push/sim.py                           Chrono scene: arm, T, table, joint PD, contact report
#   src/nedm/so101_push/robot.py                         SO101 model from assets/so101, collector config loader
#   src/nedm/so101_push/geometry.py                      T shape geometry
#   src/nedm/so101_push/clearance.py                     arm-T clearance (built by the Chrono scene)
#   assets/so101/so101_robot.json, so101_hulls.npz, so101_finger_sections.npz, so101_finger_boxes.json
#                                                        SO101 links, joints and collision shapes
#   configs/so101_push/collector_v2.json                 Chrono scene settings (step, PD gains, friction, T, table)
#   configs/so101_push/collision_points_dense_2mm.npz    dense surface points of the arm links (distance checks)
#
# Use
#   so101_push_rl/reproduce.sh [full|eval|smoke] [options]
#     full   download, B1, C1, Chrono eval of the trained C1 and the released C1f (paired), score   (default)
#     eval   download, Chrono eval of the released policy, score
#     smoke  setup check: 2 PPO updates for B1 and C1 (256 environments; C1 starts from the released B1), then
#            4 Chrono tasks (val rows) for the released C1f and the smoke C1, score
#   Options
#     --steps LIST         comma list of download,b1,c1,eval,score (default: all steps of the mode). A list without
#                          download skips the download and its sha256 check (offline use: the files must already
#                          be in DIR/so101_push_rl/ of --data)
#     --out DIR            outputs (default artifacts/so101_push_rl_repro; smoke: a new smoke_<time> folder in it)
#     --data DIR           download folder; files land in DIR/so101_push_rl/... (default artifacts)
#     --device DEV         torch device of PPO (default cuda)
#     --num-envs N         PPO environments (default 16384; smoke 256)
#     --updates N          stop each PPO run after N updates (default 0 = whole curriculum; smoke 2)
#     --c1-init own|released   C1 starts from this script's B1 (default) or the released B1 policy (smoke)
#     --no-train-chrono    no background Chrono checks during PPO (--chrono-n 0; the checks never feed back into
#                          training, so the training computation does not change)
#     --chrono-workers N   workers of those background checks (default 16, as the original runs)
#     --split val|test|both    eval rows (default test; smoke val)
#     --rows N             evaluate N rows of the eval id list, spread over the goal types (default 0 = all; smoke 4)
#     --policy LIST        released policies for the eval mode: c1f_selected (default), d2f_runner_up,
#                          b1f_fresh_stage12 (comma list)
#     --workers N          Chrono eval workers (default: number of CPUs)
#     --with-nrd-data      also download the NRD training data (15.8 GB)
#   Environment
#     PY                   Python with torch, rsl-rl-lib 2.2.4, tensorboard, numpy, huggingface_hub (default python3)
#     NRD_PYTHON           Python with pychrono (Chrono 10, Bullet collision) and numpy, for the Chrono server; needed by
#                          the eval and by the background Chrono checks of PPO. It can be the same Python as PY.
#     SO101_CHRONO_ENV     optional bash file that the Chrono server sources (default: one this script writes)
# =====================================================================================================================
set -euo pipefail

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"
MODE=full
STEPS=""
OUT=""
DATA_ROOT="$CODE/artifacts"
DEVICE=cuda
NUM_ENVS=""
UPDATES=""
C1_INIT=""
TRAIN_CHRONO=1
CHRONO_WORKERS=16
SPLIT=""
ROWS=""
POLICIES=c1f_selected
WORKERS="$(nproc)"
WITH_NRD_DATA=0

die() { echo "reproduce.sh: $*" >&2; exit 1; }
log() { echo "[reproduce $(date +%H:%M:%S)] $*"; }

while [ $# -gt 0 ]; do
    case "$1" in
        full|eval|smoke) MODE="$1" ;;
        --steps) STEPS="$2"; shift ;;
        --out) OUT="$2"; shift ;;
        --data) DATA_ROOT="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        --num-envs) NUM_ENVS="$2"; shift ;;
        --updates) UPDATES="$2"; shift ;;
        --c1-init) C1_INIT="$2"; shift ;;
        --no-train-chrono) TRAIN_CHRONO=0 ;;
        --chrono-workers) CHRONO_WORKERS="$2"; shift ;;
        --split) SPLIT="$2"; shift ;;
        --rows) ROWS="$2"; shift ;;
        --policy) POLICIES="$2"; shift ;;
        --workers) WORKERS="$2"; shift ;;
        --with-nrd-data) WITH_NRD_DATA=1 ;;
        -h|--help) sed -n '3,/^# =====/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument $1 (see --help)" ;;
    esac
    shift
done

case "$MODE" in
    full)  : "${STEPS:=download,b1,c1,eval,score}" "${NUM_ENVS:=16384}" "${UPDATES:=0}" "${C1_INIT:=own}" "${SPLIT:=test}" "${ROWS:=0}" ;;
    eval)  : "${STEPS:=download,eval,score}" "${NUM_ENVS:=16384}" "${UPDATES:=0}" "${C1_INIT:=own}" "${SPLIT:=test}" "${ROWS:=0}" ;;
    smoke) : "${STEPS:=download,b1,c1,eval,score}" "${NUM_ENVS:=256}" "${UPDATES:=2}" "${C1_INIT:=released}" "${SPLIT:=val}" "${ROWS:=4}" ;;
esac
OUT="${OUT:-$CODE/artifacts/so101_push_rl_repro}"
[ "$MODE" = smoke ] && OUT="$OUT/smoke_$(date +%m%d_%H%M%S)"
mkdir -p "$OUT" "$DATA_ROOT"
OUT="$(cd "$OUT" && pwd)"
DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"
D="$DATA_ROOT/so101_push_rl"                       # the Hub folder so101_push_rl/ after the download
RUNS="$OUT/runs"
export PYTHONPATH="$CODE/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$CODE"

has_step() { case ",$STEPS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
for s in ${STEPS//,/ }; do
    case "$s" in download|b1|c1|eval|score) ;; *) die "unknown step $s (download, b1, c1, eval, score)" ;; esac
done
case "$SPLIT" in val|test|both) ;; *) die "--split must be val, test or both" ;; esac
case "$C1_INIT" in own|released) ;; *) die "--c1-init must be own or released" ;; esac
log "mode $MODE, steps $STEPS, out $OUT, data $D"

# ---------------------------------------------------------------------------------------------------- checks
need_chrono=0
has_step eval && need_chrono=1
{ has_step b1 || has_step c1; } && [ "$TRAIN_CHRONO" = 1 ] && need_chrono=1   # Chrono checks during PPO (any --updates)
if has_step b1 || has_step c1 || has_step eval; then
    "$PY" -c "import numpy, torch, rsl_rl" 2>/dev/null || die "$PY needs numpy, torch and rsl-rl-lib 2.2.4 (set PY)"
fi
if { has_step b1 || has_step c1; } && [ "$DEVICE" != cpu ]; then
    "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || die "no GPU for --device $DEVICE"
fi
if [ "$need_chrono" = 1 ]; then
    [ -n "${NRD_PYTHON:-}" ] || die "set NRD_PYTHON to a Python with pychrono: the eval and the Chrono checks during PPO need it (--no-train-chrono: PPO without Chrono)"
    "$NRD_PYTHON" -c "import numpy, pychrono" 2>/dev/null || die "$NRD_PYTHON cannot import pychrono"
    export NRD_PYTHON
    if [ -z "${SO101_CHRONO_ENV:-}" ]; then          # what the Chrono server sources before it starts
        cat > "$OUT/chrono_env.sh" <<EOF
# Chrono server runtime, written by so101_push_rl/reproduce.sh. ChronoServer of
# nedm.so101_push_rl.so101_push_chrono_tracking_env sources this file with SO101_CODE = the repository root, then runs
# \$NRD_PYTHON -m nedm.so101_push.chrono_server.
export NRD_PYTHON="$NRD_PYTHON"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="\$SO101_CODE/src:\${PYTHONPATH:-}"
cd "\$SO101_CODE"
EOF
        export SO101_CHRONO_ENV="$OUT/chrono_env.sh"
    fi
fi

# ---------------------------------------------------------------------------------------------------- 1. download
DL_GROUPS=nrd,rl_inputs,policy
[ "$WITH_NRD_DATA" = 1 ] && DL_GROUPS="$DL_GROUPS,nrd_data"
if has_step download; then
    log "download + sha256 check ($DL_GROUPS)"
    "$PY" - "$CODE/so101_push_rl/release_manifest.json" "$DATA_ROOT" "$DL_GROUPS" <<'EOF'
import hashlib, json, re, sys
from pathlib import Path

manifest, dest, groups = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]), sys.argv[3].split(",")
folders = tuple(f"{manifest['prefix']}{g}/" for g in groups)
files = {n: m for n, m in manifest["files"].items() if n.startswith(folders)}
for n in files:
    if n.startswith("/") or "\\" in n or ".." in n.split("/"):
        sys.exit(f"unsafe path in the manifest: {n}")
rev = manifest.get("hf_revision", "")
pinned = re.fullmatch(r"[0-9a-f]{40}", rev) is not None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def good(n):
    p = dest / n
    return p.is_file() and p.stat().st_size == files[n]["bytes"] and sha256(p) == files[n]["sha256"]


def hub_or_exit(n_missing):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(f"{n_missing} file(s) must be downloaded, and the download needs huggingface_hub in the Python of PY "
                 f"(pip install huggingface_hub). Offline: put the files in {dest} and run without the download step.")
    return snapshot_download


absent = [n for n in files if not ((dest / n).is_file() and (dest / n).stat().st_size == files[n]["bytes"])]
if absent and pinned:
    hub_or_exit(len(absent))                    # pre-check before the sha256 pass
ok = {n: good(n) for n in sorted(files)}       # files already in place are not downloaded again (no Hub call)
missing = [n for n in ok if not ok[n]]
if missing and pinned:
    snapshot_download = hub_or_exit(len(missing))
    print(f"download {len(missing)} of {len(files)} file(s) from {manifest['repo_id']} at revision {rev}", flush=True)
    snapshot_download(repo_id=manifest["repo_id"], repo_type="dataset", revision=rev, allow_patterns=missing,
                      local_dir=str(dest))
    ok.update({n: good(n) for n in missing})
elif not pinned:
    print(f"hf_revision is {rev!r} (no pinned Hub revision): no download, only the check of the files in {dest}")
bad = [str(dest / n) for n in ok if not ok[n]]
for n in ok:
    print(f"{'ok ' if ok[n] else 'BAD'} {dest / n}", flush=True)
if bad and not pinned:
    sys.exit(f"{len(bad)} file(s) missing or with another size or sha256, and no pinned Hub revision to download from")
if bad:
    sys.exit(f"{len(bad)} file(s) missing or with another size or sha256: delete them and run again")
EOF
fi

# ---------------------------------------------------------------------------------------------------- configs
# configs/env.json with {data} and {code} filled in; released policy folders with the same local paths (the eval reads
# collector_config, dense_points and stats_path from <run>/env_cfg.json and the network size from <run>/train_cfg.json)
RELEASED="$POLICIES"
[ "$MODE" = eval ] || RELEASED=c1f_selected
"$PY" - "$CODE" "$D" "$OUT" "$RELEASED" <<'EOF'
import json, os, shutil, sys
from pathlib import Path

code, data, out, released = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4].split(",")
PATH_KEYS = ("checkpoints", "starts_path", "banks_path", "tasks_path", "stats_path", "collector_config", "dense_points")
fill = lambda v: [fill(x) for x in v] if isinstance(v, list) else v.replace("{data}", str(data)).replace("{code}", str(code))
env = json.loads((code / "so101_push_rl/configs/env.json").read_text())
paths = {k: fill(env[k]) for k in PATH_KEYS}
(out / "configs").mkdir(parents=True, exist_ok=True)
(out / "configs/env.json").write_text(json.dumps({**env, **paths}, indent=1))
for name in filter(None, released):
    src, dst = data / "policy" / name, out / "released" / name
    if not (src / "policy.pt").is_file():
        continue                                   # not downloaded (the steps that need it stop with a clear error)
    dst.mkdir(parents=True, exist_ok=True)
    ec = json.loads((src / "env_cfg.json").read_text())
    (dst / "env_cfg.json").write_text(json.dumps({**ec, **paths}, indent=1))
    shutil.copyfile(src / "train_cfg.json", dst / "train_cfg.json")
    if (dst / "policy.pt").is_symlink() or (dst / "policy.pt").exists():
        (dst / "policy.pt").unlink()
    os.symlink(src / "policy.pt", dst / "policy.pt")
EOF

# ---------------------------------------------------------------------------------------------------- 2. / 3. PPO
VAL=(--val-tasks "$D/rl_inputs/tasks/rot_tasks_rl_val_rt.npz" --val-starts "$D/rl_inputs/starts/rot_starts_rl_val.npz"
     --val-banks "$D/rl_inputs/banks/rot_bank_rl_val.npz")
CHRONO_N=48
[ "$TRAIN_CHRONO" = 1 ] || CHRONO_N=0

run_ppo() {          # run_ppo <run name> <curriculum> [--init-from <ckpt>]
    local run="$RUNS/$1" cur="$2"; shift 2
    local extra=()
    [ "$DEVICE" = cuda ] || extra+=(--device "$DEVICE")
    [ "$UPDATES" = 0 ] || extra+=(--max-iterations "$UPDATES")
    for attempt in 1 2 3 4 5 6 7 8; do
        [ -f "$run/done.json" ] && return 0
        local resume=()
        [ -f "$run/curriculum_state.json" ] && resume=(--resume)
        log "PPO $(basename "$run") (job $attempt${resume:+, resume})"
        "$PY" scripts/so101_push_rl/train_so101_planar_ppo.py train \
            --env-cfg "$OUT/configs/env.json" --curriculum "$cur" "${VAL[@]}" \
            --out "$run" --seed 1 --num-envs "$NUM_ENVS" --start-stage 0 --chrono-every 50 --chrono-n "$CHRONO_N" \
            --chrono-workers "$CHRONO_WORKERS" --max-hours 3.6 "$@" "${extra[@]}" "${resume[@]}" \
            --set num_steps_per_env=48 algorithm.num_mini_batches=32 std_floor=0.2
        [ "$UPDATES" = 0 ] || return 0             # a capped run stops with paused.json
    done
    [ -f "$run/done.json" ] || die "$run did not finish its curriculum"
}

mkdir -p "$RUNS"
if has_step b1; then
    run_ppo B1_rt_fresh so101_push_rl/configs/curriculum_fresh.json
fi
if has_step c1; then
    if [ "$C1_INIT" = own ]; then
        INIT="$RUNS/B1_rt_fresh/stage12_end.pt"
    else
        INIT="$D/policy/b1f_fresh_stage12/policy.pt"
    fi
    [ -f "$INIT" ] || die "no B1 checkpoint $INIT (run step b1, or use --c1-init released)"
    run_ppo C1_cont_sc8 so101_push_rl/configs/curriculum_cont.json --init-from "$INIT"
fi

# ---------------------------------------------------------------------------------------------------- 4. / 5. eval, score
ARMS=()
add_released() {     # add_released <folder name> <arm name>
    [ -f "$OUT/released/$1/policy.pt" ] || die "released policy $1 is missing (run the download step)"
    ARMS+=(--policy "$2=$OUT/released/$1:$OUT/released/$1/policy.pt")
}
if [ "$MODE" = eval ]; then
    for p in ${POLICIES//,/ }; do
        case "$p" in
            c1f_selected) add_released c1f_selected C1f ;;
            d2f_runner_up) add_released d2f_runner_up D2f ;;
            b1f_fresh_stage12) add_released b1f_fresh_stage12 B1f ;;
            *) die "unknown released policy $p" ;;
        esac
    done
else
    add_released c1f_selected C1f
    if [ "$UPDATES" = 0 ]; then OWN="$RUNS/C1_cont_sc8/stage1_end.pt"; else OWN="$RUNS/C1_cont_sc8/last.pt"; fi
    if [ -f "$OWN" ]; then ARMS+=(--policy "C1=$RUNS/C1_cont_sc8:$OWN"); else log "no $OWN: the eval runs the released C1f only"; fi
fi

SPLITS=("$SPLIT")
[ "$SPLIT" = both ] && SPLITS=(val test)
for sp in "${SPLITS[@]}"; do
    if [ "$sp" = val ]; then
        TASKS="$D/rl_inputs/tasks/rot_tasks_rl_val_rt.npz"; STARTS="$D/rl_inputs/starts/rot_starts_rl_val.npz"
        BANKS="$D/rl_inputs/banks/rot_bank_rl_val.npz"
    else
        TASKS="$D/rl_inputs/tasks/rot_tasks_rl_test_rt.npz"; STARTS="$D/rl_inputs/starts/starts_rl_test_all.npz"
        BANKS="$D/rl_inputs/banks/rl_test.npz"
    fi
    IDS="$D/rl_inputs/tasks/eval_ids_$sp.json"
    EVAL_DIR="$OUT/eval_$sp"
    if has_step eval; then
        # rows of the released eval id list (checked against the task table's sha256); --rows N: N rows, one goal type
        # after the other
        ROW_IDS=$("$PY" - "$IDS" "$TASKS" "$ROWS" <<'EOF'
import hashlib, itertools, json, sys
from pathlib import Path

ids, table, n = json.loads(Path(sys.argv[1]).read_text()), sys.argv[2], int(sys.argv[3])
if hashlib.sha256(Path(table).read_bytes()).hexdigest() != ids["tasks_sha256"]:
    sys.exit(f"{table} is not the task table of {sys.argv[1]}")
if n <= 0:
    rows = list(ids["all_ids"])
else:
    per_class = [[r for r in itertools.chain(*itertools.zip_longest(*groups.values())) if r is not None]
                 for groups in ids["classes"].values()]
    rows = [r for r in itertools.chain(*itertools.zip_longest(*per_class)) if r is not None][:n]
print(" ".join(str(int(r)) for r in rows))
EOF
)
        log "Chrono eval, $sp split, $(wc -w <<< "$ROW_IDS") rows, arms: ${ARMS[*]}"
        # shellcheck disable=SC2086
        "$PY" scripts/so101_push_rl/eval_so101_planar_chrono.py "${ARMS[@]}" --tasks "$TASKS" --starts "$STARTS" \
            --banks "$BANKS" --task-ids $ROW_IDS --stage-env '{"decoder": {"ws_r_min": 0.10}}' --workers "$WORKERS" \
            --out "$EVAL_DIR" > "$EVAL_DIR.log"
    fi
    if has_step score; then
        [ -f "$EVAL_DIR/episodes.jsonl" ] || die "no eval output in $EVAL_DIR (run the eval step)"
        log "score, $sp split -> $EVAL_DIR/score.md"
        "$PY" scripts/so101_push_rl/score_eval.py "$EVAL_DIR" --ids "$IDS" --out "$EVAL_DIR/score" > /dev/null
        awk '/^\| Class/ {p = 1} p && /^$/ {exit} p && !/\| 0 \([0-9]+ missing\) \|/' "$EVAL_DIR/score.md"
    fi
done
log "done: $OUT"
