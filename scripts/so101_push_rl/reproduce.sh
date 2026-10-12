#!/usr/bin/env bash
# =====================================================================================================================
# scripts/so101_push_rl/reproduce.sh: train the SO101 push-T policy C1f inside the NRD (so101_push_rl/README.md).
#
# A 5-joint SO101 arm pushes a T block to a goal pose (x, y, yaw). PPO trains the policy inside a learned simulator,
# the NRD below. No step of this script needs Chrono. As for contact_nrd/, this repository holds only the NRD side: the
# Chrono evaluations of the released policies are data on the Hub (so101_push_rl/results/; --with-results).
#
# NRD (every PPO run below steps this network)
#   Hub      harryzhang1018/NeDM (dataset), so101_push_rl/nrd/rot_sc8_s61/best.pt, at the revision pinned in
#            so101_push_rl/release_manifest.json
#   sha256   425d836a19ed4113a2f18ce1536fa18103999119ada75dd832bd421611f784d7 (10,248,031 bytes)
#   Design   the contact NRD design of contact_nrd/ (case so101_push_t): core, collision and contact Transformer
#            networks, 2,420,190 parameters, history 4, 20 ms step, seed 61; the same state, channel masks, step rule
#            and file format. nedm.contact_nrd.model.load_model loads it with no change. Only best.pt is read;
#            run_config.json and contact_scale.json next to it are provenance.
#   Differs  It is NOT the released contact_nrd/so101_push_t/models/seed61.pt:
#            data      144,112 train / 7,828 validation episodes (seed61: 51,200 / 6,400 scripted pushes); see Data.
#            training  step 1 = configs/contact_nrd/train.json with "check_episodes": 2500 (not 1500) on the full_v1
#                      and rot_v1 episodes (rot_scratch_s61, best at 80,000 of 120,000 updates). Step 2 = 7 fine-tune
#                      rounds rot_sc2_s61 ... rot_sc8_s61, each from the round before: 20,000 core and contact
#                      updates, lr 1e-4, 500 warmup, own-history examples from the first update, rollout horizon
#                      0.5 -> 1 s; the collision network, the input and output scales and the channel masks stay at
#                      their step-1 values. Before each round, Chrono episodes of the newest PPO policies went into
#                      the data. nedm.contact_nrd.train can run step 1, not step 2 (it cannot start from a checkpoint);
#                      the experiment code that ran both steps is not in this repository.
#            values    the trained weights, input scales, loss scales and history noise were fitted on this data, so
#                      they differ from seed61. Config, data sha256 and checkpoint sha256 of every round:
#                      provenance.nrd.fine_tune_chain in the Hub copy of so101_push_rl/release_manifest.json.
#   Data     so101_push_rl/nrd_data/train/ = the data of the last round (15.8 GB; unified_data.npz sha256
#            8aeb70cd9af52200b1585689dc14b557ff857e42ff2d9e6b028ee9b9501e2dc2), train / validation episodes:
#              full_v1  51,200 / 1,000   scripted pushes, the same as contact_nrd/so101_push_t/train (validation capped)
#              rot_v1   39,320 / 2,500   scripted walks and turns of the T (4 s)
#              r1-r8    53,592 / 4,328   Chrono episodes of 10 checkpoints of 9 earlier PPO runs (up to 8.56 s)
#            Step 1 used the full_v1_* and rot_v1_* episodes (index.json). This script does not need the data
#            (--with-nrd-data downloads it).
#
# PPO (rsl_rl 2.2.4 PPO, 16,384 NRD environments, one decision = 0.1 s, seed 1)
#   B1   new policy; curriculum configs/so101_push_rl/curriculum_fresh.json (13 stages); 400 updates
#        -> B1_rt_fresh/stage12_end.pt (released: so101_push_rl/policy/b1f_fresh_stage12/policy.pt, sha256 57fffa6f...)
#   C1   --init-from B1 stage12_end.pt (weights only); curriculum configs/so101_push_rl/curriculum_cont.json (2 stages);
#        250 updates -> C1_cont_sc8/stage1_end.pt = the selected policy C1f
#        (released: so101_push_rl/policy/c1f_selected/policy.pt, sha256 ec2eff8a...)
#   Both use configs/so101_push_rl/env.json: NRD rot_sc8_s61, train task table, train starts, rl_train bank, obs stats.
#   Arguments = those of the original runs (one MI350X each; B1 2 h 32 min, C1 1 h 34 min). The trainer stops after
#   3.6 h (--max-hours); this script then runs it again with --resume, as the original resume jobs did.
#   The original runs also started Chrono checks in the background (--chrono-every 50 --chrono-n 48). The checks never
#   fed back into training (they only read saved checkpoints), and this trainer has no such check.
#
# Steps
#   download  the NRD, the RL inputs and the released policies (47 files, 191 MB) at the pinned revision; size and
#             sha256 check of every file; files already in place are not downloaded again
#   b1        PPO run B1
#   c1        PPO run C1
#   summary   the NRD validation of each run at each stage end (the trainer validates every 25 updates on the
#             validation task table), next to the same numbers of the released run (policy/<name>/done.json)
#
# Files this script uses (paths from the repository root)
#   so101_push_rl/release_manifest.json                     Hub revision, bytes and sha256 of every released file
#   configs/so101_push_rl/env.json                          environment config; {data} / {code} are filled in below
#   configs/so101_push_rl/curriculum_fresh.json             curriculum of B1 (13 stages)
#   configs/so101_push_rl/curriculum_cont.json              curriculum of C1 (2 stages)
#   configs/so101_push_rl/collector_v2.json                 arm and T settings (robot files, PD gains, step, T size)
#   assets/so101/collision_points_dense_2mm.npz             dense surface points of the fingertips (distance checks)
#   scripts/so101_push_rl/train_so101_planar_ppo.py         PPO trainer: curriculum, NRD validation, checkpoints
#   src/nedm/__init__.py                                    package marker
#   src/nedm/contact_nrd/__init__.py                        package marker
#   src/nedm/contact_nrd/model.py                           the NRD networks and load_model (from contact_nrd/)
#   src/nedm/so101_push_rl/__init__.py                      package marker
#   src/nedm/so101_push_rl/so101_planar_env.py              NRD environment for PPO: tasks, decoder, reward, validity
#   src/nedm/so101_push_rl/so101_planar_common.py           action map, observation, validity rules, goal coverage
#   src/nedm/so101_push_rl/so101_planar_decoder.py          action -> 20 ms joint commands (planar finger target, IK)
#   src/nedm/so101_push_rl/so101_direct_common.py           finger-T distance geometry and observation pieces
#   src/nedm/so101_push_rl/so101_goal_common.py             goal error, potential, "T settled inside the goal" test
#   src/nedm/so101_push_rl/so101_push_common.py             history length, state layout, obs normalisation, action map
#   src/nedm/so101_push_rl/so101_fk.py                      batched forward kinematics, dense collision points (torch)
#   src/nedm/so101_push_rl/so101_kin_torch.py               batched arm kinematics, command law and IK (torch)
#   src/nedm/so101_push_rl/so101_robot.py                   SO101 arm model from assets/so101 (numpy), config loader
#   src/nedm/so101_push_rl/so101_geometry.py                T shape geometry
#   assets/so101/so101_robot.json                           SO101 links, joints, TCP and joint limits
#   assets/so101/so101_hulls.npz                            convex hulls of the links base .. wrist
#   assets/so101/so101_finger_sections.npz                  convex hulls of the upper finger parts
#   assets/so101/so101_finger_boxes.json                    fingertip collision boxes
# Hub files the runs read (in DIR/so101_push_rl/ of --data; the download step checks all files of its groups)
#   nrd/rot_sc8_s61/best.pt                                 the NRD
#   rl_inputs/tasks/rot_tasks_rl_train_rt.npz               train task table (start, goal, time, level)
#   rl_inputs/starts/starts_rl_train_all.npz                train start states
#   rl_inputs/banks/rl_train.npz                            recorded Chrono episodes of the train starts
#   rl_inputs/stats/stats.pt                                observation scales
#   rl_inputs/tasks/rot_tasks_rl_val_rt.npz                 validation task table (NRD validation in the trainer)
#   rl_inputs/starts/rot_starts_rl_val.npz                  validation start states
#   rl_inputs/banks/rot_bank_rl_val.npz                     recorded Chrono episodes of the validation starts
#   policy/b1f_fresh_stage12/policy.pt                      released B1 (start of C1 with --c1-init released)
#   policy/b1f_fresh_stage12/done.json                      released B1 stage ends (summary)
#   policy/c1f_selected/done.json                           released C1 stage ends (summary)
#
# Use
#   scripts/so101_push_rl/reproduce.sh [full|smoke] [options]
#     full   download, B1, C1, NRD validation summary   (default)
#     smoke  setup check: download, 2 PPO updates for B1 and C1 (256 environments; C1 starts from the released B1),
#            summary
#   Options
#     --steps LIST         comma list of download,b1,c1,summary (default: all steps of the mode). A list without
#                          download skips the download and its sha256 check (offline use: the files must already
#                          be in DIR/so101_push_rl/ of --data)
#     --out DIR            outputs (default artifacts/so101_push_rl_repro; smoke: a new smoke_<time> folder in it)
#     --data DIR           download folder; files land in DIR/so101_push_rl/... (default artifacts)
#     --device DEV         torch device of PPO (default cuda)
#     --num-envs N         PPO environments (default 16384; smoke 256)
#     --updates N          stop each PPO run after N updates (default 0 = whole curriculum; smoke 2)
#     --c1-init own|released   C1 starts from this script's B1 (default) or the released B1 policy (smoke)
#     --with-nrd-data      also download the NRD training data (15.8 GB)
#     --with-results       also download the released Chrono evaluations (so101_push_rl/results/, 16 files, 18.6 MB)
#   Environment
#     PY                   Python with torch, rsl-rl-lib 2.2.4, tensorboard, numpy, huggingface_hub (default python3)
# =====================================================================================================================
set -euo pipefail

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python3}"
MODE=full
STEPS=""
OUT=""
DATA_ROOT="$CODE/artifacts"
DEVICE=cuda
NUM_ENVS=""
UPDATES=""
C1_INIT=""
WITH_NRD_DATA=0
WITH_RESULTS=0

die() { echo "reproduce.sh: $*" >&2; exit 1; }
log() { echo "[reproduce $(date +%H:%M:%S)] $*"; }

while [ $# -gt 0 ]; do
    case "$1" in
        full|smoke) MODE="$1" ;;
        --steps) STEPS="$2"; shift ;;
        --out) OUT="$2"; shift ;;
        --data) DATA_ROOT="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        --num-envs) NUM_ENVS="$2"; shift ;;
        --updates) UPDATES="$2"; shift ;;
        --c1-init) C1_INIT="$2"; shift ;;
        --with-nrd-data) WITH_NRD_DATA=1 ;;
        --with-results) WITH_RESULTS=1 ;;
        -h|--help) sed -n '3,/^# =====/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument $1 (see --help)" ;;
    esac
    shift
done

case "$MODE" in
    full)  : "${STEPS:=download,b1,c1,summary}" "${NUM_ENVS:=16384}" "${UPDATES:=0}" "${C1_INIT:=own}" ;;
    smoke) : "${STEPS:=download,b1,c1,summary}" "${NUM_ENVS:=256}" "${UPDATES:=2}" "${C1_INIT:=released}" ;;
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
    case "$s" in download|b1|c1|summary) ;; *) die "unknown step $s (download, b1, c1, summary)" ;; esac
done
case "$C1_INIT" in own|released) ;; *) die "--c1-init must be own or released" ;; esac
log "mode $MODE, steps $STEPS, out $OUT, data $D"

# ---------------------------------------------------------------------------------------------------- checks
if has_step b1 || has_step c1; then
    "$PY" -c "import numpy, torch, rsl_rl" 2>/dev/null || die "$PY needs numpy, torch and rsl-rl-lib 2.2.4 (set PY)"
    if [ "$DEVICE" != cpu ]; then
        "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || die "no GPU for --device $DEVICE"
    fi
fi

# ---------------------------------------------------------------------------------------------------- 1. download
DL_GROUPS=nrd,rl_inputs,policy
[ "$WITH_NRD_DATA" = 1 ] && DL_GROUPS="$DL_GROUPS,nrd_data"
[ "$WITH_RESULTS" = 1 ] && DL_GROUPS="$DL_GROUPS,results"
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
# configs/so101_push_rl/env.json with {data} and {code} filled in
"$PY" - "$CODE" "$D" "$OUT" <<'EOF'
import json, sys
from pathlib import Path

code, data, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
PATH_KEYS = ("checkpoints", "starts_path", "banks_path", "tasks_path", "stats_path", "collector_config", "dense_points")
fill = lambda v: [fill(x) for x in v] if isinstance(v, list) else v.replace("{data}", str(data)).replace("{code}", str(code))
env = json.loads((code / "configs/so101_push_rl/env.json").read_text())
(out / "configs").mkdir(parents=True, exist_ok=True)
(out / "configs/env.json").write_text(json.dumps({**env, **{k: fill(env[k]) for k in PATH_KEYS}}, indent=1))
EOF

# ---------------------------------------------------------------------------------------------------- 2. / 3. PPO
VAL=(--val-tasks "$D/rl_inputs/tasks/rot_tasks_rl_val_rt.npz" --val-starts "$D/rl_inputs/starts/rot_starts_rl_val.npz"
     --val-banks "$D/rl_inputs/banks/rot_bank_rl_val.npz")

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
            --out "$run" --seed 1 --num-envs "$NUM_ENVS" --start-stage 0 --max-hours 3.6 "$@" "${extra[@]}" "${resume[@]}" \
            --set num_steps_per_env=48 algorithm.num_mini_batches=32 std_floor=0.2
        [ "$UPDATES" = 0 ] || return 0             # a capped run stops with paused.json
    done
    [ -f "$run/done.json" ] || die "$run did not finish its curriculum"
}

mkdir -p "$RUNS"
if has_step b1; then
    run_ppo B1_rt_fresh configs/so101_push_rl/curriculum_fresh.json
fi
if has_step c1; then
    if [ "$C1_INIT" = own ]; then
        INIT="$RUNS/B1_rt_fresh/stage12_end.pt"
    else
        INIT="$D/policy/b1f_fresh_stage12/policy.pt"
    fi
    [ -f "$INIT" ] || die "no B1 checkpoint $INIT (run step b1, or use --c1-init released)"
    run_ppo C1_cont_sc8 configs/so101_push_rl/curriculum_cont.json --init-from "$INIT"
fi

# ---------------------------------------------------------------------------------------------------- 4. summary
if has_step summary; then
    log "NRD validation at each stage end: this run vs the released run"
    "$PY" - "$RUNS" "$D" <<'EOF'
import json, sys
from pathlib import Path

runs, data = Path(sys.argv[1]), Path(sys.argv[2])
for run, released in (("B1_rt_fresh", "b1f_fresh_stage12"), ("C1_cont_sc8", "c1f_selected")):
    state = next((runs / run / f for f in ("done.json", "paused.json", "curriculum_state.json") if (runs / run / f).is_file()), None)
    ref_path = data / "policy" / released / "done.json"
    if state is None:
        print(f"{run}: no run in {runs / run}")
        continue
    own = json.loads(state.read_text())
    mine = {e["stage"]: e for e in own.get("stage_ends", [])}
    ref = {e["stage"]: e for e in json.loads(ref_path.read_text())["stage_ends"]} if ref_path.is_file() else {}
    print(f"\n{run} ({state.name}) vs released {released}: NRD validation at each stage end ('-': stage not reached)")
    print("| Stage | Update (this / released) | Success (this / released) | Invalid (this / released) |")
    print("|---|---|---|---|")
    get = lambda e, k: "-" if e is None else (str(e[k]) if k == "iteration" else f"{e['val'][k]:.3f}")
    for st in sorted(set(mine) | set(ref)):
        a, b = mine.get(st), ref.get(st)
        print(f"| {st} | {get(a, 'iteration')} / {get(b, 'iteration')} | {get(a, 'success')} / {get(b, 'success')} | "
              f"{get(a, 'invalid')} / {get(b, 'invalid')} |")
    last = own["history"][-1] if own.get("history") else None
    if last:
        print(f"last validation: update {last['iteration']}, stage {last['stage']}, success {last['success']:.3f}, "
              f"by level {last['by_level']}")
EOF
fi
log "done: $OUT"
