#!/usr/bin/env bash
# =====================================================================================================================
# scripts/g1_footstep_rl/reproduce.sh: train the G1 footstep policy inside the NRD and check it in MuJoCo
# (g1_footstep_rl/README.md).
#
# The frozen HOMIE walking policy drives a Unitree G1. The footstep policy sets HOMIE's velocity command (vx, vy, yaw
# rate) every 20 ms so that the feet land on given footprints. PPO trains it inside a learned simulator only (the NRD
# below); CPU MuJoCo checks it on new courses. Unlike so101_push_rl/, this repository holds the MuJoCo evaluation
# (scene, runner, evaluation script), so the MuJoCo tables can be run again; the data collector of the NRD is not here.
#
# NRD (every PPO run below steps these networks)
#   Hub      harryzhang1018/NeDM (dataset), contact_nrd/g1_locomotion/models/seed61.pt and seed62.pt, at the revision
#            of case g1_locomotion in contact_nrd/release_manifest.json ("case_revisions")
#   sha256   04ed134c2f22f416399de2797046dc133538327a0f4b44f068708400efad9ba0 (seed 61, 11,051,523 bytes)
#            9706d61b38ad857623c27171bad64c54881a7d5a7d2008a63cac8b868b2e3d11 (seed 62, 11,051,523 bytes)
#   Design   the contact NRD design of contact_nrd/ (case g1_locomotion, contact_nrd/g1_locomotion/README.md): core,
#            collision and contact networks, history 4, 20 ms step, one 67-channel robot, 27 joint targets, two pairs
#            (left foot - floor, right foot - floor); trained with configs/contact_nrd/train_g1_locomotion.json
#            (ridge 0.1). The two checkpoints are the two members: each training episode uses one member, the
#            validation uses both. nedm.contact_nrd.model.load_model loads them with no change.
#   Data     contact_nrd/g1_locomotion/train/ (19.3 GB; HOMIE walking the G1 in MuJoCo): 30,685 training and 3,413
#            validation episodes. The steps below do not read it; scripts/g1_footstep_rl/build_start_bank.py rebuilds
#            the start bank from it. --with-nrd-data downloads it and the NRD test data (2.0 GB).
#
# HOMIE (not in this repository and not on the Hub; OpenHomie has its own license)
#   scripts/g1_footstep_rl/fetch_homie.py fetches OpenHomie at commit 33071f79f1884bb122bd45631c8c60d61dac1eb8 into
#   artifacts/g1_footstep_rl/vendor/OpenHomie, checks 71 files by size and SHA-256 (configs/g1_footstep_rl/
#   homie_sources.json) and writes artifacts/g1_footstep_rl/homie/homie_weights.npz (sha256 9a4ade34..., the value in
#   the manifests of both released runs). It needs git and network access to github.com.
#
# PPO (rsl_rl 2.2.4 PPO, 16,384 NRD environments, 32 steps per update, one decision = 20 ms)
#   fs_cmd_s1  seed 1; curriculum configs/g1_footstep_rl/footsteps_curriculum.json (stages S0-S5); stage ends at updates
#              50, 150, 225, 275, 450, 2,375 (all promoted); released: g1_footstep_rl/policy/fs_cmd_s1/policy.pt
#              (= stage5_end.pt, sha256 8e0796d7...); the default policy of the evaluation
#   fs_cmd_s2  seed 2; the same; stage ends 50, 100, 200, 250, 525, 2,525 (S5 at its update limit); released:
#              g1_footstep_rl/policy/fs_cmd_s2/policy.pt (sha256 b7c9cf49...)
#   Both use configs/g1_footstep_rl/footsteps_env.json (command-only actions, action.alpha 0) with the data paths of
#   $OUT/configs/data_paths.json, --val-members 0 1 --val-n-per-level 128 --val-every 25. Arguments = those of the
#   original runs (fs_cmd_s1: two consecutive jobs on two MI300X nodes, 4.2 h; fs_cmd_s2: 3.1 h on one MI350X). The
#   trainer stops after 3.6 h (--max-hours); this script then runs it again with --resume, as the original jobs did.
#   A resumed run is not bitwise the same as an uninterrupted one (the environment generator starts again).
#
# MuJoCo evaluation (CPU; nothing of it feeds back into training)
#   Cells F1 F2 MP MN ES, 64 courses each, development seeds 1000-1063 (the released results/dev/). The footstep
#   policies (released, or this script's with --policy own), with the same episodes also inside the NRD from the
#   MuJoCo hand-over states (--nrd, both members), and the H-FS baseline (HOMIE + a footstep-to-speed controller;
#   parameters {"g_v": 0.8, "lookahead": 2, "k_y": 2.0, "v_min": 0.2} from scripts/g1_footstep_rl/footstep_baselines.py
#   tune on the tuning seeds 0-15). MuJoCo rows are bitwise equal to the released rows only on the same CPU type and
#   package versions (the released ones: AMD Ryzen 7 5800X, numpy 2.4.6, torch 2.12.0, mujoco 3.15.0, onnxruntime
#   1.30.0, results/dev/environment.json): elsewhere compare the rates. environment.json records those of this run.
#
# Steps
#   download  the NRD checkpoints, the RL inputs, the released policies and their development results
#             (g1_footstep_rl/policy/, rl_inputs/, results/dev/) at the pinned revisions; size and sha256 check of
#             every file; files already in place are not downloaded again.
#             Without a pinned revision ("PENDING") the step only checks the files that are already in artifacts/
#   homie     scripts/g1_footstep_rl/fetch_homie.py
#   ppo_s1    PPO run fs_cmd_s1
#   ppo_s2    PPO run fs_cmd_s2
#   eval      the MuJoCo evaluation above (H-FS first; then the footstep policies with --nrd and the paired tests
#             against H-FS)
#   summary   the NRD validation at each stage end next to the released run (policy/<name>/done.json); the MuJoCo cell
#             rates next to the released ones (results/dev/<name>/summary.json, from the download step); smoke: the
#             MuJoCo and NRD rows next to the released dev rows of the same courses
#
# Files this script uses (paths from the repository root)
#   g1_footstep_rl/release_manifest.json                  Hub revision, bytes and sha256 of every released footstep file
#   contact_nrd/release_manifest.json                     the same for the NRD case (its revision: "case_revisions")
#   configs/g1_footstep_rl/footsteps_env.json             env config of both runs (paths: repository-relative defaults)
#   configs/g1_footstep_rl/footsteps_curriculum.json      curriculum S0-S5 (= the released curriculum.json)
#   configs/g1_footstep_rl/g1_scene.json                  MuJoCo scene: physics, contact rule, vendor files and hashes
#   configs/g1_footstep_rl/homie_sources.json             OpenHomie URL, commit and the 71 pinned files
#   scripts/g1_footstep_rl/train_footsteps_ppo.py         PPO trainer: curriculum, NRD validation, checkpoints
#   scripts/g1_footstep_rl/eval_footsteps_mujoco.py       MuJoCo evaluation, the NRD part, reports
#   scripts/g1_footstep_rl/fetch_homie.py                 HOMIE fetch and checks
#   src/nedm/__init__.py, src/nedm/contact_nrd/__init__.py   package markers
#   src/nedm/contact_nrd/model.py, data.py                the NRD networks and load_model; step-average rates
#   src/nedm/contact_nrd/download.py                      download of the NRD case (download step)
#   src/nedm/g1_footstep_rl/*.py, *.json                  environment, task, scorer, NRD simulator, HOMIE copy, leg
#                                                         kinematics, MuJoCo scene and runner (module table in
#                                                         src/nedm/g1_footstep_rl/__init__.py)
# Hub files the steps read (in artifacts/; the download step checks all files of its folders)
#   contact_nrd/g1_locomotion/models/seed61.pt, seed62.pt                  the NRD members
#   g1_footstep_rl/rl_inputs/start_bank/bank_v1.npz, bank_v1.json          200,000 start states (training split)
#   g1_footstep_rl/rl_inputs/courses/course_bank_v4_train.npz, _val.npz    training and validation courses
#   g1_footstep_rl/policy/fs_cmd_s{1,2}/policy.pt + run files              released policies (eval, summary)
#   g1_footstep_rl/results/dev/*/summary.json, rows.json, nrd_rows.json    released MuJoCo results (summary)
#
# Use
#   scripts/g1_footstep_rl/reproduce.sh [full|smoke] [options]
#     full   download, homie, ppo_s1, ppo_s2, eval, summary   (default; PPO about 4 + 3 GPU hours on one MI300X / MI350X)
#     smoke  setup check: download, homie, 2 PPO updates per seed (256 environments), MuJoCo of the released policy
#            fs_cmd_s1 and of H-FS on dev seeds 1000-1001 of each cell (with the NRD part), summary
#   Options
#     --steps LIST         comma list of download,homie,ppo_s1,ppo_s2,eval,summary (default: all steps of the mode). A
#                          list without download skips the download and its sha256 check (offline use: the files
#                          must already be in artifacts/)
#     --out DIR            outputs (default artifacts/g1_footstep_rl_repro; smoke: a new smoke_<time> folder in it)
#     --device DEV         torch device of PPO (default cuda)
#     --num-envs N         PPO environments (default 16384; smoke 256)
#     --updates N          stop each PPO run after N updates (default 0 = whole curriculum; smoke 2)
#     --policy released|own   policies of the eval step: the released ones (default) or this script's PPO runs
#                          ($OUT/runs/fs_cmd_s{1,2}/stage5_end.pt; full mode only)
#     --eval-workers N     MuJoCo worker processes (default 8; the released footstep results used 12, H-FS 8; the
#                          count does not change the rows)
#     --with-nrd-data      also download the NRD training and test data (21.3 GB)
#     --with-results       also download the H-FS parameter search (g1_footstep_rl/results/hfs_tuning/)
#   Environment
#     PY                   Python with torch, rsl-rl-lib 2.2.4, tensorboard, numpy, mujoco 3.15.0, onnx,
#                          onnxruntime 1.30.0 and huggingface_hub (default python3); git for the homie step
#   All data go to artifacts/ of this checkout (the code reads them there); link artifacts/ to another disk if needed.
# =====================================================================================================================
set -euo pipefail

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python3}"
MODE=full
STEPS=""
OUT=""
DEVICE=cuda
NUM_ENVS=""
UPDATES=""
POLICY=""
EVAL_WORKERS=8
WITH_NRD_DATA=0
WITH_RESULTS=0
HFS='{"g_v": 0.8, "lookahead": 2, "k_y": 2.0, "v_min": 0.2}'
CELLS=(F1 F2 MP MN ES)

die() { echo "reproduce.sh: $*" >&2; exit 1; }
log() { echo "[reproduce $(date +%H:%M:%S)] $*"; }

while [ $# -gt 0 ]; do
    case "$1" in
        full|smoke) MODE="$1" ;;
        --steps) STEPS="$2"; shift ;;
        --out) OUT="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        --num-envs) NUM_ENVS="$2"; shift ;;
        --updates) UPDATES="$2"; shift ;;
        --policy) POLICY="$2"; shift ;;
        --eval-workers) EVAL_WORKERS="$2"; shift ;;
        --with-nrd-data) WITH_NRD_DATA=1 ;;
        --with-results) WITH_RESULTS=1 ;;
        -h|--help) sed -n '3,/^# =====/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument $1 (see --help)" ;;
    esac
    shift
done

case "$MODE" in
    full)  : "${STEPS:=download,homie,ppo_s1,ppo_s2,eval,summary}" "${NUM_ENVS:=16384}" "${UPDATES:=0}" "${POLICY:=released}" ;;
    smoke) : "${STEPS:=download,homie,ppo_s1,ppo_s2,eval,summary}" "${NUM_ENVS:=256}" "${UPDATES:=2}" "${POLICY:=released}" ;;
esac
case "$POLICY" in own|released) ;; *) die "--policy must be own or released" ;; esac
[ "$MODE" = smoke ] && [ "$POLICY" = own ] && die "smoke mode evaluates the released policy (2 PPO updates make no walker)"
OUT="${OUT:-$CODE/artifacts/g1_footstep_rl_repro}"
[ "$MODE" = smoke ] && OUT="$OUT/smoke_$(date +%m%d_%H%M%S)"
A="$CODE/artifacts"                                  # the one data root: the code reads every input here
mkdir -p "$OUT" "$A"
OUT="$(cd "$OUT" && pwd)"
G="$A/g1_footstep_rl"                                # the Hub folder g1_footstep_rl/ after the download
RUNS="$OUT/runs"
export PYTHONPATH="$CODE/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$CODE"

has_step() { case ",$STEPS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
for s in ${STEPS//,/ }; do
    case "$s" in download|homie|ppo_s1|ppo_s2|eval|summary) ;;
        *) die "unknown step $s (download, homie, ppo_s1, ppo_s2, eval, summary)" ;; esac
done
log "mode $MODE, steps $STEPS, out $OUT, data $A"

# ---------------------------------------------------------------------------------------------------- checks
if has_step homie; then
    command -v git > /dev/null || die "the homie step needs git"
    "$PY" -c "import numpy, torch, onnx" 2>/dev/null || die "$PY needs numpy, torch and onnx for the homie step (set PY)"
fi
if has_step ppo_s1 || has_step ppo_s2; then
    "$PY" -c "import numpy, torch, rsl_rl" 2>/dev/null || die "$PY needs numpy, torch and rsl-rl-lib 2.2.4 (set PY)"
    if [ "$DEVICE" != cpu ]; then
        "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || die "no GPU for --device $DEVICE"
    fi
fi
if has_step eval; then
    "$PY" -c "import numpy, torch, rsl_rl, mujoco, onnxruntime" 2>/dev/null \
        || die "$PY needs torch, rsl-rl-lib 2.2.4, mujoco 3.15.0 and onnxruntime 1.30.0 for the eval step (set PY)"
fi

# ---------------------------------------------------------------------------------------------------- 1. download
NRD_PARTS=models
[ "$WITH_NRD_DATA" = 1 ] && NRD_PARTS="models,train,test"
DL_GROUPS=policy,rl_inputs,results/dev                # results/dev: the released rows and rates of the summary step
[ "$WITH_RESULTS" = 1 ] && DL_GROUPS="policy,rl_inputs,results"
if has_step download; then
    log "download + sha256 check (contact_nrd/g1_locomotion: $NRD_PARTS; g1_footstep_rl: $DL_GROUPS)"
    "$PY" - "$CODE" "$A" "$NRD_PARTS" "$DL_GROUPS" <<'EOF'
import hashlib, json, re, subprocess, sys
from pathlib import Path

code, dest, nrd_parts, groups = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split(","), sys.argv[4].split(",")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hub_or_exit(n_missing):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(f"{n_missing} file(s) must be downloaded, and the download needs huggingface_hub in the Python of PY "
                 f"(pip install huggingface_hub). Offline: put the files in {dest} and run without the download step.")
    return snapshot_download


def fetch(manifest_path, folders, revision, download):
    """Check the files of the manifest under the given folders; download the missing ones (pinned revision only)."""
    manifest = json.loads(manifest_path.read_text())
    files = {n: m for n, m in manifest["files"].items() if n.startswith(folders)}
    for f in folders:
        if not any(n.startswith(f) for n in files):
            sys.exit(f"{manifest_path}: no file under {f}")
    for n in files:
        if n.startswith("/") or "\\" in n or ".." in n.split("/"):
            sys.exit(f"unsafe path in the manifest: {n}")
    pinned = re.fullmatch(r"[0-9a-f]{40}", revision) is not None
    good = lambda n: ((dest / n).is_file() and (dest / n).stat().st_size == files[n]["bytes"]
                      and sha256(dest / n) == files[n]["sha256"])
    ok = {n: good(n) for n in sorted(files)}      # files already in place are not downloaded again (no Hub call)
    missing = [n for n in ok if not ok[n]]
    if missing and pinned:
        download(manifest, missing, revision)
        ok.update({n: good(n) for n in missing})
    elif not pinned:
        print(f"{manifest_path.relative_to(code)}: revision {revision!r} is not a pinned Hub commit: no download, only "
              f"the check of the files in {dest}")
    for n in ok:
        print(f"{'ok ' if ok[n] else 'BAD'} {dest / n}", flush=True)
    bad = [n for n in ok if not ok[n]]
    if bad:
        sys.exit(f"{len(bad)} file(s) missing or with another size or sha256" +
                 ("; no pinned Hub revision to download from" if not pinned else ": delete them and run again"))


def nrd_download(manifest, missing, revision):            # the contact NRD case: its own downloader
    hub_or_exit(len(missing))
    print(f"download contact_nrd/g1_locomotion ({', '.join(nrd_parts)}) at revision {revision}", flush=True)
    subprocess.run([sys.executable, "-m", "nedm.contact_nrd.download", "--case", "g1_locomotion", "--part",
                    *nrd_parts, "--dest", str(dest)], check=True)


def footstep_download(manifest, missing, revision):
    snapshot_download = hub_or_exit(len(missing))
    print(f"download {len(missing)} file(s) from {manifest['repo_id']} at revision {revision}", flush=True)
    snapshot_download(repo_id=manifest["repo_id"], repo_type="dataset", revision=revision, allow_patterns=missing,
                      local_dir=str(dest))


cn = code / "contact_nrd/release_manifest.json"
m = json.loads(cn.read_text())
fetch(cn, tuple(f"{m['prefix']}g1_locomotion/{p}/" for p in nrd_parts),
      m.get("case_revisions", {}).get("g1_locomotion", m.get("hf_revision", "")), nrd_download)
fs = code / "g1_footstep_rl/release_manifest.json"
m = json.loads(fs.read_text())
fetch(fs, tuple(f"{m['prefix']}{g}/" for g in groups), m.get("hf_revision", ""), footstep_download)
EOF
fi

# ---------------------------------------------------------------------------------------------------- 2. HOMIE
if has_step homie; then
    log "HOMIE: fetch OpenHomie, check the pinned files, write homie_weights.npz"
    "$PY" -B scripts/g1_footstep_rl/fetch_homie.py
fi

# ---------------------------------------------------------------------------------------------------- configs
# absolute data paths for the PPO runs (the defaults in footsteps_env.json are repository-relative)
mkdir -p "$OUT/configs"
"$PY" - "$A" "$OUT/configs/data_paths.json" <<'EOF'
import json, sys
from pathlib import Path

a, out = Path(sys.argv[1]), Path(sys.argv[2])
paths = dict(checkpoints=[str(a / f"contact_nrd/g1_locomotion/models/seed{s}.pt") for s in (61, 62)],
             homie_weights=str(a / "g1_footstep_rl/homie/homie_weights.npz"),
             bank_path=str(a / "g1_footstep_rl/rl_inputs/start_bank/bank_v1.npz"),
             course_bank=str(a / "g1_footstep_rl/rl_inputs/courses/course_bank_v4_train.npz"),
             val_course_bank=str(a / "g1_footstep_rl/rl_inputs/courses/course_bank_v4_val.npz"))
out.write_text(json.dumps(dict(note="data paths of reproduce.sh (absolute)", **paths), indent=1) + "\n")
EOF

# ---------------------------------------------------------------------------------------------------- 3. / 4. PPO
run_ppo() {          # run_ppo <seed>
    local run="$RUNS/fs_cmd_s$1"
    local extra=()
    [ "$UPDATES" = 0 ] || extra+=(--max-iterations "$UPDATES")
    for attempt in 1 2 3 4 5 6 7 8; do
        [ -f "$run/done.json" ] && return 0
        if [ -f "$run/curriculum_state.json" ]; then
            log "PPO fs_cmd_s$1 (job $attempt, resume)"
            "$PY" -B scripts/g1_footstep_rl/train_footsteps_ppo.py train --resume --out "$run" --device "$DEVICE" \
                --val-members 0 1 --val-n-per-level 128 --val-every 25 --max-hours 3.6 "${extra[@]}"
        else
            log "PPO fs_cmd_s$1 (job $attempt)"
            "$PY" -B scripts/g1_footstep_rl/train_footsteps_ppo.py train --device "$DEVICE" --val-members 0 1 \
                --val-n-per-level 128 --val-every 25 --max-hours 3.6 --num-envs "$NUM_ENVS" --seed "$1" \
                --env-cfg configs/g1_footstep_rl/footsteps_env.json \
                --curriculum configs/g1_footstep_rl/footsteps_curriculum.json \
                --env-overrides "$OUT/configs/data_paths.json" --out "$run" --set env.action.alpha=0 "${extra[@]}"
        fi
        [ "$UPDATES" = 0 ] || return 0             # a capped run stops with paused.json
    done
    [ -f "$run/done.json" ] || die "$run did not finish its curriculum"
}

mkdir -p "$RUNS"
if has_step ppo_s1; then run_ppo 1; fi
if has_step ppo_s2; then run_ppo 2; fi

# ---------------------------------------------------------------------------------------------------- 5. MuJoCo
if [ "$MODE" = smoke ]; then
    EVAL_DIR="$OUT/eval/smoke"; SEED_ARG=1000-1001; SUFFIX=smoke; EVAL_POLICIES=(1)
else
    EVAL_DIR="$OUT/eval/dev"; SEED_ARG=dev; SUFFIX=dev; EVAL_POLICIES=(1 2)
fi
if has_step eval; then
    mkdir -p "$EVAL_DIR"
    E=scripts/g1_footstep_rl/eval_footsteps_mujoco.py
    log "MuJoCo: H-FS on ${CELLS[*]}, seeds $SEED_ARG"
    rm -rf "$EVAL_DIR/HFS_$SUFFIX"
    "$PY" -B "$E" eval --hfs "$HFS" --levels "${CELLS[@]}" --seeds "$SEED_ARG" --workers "$EVAL_WORKERS" \
        --name "HFS_$SUFFIX" --out "$EVAL_DIR/HFS_$SUFFIX"
    for s in "${EVAL_POLICIES[@]}"; do
        if [ "$POLICY" = released ]; then CKPT="$G/policy/fs_cmd_s$s/policy.pt"; else CKPT="$RUNS/fs_cmd_s$s/stage5_end.pt"; fi
        [ -f "$CKPT" ] || die "no policy $CKPT (run ppo_s$s, or use --policy released)"
        log "MuJoCo + NRD: $CKPT on ${CELLS[*]}, seeds $SEED_ARG"
        rm -rf "$EVAL_DIR/FS_cmd_s${s}_$SUFFIX"
        "$PY" -B "$E" eval --ckpt "$CKPT" --levels "${CELLS[@]}" --seeds "$SEED_ARG" --workers "$EVAL_WORKERS" \
            --nrd --nrd-members 0 1 --baseline "HFS=$EVAL_DIR/HFS_$SUFFIX" --name "FS_cmd_s${s}_$SUFFIX" \
            --out "$EVAL_DIR/FS_cmd_s${s}_$SUFFIX"
    done
    # versions, CPU and code commit of this evaluation (the rows are bitwise reproducible only with the same ones)
    "$PY" - "$CODE" "$EVAL_DIR/environment.json" <<'EOF'
import json, platform, socket, subprocess, sys
from importlib import metadata
from pathlib import Path

code, out = Path(sys.argv[1]), Path(sys.argv[2])
ver = {}
for p in ("torch", "numpy", "mujoco", "onnxruntime", "onnx", "rsl-rl-lib"):
    try:
        ver[p] = metadata.version(p)
    except metadata.PackageNotFoundError:
        ver[p] = None
cpu = platform.processor()
if Path("/proc/cpuinfo").exists():
    cpu = next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                if l.startswith("model name")), cpu)
git = lambda *a: subprocess.run(["git", "-C", str(code), *a], capture_output=True, text=True).stdout.strip() or None
out.write_text(json.dumps(dict(python=platform.python_version(), packages=ver, cpu=cpu, host=socket.gethostname(),
                               platform=platform.platform(), git_commit=git("rev-parse", "HEAD"),
                               git_dirty=bool(git("status", "--porcelain", "--untracked-files=no"))), indent=1) + "\n")
print(f"wrote {out}")
EOF
fi

# ---------------------------------------------------------------------------------------------------- 6. summary
if has_step summary; then
    log "summary: NRD validation at the stage ends and MuJoCo cell rates, this run vs the released files"
    "$PY" - "$RUNS" "$G" "$EVAL_DIR" "$SUFFIX" <<'EOF'
import json, sys
from pathlib import Path

runs, g, ev, suffix = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
load = lambda p: json.loads(p.read_text()) if p.is_file() else None
for seed in (1, 2):
    run, released = runs / f"fs_cmd_s{seed}", g / f"policy/fs_cmd_s{seed}/done.json"
    state = next((load(run / f) for f in ("done.json", "paused.json", "curriculum_state.json") if (run / f).is_file()), None)
    if state is None:
        continue
    mine = {e["stage"]: e for e in state.get("stage_ends", [])}
    ref = {e["stage"]: e for e in (load(released) or {}).get("stage_ends", [])}
    print(f"\nfs_cmd_s{seed}: stage ends, NRD validation of the promotion rules ('-': stage not reached)")
    print("| Stage | Update (this / released) | End | Rules: value this / released |")
    print("|---|---|---|---|")
    for st in sorted(set(mine) | set(ref)):
        a, b = mine.get(st), ref.get(st)
        rules = (b or a)["rules"]
        val = lambda e, i: "-" if e is None else f"{e['rules'][i]['value']:.3f}"
        txt = "; ".join(f"{r['level']} {r['metric']} {val(a, i)} / {val(b, i)}" for i, r in enumerate(rules))
        print(f"| {st} | {a['iteration'] if a else '-'} / {b['iteration'] if b else '-'} | "
              f"{a['reason'] if a else '-'} / {b['reason'] if b else '-'} | {txt} |")
    if state.get("history"):
        h = state["history"][-1]
        print(f"last validation: update {h['iteration']}, stage {h['stage']}, completion {h['completion']:.3f}")
rate = lambda c, k: ("-" if not c or (c.get(k) or {}).get("rate") is None   # e.g. no scored step in the cell
                     else f"{100 * c[k]['rate']:.1f}")
for d in sorted(p for p in ev.glob("*") if (p / "summary.json").is_file()):
    sm = load(d / "summary.json")
    rd = g / "results/dev" / d.name.replace("_smoke", "_dev")
    ref = load(rd / "summary.json")
    print(f"\n{d.name}: MuJoCo cells, this run / released {rd.relative_to(g.parent) if ref else '(not downloaded)'}")
    print("| Cell | n | Completion % | Clean completion % | Success@5 % | Hit@5 % (step) | Falls |")
    print("|---|---|---|---|---|---|---|")
    for lv, c in sm["cells"].items():
        r = (ref or {}).get("cells", {}).get(lv)
        two = lambda k: f"{rate(c, k)} / {rate(r, k)}"
        falls = f"{c['fell']['k']} / {r['fell']['k'] if r else '-'}"
        print(f"| {lv} | {c['n']} / {r['n'] if r else '-'} | {two('completed')} | {two('clean_completion')} | "
              f"{two('success5')} | {two('hit5')} | {falls} |")
    if suffix == "smoke":
        # per-episode rows of the same courses (path-derived names and the run's episode numbers aside)
        skip = {"host", "controller", "policy", "episode_id"}
        for name, key in (("rows.json", ("level", "seed")), ("nrd_rows.json", ("level", "seed", "member"))):
            mine, theirs = load(d / name), load(rd / name)
            if mine is None or theirs is None:
                continue
            theirs = {tuple(r[k] for k in key): r for r in theirs}
            strip = lambda r: {k: v for k, v in r.items() if k not in skip}
            same = [strip(r) == strip(theirs[tuple(r[k] for k in key)]) for r in mine if tuple(r[k] for k in key) in theirs]
            print(f"{name}: {sum(same)} of {len(same)} equal to the released dev rows of the same courses "
                  "(bitwise only on the same CPU type and package versions; elsewhere compare the rates)")
EOF
fi
log "done: $OUT"
