#!/usr/bin/env bash
# =====================================================================================================================
# scripts/so101_pick_place/reproduce.sh: the SO101 pick-and-place chain NRD -> state teacher (PPO in the NRD) ->
# camera student (online DAgger in the NRD with ray-traced images) -> sim checks -> robot bundle
# (so101_pick_place/README.md).
#
# A real SO101 arm picks a 30 mm cube and places it on a plate. Every training step of every policy ran inside the
# NRD below; no real image was a training sample. This script repeats the steps from the released files; the robot
# test itself is a manual step on the robot computer (README "Deploy").
#
# NRD (every teacher and student step below runs in it)
#   Hub      harryzhang1018/NeDM (dataset), so101_pick_place/nrd/c1_s62/best.pt at the revision pinned in
#            so101_pick_place/release_manifest.json; sha256 2d1a24926d463c5432a8dcb6897c05fe1082f04095c773ec914dd53986d66abc
#            (10,893,663 bytes). Main's nedm.contact_nrd.model.load_model loads it with no code change: it is the
#            study's file (original/best.pt, sha256 f45ee8be...) with the two training-only geometry heads removed;
#            its rollouts are bit-identical to the original's in the training code (README "The NRD").
#   Base     so101_pick_place/nrd/c0_s62/ (the start weights of c1_s62; the same layout).
#   Data     so101_pick_place/nrd_data/{production_plus,noisy_v1,expert_v2,policy_v3}/ (10.6 GB, --with-nrd-data);
#            sealed test so101_pick_place/nrd_test/ (3,050 scripted episodes, 0.34 GB).
#
# Steps (each step reads the released parents unless --parents own and this script's own run exists in OUT/runs)
#   setup        contact_nrd/so101_grasp/fetch_sources.py: the MuJoCo Menagerie SO101 model at the pinned revision
#                (vendor/, 27 MB; needs git and github.com); exact
#   download     scripts/so101_pick_place/download.py: the Hub files (size + sha256 of every file); --hf-source DIR
#                (or SO101_PP_HF_SOURCE) takes them from a local folder with the Hub layout instead of the Hub
#   geometry     kin.save_points and TaskGeometry.from_mujoco rebuild rl_inputs/geometry/ (exact)
#   nrd          main's load_model on both NRDs; a 100-step rollout of 4 test episodes; the sha256 of the files
#   nrd_test     main's python -m nedm.contact_nrd.test on nrd_test/ (smoke: 2 episodes per family), next to the
#                released results/nrd/sealed_test.json (the README numbers)
#   nrd_collect  the collector (nedm.contact_nrd.collect.so101_grasp) regenerates the first episode of every family
#                of nrd_test/ and compares the arrays (exact); chain: the 4 scripted campaigns in full
#   nrd_train    chain only: base NRD and fine-tune with the training code of commit f6f05c21c (branch
#                so101-pick-place-study; a git worktree under OUT; main's train.py lacks its options)
#   t0 .. t6     the teacher stages (README "Training chain"): t0 decoder PPO from scratch (train_ppo.py), t1 expert
#                data, t2 behaviour cloning, t3 / t4 joint-target PPO, t5 / t6 the robot-round fine-tunes
#   s0 .. s5     the student stages: s0 offline rounds from ImageNet ResNet18, s1 .. s5 online DAgger
#   simcheck     scripts/so101_pick_place/check/run_gates.sh on the released (or own) tb10 and on D; gates_summary
#   bundle       scripts/so101_policy/real2sim/hw/make_orin_bundle.sh -> OUT/bundle/so101_hw_v13_tb10, compared with the
#                released bundle (MANIFEST.txt)
#   fake         the bundle on this machine with a fake robot (no device opened): robot_test.sh check, eval 1 6 and
#                replay of 3 recordings; the replay is repeated with the released bundle and the action chunks are
#                compared
#   robot        (--with-robot) the robot result tables from the released logs (build_results.py), next to the
#                released results.json
#   summary      the numbers of this run next to the released ones
#
# Modes
#   smoke  every step of full with tiny sizes (about 3 min on one RTX 5090 once the files are local): a setup check
#   full   setup, download, geometry, nrd, nrd_test, nrd_collect, t6, s5, simcheck, bundle, fake, summary (default):
#          the last teacher and student stages from the released parents (T5, S4), the sim checks, the bundle.
#          About 2 GPU-h (t6 0.8, s5 0.5 + its in-run evaluations, simcheck 21 evaluations) + CPU
#   chain  every stage from scratch: nrd_collect (full), nrd_train, t0 .. t6, s0 .. s5, simcheck, bundle, fake.
#          About 22 GPU-h on one MI350X / MI300X per job (README table); a cluster is advised. Needs
#          --with-nrd-data (and the expert data come from t1; --with-expert-data downloads the released ones)
#
# Options
#   --steps LIST        comma list of steps (default: those of the mode). Without download no file is fetched or checked
#   --out DIR           outputs (default artifacts/so101_pick_place_repro; smoke: a new smoke_<time> folder in it)
#   --hf-source DIR     take the Hub files from DIR (a folder with the layout of the Hub folder so101_pick_place/)
#   --parents own|released   start each stage from this script's parent run (own) or from the released parent
#                       (default released; chain: own)
#   --device DEV        torch device (default cuda)
#   --workers N         CPU MuJoCo worker processes of the evaluations (default 8)
#   --with-nrd-data     also download the 4 NRD data parts (10.6 GB; nrd_collect / nrd_train / t0 / t1 need them)
#   --with-expert-data  also download the released expert data of t1 (0.83 GB; t2 with --parents released)
#   --with-robot        also download the robot logs and videos (1.15 GB; robot step)
# Environment
#   PY                  Python with torch, torchvision, numpy, scipy, mujoco 3.15.0, warp-lang 1.13.0, newton 1.2.0,
#                       huggingface_hub (default python3); requirements-so101-pick-place.txt. The renderer needs an
#                       NVIDIA GPU with CUDA (or the ROCm build of Warp on AMD GPUs)
#   SO101_PP_HF_SOURCE  default of --hf-source
# Files: all data go to artifacts/so101_pick_place of this checkout (nedm.so101_policy.paths: the code reads its
# defaults there; link artifacts/ to another disk if needed). Run records: OUT/runs/<stage>/.
# =====================================================================================================================
set -euo pipefail

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python3}"
MODE=full
STEPS=""
OUT=""
SOURCE="${SO101_PP_HF_SOURCE:-}"
PARENTS=""
DEVICE=cuda
WORKERS=8
WITH_NRD_DATA=0
WITH_EXPERT=0
WITH_ROBOT=0

die() { echo "reproduce.sh: $*" >&2; exit 1; }
log() { echo "[reproduce $(date +%H:%M:%S)] $*"; }

while [ $# -gt 0 ]; do
    case "$1" in
        full|smoke|chain) MODE="$1" ;;
        --steps) STEPS="$2"; shift ;;
        --out) OUT="$2"; shift ;;
        --hf-source) SOURCE="$2"; shift ;;
        --parents) PARENTS="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        --workers) WORKERS="$2"; shift ;;
        --with-nrd-data) WITH_NRD_DATA=1 ;;
        --with-expert-data) WITH_EXPERT=1 ;;
        --with-robot) WITH_ROBOT=1 ;;
        -h|--help) sed -n '3,/^# =====/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument $1 (see --help)" ;;
    esac
    shift
done

ALL_STEPS="setup download geometry nrd nrd_test nrd_collect nrd_train t0 t1 t2 t3 t4 t5 t6 s0 s1 s2 s3 s4 s5 simcheck bundle fake robot summary"
case "$MODE" in
    full)  : "${STEPS:=setup,download,geometry,nrd,nrd_test,nrd_collect,t6,s5,simcheck,bundle,fake,summary}" "${PARENTS:=released}" ;;
    smoke) : "${STEPS:=setup,download,geometry,nrd,nrd_test,nrd_collect,t6,s5,simcheck,bundle,fake,summary}" "${PARENTS:=released}" ;;
    chain) : "${STEPS:=setup,download,nrd_collect,nrd_train,t0,t1,t2,t3,t4,t5,t6,s0,s1,s2,s3,s4,s5,simcheck,bundle,fake,summary}" "${PARENTS:=own}" ;;
esac
case "$PARENTS" in own|released) ;; *) die "--parents must be own or released" ;; esac
[ "$WITH_ROBOT" = 1 ] && [[ ",$STEPS," != *",robot,"* ]] && STEPS="$STEPS,robot"
for s in ${STEPS//,/ }; do [[ " $ALL_STEPS " == *" $s "* ]] || die "unknown step $s ($ALL_STEPS)"; done
has_step() { case ",$STEPS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

OUT="${OUT:-$CODE/artifacts/so101_pick_place_repro}"
[ "$MODE" = smoke ] && OUT="$OUT/smoke_$(date +%m%d_%H%M%S)"
A="$CODE/artifacts/so101_pick_place"                 # the Hub folder so101_pick_place/ after the download
mkdir -p "$OUT" "$A"
OUT="$(cd "$OUT" && pwd)"
RUNS="$OUT/runs"
TINY=0; [ "$MODE" = smoke ] && TINY=1
export PYTHONPATH="$CODE/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
cd "$CODE"
mkdir -p "$RUNS" "$OUT/configs" "$OUT/tasks"
log "mode $MODE, steps $STEPS, parents $PARENTS, out $OUT, data $A${SOURCE:+, Hub files from $SOURCE}"

# ---------------------------------------------------------------------------------------------------- inputs
NRD=$A/nrd/c1_s62/best.pt
T=$A/rl_inputs/tasks
B=$A/rl_inputs/banks
SCENE=$CODE/configs/so101_pick_place/real_scene_v2.json     # sha256 4bce0b8d... (pinned in every student)
CALIB=$CODE/configs/so101_pick_place/calib_v1.json          # sha256 3b7a2124... (pinned in every student)
# the scene file names the study's home path of the calibration: every student run passes both (as the runs did)
SETS=(jaw_lead=0 "scene=\"$SCENE\"" "calib=\"$CALIB\"")
REL_T=(t0:teacher/t0_nC1_s62_b/model_00171.pt t2:teacher/t2_bc_all/bc.pt t3:teacher/t3_jt_b140/model_00140.pt
       t4:teacher/t4_jt_h110/model_00110.pt t5:teacher/t5_tppo_roll_v8_s21/model_00200.pt
       t6:teacher/t6_tppo_v12_fr10_s122/model_00200.pt s0:student/s0_v2_start_s2/r2/last.pt
       s1:student/s1_v2_online_s18/it_03000.pt s2:student/s2_v2_online2_s28/it_03000.pt
       s3:student/s3_v4_relabel_s63/it_02000.pt s4:student/s4_v8_roll_s801/it_02000.pt
       s5:student/s5_v12_tb10_s1210/it_01000.pt)
OWN_T=(t0:t0_nC1_s62_b/ckpt/model_00171.pt t2:t2_bc_all/bc.pt t3:t3_jt_b140/ckpt/model_00140.pt
       t4:t4_jt_h110/ckpt/model_00110.pt t5:t5_tppo_roll_v8_s21/ckpt/model_00200.pt
       t6:t6_tppo_v12_fr10_s122/ckpt/model_00200.pt s0:s0_v2_start_s2/r2/last.pt s1:s1_v2_online_s18/ckpt/it_03000.pt
       s2:s2_v2_online2_s28/ckpt/it_03000.pt s3:s3_v4_relabel_s63/ckpt/it_02000.pt s4:s4_v8_roll_s801/ckpt/it_02000.pt
       s5:s5_v12_tb10_s1210/ckpt/it_01000.pt)
lookup() { local k=$1; shift; for e in "$@"; do [ "${e%%:*}" = "$k" ] && { echo "${e#*:}"; return 0; }; done; return 1; }
ckpt() {             # ckpt <stage>: the parent checkpoint (own run if --parents own and it exists, else released)
    local own; own="$RUNS/$(lookup "$1" "${OWN_T[@]}")"
    if [ "$PARENTS" = own ] && [ "$TINY" = 0 ]; then
        [ -f "$own" ] || die "no own checkpoint $own (run step $1, or --parents released)"
        echo "$own"
    elif [ "$PARENTS" = own ] && [ -f "$own" ]; then echo "$own"
    else echo "$A/$(lookup "$1" "${REL_T[@]}")"; fi
}
tiny_or() { [ "$TINY" = 1 ] && echo "$1" || echo "$2"; }

# ---------------------------------------------------------------------------------------------------- checks
if has_step t0 || has_step t1 || has_step t2 || has_step t3 || has_step t4 || has_step t5 || has_step t6 || \
   has_step s0 || has_step s1 || has_step s2 || has_step s3 || has_step s4 || has_step s5 || has_step simcheck; then
    "$PY" -c "import torch, torchvision, numpy, scipy, mujoco" 2>/dev/null \
        || die "$PY needs torch, torchvision, numpy, scipy and mujoco 3.15.0 (requirements-so101-pick-place.txt; set PY)"
    if [ "$DEVICE" != cpu ]; then
        "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || die "no GPU for --device $DEVICE"
    fi
fi
if has_step s0 || has_step s1 || has_step s2 || has_step s3 || has_step s4 || has_step s5 || has_step simcheck; then
    "$PY" -c "import warp, newton" 2>/dev/null || die "$PY needs warp-lang 1.13.0 and newton 1.2.0 for the renderer"
fi

# ---------------------------------------------------------------------------------------------------- setup
if has_step setup; then
    log "setup: MuJoCo Menagerie SO101 at the pinned revision (contact_nrd/so101_grasp/vendor/)"
    "$PY" -I contact_nrd/so101_grasp/fetch_sources.py
fi

# ---------------------------------------------------------------------------------------------------- download
DL_GROUPS=nrd,nrd_test,teacher,student,rl_inputs,results,bundle,recordings
[ "$WITH_NRD_DATA" = 1 ] && DL_GROUPS="$DL_GROUPS,nrd_data"
[ "$WITH_EXPERT" = 1 ] && DL_GROUPS="$DL_GROUPS,expert_data"
[ "$WITH_ROBOT" = 1 ] && DL_GROUPS="$DL_GROUPS,robot"
if has_step download; then
    log "download + sha256 check ($DL_GROUPS)"
    "$PY" -I scripts/so101_pick_place/download.py --groups "$DL_GROUPS" --dest "$CODE/artifacts" --quiet \
        ${SOURCE:+--source "$SOURCE"}
fi

# smoke task files: the first rows of the released sets (dev100 rows 0-3 = tasks --kind C --n 4 --dev, bit for bit)
TS="$T"
if [ "$TINY" = 1 ] && [ ! -f "$T/dev100.npz" ]; then
    for s in t1 t2 t3 t4 t5 t6 s0 s1 s2 s3 s4 s5; do
        has_step $s && die "no task files in $T (step $s): run the download step first"
    done
elif [ "$TINY" = 1 ]; then
    "$PY" - "$T" "$OUT/tasks" <<'EOF'
import sys
from pathlib import Path
import numpy as np
src, out = Path(sys.argv[1]), Path(sys.argv[2])
for name, n in (("dev100", 4), ("home45_dev100", 4), ("home_dev100", 4), ("train_mix45_16k", 256)):
    if not (src / f"{name}.npz").is_file():
        continue
    t = dict(np.load(src / f"{name}.npz", allow_pickle=True))
    N = len(t["seeds"])
    np.savez(out / f"{name}.npz", **{k: (v[:n] if np.ndim(v) and len(v) == N else v) for k, v in t.items()})
EOF
    TS="$OUT/tasks"
fi

# ---------------------------------------------------------------------------------------------------- geometry
if has_step geometry; then
    log "geometry: rebuild the kin points and the task geometry from MuJoCo, compare with rl_inputs/geometry/"
    mkdir -p "$OUT/geometry"
    "$PY" - "$A/rl_inputs/geometry" "$OUT/geometry" <<'EOF'
import hashlib, sys
from pathlib import Path
import numpy as np
from nedm.so101_policy import kin, task
rel, out = Path(sys.argv[1]), Path(sys.argv[2])
kin.save_points(out / "kin_points_v1.npz")
task.TaskGeometry.from_mujoco().save(out / "task_geom_v2.npz")
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for f in ("kin_points_v1.npz", "task_geom_v2.npz"):
    a, b = np.load(out / f), np.load(rel / f)
    same = sorted(a.files) == sorted(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
    print(f"{f}: arrays {'EQUAL' if same else 'DIFFER'} to the released file; sha256 "
          f"{'equal' if sha(out / f) == sha(rel / f) else 'differs (zip records)'}")
    if not same:
        sys.exit(f"{f}: the rebuilt arrays differ from rl_inputs/geometry/{f}")
EOF
fi

# ---------------------------------------------------------------------------------------------------- nrd
if has_step nrd; then
    log "nrd: main's nedm.contact_nrd.model.load_model on c1_s62 and c0_s62, 100-step rollouts of 4 test episodes"
    "$PY" - "$A" "$DEVICE" <<'EOF'
import hashlib, json, sys
from pathlib import Path
import torch
from nedm.contact_nrd import data as D
from nedm.contact_nrd.model import load_model
A, dev = Path(sys.argv[1]), sys.argv[2]
man = json.loads((Path("so101_pick_place/release_manifest.json")).read_text())["files"]
data = None
for run in ("c1_s62", "c0_s62"):
    f = A / f"nrd/{run}/best.pt"
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    model, packet = load_model(str(f), dev)
    n = sum(p.numel() for p in model.parameters())
    if data is None:
        root = A / "nrd_test"
        data = D.step_average_rates(D.load_data(str(root), dev, model_step_s=model.dt), set(model.config["integrate"]["kinds"]))
    s = data["stride"]
    st, act = data["states"][:4, ::s], data["actions"][:4, ::s]
    with torch.no_grad():
        traj = model.rollout(st[:, 0], act[:, :100])
    ic = [b["name"] for b in model.system["bodies"] if b["moving"]].index("cube")
    err = (traj[:, 100, ic, :3] - st[:, 100, ic, :3]).norm(dim=-1) * 1e3
    print(f"{run}: sha256 {sha[:16]} ({'= manifest' if sha == man['so101_pick_place/nrd/' + run + '/best.pt']['sha256'] else 'NOT the manifest'}), "
          f"{n:,} parameters, update {packet.get('update')}, rollout {tuple(traj.shape)} finite {bool(torch.isfinite(traj).all())}, "
          f"cube position error at 2 s (mm): {[round(float(e), 2) for e in err]}")
EOF
fi

# ---------------------------------------------------------------------------------------------------- nrd_test
if has_step nrd_test; then
    D_TEST="$A/nrd_test"
    if [ "$TINY" = 1 ]; then          # 2 episodes per family (24 of 3,050)
        D_TEST="$OUT/nrd_test_subset"
        "$PY" - "$A/nrd_test" "$D_TEST" <<'EOF'
import json, shutil, sys
from pathlib import Path
import numpy as np
src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
with np.load(src / "unified_data.npz") as f:
    fam = f["family"]
    rows = sorted(r for k in np.unique(fam) for r in np.flatnonzero(fam == k)[:2].tolist())
    np.savez(out / "unified_data.npz", **{k: f[k][rows] for k in f.files})
shutil.copy2(src / "system.json", out / "system.json")
(out / "rows.json").write_text(json.dumps(rows))
EOF
    fi
    log "nrd_test: python -m nedm.contact_nrd.test on $D_TEST (main's code, unchanged)"
    mkdir -p "$OUT/nrd"
    "$PY" -P -m nedm.contact_nrd.test --data "$D_TEST" --model c1_s62="$A/nrd/c1_s62" c0_s62="$A/nrd/c0_s62" \
        --output "$OUT/nrd/nrd_test.json" --device "$DEVICE" > "$OUT/nrd/nrd_test.txt"
    "$PY" - "$OUT/nrd/nrd_test.json" "$A/results/nrd/sealed_test.json" "$D_TEST" <<'EOF'
import json, sys
from pathlib import Path
mine, ref = json.loads(Path(sys.argv[1]).read_text()), json.loads(Path(sys.argv[2]).read_text())
rows_f = Path(sys.argv[3]) / "rows.json"
rows = json.loads(rows_f.read_text()) if rows_f.exists() else None
for m in ("c1_s62", "c0_s62"):
    a, b = mine["models"][m], ref["models"][m]
    if rows is None:
        print(f"{m}: selected score {a['selected']['selection_score']:.4f} (released {b['selected']['selection_score']:.4f}); "
              f"all {a['selection_score']:.4f} (released {b['selection_score']:.4f})")
    else:            # per-episode free-rollout errors of the same episodes (the batch differs: float noise possible)
        def flat(x, p=""):
            if isinstance(x, dict):
                for k, v in x.items():
                    yield from flat(v, f"{p}/{k}")
            elif isinstance(x, (int, float)) and not isinstance(x, bool):
                yield p, float(x)
        diff, n = 0.0, 0
        for j, r in enumerate(rows):
            ea, eb = dict(flat(a["per_episode"][j]["free"])), dict(flat(b["per_episode"][r]["free"]))
            for k in ea.keys() & eb.keys():
                diff, n = max(diff, abs(ea[k] - eb[k])), n + 1
        print(f"{m}: {len(rows)} test episodes, {n} per-episode free-rollout values vs the released test: max |diff| {diff:.3g}")
EOF
fi

# ---------------------------------------------------------------------------------------------------- nrd_collect
if has_step nrd_collect; then
    log "nrd_collect: regenerate the first episode of every family of nrd_test/ with the collector (exact check)"
    "$PY" -P - "$A/nrd_test" <<'EOF'
import json, sys, time
import numpy as np
from nedm.contact_nrd.collect import so101_grasp as g
data = sys.argv[1]
cfg = g.load_config(); s = g.Scene(cfg)
master = cfg["campaigns"]["test"]["master_seed"]
with np.load(f"{data}/unified_data.npz") as f:
    fam_all, fidx, lengths, seeds = f["family"], f["family_index"], f["lengths"], f["seed"]
    rows = sorted(int(np.flatnonzero(fam_all == k)[0]) for k in np.unique(fam_all))
    a = {key: f[key][rows] for key in ("arm", "cube", "action", "contacts")}
same = []
for j, row in enumerate(rows):
    ep = g.make_episode(s, int(fam_all[row]), int(fidx[row]), master)
    n = int(lengths[row])
    ok = (len(ep["action"]) == n and int(ep["seed"]) == int(seeds[row])
          and np.array_equal(ep["states"]["arm"][:n].astype(np.float32), a["arm"][j, :n])
          and np.array_equal(ep["states"]["cube"][:n].astype(np.float32), a["cube"][j, :n])
          and np.array_equal(ep["action"][:n].astype(np.float32), a["action"][j, :n])
          and np.array_equal(ep["contacts"], a["contacts"][j, :n - 1]))
    same.append(ok)
    print(f"  {cfg['families'][int(fam_all[row])]['name']:14s} row {row}: {'identical' if ok else 'DIFFERENT'}", flush=True)
print(f"{sum(same)} of {len(same)} test episodes regenerated bit for bit (campaign test, master seed {master})")
if not all(same):
    sys.exit("the collector does not regenerate the released test episodes on this machine (MuJoCo version?)")
EOF
    if [ "$MODE" = chain ]; then
        for c in production extra noisy_v1 test; do
            log "nrd_collect: campaign $c (CPU MuJoCo; about 15 min on 24 workers each)"
            [ -f "$OUT/nrd_data/$c/system.json" ] || "$PY" -P -m nedm.contact_nrd.collect.so101_grasp collect \
                --campaign "$c" --workers "$WORKERS" --out "$OUT/nrd_data/$c"
            "$PY" -P -m nedm.contact_nrd.collect.validate "$OUT/nrd_data/$c" --output "$OUT/nrd_data/$c/validate.json"
        done
        log "nrd_collect: expert_v2 and policy_v3 are not re-collected (their collector is in no commit / needs the 7 "
        log "             policies; README): the released parts are used by nrd_train"
    fi
fi

# ---------------------------------------------------------------------------------------------------- nrd_train
if has_step nrd_train; then
    # The training options of these two NRDs are not in main's train.py: train with the study code (README).
    ST="$OUT/study_f6f05c21c"
    if [ ! -d "$ST" ]; then
        git -C "$CODE" cat-file -e f6f05c21c^{commit} 2>/dev/null || git -C "$CODE" fetch origin so101-pick-place-study \
            || die "commit f6f05c21c not found: git fetch origin so101-pick-place-study"
        git -C "$CODE" worktree add --detach "$ST" f6f05c21c
        ln -s "$CODE/artifacts" "$ST/artifacts"
        cp -r "$CODE/contact_nrd/so101_grasp/vendor" "$ST/contact_nrd/so101_grasp/" 2>/dev/null || true
    fi
    PP="$A/nrd_data/production_plus"
    if [ -f "$OUT/nrd_data/production/system.json" ] && [ -f "$OUT/nrd_data/extra/system.json" ]; then
        PP="$OUT/nrd_data/production_plus"
        [ -f "$PP/system.json" ] || (cd "$ST" && "$PY" scripts/contact_nrd/merge_data.py "$PP" "$OUT/nrd_data/production" "$OUT/nrd_data/extra")
    fi
    log "nrd_train: base NRD c0_s62 from scratch on $PP (about 8.5 GPU-h; the study ran it with the uncommitted code of provenance/code_v16_base_nrd.tar.gz: recipe only)"
    (cd "$ST" && PYTHONPATH="$ST/src" "$PY" -P -m nedm.contact_nrd.train --data "$PP" --seed 62 \
        --config "$CODE/configs/so101_pick_place/nrd/base_c0.json" --output "$RUNS/nrd_c0_s62" --device "$DEVICE")
    M="$OUT/nrd_data/merged_r3_pp_noisy_expert_policy_x2.5"
    log "nrd_train: merge production_plus + noisy_v1 + expert_v2 + policy_v3 (train rows x 2.5) -> $M (13.6 GB)"
    [ -f "$M/system.json" ] || (cd "$ST" && PYTHONPATH="$ST/src" "$PY" -P scripts/so101_policy/collect_round.py merge \
        --base "$PP" --extra "$A/nrd_data/noisy_v1" "$A/nrd_data/expert_v2" "$A/nrd_data/policy_v3" \
        --repeat "$A/nrd_data/policy_v3=2.5" --out "$M")
    log "nrd_train: fine-tune c1_s62 (20,000 updates, best at update 2,500; 1.6 GPU-h); init_from = the released"
    log "           nrd/c0_s62/original/best.pt (edit configs/so101_pick_place/nrd/ft_c1_s62.json for an own base run)"
    (cd "$ST" && PYTHONPATH="$ST/src" "$PY" -P -m nedm.contact_nrd.train --data "$M" --output "$RUNS/nrd_c1_s62" \
        --config "$CODE/configs/so101_pick_place/nrd/ft_c1_s62.json" --device "$DEVICE")
fi

# ---------------------------------------------------------------------------------------------------- teachers
# smoke env configs: the stage's env config with 200-step episodes
smoke_env() {        # smoke_env <config> -> a copy with horizon 200 (smoke) or the config itself
    if [ "$TINY" = 1 ]; then
        local o="$OUT/configs/$(basename "$1")"
        "$PY" -c "import json, sys; c = json.load(open(sys.argv[1])); c['horizon'] = 200; json.dump(c, open(sys.argv[2], 'w'), indent=1)" "$1" "$o"
        echo "$o"
    else echo "$1"; fi
}
teacher_ppo() {      # teacher_ppo <run> <init> <env config> <train config> <critic warmup> <iterations> <eval/save every>
                     #             <seed> <dev set> [<more dev sets> (then also --dev-match-env)]
    local run="$RUNS/$1" init="$2" ecfg; ecfg="$(smoke_env "$3")"; local tc="$4" cw="$5" it="$6" every="$7" seed="$8"
    shift 8
    local dev=(--dev-tasks "$1"); shift
    [ $# -gt 0 ] && dev+=(--dev-tasks-extra "$@" --dev-match-env)
    local size=(--envs 8192 --iterations "$it" --eval-every "$every" --save-every "$every" --critic-warmup "$cw")
    [ "$TINY" = 1 ] && size=(--envs 256 --iterations 2 --eval-every 2 --save-every 2 --critic-warmup 1)
    [ -e "$run/ckpt" ] && { log "skip $run (exists)"; return 0; }
    log "teacher PPO $(basename "$run"): init $init, env $ecfg"
    env -u NEDM_SO101_DYN "$PY" -P scripts/so101_policy/vision_viral_teacher.py ppo --nrd-ckpt "$NRD" --init "$init" \
        --env-config "$ecfg" --train-config "$tc" "${size[@]}" "${dev[@]}" --seed "$seed" --out "$run" \
        --mj-python "$PY" --device "$DEVICE"
}
if has_step t0; then
    run="$RUNS/t0_nC1_s62_b"
    size=(--num-envs 16384 --max-hours 3.3 --val-every 10 --save-every 10 --val-n 512 --print-every 5)
    [ "$TINY" = 1 ] && size=(--num-envs 256 --max-iterations 1 --val-every 1 --save-every 1 --val-n 16 --print-every 1)
    log "t0: decoder PPO from scratch in the NRD (train_ppo.py; study: 171 iterations in 3.3 h on one MI350X, model_00171)"
    [ -e "$run/done.json" ] || "$PY" -P scripts/so101_policy/train_ppo.py train --world nrd --out "$run" \
        --env-cfg configs/so101_policy/ppo/env_v2.json --train-cfg configs/so101_policy/ppo/train_v2.json \
        --curriculum configs/so101_policy/ppo/curriculum_tol_v3.json "${size[@]}" --starts "$B/bank_pp_s0.npz" \
        --set env.geom_path="\"$A/rl_inputs/geometry/task_geom_v2.npz\"" env.kin_points="\"$A/rl_inputs/geometry/kin_points_v1.npz\"" \
        --ckpt "$NRD" --seed 2 --set 'env.grasp_shaping={"w_x":20.0,"w_y":10.0,"w_yaw":2.0}' algorithm.desired_kl=0.02 \
        --device "$DEVICE"
fi
if has_step t1; then
    log "t1: expert data, T0 + the action decoder in the NRD (recipe only: the study code was not committed)"
    mkdir -p "$RUNS/t1_expert_data"
    for spec in "e0_n1 1.0 0 0" "e1_n2 2.0 1 8192" "e2_n15 1.5 2 16384"; do
        set -- $spec
        [ -f "$RUNS/t1_expert_data/$1.npz" ] || "$PY" -P scripts/so101_policy/vision_viral_teacher.py expert \
            --teacher "$(ckpt t0)" --nrd-ckpt "$NRD" --n "$(tiny_or 64 8192)" --noise "$2" --noise-frac 0.5 --seed "$3" \
            --first "$4" --mj-python "$PY" --device "$DEVICE" --out "$RUNS/t1_expert_data/$1.npz"
    done
fi
if has_step t2; then
    E="$RUNS/t1_expert_data"          # own t1 data if this run made them (smoke: always), else the released data
    { [ "$PARENTS" = released ] && [ "$TINY" = 0 ]; } || [ ! -f "$E/e2_n15.npz" ] && E="$A/teacher/t1_expert_data"
    [ -f "$E/e2_n15.npz" ] || die "t2 needs the expert data: run step t1, or download them (--with-expert-data)"
    log "t2: behaviour cloning of the joint-target teacher on $E (recipe only)"
    [ -f "$RUNS/t2_bc_all/bc.pt" ] || "$PY" -P scripts/so101_policy/vision_viral_teacher.py bc \
        --data "$E/e0_n1.npz" "$E/e1_n2.npz" "$E/e2_n15.npz" --steps "$(tiny_or 20 60000)" --batch "$(tiny_or 512 8192)" \
        --hidden 1024 512 256 --lr 0.001 --seed 0 --device "$DEVICE" --out "$RUNS/t2_bc_all"
fi
# t3 / t4: the study planned 400 / 200 iterations and cancelled the jobs; the adaptive-lr loop does not depend on the
# planned count, so this script stops at the released checkpoint's iteration (140 / 110)
if has_step t3; then
    teacher_ppo t3_jt_b140 "$(ckpt t2)" configs/so101_pick_place/teacher/env_jt_b140.json \
        configs/so101_policy/ppo/joint_ppo_v3.json 20 140 10 0 "$TS/dev100.npz"
fi
if has_step t4; then
    teacher_ppo t4_jt_h110 "$(ckpt t3)" configs/so101_pick_place/teacher/env_jt_h110.json \
        configs/so101_policy/ppo/joint_ppo_v3.json 5 110 10 0 "$TS/dev100.npz"
fi
if has_step t5; then
    teacher_ppo t5_tppo_roll_v8_s21 "$(ckpt t4)" configs/so101_policy/ppo/r2s/teacher_roll_v8.json \
        configs/so101_policy/ppo/r2s/joint_ppo_v3_kl02.json 20 200 20 21 "$TS/home45_dev100.npz" "$TS/dev100.npz"
fi
if has_step t6; then
    teacher_ppo t6_tppo_v12_fr10_s122 "$(ckpt t5)" configs/so101_policy/ppo/r2s/teacher_roll_v12_fr10.json \
        configs/so101_policy/ppo/r2s/joint_ppo_v3_kl02.json 20 200 20 122 "$TS/home45_dev100.npz" "$TS/dev100.npz" \
        "$TS/home_dev100.npz"
fi

# ---------------------------------------------------------------------------------------------------- students
GUARD=("$T/fresh100.npz" "$T/home45_fresh100.npz")
student_online() {   # student_online <run> <init> <teacher> <online config> <seed> [--oset ...]
    local run="$RUNS/$1" init="$2" jt="$3" oc="$4" seed="$5"; shift 5
    local oset=("$@") size=(--render-worlds 128 --eval-workers 16)
    local ev=(--eval-tasks "$T/dev100.npz" "$T/home45_dev100.npz")
    if [ "$TINY" = 1 ]; then
        size=(--render-worlds 16 --eval-workers 2); ev=(--eval-tasks "$TS/dev100.npz")
        oset=(--oset iterations=12 eval_every=12 save_every=12 log_every=4 envs=64 buffer=4096
              $(for o in "$@"; do case "$o" in roll_hi_deg=*) echo "$o" ;; esac; done))
    fi
    [ -e "$run/curve.csv" ] && { log "skip $run (exists)"; return 0; }
    log "student online $(basename "$run"): init $init, teacher $jt, $(basename "$oc")"
    "$PY" -P scripts/so101_policy/vision_viral.py online --init "$init" --out "$run" --jt "$jt" --nrd-ckpt "$NRD" \
        --mj-python "$PY" --tasks "$TS/train_mix45_16k.npz" "${ev[@]}" --guard-tasks "${GUARD[@]}" \
        --config configs/so101_policy/vision/viral_r2s_v1.json --set "${SETS[@]}" --online-config "$oc" \
        --eval-randomise "${size[@]}" --seed "$seed" --device "$DEVICE" "${oset[@]}"
}
if has_step s0; then
    run="$RUNS/s0_v2_start_s2"
    size=(--r0-tasks 4096 --rounds 2 --round-tasks 2048 --train-steps0 10000 --train-steps 4000 --render-worlds 256
          --eval-workers 16 --eval-tasks "$T/dev100.npz" "$T/home45_dev100.npz")
    [ "$TINY" = 1 ] && size=(--r0-tasks 32 --rounds 1 --round-tasks 16 --train-steps0 4 --train-steps 2 --render-worlds 16
                             --eval-workers 2 --eval-tasks "$TS/dev100.npz")
    log "s0: first student (offline rounds, ImageNet ResNet18 encoders from torchvision), labels of T4"
    [ -e "$run/r$([ "$TINY" = 1 ] && echo 1 || echo 2)/last.pt" ] || "$PY" -P scripts/so101_policy/vision_viral.py run \
        --out "$run" --jt "$(ckpt t4)" --nrd-ckpt "$NRD" --mj-python "$PY" --tasks "$TS/train_mix45_16k.npz" \
        --guard-tasks "${GUARD[@]}" --config configs/so101_policy/vision/viral_r2s_v1.json \
        --set "scene=\"$SCENE\"" "calib=\"$CALIB\"" --teacher-share 0.5 --noise 1.0 --r0-noise-frac 0.5 \
        --collect-randomise --eval-randomise --seed 2 --device "$DEVICE" "${size[@]}"
fi
V=configs/so101_policy/vision
if has_step s1; then student_online s1_v2_online_s18 "$(ckpt s0)" "$(ckpt t4)" $V/viral_r2s_online_v1.json 18; fi
if has_step s2; then student_online s2_v2_online2_s28 "$(ckpt s1)" "$(ckpt t4)" $V/viral_r2s_online_v1.json 28; fi
if has_step s3; then        # exact code of this stage: commit 977863c9e (550a58cb6 changed the relabel mask; README)
    student_online s3_v4_relabel_s63 "$(ckpt s2)" "$(ckpt t4)" $V/viral_r2s_online_relabel_v4.json 63
fi
if has_step s4; then
    student_online s4_v8_roll_s801 "$(ckpt s3)" "$(ckpt t5)" $V/viral_r2s_online_relabel_roll_v8.json 801 \
        --oset iterations=2000
fi
if has_step s5; then        # the teacher: --parents own and an own t6 run, else the released T6
    student_online s5_v12_tb10_s1210 "$(ckpt s4)" "$(ckpt t6)" $V/viral_r2s_online_v12_tb10.json 1210 \
        --oset iterations=1500 eval_every=750 save_every=250 roll_hi_deg=98
fi

# ---------------------------------------------------------------------------------------------------- simcheck
STUDENT="$(ckpt s5)"
if has_step simcheck; then
    G="$OUT/simcheck"
    if [ "$TINY" = 1 ]; then
        log "simcheck (smoke): G2-type evaluation of $STUDENT, 4 J224 tasks, 300 steps"
        ONLY=g2_a TASK_ROWS=4 HORIZON=300 WORKERS=4 PY="$PY" scripts/so101_pick_place/check/run_gates.sh run \
            v12_tb10_s1210_it1000 "$STUDENT" "$G"
        "$PY" - "$G/runs/gates/local/v12_tb10_s1210_it1000/g2_a.npz" <<'EOF'
import json, sys
sys.path.insert(0, "scripts/so101_pick_place/check")
import numpy as np
import check_analyze as CA
f = sys.argv[1]
rows, info = CA.run_episodes(f)
t = CA.j_task(f, tuple(info["plane"]) if info.get("plane") else None, info)
print("g2_a smoke:", json.dumps(t["all"]), "| check info: look", info.get("look"), "scene", info.get("scene_sha256", "")[:8])
EOF
    else
        log "simcheck: D's reference runs (12 evaluations), then tb10 (9); about 8-10 GPU-h of renders + CPU MuJoCo"
        WORKERS="$WORKERS" PY="$PY" scripts/so101_pick_place/check/run_gates.sh run D "$A/student/s4_v8_roll_s801/it_02000.pt" "$G" ref
        WORKERS="$WORKERS" PY="$PY" scripts/so101_pick_place/check/run_gates.sh run v12_tb10_s1210_it1000 "$STUDENT" "$G"
        PY="$PY" scripts/so101_pick_place/check/run_gates.sh summary v12_tb10_s1210_it1000 "$G"
    fi
fi

# ---------------------------------------------------------------------------------------------------- bundle
BUNDLE="$OUT/bundle/so101_hw_v13_tb10"
if has_step bundle; then
    log "bundle: make_orin_bundle.sh $STUDENT -> $BUNDLE"
    SO101_KIN_POINTS="$A/rl_inputs/geometry/kin_points_v1.npz" PYTHON="$PY" \
        scripts/so101_policy/real2sim/hw/make_orin_bundle.sh "$STUDENT" "$BUNDLE" "$CALIB"
    "$PY" - "$BUNDLE/MANIFEST.txt" "$A/bundle/so101_hw_v13_tb10/MANIFEST.txt" <<'EOF'
import sys
read = lambda p: {l.split(None, 1)[1].strip(): l.split()[0] for l in open(p).read().splitlines()[3:]}
mine, ref = read(sys.argv[1]), read(sys.argv[2])
same = sorted(k for k in mine.keys() & ref.keys() if mine[k] == ref[k])
diff = sorted(k for k in mine.keys() & ref.keys() if mine[k] != ref[k])
print(f"bundle vs the released bundle: {len(same)} files equal, {len(diff)} differ {diff}, only here "
      f"{sorted(mine.keys() - ref.keys())}, only in the released one {sorted(ref.keys() - mine.keys())}")
EOF
fi

# ---------------------------------------------------------------------------------------------------- fake robot
if has_step fake; then
    [ -d "$BUNDLE" ] || die "no bundle $BUNDLE (run the bundle step)"
    FB="$OUT/fakebin"; mkdir -p "$FB"
    PYABS="$(command -v "$PY")"
    cat > "$FB/python" <<EOF
#!/usr/bin/env bash
# fake robot (reproduce.sh): run_student_real.py gets --backend fake --auto-confirm --device cpu and the LeRobot
# motor calibration of the lab robot; no device is opened
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
if [[ "\${1:-}" == *run_student_real.py ]]; then
  exec "$PYABS" "\$@" --backend fake --auto-confirm --device cpu --lerobot-calib "$CODE/configs/so101_pick_place/so101_follower_lerobot_calib.json"
fi
exec "$PYABS" "\$@"
EOF
    chmod +x "$FB/python"
    F="$OUT/fake"; mkdir -p "$F"
    REC="$A/robot/lerobot_recordings"
    for b in "$BUNDLE" "$A/bundle/so101_hw_v13_tb10"; do
        tag=$([ "$b" = "$BUNDLE" ] && echo built || echo released)
        [ "$tag" = released ] && { rm -rf "$F/released_bundle"; cp -r "$b" "$F/released_bundle"; b="$F/released_bundle"; }
        (cd "$b" && sed -n '4,$p' MANIFEST.txt | sha256sum -c --quiet) || die "$tag bundle: MANIFEST.txt check failed"
        echo "$tag bundle: MANIFEST OK"
        if [ "$tag" = built ]; then
            log "fake: robot_test.sh check (the sections that need the robot computer fail here: CUDA, port, cameras, LeRobot calibration)"
            rc=0; (cd "$b" && PATH="$FB:$PATH" CONDA_DEFAULT_ENV=lerobot timeout 300 ./robot_test.sh check > "$F/check.log" 2>&1) || rc=$?
            echo "check exit $rc (nonzero expected off the robot computer); $(grep -c -i 'fail\|missing' "$F/check.log") lines with FAIL / missing"
            log "fake: robot_test.sh eval 1 6 (one 6 s episode on the fake robot)"
            rc=0; (cd "$b" && PATH="$FB:$PATH" CONDA_DEFAULT_ENV=lerobot RUNS="$F/runs" timeout 900 \
                script -qec "./robot_test.sh eval 1 6" /dev/null > "$F/eval.log" 2>&1) || rc=$?
            echo "eval exit $rc"; [ "$rc" = 0 ] || die "fake eval failed ($F/eval.log)"
        fi
        log "fake: robot_test.sh replay with the $tag bundle (3 recordings)"
        rc=0; (cd "$b" && PATH="$FB:$PATH" CONDA_DEFAULT_ENV=lerobot RUNS="$F/runs_$tag" DATA_ROOT="$REC" timeout 1800 \
            ./robot_test.sh replay > "$F/replay_$tag.log" 2>&1) || rc=$?
        echo "replay ($tag) exit $rc"; [ "$rc" = 0 ] || die "fake replay failed ($F/replay_$tag.log)"
    done
    "$PY" - "$F/runs_built" "$F/runs_released" <<'EOF'
import sys
from pathlib import Path
import numpy as np
def chunks(root):
    out = {}
    for f in sorted(Path(root).rglob("*.npz")):
        z = np.load(f, allow_pickle=False)
        key = str(f.relative_to(root)).split("/", 1)[-1]
        out[key] = {k: z[k] for k in ("q_sim", "qd_sim", "qry_step", "qry_thumbs", "qry_chunk") if k in z.files}
    return out
a, b = chunks(sys.argv[1]), chunks(sys.argv[2])
if not a or a.keys() != b.keys():
    sys.exit(f"replay outputs differ in their files: {sorted(a)} vs {sorted(b)}")
worst, n = 0.0, 0
for k in a:
    for name in a[k].keys() & b[k].keys():
        x, y = a[k][name], b[k][name]
        n += 1
        worst = max(worst, float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max()) if x.shape == y.shape else float("inf"))
print(f"replay, built vs released bundle: {len(a)} recordings, {n} arrays compared, max |diff| {worst:.3g}")
EOF
fi

# ---------------------------------------------------------------------------------------------------- robot results
if has_step robot; then
    log "robot: the result tables from the released logs (build_results.py), next to results/robot/results.json"
    mkdir -p "$OUT/robot"
    "$PY" -I scripts/so101_pick_place/robot/build_results.py --logs "$A/robot/logs" \
        --stress "$A/results/robot/stress_episodes.json" --out "$OUT/robot" > "$OUT/robot/build.log"
    "$PY" - "$OUT/robot/results.json" "$A/results/robot/results.json" <<'EOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
for d in (a, b):
    for k in ("built_by", "logs_root", "stress_meta"):
        d.pop(k)
    for s in d["students"].values():
        s.pop("local_bundle")
t = a["per_student"]["tb10"]["stress_20261027"]
print(f"robot tables: equal to the released results.json (source paths aside): {a == b}; tb10 stress {t['placed']}/{t['n']}")
EOF
fi

# ---------------------------------------------------------------------------------------------------- summary
if has_step summary; then
    log "summary: this run next to the released records"
    "$PY" - "$RUNS" "$A" <<'EOF'
import json, sys
from pathlib import Path
runs, A = Path(sys.argv[1]), Path(sys.argv[2])
REL = {"t3_jt_b140": None, "t4_jt_h110": None, "t5_tppo_roll_v8_s21": "teacher/t5_tppo_roll_v8_s21/dev_evals.jsonl",
       "t6_tppo_v12_fr10_s122": "teacher/t6_tppo_v12_fr10_s122/dev_evals.jsonl"}
for name, rel in REL.items():
    f = runs / name / "dev_evals.jsonl"
    if not f.is_file():
        continue
    last = json.loads(f.read_text().splitlines()[-1])
    ref = json.loads((A / rel).read_text().splitlines()[-1]) if rel else None
    for s, v in last["sets"].items():
        r = (ref or {}).get("sets", {}).get(s, {}) if ref else {}
        print(f"{name} it {last['iteration']} {s}: lenient {v['lenient']:.2f} strict {v['strict']:.2f} | released it "
              f"{ref['iteration'] if ref else '-'}: {r.get('lenient', '-')} / {r.get('strict', '-')}")
for d in sorted(runs.glob("s[1-5]_*")):
    f = d / "progress.jsonl"
    if not f.is_file():
        continue
    rel = A / "student" / d.name / "progress.jsonl"
    ev = lambda p: [json.loads(l) for l in p.read_text().splitlines() if '"stage": "eval' in l]
    mine, ref = ev(f), (ev(rel) if rel.is_file() else [])
    for e in mine[-2:]:
        print(f"{d.name} {e['stage']}: lenient {e['lenient']:.2f} strict {e['strict']:.2f}")
    for e in ref[-2:]:
        print(f"{d.name} released {e['stage']}: lenient {e['lenient']:.2f} strict {e['strict']:.2f}")
EOF
    G="$OUT/simcheck/out/gates/local/v12_tb10_s1210_it1000.json"
    if [ -f "$G" ]; then
        "$PY" - "$G" "$A/results/sim_checks/v12_tb10_s1210_it1000.json" <<'EOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
for k in ("G2",):
    for m in ("grasp", "place", "carry_stop"):
        print(f"sim check G2 {m}: {a['gates'][k][m]['cand']} (released, AMD MI350X: {b['gates'][k][m]['cand']})")
for p in ("g3_m3", "g3_p3"):
    print(f"sim check G3 {p}: " + ", ".join(f"{m} {a['gates']['G3']['planes'][p][m]['cand']} / {b['gates']['G3']['planes'][p][m]['cand']}"
                                          for m in ("grasp", "place", "carry_stop")))
print(f"G1 {a['gates']['G1']['cand']} % (released {b['gates']['G1']['cand']} %); veto {a['vetoed']} (released {b['vetoed']})")
EOF
    fi
fi
log "done: $OUT"
