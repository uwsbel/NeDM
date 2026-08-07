#!/usr/bin/env bash
# Chrono policy-in-sim eval of the L8 RL run's model_500.pt, mirroring the legacy
# anchor run's model_500 evals EXACTLY (rigid + bumpy; CRM held — GPU busy w/ RL).
# Runs on NEWTON (idle 4090, anaconda nedm/pychrono) so it never contends with the
# live L8 RL training on luffy. Chrono eval is policy-in-real-physics (no dynamics
# NN, deterministic) so the box doesn't affect the comparison.
#
# Legacy baseline (anchor last.pt, model_500): rigid mean 0.168 / median 0.125 m.
#
#   bash scripts/ablations/run_l8_chrono_eval_newton.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON=/home/harry/anaconda3/envs/nedm/bin/python
# NOTE: do NOT export the nedm libstdc++/OpenSSL path before rsync — it breaks
# ssh with "OpenSSL version mismatch". Set it only around the pychrono eval below.
NEDM_LIB=/home/harry/anaconda3/envs/nedm/lib
LUFFY=luffy
L8=hmmwv_rl_15d_crm2000mix25_onehot_ofatL8_bestval51_flat20crm20_K16_64steps_ar02_state_vxvyyr_pos2_yaw2_steerlim010
RUN=artifacts/rl_runs/$L8
CKPT=$RUN/model_500.pt
STEERLIM=0.1
NREF=20

RIGID_REF=/home/harry/NeDM/artifacts/rl_reference_sets/hmmwv_tire_normal_force_omega_val_refs_20_1100_rest_start.npz
BUMPY_REF=/home/harry/NeDM/artifacts/rl_reference_sets/hmmwv_bumpy_10g_normal_force_omega_val_refs_20_1100_rest_start.npz

# --- 1) pull the L8 run's model_500 + configs from luffy -----------------------
mkdir -p "$RUN"
echo "[sync] pulling model_500 + cfgs from luffy..."
rsync -a "$LUFFY:/home/harry/NeDM/$RUN/model_500.pt" \
         "$LUFFY:/home/harry/NeDM/$RUN/env_cfg.json" \
         "$LUFFY:/home/harry/NeDM/$RUN/train_cfg.json" "$RUN/" || { echo "sync failed"; exit 1; }

# pychrono needs the nedm env's libstdc++ (CXXABI) — safe to export now (post-rsync)
export LD_LIBRARY_PATH=$NEDM_LIB:${LD_LIBRARY_PATH:-}

# --- 2) build eval-cfg dirs (terrain=flat one-hot, rest-start ref, relaxed term) ---
build_evalcfg () {  # $1=eval-cfg subdir  $2=reference_path
  local dst="$RUN/$1"; mkdir -p "$dst"
  cp "$RUN/train_cfg.json" "$dst/train_cfg.json"
  "$PYTHON" - "$RUN/env_cfg.json" "$dst/env_cfg.json" "$2" <<'PY'
import json, sys
src, dst, ref = sys.argv[1], sys.argv[2], sys.argv[3]
c = json.load(open(src))
c["reference_path"] = ref
c["terrain"] = [1.0, 0.0]          # flat one-hot key (bumpy uses the flat key too)
c["terrain_mix"] = None
c.setdefault("termination", {})["max_position_error_m"] = 20.0
json.dump(c, open(dst, "w"), indent=2)
print("wrote", dst)
PY
}
build_evalcfg eval_cfg_rigid20_val_rest_start_flatkey "$RIGID_REF"
build_evalcfg eval_cfg_bumpy20_val_rest_start_flatkey "$BUMPY_REF"

# --- 3) run one group: 20 refs, one process each (stack-smash-safe), aggregate --
run_group () {  # $1=eval-cfg subdir  $2=chrono-config  $3=output subdir
  local cfgdir="$RUN/$1" chrono="$2" out="$RUN/$3"
  mkdir -p "$out"; : > "$out/rollouts.jsonl"
  echo "[eval] group $3  chrono=$chrono  ($NREF refs)"
  for i in $(seq 0 $((NREF-1))); do
    echo "[eval] $3 ref $i/$((NREF-1))"
    "$PYTHON" scripts/evaluation/eval_hmmwv_rl_chrono_tracking.py \
      --run-dir "$cfgdir" \
      --policy-checkpoint "$CKPT" \
      --chrono-config "$chrono" \
      --steering-rate-limit "$STEERLIM" \
      --reference-index "$i" \
      --ignore-dones \
      --no-plots \
      --output-dir "$out" > "$out/ref_${i}.log" 2>&1
    # per-ref metrics line printed as compact json.dumps(metrics) starting {"steps"
    grep -h '^{"steps"' "$out/ref_${i}.log" | tail -1 >> "$out/rollouts.jsonl" \
      || echo "[eval] WARN no metrics line for ref $i (see $out/ref_${i}.log)"
  done
  # aggregate to summary.json (legacy format)
  "$PYTHON" - "$out/rollouts.jsonl" "$out/summary.json" "$CKPT" "$chrono" "$STEERLIM" <<'PY'
import json, sys, statistics
jsonl, out, ckpt, chrono, steer = sys.argv[1:6]
rows = [json.loads(l) for l in open(jsonl) if l.strip()]
xy = [r["xy_rmse_m"] for r in rows if "xy_rmse_m" in r]
agg = {
  "backend": "chrono_hmmwv", "policy_checkpoint": ckpt, "chrono_config": chrono,
  "steering_rate_limit": float(steer), "num_rollouts": len(rows),
  "mean_xy_rmse_m": (sum(xy)/len(xy)) if xy else None,
  "median_xy_rmse_m": statistics.median(xy) if xy else None,
  "rollouts": rows,
}
json.dump(agg, open(out, "w"), indent=2)
print(f"[agg] {out}: n={len(rows)} mean={agg['mean_xy_rmse_m']} median={agg['median_xy_rmse_m']}")
PY
}

run_group eval_cfg_rigid20_val_rest_start_flatkey configs/hmmwv_overfit_v1.json \
          chrono_eval_tracking_model_500_rigid_val_rest_start_steerlim010
run_group eval_cfg_bumpy20_val_rest_start_flatkey configs/hmmwv_bumpy_eval.json \
          chrono_bumpy_eval_model_500_val20_rest_start_steerlim010

echo "[done] L8 model_500 rigid + bumpy chrono eval complete. CRM held."
