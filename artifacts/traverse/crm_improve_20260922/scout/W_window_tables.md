Command: `PYTHONPATH=src:scripts OMP_NUM_THREADS=6 python scripts/ci_window_probe.py --full-coverage --common --ablate 10:5 10:10 20:20 60:40 --out artifacts/traverse/crm_improve_20260922/scout/W_window`

Sources per k: k 10: short_anchor.npz, k 20: short_anchor.npz, k 30: short_anchor.npz, k 40: mixed_reanchor.npz, k 60: anchor_k40_60_80.npz, k 80: mixed_reanchor.npz. * = window longer than the time since the start (only k frames valid).

### GRU val AUC, mean of 3 seeds

| decision frame k (time) | pairs (val) | L = 5 (0.25 s) | L = 10 (0.50 s) | L = 20 (1.00 s) | L = 40 (2.00 s) |
|---|---|---|---|---|---|
| 10 (0.50 s) | 14331 (702) | 0.9994 | 0.9997 | 0.9999* | 0.9999* |
| 20 (1.00 s) | 14331 (702) | 0.9997 | 0.9998 | 0.9997 | 0.9996* |
| 30 (1.50 s) | 14234 (700) | 0.9987 | 0.9984 | 0.9998 | 0.9998* |
| 40 (2.00 s) | 14044 (693) | 0.9996 | 0.9991 | 0.9990 | 0.9995 |
| 60 (3.00 s) | 14039 (688) | 0.9997 | 1.0000 | 1.0000 | 1.0000 |
| 80 (4.00 s) | 3010 (151) | 0.9969 | 0.9983 | 0.9999 | 0.9983 |

### GRU val AUC, worst seed

| decision frame k (time) | pairs (val) | L = 5 (0.25 s) | L = 10 (0.50 s) | L = 20 (1.00 s) | L = 40 (2.00 s) |
|---|---|---|---|---|---|
| 10 (0.50 s) | 14331 (702) | 0.9992 | 0.9995 | 0.9998* | 0.9999* |
| 20 (1.00 s) | 14331 (702) | 0.9995 | 0.9998 | 0.9996 | 0.9995* |
| 30 (1.50 s) | 14234 (700) | 0.9984 | 0.9976 | 0.9997 | 0.9997* |
| 40 (2.00 s) | 14044 (693) | 0.9994 | 0.9989 | 0.9979 | 0.9993 |
| 60 (3.00 s) | 14039 (688) | 0.9996 | 1.0000 | 1.0000 | 1.0000 |
| 80 (4.00 s) | 3010 (151) | 0.9963 | 0.9969 | 0.9999 | 0.9974 |

### GRU seed-ensemble val AUC, 95 % group-bootstrap lower bound

| decision frame k (time) | pairs (val) | L = 5 (0.25 s) | L = 10 (0.50 s) | L = 20 (1.00 s) | L = 40 (2.00 s) |
|---|---|---|---|---|---|
| 10 (0.50 s) | 14331 (702) | 0.9988 | 0.9999 | 0.9999* | 0.9999* |
| 20 (1.00 s) | 14331 (702) | 0.9996 | 0.9996 | 0.9992 | 0.9992* |
| 30 (1.50 s) | 14234 (700) | 0.9961 | 0.9965 | 0.9993 | 0.9994* |
| 40 (2.00 s) | 14044 (693) | 0.9993 | 0.9977 | 0.9979 | 0.9986 |
| 60 (3.00 s) | 14039 (688) | 0.9994 | 1.0000 | 1.0000 | 1.0000 |
| 80 (4.00 s) | 3010 (151) | 0.9936 | 0.9968 | 1.0000 | 0.9989 |

### Logistic on window means + stds (30 features), val AUC

| decision frame k (time) | pairs (val) | L = 5 (0.25 s) | L = 10 (0.50 s) | L = 20 (1.00 s) | L = 40 (2.00 s) |
|---|---|---|---|---|---|
| 10 (0.50 s) | 14331 (702) | 0.995 | 0.995 | 0.995* | 0.995* |
| 20 (1.00 s) | 14331 (702) | 0.992 | 0.993 | 0.994 | 0.994* |
| 30 (1.50 s) | 14234 (700) | 0.980 | 0.987 | 0.992 | 0.992* |
| 40 (2.00 s) | 14044 (693) | 0.987 | 0.988 | 0.993 | 0.994 |
| 60 (3.00 s) | 14039 (688) | 0.953 | 0.974 | 0.988 | 0.997 |
| 80 (4.00 s) | 3010 (151) | 0.928 | 0.945 | 0.978 | 0.997 |

### Last-step logistic (state + action at frame k only), val AUC

| k | 10 | 20 | 30 | 40 | 60 | 80 |
|---|---|---|---|---|---|---|
| val AUC | 0.903 | 0.985 | 0.964 | 0.977 | 0.930 | 0.888 |

### Variant full_coverage: GRU val AUC mean of 3 seeds (k >= 40 from anchor_k40_60_80.npz: every admitted anchor, no speed-stratified selection)

| decision frame k (time) | pairs (val) | L = 5 (0.25 s) | L = 10 (0.50 s) | L = 20 (1.00 s) | L = 40 (2.00 s) |
|---|---|---|---|---|---|
| 40 (2.00 s) | 14044 (693) | 0.9993 | 0.9992 | 0.9988 | 0.9994 |
| 60 (3.00 s) | 14039 (688) | 0.9997 | 1.0000 | 1.0000 | 1.0000 |
| 80 (4.00 s) | 13810 (672) | 0.9992 | 0.9999 | 1.0000 | 1.0000 |

### Variant full_coverage: logistic on window means + stds, val AUC

| decision frame k (time) | pairs (val) | L = 5 (0.25 s) | L = 10 (0.50 s) | L = 20 (1.00 s) | L = 40 (2.00 s) |
|---|---|---|---|---|---|
| 40 (2.00 s) | 14044 (693) | 0.987 | 0.988 | 0.993 | 0.994 |
| 60 (3.00 s) | 14039 (688) | 0.953 | 0.974 | 0.988 | 0.997 |
| 80 (4.00 s) | 13810 (672) | 0.902 | 0.930 | 0.964 | 0.993 |

### Variant common: GRU val AUC mean of 3 seeds (episodes paired at every k in [10, 20, 30, 40, 60, 80] (13653 episodes); k >= 40 from anchor_k40_60_80.npz)

| decision frame k (time) | pairs (val) | L = 5 (0.25 s) | L = 10 (0.50 s) | L = 20 (1.00 s) | L = 40 (2.00 s) |
|---|---|---|---|---|---|
| 10 (0.50 s) | 13653 (668) | 0.9994 | 0.9999 | 0.9999* | 0.9999* |
| 20 (1.00 s) | 13653 (668) | 0.9995 | 0.9997 | 0.9996 | 0.9996* |
| 30 (1.50 s) | 13653 (668) | 0.9984 | 0.9985 | 0.9997 | 0.9997* |
| 40 (2.00 s) | 13653 (668) | 0.9993 | 0.9991 | 0.9989 | 0.9981 |
| 60 (3.00 s) | 13653 (668) | 0.9997 | 0.9999 | 1.0000 | 1.0000 |
| 80 (4.00 s) | 13653 (668) | 0.9991 | 0.9998 | 1.0000 | 1.0000 |

### Variant common: logistic on window means + stds, val AUC

| decision frame k (time) | pairs (val) | L = 5 (0.25 s) | L = 10 (0.50 s) | L = 20 (1.00 s) | L = 40 (2.00 s) |
|---|---|---|---|---|---|
| 10 (0.50 s) | 13653 (668) | 0.995 | 0.995 | 0.995* | 0.995* |
| 20 (1.00 s) | 13653 (668) | 0.992 | 0.992 | 0.994 | 0.994* |
| 30 (1.50 s) | 13653 (668) | 0.980 | 0.986 | 0.992 | 0.992* |
| 40 (2.00 s) | 13653 (668) | 0.987 | 0.988 | 0.992 | 0.993 |
| 60 (3.00 s) | 13653 (668) | 0.959 | 0.977 | 0.989 | 0.996 |
| 80 (4.00 s) | 13653 (668) | 0.902 | 0.930 | 0.963 | 0.995 |

### Channel-group ablation at k:L = 10:5 (GRU, 1 seed, val AUC)

| channels kept | val AUC |
|---|---|
| actions (steer, throttle, brake) | 0.7333 |
| body_vel (vx, vy) | 0.9546 |
| attitude (roll, pitch) | 0.8887 |
| ang_rates (roll_rate, pitch_rate, yaw_rate) | 0.9712 |
| omegas (omega_fl, omega_fr, omega_rl, omega_rr) | 0.9899 |
| engine (engine_speed) | 0.9856 |
| state12 (vx, vy, roll, pitch, roll_rate, pitch_rate, yaw_rate, omega_fl, omega_fr, omega_rl, omega_rr, engine_speed) | 0.9996 |
| no_engine (vx, vy, roll, pitch, roll_rate, pitch_rate, yaw_rate, omega_fl, omega_fr, omega_rl, omega_rr, steer, throttle, brake) | 0.9994 |
| no_omegas (vx, vy, roll, pitch, roll_rate, pitch_rate, yaw_rate, engine_speed, steer, throttle, brake) | 0.9991 |
| vx_only (vx) | 0.9154 |
| longitudinal (vx, pitch, throttle, brake) | 0.9613 |
| no_rates (vx, vy, roll, pitch, omega_fl, omega_fr, omega_rl, omega_rr, engine_speed, steer, throttle, brake) | 0.9995 |
| no_engine_brake (vx, vy, roll, pitch, roll_rate, pitch_rate, yaw_rate, omega_fl, omega_fr, omega_rl, omega_rr, steer, throttle) | 0.9994 |
| motion_no_actions (vx, vy, roll, pitch, roll_rate, pitch_rate, yaw_rate, omega_fl, omega_fr, omega_rl, omega_rr) | 0.9994 |

### Channel-group ablation at k:L = 10:10 (GRU, 1 seed, val AUC)

| channels kept | val AUC |
|---|---|
| actions (steer, throttle, brake) | 0.8522 |
| body_vel (vx, vy) | 0.9673 |
| attitude (roll, pitch) | 0.9011 |
| ang_rates (roll_rate, pitch_rate, yaw_rate) | 0.9901 |
| omegas (omega_fl, omega_fr, omega_rl, omega_rr) | 0.9975 |
| engine (engine_speed) | 0.9175 |
| state12 (vx, vy, roll, pitch, roll_rate, pitch_rate, yaw_rate, omega_fl, omega_fr, omega_rl, omega_rr, engine_speed) | 0.9998 |
| no_engine (vx, vy, roll, pitch, roll_rate, pitch_rate, yaw_rate, omega_fl, omega_fr, omega_rl, omega_rr, steer, throttle, brake) | 0.9998 |
| no_omegas (vx, vy, roll, pitch, roll_rate, pitch_rate, yaw_rate, engine_speed, steer, throttle, brake) | 0.9979 |
| vx_only (vx) | 0.9313 |
| longitudinal (vx, pitch, throttle, brake) | 0.9770 |
| no_rates (vx, vy, roll, pitch, omega_fl, omega_fr, omega_rl, omega_rr, engine_speed, steer, throttle, brake) | 0.9991 |
| no_engine_brake (vx, vy, roll, pitch, roll_rate, pitch_rate, yaw_rate, omega_fl, omega_fr, omega_rl, omega_rr, steer, throttle) | 0.9998 |
| motion_no_actions (vx, vy, roll, pitch, roll_rate, pitch_rate, yaw_rate, omega_fl, omega_fr, omega_rl, omega_rr) | 0.9995 |

### Channel-group ablation at k:L = 20:20 (GRU, 1 seed, val AUC)

| channels kept | val AUC |
|---|---|
| actions (steer, throttle, brake) | 0.9602 |
| body_vel (vx, vy) | 0.9897 |
| attitude (roll, pitch) | 0.9106 |
| ang_rates (roll_rate, pitch_rate, yaw_rate) | 0.9985 |
| omegas (omega_fl, omega_fr, omega_rl, omega_rr) | 0.9936 |
| engine (engine_speed) | 0.9913 |
| state12 (vx, vy, roll, pitch, roll_rate, pitch_rate, yaw_rate, omega_fl, omega_fr, omega_rl, omega_rr, engine_speed) | 0.9995 |
| no_engine (vx, vy, roll, pitch, roll_rate, pitch_rate, yaw_rate, omega_fl, omega_fr, omega_rl, omega_rr, steer, throttle, brake) | 0.9993 |
| no_omegas (vx, vy, roll, pitch, roll_rate, pitch_rate, yaw_rate, engine_speed, steer, throttle, brake) | 0.9995 |
| vx_only (vx) | 0.9800 |
| longitudinal (vx, pitch, throttle, brake) | 0.9946 |
| no_rates (vx, vy, roll, pitch, omega_fl, omega_fr, omega_rl, omega_rr, engine_speed, steer, throttle, brake) | 0.9997 |
| no_engine_brake (vx, vy, roll, pitch, roll_rate, pitch_rate, yaw_rate, omega_fl, omega_fr, omega_rl, omega_rr, steer, throttle) | 0.9991 |
| motion_no_actions (vx, vy, roll, pitch, roll_rate, pitch_rate, yaw_rate, omega_fl, omega_fr, omega_rl, omega_rr) | 0.9986 |

### Channel-group ablation at k:L = 60:40 (GRU, 1 seed, val AUC)

| channels kept | val AUC |
|---|---|
| actions (steer, throttle, brake) | 0.9652 |
| body_vel (vx, vy) | 0.9683 |
| attitude (roll, pitch) | 0.7191 |
| ang_rates (roll_rate, pitch_rate, yaw_rate) | 0.9975 |
| omegas (omega_fl, omega_fr, omega_rl, omega_rr) | 0.9445 |
| engine (engine_speed) | 0.9253 |
| state12 (vx, vy, roll, pitch, roll_rate, pitch_rate, yaw_rate, omega_fl, omega_fr, omega_rl, omega_rr, engine_speed) | 0.9998 |
| no_engine (vx, vy, roll, pitch, roll_rate, pitch_rate, yaw_rate, omega_fl, omega_fr, omega_rl, omega_rr, steer, throttle, brake) | 1.0000 |
| no_omegas (vx, vy, roll, pitch, roll_rate, pitch_rate, yaw_rate, engine_speed, steer, throttle, brake) | 0.9999 |
| vx_only (vx) | 0.8908 |
| longitudinal (vx, pitch, throttle, brake) | 0.9976 |
| no_rates (vx, vy, roll, pitch, omega_fl, omega_fr, omega_rl, omega_rr, engine_speed, steer, throttle, brake) | 1.0000 |
| no_engine_brake (vx, vy, roll, pitch, roll_rate, pitch_rate, yaw_rate, omega_fl, omega_fr, omega_rl, omega_rr, steer, throttle) | 1.0000 |
| motion_no_actions (vx, vy, roll, pitch, roll_rate, pitch_rate, yaw_rate, omega_fl, omega_fr, omega_rl, omega_rr) | 0.9998 |
