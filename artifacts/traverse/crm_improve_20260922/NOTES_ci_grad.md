# Module note: `scripts/ci_grad.py` (gradient route refinement, night-2 arm G, from a moving decision state)

2026-09-22 12:00-12:45. New files: `scripts/ci_grad.py` and this note; self-test outputs under
`artifacts/traverse/crm_improve_20260922/ci_grad_selftest/`. No existing repo file was edited (`git status`: only
`scripts/ci_grad.py` is new under `scripts/`); `ci_planner.py`, `f104_n2_grad.py`, `ga_planner.py`, `f104_n2_iter.py`,
`f104_n2_sampler.py`, `gen_planner.py` are imported. No Chrono, no cluster job. GPU: 0.54 GB peak, ~3 s per decision.

## What it is

The night-2 gradient refinement (arm G of `crm_night2_v1`, REPORT section 6) rebuilt for the new
history-conditioned ensembles and for a decision taken while the vehicle is moving. For one group it

1. reads the decision state exactly as `ci_planner` does (`ga_planner.decision_for` with the `--poses` entry: pose,
   history window npz, base route `gen_planner.base_route(pose, goal)`), and records the vehicle speed v0 with
   `ci_planner.decision_v0` (the candidate family is the unchanged free family, so v0 is recorded, not imposed);
2. builds the scorer (`ci_planner.CIScorer`; `ga_planner.Scorer` for `ga_train` / legacy members), which encodes the
   history context of every member ONCE per decision (16-d code, or the joint transformer's history tokens, plus the
   velocity context and the domain tag);
3. plans arm B = the CEM 4 x 64 pick with `ci_planner`'s own planning loop in process (`plan_iter_family`, family
   `free`, seed `IT.seed(group, 'n2iter_cem4x64')`), which gives the same pick as running `ci_planner.py --arms B`
   (self-test T2);
4. takes 17 starts: the B pick plus the best 16 routes of the CEM's round-0 pool (the designed anchors and the prior
   draws, ranked by the ensemble-mean logit; B itself is not repeated). Night 2 took the top 16 of the one-shot 256
   pool plus the lateral mirror of the argmin. Every start keeps its exact parameters: for samples and CEM means the
   route's own theta (3 lateral sine amplitudes + 4 speed knots on the base route's 2 m/s profile), for an anchor
   a = dv = 0 with the lateral profile `offset * sin^2(pi f)` and a constant cruise-speed base profile. The float64
   re-shape of those parameters reproduces each start route bit for bit (checked in every group, `start_reproduced`);
5. runs 60 Adam steps on all 17 starts in ONE GPU batch (lr 0.02 lateral / 0.10 speed, betas (0.9, 0.99), per-row
   gradient-norm clip 10, projection onto the curvature-derived amplitude caps and the +-4 m/s knot clip after every
   step, early stop when no row has improved for 15 steps) on the ensemble-mean route logit plus the night-2 penalties
   (curvature `1e5 * relu(kappa - 0.95 * 0.125)^2`, arena `10 * relu(|xy| - 37 m)^2`), through the differentiable
   corridor chain and the member networks with the history context held fixed. Keep-best per row by the pessimistic
   (max over members) score, `--keep mean` for the ensemble mean, 1e-3 hysteresis;
6. re-shapes each row's kept parameters in float64 with the family's own numpy functions (a row whose best iterate is
   its start keeps its start route object), checks them with the planner validator (`gen_planner.safe_validate`
   anchored at the pose: curvature, speeds, acceleration / deceleration, arena) and the route contract (start within
   0.25 m of the pose, end within 0.25 m of the goal, finite, speeds in [0, 6]), and re-scores the B pick, all 17 start
   routes and every valid final in ONE deployed-scorer batch (numpy corridors, float16 rounding, same ensemble and
   history context). G = the best valid final by the keep criterion when it beats B by at least `--abstain-logit`
   (0.3); otherwise G = B and the group is marked `abstained`.

### The chain is written deterministically

The torch mirrors of `f104_n2_grad` are re-implemented here so that no backward pass uses atomic adds:

- map sampling is the arithmetic of `f104_n2_dataset.sample_map` (cell index `floor` clipped to [0, n-2], fractions
  clipped to [0, 1], valid = all four corners above -1.999) with the map values gathered at detached integer indices,
  instead of `grid_sample` (whose backward accumulates with atomics);
- `np.interp` and the speed-knot lookup use one-hot products instead of `gather` / advanced indexing (their backward is
  a scatter-add);
- cuDNN runs in deterministic mode and attention uses the math kernel, over the forward AND the backward pass (the
  original `cudnn.flags` context only covered the forward, which is where the first version still drifted).

In float64 this chain equals `f104_n2_grad`'s own chain to 1e-13 in the corridor, the logits and the gradient (T1), and
two runs of the same command now give identical picks (T6). Before the change, two runs ended 0.1-0.2 apart in theta
and produced different routes. The global flags (used by the CEM and the deployed scorer) are never changed, so the B
picks stay bit-identical to `ci_planner`'s.

## Commands

    PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts OMP_NUM_THREADS=6
    cd /home/harry/NeDM-traverse_mppi
    K1=artifacts/traverse/generalist_20260921/A_adapt; K2=artifacts/traverse/crm_improve_20260922
    $PY scripts/ci_grad.py --selftest $K2/ci_grad_selftest          # T1-T6, 3 ensembles, 10 suite groups, 3 min 17 s

Full run, one ensemble, the 800 suite decision states of the 1 s approach (joint transformer X; swap the glob for the
GRU ensembles):

    $PY scripts/ci_grad.py --cases $K1/suite/cases --map-root artifacts/traverse/crm_f104_v1/map_root \
        --models "$K2/deploy_v1/deploy_a3_haux_txjoint_s*.pt" --world crm --domain crm \
        --poses $K2/a5data/decision_suite_L1/poses_crm_all.json --task-root $K2 \
        --out $K2/s4/picks_crm_XG --ref-b-picks $K2/s2/L1/picks_crm_X/picks

Measured cost per group (10 groups each, RTX 5090 shared, nothing else on the GPU): joint transformer 3.36 s median
(3.49 mean, 4.62 s worst = the first group's warm-up), GRU history ensembles 2.59 / 2.63 s median (2.72 / 2.76 mean).
So a full 800-group run takes about 45 min with the joint transformer and about 35 min with a GRU ensemble. Of that,
the CEM pick is 0.36-0.41 s, the 60 Adam steps 2.15-2.93 s and the float64 re-shape + validation + deployed re-score
0.05-0.06 s. GPU 0.54 GB reserved peak (0.48 GB for the GRU ensembles).

Useful options: `--steps` (60), `--starts` (17), `--lr 0.02,0.10`, `--keep pessimistic|mean`, `--abstain-logit 0.3`,
`--patience 15`, `--clip-grad 10`, `--groups` (comma list or `@file`), `--limit`, `--ref-b-picks DIR` (checks arm B
against another ci_planner / ga_planner pick set), `--world rigid` (adds arena / shard to the task rows).

## Outputs

`picks/<g>.json` (arms `B` = the CEM pick entry in ga_planner's format, `G` = route id, abstained, gain, z_mean,
z_pess, P, P_pess, per-member logits, which start it came from, the best step, the theta, the contract numbers and the
night-2 exploitation flags; plus the `family` block with v0 and the start speed / speed step / start heading of both
arms, the `grad` block with one row per start and the timings), `routes/<g>__G.json` (always; the B route when G
abstained) and `routes/<g>__B.json` (only when B differs from G), `tasks.json` (one row per group for arm G in
ga_planner's row format, so `ga_a5_pass2_tasks.py --arms <ARM>` reads it when the out dir is named
`picks_<world>_<ARM>`), `tasks_new_only.json` (abstained rows get `run: false`, `ref_arm: B`), `summary.json`,
`PICKS_LOCKED.sha256` (route file name + content, sorted, as ga_planner).

Checked downstream: the 10-group self-test output copied to `/tmp/pd/picks_crm_HG` and fed to
`ga_a5_pass2_tasks.py --world crm --arms HG --picks-dir /tmp/pd --branch-frame 20 ...` produced 10 pass-2 rows and
10 route files without a change to that script (dry run into /tmp; nothing was written to the artefact tree).

## Self-tests (`ci_grad_selftest/RESULTS.json`, all passed; 10 suite groups = every 80th of the 800, 3 ensembles)

Ensembles: `X` = joint transformer `deploy_v1/deploy_a3_haux_txjoint_s*.pt`, `Hn` = GRU history
`deploy_v1/deploy_a1_haux_gru_s*.pt`, `H` = K1 GRU `generalist_20260921/A_adapt/train/deploy_v1/H_deploy_s*.pt`.
The suite groups are evaluation data: they are used here only to check the machinery, nothing was tuned on them.

- **T1 chain and gradients** (2 groups x 4 rows per ensemble: the B pick, the best anchor, the two best samples).
  - float64 re-shape vs the torch route: waypoints 3.6e-15, speeds 1e-6 (the terminal `sqrt(clamp_min(1e-12))`);
    corridor vs `f104_n2_iter.corridors` (the deployed numpy builder) 1.2e-7 elevation, 1.5e-8 grade / cross-slope,
    1e-6 speed channel; in float32 1.1e-5 m on the waypoints and 2.7e-5 in the corridor.
  - autograd vs central differences (float64, h 1e-5) of the ensemble-mean logit in the 7 parameters: at a generic
    point near each start the cosine per row is 0.999992-1.0 (worst of all 24 rows 0.999992), the median component
    error 6.6e-10 to 9.3e-8, the worst row's relative gradient-norm error 6.6e-3. At the start parameters themselves
    the anchors sit on kinks of the family (speed clamped at 6 m/s, zero amplitudes), where one-sided and central
    differences must differ: cosine 0.948-0.996 for those rows, 1.0 for every sample row.
  - torch networks vs the deployed scorer on identical float16-rounded corridors (one 64-route batch): 4.8e-6 max
    over members in float32. The same comparison in float64 gives 6e-3 to 1.9e-2 because the deployed scorer's
    convolutions run in TF32; for scale, the deployed scorer's own logits move by 5.7e-3 to 1.1e-2 when the same
    routes are scored in 4-row instead of 64-row batches (cuDNN / TF32 algorithm choice). This is why B, the starts
    and the finals are re-scored inside one batch, and why the 0.3 abstain margin is far above the scoring noise.
  - deterministic chain vs `f104_n2_grad`'s own chain (float64): corridor 1.2e-14, logits 3.4e-13, gradient 2.8e-13.
- **T2 `--steps 0`** (10 groups per ensemble, `--keep mean`): every row keeps its start (`n_refined_rows` 0), every
  start reproduces its route, G equals B in 10/10 groups and abstains, and B equals the existing `ci_planner --family
  free --arms B` pick of the same ensemble and the same decision states (`s2/L1/picks_crm_{X,Hn,H}/picks`): route
  sha256 and z_mean equal in 30/30 group-ensemble pairs, and the route file arrays are equal to that set's
  `routes/<g>__B.json`.
  - T2p is the same run with the default `--keep pessimistic`: G still differs from B in 1 (X), 1 (Hn) and 2 (H) of
    the 10 groups at zero steps, because B is the argmin of the ensemble MEAN while G is chosen by the worst member,
    and a round-0 start can be better by more than 0.3 on that criterion (gains 0.48-1.21). Expected, not a failure -
    but it means "G changed the pick" is not by itself evidence that the gradient did anything; `best_step` and
    `refined` in the pick file say whether the route was actually moved.
- **T2b** `ci_planner.plan_decision(arm='B', family='free')` in process, 10 groups x 3 ensembles: same route sha256 and
  the same z_mean as this module's arm B, 30/30.
- **T3 keep-best** (60 steps, 10 groups x 3 ensembles, 510 start rows): the kept iterate's torch objective is never
  above its start's (max difference 0.0); re-scored through the deployed pipeline in the same batch, no kept final is
  worse than its own start (max change 0.0000, median -1.8 to -4.6 logit); and G is never worse than B in the keep
  criterion by construction (abstain otherwise).
- **T4 validity** (the 30 G routes): all pass the planner validator and the contract - start distance to the pose
  0.0 m, end distance to the goal <= 3.6e-15 m, curvature <= 0.121 (limit 0.125), speeds <= 6.0, acceleration within
  [-2.0, 1.5], the route file's arrays hash to the pick's `route_sha256`. v0 came from the recorded history window in
  30/30 (`history_window`), and its cross-check against the pass-1 recording was 0.0 in every group. The free family
  does not tie the start speed to v0: the picked G routes start 0.6 m/s below to 4.3 m/s above the vehicle speed
  (median +2.2) and up to 51 deg off the vehicle heading, as the free-family B picks do.
- **T5 cost**: see the numbers above; `--ref-b-picks` matched 10/10 in every run.
- **T6 determinism**: three groups replanned with the same command give the same G route sha256 in 3/3 for each
  ensemble (and the same J trace step by step).

## What the refinement does on these 10 groups (diagnostic only, 10 evaluation groups)

| ensemble | picks changed | abstained | gain (logit, median / max) | mean z (B -> G) | mean P (B -> G) | route time B -> G |
|---|---|---|---|---|---|---|
| X joint transformer | 2 / 10 | 8 | 0.07 / 5.15 | -7.75 -> -8.04 | 7.7e-4 -> 3.3e-4 | 14.8 -> 13.3 s |
| Hn GRU history | 8 / 10 | 2 | 1.75 / 10.39 | -10.69 -> -12.30 | 2.4e-3 -> 1.2e-5 | 18.5 -> 16.2 s |
| H K1 GRU | 10 / 10 | 0 | 2.54 / 6.79 | -9.07 -> -10.59 | 6.9e-4 -> 6.2e-5 | 19.4 -> 17.8 s |

The joint transformer rates its own CEM picks at z about -8 (P 3e-4) and the refinement cannot go below that on 8 of
10 groups - its logit looks floored, where the GRU ensembles keep going down to -12 or -13. Whether that floor is
calibration or saturation is a question for the closed loop, not for this module.

## Known limits / things a reviewer must know

- Only the `free` family is supported (night-2 arm G's family). The speed-continuous families of `ci_planner` (`cont`,
  `cont_head`) are not implemented here; E2 refuted the handover hypothesis, so they were not needed. Adding `cont`
  would need a torch mirror of the speed transform inside the chain.
- The pick is only as good as the model: `--keep pessimistic` optimises the ensemble mean but selects on the worst
  member, so it can pick a route that most members rate worse than B. On these 10 groups that happened twice with the
  K1 GRU ensemble (`flags.minority_members`: 2, e.g. 1 of 5 members improved while the worst member improved by 1.3).
  `--keep mean` selects on the mean instead. The flags in each pick (`mean_worse`, `minority_members`, `lateral_clip`,
  `dv_clip`, `fast`, `edge`) are diagnostics only; nothing is filtered on them.
- The refinement is free to push the lateral profile onto its +-10 m clip and the speed onto 6 m/s (one of the 30 G
  routes here is at the lateral clip, one has a mean speed above 5 m/s). The validator and the contract still hold,
  but these are the routes night 2 flagged as exploitation candidates.
- Starts come from the CEM's round-0 pool (64 candidates: the valid anchors plus prior draws), not from a one-shot 256
  pool as in night 2. They are already scored by the CEM, so the extra starts cost nothing.
- Early stopping never triggered on these groups (all 10 x 3 runs used all 60 steps), so the cost per group is the
  full 60 steps.
- The deployed re-score and the torch chain differ by the float16 corridor rounding and by TF32 convolutions (about
  5e-3 in logit); the torch objective is what the optimiser sees, the deployed numbers are what the pick file reports.
- No route of this module has been driven. Nothing was written outside `artifacts/traverse/crm_improve_20260922/` and
  `scripts/ci_grad.py`, and no cluster job was submitted.
