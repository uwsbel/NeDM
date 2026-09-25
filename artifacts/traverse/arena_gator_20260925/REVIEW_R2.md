# Review R2 of PLAN.md: feasibility and engineering (2026-09-25, 01:30)

Reviewer role: can the plan run tonight as written, on this code and this cluster? I checked the code, the cluster
(read-only: `squeue`, `sinfo`, `scontrol`, file hashes) and ran one local plumbing smoke of the planner with
untrained models (scratch outputs in `/tmp/r2_review/`, nothing in the repository was edited). E1 had already added the
new suite patterns to `ci_train.py`, `ga_build_mixed.py`, `ci_a5data.py` and four arenas to `gen_arenas.json`; I read
the files in that state and changed nothing.

**Verdict.** The pipeline pieces the plan relies on all exist and mostly work as assumed. Two things are wrong or
missing in the plan and would bite silently: (1) the soil worker picks ONE collector per job, so "HMMWV rows unchanged,
Gator rows switched by extra arguments" only works through a dispatching collector whose file name contains
`crm_collect`; (2) `ga_planner.py --fixed2` cannot load `ci_train` checkpoints. The real risk is time: the soil work in
the plan is about 190-240 simulated hours of collection plus about 55 hours of evaluation drives on a GPU pool that
delivers about 15-17 simulated hours per wall hour today. The collection alone runs to 13:30-18:30 if nothing is cut,
and results arrive in the late afternoon even with the 12:00 cut.

## 1. What the code and the cluster actually do

### 1.1 Soil worker: claims, relaunch, stop file, per-row options (`scripts/crm_worker.py`)

- The task file is read **once** when a job starts (`:30`). A running job never sees rows added later.
- Order: each worker sorts its (shuffled) rows by `tier` (`:79`), so all of tier 0 before tier 1, and so on. Rows
  without `tier` count as tier 0. Negative tiers are allowed and run first.
- A claim is a directory `<OUT>/claims/<id>`; a run is complete when `<OUT>/runs/<id>/episode_complete.json` exists.
  Claims are never removed after success, and completed ids are skipped. So **a relaunch with an extended file does
  not duplicate episodes, provided it uses the same `CRM_OUT` and the ids are unique and unchanged.** A different
  output folder would re-run everything.
- A job killed mid-episode leaves its claim; another worker can take it only after `CRM_STALE_S` = 1,500 s (`:23`).
  The killed episode restarts from scratch (the collector deletes the partial files, `crm_collect.py:408`). A killed
  job records no failure.
- `STOP_CLAIMS` is one file per output folder (`:84`, `:148`): every job on that folder, old or new, stops claiming,
  finishes its current episode and exits. It stays in force until deleted, so new jobs exit at once if it is still
  there. It cannot stop one arena or one vehicle. To drop a class of rows you have to write a new file with those
  rows set to `run: false` and relaunch.
- **The collector is per job, not per row**: `CRM_COLLECTOR` from the environment (`:91`). Per row the worker
  appends `extra` after its own `--source-root/--case/--route/--out/--chrono-data/--horizon-s` (`:94`). Then it adds
  `--episode-seed` and, **only if the collector path contains the text `crm_collect`**, `--crm-config` (`:98`).
  Consequences:
  - `extra` can override the worker's own options, because argparse keeps the last value. That includes
    `--source-root`. It cannot override `--crm-config` or `--episode-seed`, which come after it.
  - A per-row `config` works too, but only under the same file-name rule.
  - A Gator collector named, for example, `ag_soil_collect.py` would silently run with the built-in defaults: a
    0.5 ms step instead of the production 1 ms (`crm_collect.py:38`). That means twice the cost and different
    physics from the HMMWV.
- Failures: each failure counts as one attempt (maximum 2, `:24`), and three failures in a row stop that worker for
  the rest of the sweep (`:131`). Soil runs are bit-identical on repeat, so a failure that depends on the Gator
  itself (for example a failed launch check) repeats on retry and wastes GPU time.
- The HMMWV soil collector on the cluster that produced `collect_v1`
  (`crm_f104_20260916/source/scripts/crm_collect.py`, sha `26634a90`) differs from the repository copy (`cb6792be`,
  the one in `crm_improve_20260922/source`) only by API-name fallbacks and optional visualisation hooks. The two
  behave identically. The repository copy has `main(argv=None, ...)`, which makes a dispatching wrapper easy to
  write. `crm_collect.py:436` records the sha256 of `crm_collect.py` itself, **not** of any wrapper.
- `crm_collect.sbatch:11` fixes `CRM_ROOT` to `crm_f104_20260916`, whose `source/assets/traverse` holds only f104.
  `crm_launch.sh` submits 47 array tasks (`:15-21`). If a submission fails it prints "submit failed" and carries on
  (`:12`). It uses `--export=ALL`, so a `CRM_COLLECTOR` left over in the submitting shell is silently inherited.

### 1.2 Queue cap and GPUs now (01:20-01:29)

- My queue is empty. The 50-task limit could not be re-read: the submit filter is not readable. The only evidence is
  the record in the memory note. Treat it as real.
- Idle GPUs: mi2101x 13 nodes (16 at 01:20, others are taking them), mi2104x 7 nodes x 4 = 28 MI210, mi3501x 7 x
  MI350X (one node drained). That is **about 48 GPUs, 41 of them MI210.**
- Every 8-GPU node is allocated and has a queue of other users' jobs. Another user has 12-hour arrays on mi2101x
  waiting on dependencies, and these can take nodes at any time.
- PyTorch 2.10 crashes on MI210 compute nodes (memory note), so **training can only use the MI350X nodes (mi3501x,
  4-hour cap), the same nodes the soil launch wants.**
- Nodes are allocated whole. Rigid CPU jobs on mi2104x or mi2101x take whole nodes away from soil. A mi2104x node has
  128 cores. Soil uses about 8 threads per GPU (4 OpenMP + 4 multibody), so about 32 cores, and about 90 cores sit
  idle during soil runs.

### 1.3 Training: `ci_train.py --cond none --domain-filter crm|rigid` on `ga_build_mixed` files

Works.
- With `cond none` and `ctx geom`, no history window is loaded; only the required keys are needed.
- The world filter keeps rows by the `domain` array (`:290-291`). `--mode deploy` fits every `train` row and needs at
  least one `val` row.
- The fit groups are checked against the suite patterns, which now include the new suites.
- Local smoke (synthetic file, 1 epoch, 2 seeds): both worlds train and write checkpoints.
- Precedent: the same model type on the same kind of rows was trained in the generalist effort with
  `ga_train.py`, not `ci_train.py` (`Sp_crm_deploy`, 52,173 soil rows fitted). Its offline start-up ranking was
  0.987. It was **never driven from a standing start.**
- Note: the re-anchored files have no `domain` array and no `@crm/@rigid` id suffix, so they cannot go into
  `ci_train` directly. `ga_build_mixed.py` (or a small new tool) must add them.

### 1.4 Planner: `ci_planner.py --family free` with these models, standing start

Works (smoke with the untrained models above, two f104 suite groups):
- soil world, arm B: CEM 4 x 64, start from the case pose at rest, no history, 0.54 s per group;
- rigid world with `--fixed2`: picks at 2.0 m/s mean, 0.54 s per group.

`ci_planner.py` passes `--fixed2`, `--arena-tag` and `--shards` through to `ga_planner.main`.

**`ga_planner.py --fixed2` called directly fails** for these checkpoints: `ValueError: model_kind 'ci_train' is not
loadable here` (`ga_planner.py:228`). The fixed-2 read-out must go through `ci_planner.py --family free --fixed2`.

Do not use `--family cont*` from a standing start. It starts routes at 0 m/s, and the vehicle would not move
(`ci_planner.py` docstring).

Planner task rows set `tier` = group index (`ga_planner.py:542`). That clashes with collection tiers if these rows
are merged into a shared soil task file.

### 1.5 Rigid per-node determinism

`gen_runner_g.py` runs every row of one `shard` in one array task, on one node, and the planner writes
`shard = md5(group) % --shards`. So paired rigid arms share a node **only if** all arms of a group are in one task
file, use the same `--shards`, and go out in one submission.

Arms added later go out as a new submission and may land on another node, where roughly 5 % of marginal outcomes can
flip: M1b/M3b "if time allows", or a straight arm built by another script.

The rigid runtime-fingerprint gate only checks that the listed file names include HMMWV data
(`gen_collect_ext.py:545`). A Gator run passes with the HMMWV fingerprint, but its provenance record is then wrong.

### 1.6 Dataset chain for a new arena

`f104_n2_dataset.py` -> `n2_reanchor_dataset.py` -> `ga_build_mixed.py` works for a new arena as long as:
- **One map per call and no cross-check:**
  - The builders take one `--root` per process and use every run dir the glob matches.
  - Nothing compares the map's `arena_bmp_sha256` (now present in every new `observation.json`) with the case's arena.
  - With g203, g228 and Gator runs in one soil output folder, a loose glob labels runs with the wrong map and nothing
    fails.
- **Run ids:**
  - Profiles are parsed as `int(rid.split('_route_')[1])` (`f104_n2_dataset.py:119`, `n2_reanchor_dataset.py:73`).
  - On-policy runs are recognised by `'_op_' in rid` (`crm_qa.py:31`).
  - A vehicle tag must therefore be a **prefix** of the id, not a suffix. Gator ids must differ from every other id
    in the same output folder.
- **`ga_build_mixed.py`:**
  - It needs both worlds with identical key lists (`:107`).
  - It needs every raw run locally, including `crm_extra.npz` for soil.
  - It exits with code 2 but still writes the file when runs are missing (`:194`), so check the exit code.
- **Rows behind the f104 models:**
  - The existing f104 re-anchored and mixed files hold only the 15,024 twin episodes per world, 12-13 routes per
    group. They do not hold the 24,000-route rigid pool.
  - The plan's rigid design gives new arenas 20 routes per group. Its Gator rigid set is the 24,000 pool.
  - So "M1 rigid", and "H" as the HMMWV partner of the Gator model, are undefined until someone either rebuilds f104
    rigid from all 24,000 runs or cuts everything rigid to the soil tier range.
  - All 14,400 + 9,600 f104 rigid runs are local (1.2 + 1.3 GB), so the rebuild is possible.
  - The 211 soil ids that lacked a rigid twin can now be paired too.
- **Re-anchoring flags:** `n2_reanchor_dataset.py` defaults (`--anchors 4 --min-remaining 12`) must match the flags
  behind the f104 files. The night-2 report describes "standing start plus up to 3", which is consistent with the
  defaults.
- **The 95.8 % reference:** it belongs to the older soil ensemble `crm_f104_v1/train_v1/deploy/CRM_N2_s*.pt`. That
  ensemble was trained by `crm_train.py` on standing-start rows only (13,821 fitted) and planned with
  `ga_planner.py`. **It is not "this protocol".** M1 as specified is a new recipe that has never been driven; the
  800-group HMMWV run in task B becomes its first closed-loop anchor.

### 1.7 Timing (recomputed)

- **Soil collection:**
  - g203 + g228: 2 x 606 groups x ~12.7 routes ≈ 15,400 drives, about 92-100 simulated h at f104's 21.6 s mean.
  - Gator: 15,235 drives, about 91-137 h at 1.0-1.5 x the HMMWV length.
  - Pilots and the dev check: about 3 h.
  - Total: **about 186-240 simulated h.**
- **Soil evaluation:** about 11,000-12,000 drives x ~17 s ≈ 52-57 h.
- **Pool:** about 41 MI210 x 0.32 + 4-7 MI350X x 0.5 ≈ 15-17 simulated h per wall hour, before per-episode
  overheads.
  - Collection alone: **11-16 wall hours** (02:30 start -> 13:30-18:30).
  - With the 12:00 cut: about 150 h collected, i.e. 65-80 % of the target, about 8-10 of the 12.7 routes per group.
  - Then data building and training (~1.5 h), picks (~0.5 h), soil evaluation (3-4 h) and analysis.
  - **Final numbers around 17:00-19:00.** Only a freed 8-GPU partition shortens this.
- **Rigid CPU:** one 24,000-route pool ≈ 14,400 x 85 s + 9,600 x 160 s ≈ 770 CPU-hours.
  - Three pools (g203, g228, Gator) ≈ 2,300 CPU-hours ≈ 18 mi2104x node-hours on the same nodes as soil.
  - That is about 70 MI210 GPU-hours ≈ 23 simulated soil hours ≈ 1.5 wall hours of the whole pool, unless the rigid
    work is packed onto spare cores.
- **Training** (MI350X only):
  - About 14 ensembles x 5 seeds.
  - Soil ensembles ~50-60k fitted rows (~3-5 min per seed).
  - Rigid A3 ~250-270k rows (~15-17 min per seed).
  - About 8-10 MI350X-hours in total. The soil ensembles sit on the critical path right after the cut.
- **Billing:** about 100-125 node-hours, under the soft cap. Budget does not bind; wall time does.

### 1.8 What is most likely to fail tonight (in order)

1. **Soil wall time.** Near certain: the collection does not finish by 12:00, and the evaluation runs into the
   afternoon.
2. **The Gator soil path inside the shared job.**
   - A collector name without `crm_collect`: silent 0.5 ms step.
   - A wrapper that patches HMMWV rows too.
   - A wrapper edited while jobs run: every episode re-reads the file, and the change is not recorded.
   - Gator-specific launch or breakthrough failures that retire workers under the three-in-a-row rule.
3. **Slot and node conflicts.** A 47-task soil launch plus rigid shards plus pilots plus training goes over 50 tasks,
   and the failed submission is only printed. Rigid jobs block mi2104x/mi2101x. Soil on mi3501x blocks training for
   up to 4 h.
4. **A silent map/arena mismatch** in a builder or planner call (seven arenas, one shared soil folder, no check).
5. **Undefined f104 rigid rows** (twin rows vs 24,000 pool) for M1 and H, so "matched" rigid designs would differ in
   routes per group.

## 2. Amendments, ranked by importance

1. **Soil collector: one dispatcher, named so the config arrives.**
   - One collector serves all soil rows: `scripts/ag_crm_collect.py`. The name must contain `crm_collect`.
   - Rows without a vehicle flag (for example `extra: ["--vehicle", "gator"]` absent) call the unchanged
     `crm_collect.main(argv)` with **no patching at all**. HMMWV rows are then the frozen code by construction; keep
     the planned one-episode bit-identity check.
   - The wrapper strips its own flags before delegating.
   - It asserts `cfg["step_s"] == 0.001`, i.e. the config really arrived.
   - It writes its own sha256, the vehicle and the wheel geometry into each Gator run, because `crm_collect.py` records
     only its own hash.
   - Freeze it (read-only, sha recorded in NOTES) before production rows run. Every episode re-reads the file.
   - Set `CRM_COLLECTOR` explicitly in `--export` (never rely on `--export=ALL` inheritance). Use a new
     `ag_crm_collect.sbatch` if `CRM_ROOT` is G3.
   - Same rule for rigid: `gen_runner_g.py` also takes one `GEN_COLLECTOR` per job. HMMWV and Gator rigid rows go out
     as separate submissions, or through the same kind of dispatcher.
2. **Face the timeline now; do not discover it at 12:00.**
   - Launch the g203/g228 soil rows as soon as the tree is staged. Cases and on-policy routes already exist; do not
     wait for 03:00 or for the rigid waves.
   - Put **no row above the target tier** in any file (tiers 0-12 only), so nothing runs past the target.
   - Write the cut rule into the plan: at the cut, drop the last, partly filled tier on every arena and on the Gator.
     Matched designs then use complete tiers, and M1 is trained after the cut on the same tier range.
   - Declare now what gets dropped next if 12:00 is still too late, and in which order: second ensembles, the
     g203/g228 in-distribution arms, rigid speed-free arms.
   - Tell the user results come in the late afternoon, not in the morning.
3. **Cluster slots and nodes: a sized launch, not `crm_launch.sh`.**
   - Write an `ag_` launcher with arrays sized to the idle nodes: mi2101x ≤ 13-16, mi2104x ≤ 7, mi3501x ≤ 3-4, at
     most one task per 8-GPU partition. About 25-30 tasks, leaving about 20 for rigid shards, pilots and training.
   - Keep **at least 3 MI350X nodes out of soil** from the start. Training cannot run on MI210, and soil tasks on
     mi3501x hold the node for 4 h.
   - Check that each `sbatch` returned a job id, and count tasks before every submission.
   - Resubmit the mi3501x soil tasks every 4 h only if training does not need the nodes.
4. **Stop the rigid collections from starving soil.** Pick one of these (the first is simplest):
   - (a) Cut every rigid training pool to the soil tier range: tiers 0-12, about 15,600 routes per new arena. The
     Gator's rigid set becomes the soil ids. This saves about 35 % of rigid CPU and gives rigid and soil the same
     routes per group (see 5).
   - (b) Run rigid CPU workers on the idle ~90 cores of each soil node inside the same job. Precedent: the rigid run
     driven through `crm_worker.py` with `CRM_NO_GPU_BIND=1`, job 425329. Rigid needs its own environment (plain
     Chrono build, software Vulkan renderer, `FDM_RUNTIME_FINGERPRINT`). Check that the soil real-time factor does
     not drop.
   - Either way, do not put rigid shards on mi2104x/mi2101x while soil tasks are waiting for those nodes.
5. **Pin down the f104 rows for M1 and for H.**
   - Decide in the plan which f104 episodes and tiers M1 and H use in each world. Today's re-anchored f104 files are
     twin-only (12-13 routes per group).
   - If rigid keeps 20 tiers anywhere, rebuild f104 rigid from all 24,000 local runs with the same re-anchoring
     flags. H (HMMWV) and G (Gator) must use the same ids and routes per group in each world.
   - Rebuild the twin id list: all 9,600 on-policy rigid runs now exist, so the twin set can be 15,235.
6. **Guard against wrong maps and mixed arenas.**
   - Add a small check, run before every dataset-builder and planner call: each case's arena BMP sha256 must equal
     the map root's `observation.json` `arena_bmp_sha256`, and a call may contain only one arena.
   - Glob runs by arena prefix (`runs/g203_v2_group_*_route_*`, ...).
   - Give Gator ids a **prefix** (e.g. `gator__f104_v2_group_0001_route_03`), never a suffix. The id parsers need the
     trailing `_route_NN` / `_op_NN`.
7. **Fixed 2 m/s through `ci_planner.py`.**
   - Replace "`ga_planner --fixed2`" with `ci_planner.py --family free --fixed2 --world rigid --arms B` (verified).
   - Speed-free runs use `ci_planner.py --family free --world crm|rigid --arms B`.
   - One call per arena, each with that arena's `--map-root`, `--cases`, `--arena-tag` and a fixed `--shards`.
8. **Write the relaunch recipe into the plan.**
   - Always the same `CRM_OUT`.
   - The extended file goes to a new path, and the old file stays locked.
   - Cancel only the **pending** tasks of the old launch, then submit the new ones. Running tasks drain their old rows
     and exit.
   - Delete `STOP_CLAIMS` before relaunching.
   - Expect a delay of up to 25 min on episodes of killed jobs.
   - Give priority rows (Gator pilot, dev headroom drives, evaluation drives) negative or low tiers in the same file,
     so they need no separate launch.
   - Rewrite the planner's `tier = group index` when merging planner rows.
   - Put Gator rows into the shared file only after the soil pilot shows fewer than about 5 % failed or rejected
     launches. Deterministic Gator failures otherwise retire GPUs (three in a row) and are retried to no purpose.
9. **Correct the reference claim and name the headroom model.**
   - The 95.8 % belongs to the older standing-start soil ensemble (`CRM_N2_s*`, `crm_train.py`, standing rows only),
     planned with `ga_planner.py --world crm --arms B`. It is not the ci_train re-anchored recipe.
   - Use exactly that ensemble and command for the g217 headroom check, so it reads like for like.
   - Say in the plan that M1 is a new, never-driven recipe, anchored by the task-B HMMWV run on the 800 groups.
10. **Plan the training capacity and skip the raw-run round trip.**
    - About 8-10 MI350X-hours in total. Train the rigid ensembles as soon as rigid data exists. Run the soil ensembles
      in parallel on the reserved MI350X nodes (several processes per GPU are fine; 288 GB) right after the cut.
    - Build the per-arena dataset files on the cluster. The builders need only numpy and the runs are already there.
      Sync only the dataset files and a QA sample back, instead of pulling ~10 GB of small files before training.
    - For `cond none`, `ga_build_mixed.py`'s history windows are not needed. Its only role is to add `domain` and the
      id suffix. Check its exit code.
11. **Rigid evaluation pairing and provenance.**
    - All rigid arms of an arena set go in one task file with one `--shards` and one submission.
    - Any arm added later (second ensembles, extra straight arms) re-drives its reference arm in the same job, or
      carries the cross-node caveat.
    - Write a Gator runtime fingerprint that lists `data/vehicle/gator/*`. The existing gate passes on the HMMWV file,
      but the provenance would be wrong.
12. **Report per arena and per vehicle.**
    - `crm_qa.py` pools a whole output folder, so run it per id prefix, or post-filter its rows, for validated hours
      per arena and vehicle.
    - Record in NOTES which cluster source tree ran which rows. The f104 collection used the `crm_f104_20260916` tree;
      new rows will use G3 or that tree with the new arena folders copied in.
