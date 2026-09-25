# Overnight 2026-09-25: more training arenas (task A) and the Gator on f104 (task B)

Branch `arena_gator_v1` (from `crm_improve_v1` 4bc44386d). Local root K3 = `artifacts/traverse/arena_gator_20260925/`.
Cluster root G3 = `/work1/dannegrut/harry/experiments/arena_gator_20260925`. Scout reports: `scout/S1..S4*.md`.
Earlier roots are read-only: `fdm_f104_50h_20260909`, `crm_f104_v1`, `crm_night2_v1`, `generalist_20260921`,
`crm_improve_20260922`.

## 0. Questions and the answer each part must give

- **A.** The f104 results show that one arena plus enough data gives a good planner on that arena. Does training on
  more arenas (1 -> 2 -> 3) make the planner better on arenas it has never seen? Answer separately for **more arenas
  at the same total data** (diversity) and **more arenas with more data** (what collecting on new arenas buys), in
  soil (primary: rigid speed-free goal reaching is at its ceiling) and on rigid ground (fixed 2 m/s, where gen_v1
  found a generalisation gap).
- **B.** With the Chrono Gator instead of the HMMWV on f104, can the same amount of data be collected (same task
  ids), does it train a planner that works, and how does it compare with the HMMWV planner and with the frozen
  HMMWV planner driven by the Gator?

## 1. Fixed decisions (made without the user; reasons in the scout reports)

1. **Model and protocol for both tasks: per-world specialists from a standing start.** `ci_train.py --cond none
   --domain-filter crm|rigid`, CNN-GRU (`--arch gru`), `--ctx geom`, 5 seeds x 30 epochs, trained on the re-anchored
   rows (standing-start k = 0 plus every 2 s) of each arena/vehicle. Planning: CEM 4 x 64 from the case pose at rest
   (arm B of `ci_planner.py --family free`), picks hashed before driving. Reason: the moving-start tooling is
   hard-wired to f104 and doubles the drives; the question is about data, not the decision protocol. The HMMWV f104
   reference at this protocol exists (soil specialist 95.8 % on the 800-group suite). Every arm is retrained with the
   same recipe; no old checkpoint is an arm (the frozen K1 soil specialist is used only for the dev-arena headroom
   check).
2. **Map input unchanged:** one OptiX overhead depth capture per arena, same camera contract, flat-ground lookup (as
   every f104 model). Its error grows with relief (0.050 m f104, 0.062 g203, 0.079 g231, S1 2.1): recorded as a caveat.
3. **Arenas.** Training: f104 (T1), g203 (T2), g228 (T3) (closest siblings, captures exist). Dev: g217 (headroom check
   and any tuning only). Test: 4 never-used arenas from new generator seeds 241-280 at difficulty 1.0, the 4 closest
   to f104 by the gen_v1 8-statistic distance, chosen by a script before anyone looks at them.
4. **Data per arena = the f104 episode design** (S3 1.5): 1,200 groups from `gen_cases.py --strata all`, 12 designed +
   8 on-policy routes per group (`f104_n2_onpolicy.py`), tiered by a per-group shuffle, `episode_seed = md5(id)`,
   splits 90/5/5 by group hash, HMMWV collector code identical to the f104 collection.
   - Rigid: all 24,000 per new arena (cheap).
   - Soil: the first **606 groups** of each new arena (>= 545 training groups), tiers in order, stopped at the tier
     count f104 reached (12-13 routes per group). This supports the matched designs below and a partial additive one.
5. **Gator = Chrono's stock `veh.Gator`** with only the changes it needs to run in our loop: spawn +0.35 m (settles;
   +0.75 m is still bouncing at the anchor), vehicle provenance recorded, soil wheel geometry = one cylinder per axle
   with a radius calibrated so the settled soil sinkage matches the HMMWV's (-0.02..-0.04 m) (its tyre meshes are not
   watertight: 94/116 markers vs 207), chassis not coupled to the soil (as the HMMWV) but the lowest hull point vs the
   surface logged. Stock driveline (limited-slip, rear-wheel drive), stock brakes (no locking), TMEASY on rigid,
   frozen follower gains, unchanged route limits, labels and stop rules. Deviations must be listed in the report.
6. **"Same amount of data" for the Gator = the same task ids**: soil = the 15,235 HMMWV `collect_v1` ids; rigid = the
   24,000-route night-2 pool (1,200 `f104_v2` groups x 20; the rows behind every current rigid model). Simulated hours
   and rows are reported as outcomes.
7. **Budget:** soft cap 150, hard stop 180 billed node-hours for this session (861 left in the allocation; scout
   estimate ~105-120). Soil wall time is the constraint (~18 simulated h per wall hour on the ~51 free GPUs, ~36 with
   the 8-GPU nodes). If collection is not done by 12:00, stop at the current tier and continue with what exists.
8. One soil task file (queue cap 50 tasks per user): all soil training tasks in one list, interleaved by tier
   (tier k of arena g203, g228 and the Gator before tier k+1), so a partial run stays balanced. Relaunching with an
   extended file is allowed (claims are per id). HMMWV rows use the unchanged collector code; Gator rows carry the
   vehicle switch in their `extra` arguments.

## 2. Task A design

### 2.1 Training sets (per world; nested, fixed data seed; counts are training groups)

| model | arenas | groups | question |
|---|---|---|---|
| M1 | f104 | 1,089 | today's recipe on one arena |
| M2 | f104 + g203 | 545 + 545 (=1,090) | 2 arenas, same total |
| M3 | f104 + g203 + g228 | 363 x 3 (=1,089) | 3 arenas, same total |
| A3 | f104 + g203 + g228 | 1,089 + all g203 + all g228 (soil ~545 each; rigid 1,089 each) | more arenas **and** more data |

Matched subsets keep the same tier range on every arena (if a new arena stops at tier k, f104 is cut to tiers < k as
well, for M1 too, recorded). Rigid A3 is 3 x 1,089. Soil A3 is 1,089 + ~545 + ~545. Second independent 5-seed
ensembles for M1 and M3 (seeds 5-9) are trained; driven if time allows (training-noise floor).

### 2.2 Test suites (locked with sha256 before any model exists)

- **Unseen:** 4 test arenas x 250 groups, `gen_cases.py --strata feature` (hill/crater, where headroom is), new
  declared seeds, ids `<arena>_test_group_*` (blacklisted from every builder). 1,000 groups per world.
- **In distribution:** f104: the 200 `f104_crm_eval_group_*` (feature strata, already blacklisted); g203, g228: 150
  feature groups each, `--avoid` the arena's training cases (2 m), ids `<arena>_heldout_group_*`.
- **Dev (g217):** 150 feature groups; the frozen K1 soil specialist and straight 6 m/s are driven on soil early as a
  headroom check (does an f104-only model lose points on an unseen arena?). Result reported, not used to change the
  primary unless there is no gap at all (then the rigid fixed-2 read-out and hill-only strata carry the weight).

### 2.3 Arms and read-outs

Soil (speed free): M1, M2, M3, A3, straight route at 6 m/s. Rigid: the same four models speed free (no-harm) and at a
fixed 2 m/s (`ga_planner --fixed2`, geometry-only CEM), plus the straight route at 2 m/s. All arms of a group plan
from the same pose; identical picks driven once; rigid arms of a group in one array task (per-node determinism).

**Primary (soil, pooled over the 4 test arenas, fail = not goal reached):**
- P1: M3 vs M1 (diversity at matched data). P2: A3 vs M1 (more arenas + more data). Holm over P1, P2. Decision: the
  one-sided 95 % lower bound of the paired group-bootstrap improvement > 0, exact McNemar reported.
- Robustness: bootstrap over (arena, nearest feature) clusters; per-arena signs (4/4 needed to call it consistent).

**Secondary:** dose response M1 -> M2 -> M3; rigid fixed-2 unsafe with the same contrasts; generalisation gap per
model (unseen minus in-distribution); in-arena effect on g203/g228 held-out groups (M1 never saw them, M3/A3 did);
no-harm on f104 (M3/A3 vs M1, 2-point non-inferiority margin); median time ratio on joint successes; offline
within-group AUC on each test arena's held-out rows is not available (no training data there), so offline AUC is
reported on the g203/g228/f104 val groups only.

## 3. Task B design

- Collection: Gator on the f104 soil ids (15,235) and rigid pool (24,000). Pilots first: local rigid check on the real
  f104 heightmap and local soil wheel calibration; then a cluster pilot (24 groups x 12 routes rigid, 24 x 6 soil).
  Go unless the pilot shows a broken vehicle (explosions, launch-check failures > 5 %, NaN states).
- Models: Gator soil and rigid specialists (G), same recipe and same task ids as the HMMWV M1 (= H).
- Suite: the 800-group f104 suite (600 `f104_pair_group` + 200 `f104_crm_eval_group`), standing start, CEM B.
- Arms per world: G on the Gator; H (HMMWV-trained M1) on the Gator (transfer); straight 6 m/s on the Gator;
  H on the HMMWV and straight 6 m/s on the HMMWV (anchors; soil straight-6 on all 800 does not exist yet). Rigid adds
  fixed-2 arms if speed-free is at the ceiling.
- Primary B: G vs H, both on the Gator, soil goal reached, 800 groups, paired bootstrap + McNemar. "Works" =
  (1) G beats straight 6 m/s on the Gator (lower bound > 0); (2) G beats or is within 2 points of H on the Gator;
  (3) offline within-group AUC on held-out twin groups >= 0.95; (4) headroom closed
  (planner - straight) / (1 - straight) comparable to the HMMWV's on the HMMWV (reported, no threshold).
- Also from the collection itself: per-route Gator vs HMMWV outcomes on identical routes (by speed profile), crash and
  launch-failure rates, simulated hours, and the belly-clearance diagnostic.

## 4. Work modules (each: builder, then an independent verifier; NOTES_<module>.md + VERIFY_<module>.md in K3)

- **E1 arenas:** generate seeds 241-280, rank, pick 4 test arenas; captures + metric grids for g203, g228, g217 and
  the 4 test arenas (local OptiX); CRM surface-orientation smoke per new arena (local, ~1 min each); cases (training
  1,200 per new arena, on-policy routes, all suites), blacklist patterns, suite lock files.
- **E2 vehicle:** Gator switch in the vehicle factory / scene builder / soil collector (default HMMWV path must stay
  bit-identical: check one rigid and one soil episode before/after), spawn height, provenance gates, runtime
  fingerprint, soil wheel calibration, local rigid check on f104.
- **E3 staging + collection:** G3 source trees (HMMWV tree = the frozen files + new arenas + allowlist; Gator tree),
  soil and rigid task builders, the tier-interleaved soil task file, launch scripts, pilots, then production;
  monitoring, QA (`crm_qa.py`), sync of the run files the builders need.
- **E4 datasets:** per arena/world/vehicle station -> re-anchor -> mixed files with an `arena` column; a subset tool
  for the matched designs; blacklist checks (no suite id in any training file).
- **E5 training:** cluster MI350X (`launch-amd-cluster-training`), deploy ensembles + holdout offline metrics.
- **E6 evaluation:** picks per arena (local 5090), rigid fixed-2 path, drive tasks, index, `ga_analyze.py` pooled
  and per arena, a new arena-clustered bootstrap, the dev headroom check, task B arms and anchors.
- **E7 report:** REPORT.md for both tasks, figures, update docs/progress.md, commit and push.

## 5. Order and timing (critical path = soil GPU hours)

1. 01:30-03:00 E1 + E2 in parallel; dev headroom soil drives as soon as g217 cases and its map exist.
2. ~02:30 rigid collections for g203/g228 (then the Gator rigid pool after its pilot).
3. ~03:00 soil launch (g203/g228 tiers), ~04:00 relaunch with the Gator ids interleaved once its pilot passes.
4. As data lands: E4 builds, E5 training on the cluster (rigid first), E6 picks.
5. After soil training data: soil training, picks, evaluation drives (~11,000 soil + ~25,000 rigid drives).
6. Analysis, report, commit, push. LOG.md gets a timestamped line at every step.

## 6. Risks and fall-backs

- GPUs stay scarce: soil A stops at fewer tiers (matched designs still valid, fewer routes per group, f104 cut to
  match); Gator soil gets priority over soil A3 extra groups if time runs short, since task B asked for the same amount.
- Gator soil saturates (almost every route fails): still a valid answer to "can it train a planner"; report the
  failure rate and the headroom.
- No soil generalisation gap on the dev arena: primary stays as declared, but the report leads with the rigid fixed-2
  read-out and says the soil comparison had no room.
- Any HMMWV-path difference after the vehicle switch (bit-identity check fails): the HMMWV collections run from an
  unmodified source tree instead.

## 7. Amendments, 02:35 (after reviews R1/R2 and the first cluster hour; PLAN.sha256 above is the pre-amendment draft)

Facts that forced them: soil throughput is ~5 simulated h per wall hour (15 GPUs; every 8-GPU node and all mi2104x nodes
are taken by other users), not 18; the frozen f104-only soil model reaches 95.3 % on the unseen dev arena g217 (f104
95.8 %), i.e. no soil generalisation gap on an f104-like arena; the chosen test arenas are even closer to f104
(distance 0.65-0.91) than g203/g228; the Gator stalled on all 4 local soil routes (rear-drive traction).

1. **Spread arenas (R1-1).** Add 4 arenas from ranks 5-40 of the 241-280 ranking, one per quarter of the ranks, the
   smallest BMP sha256 in each quarter; 250 feature groups each, new locked suite. Unseen suite = 4 near + 4 spread.
   Rigid drives all 2,000 unseen groups; soil drives a declared subset of 125 per arena (lowest md5 of the group id),
   1,000 groups. Report near vs spread and the effect against distance to the nearest training arena.
2. **Soil headroom scan first (cheap, A).** The dev check (frozen model + straight 6 m/s) is repeated on the 4 spread
   arenas, 75 groups each (600 drives), before the soil evaluation. If no arena shows a gap, the soil half of task A is
   reported as "no room to improve on arenas of this family" and its evaluation shrinks to the declared primary arms.
3. **Statistics (R1-2).** One family of four primary tests (soil M3 vs M1, soil A3 vs M1, rigid fixed-2 M3 vs M1, rigid
   fixed-2 A3 vs M1), Holm at 0.05, one-sided p-values from a bootstrap over (arena, nearest feature) clusters;
   group-level McNemar alongside. "No meaningful difference" only if the 90 % interval lies within +-2 points, else
   "inconclusive". Per-arena intervals replace the 4/4 rule. The dev result does not change the primary.
4. **In-distribution reference (R1-6):** 200 hill/crater groups of `f104_pair_group_*` (lowest md5), not the
   `f104_crm_eval` groups CEM was tuned on. Generalisation gap also relative to each arena's straight 6 m/s arm.
5. **Second ensembles (R1-4):** M1b and M3b trained for both worlds; driven on rigid always, on soil if time allows.
6. **Offline checks (R1-5, R1-8):** leave-one-arena-out at matched 545 groups; f104 learning curve 272/545/1,089 groups
   (needed to read A3 vs M1); each arena's map-lookup error recorded as a covariate. Rigid designed routes on every
   unseen test group (already in the rigid task file) give a per-arena feasibility ceiling.
7. **Task B criteria (R1-3), declared now.** "Collects the same data": >= 95 % of the ids validated, < 1 % crashed or
   NaN, < 5 % launch-check failures; the belly-in-soil flag (lowest body point > 0.05 m under the surface for > 1 s)
   on <= 10 % of soil drives, else the soil result is reported as not physically trustworthy. H is trained on exactly
   the ids the Gator validated. A wheel-radius sensitivity pilot (calibrated + 0.08 m) is run; > 15 points change in
   failure = results depend on the wheel model. Rigid unsafe is reported with and without the backward-motion clause.
8. **Compute priorities (soil, in this order):** Gator pilot (24 groups x 6 routes, + the radius sensitivity rows) ->
   spread headroom scan -> training tiers interleaved Gator : g203 : g228 = 1,200 : 606 : 606 rows per tier (Gator ids
   = the HMMWV `collect_v1` ids of that tier; Gator rows only after its pilot passes the launch criteria) ->
   evaluation (primary arms of A and B first, then M1b/M3b, HMMWV anchors, M2, in-distribution, dev). Rigid work runs
   inside the soil allocations' idle cores if that does not slow soil (> 10 %), else on free CPU nodes. The 12:00 cut
   is dropped: collection continues until the tier in progress completes after ~12:00 or until the budget cap; the
   report states the tiers reached per arena and vehicle, and every model is trained on the same tier range as its
   comparison partner (M1 cut to the new arenas' tiers; H cut to the Gator's ids).
9. **Disclosures (R1-11):** g203/g228 are the two most f104-like of the earlier 40 seeds, so M3 tests a small step in
   variety; matched groups are not matched rows (rows reported).
10. **Soil stage-1 cut (10:58, after an unplanned session shutdown 05:50-10:50; cluster jobs kept running).** Spread
    headroom: the frozen f104 soil model reaches 85.3 / 89.3 / 94.7 / 90.7 % on g241 / g258 / g263 / g268 (pooled
    90.0 %, straight 6 m/s 62.7 %) against 95.3 % on g217 and 95.8 % on f104, so the soil half of task A has room.
    Soil throughput is now ~15 simulated h per wall hour (46 MI210). Stage-1 soil models use tiers 0-6 (7 routes per
    group) on every arena and for the Gator; f104 (M1, H) is cut to the same tiers; H uses exactly the Gator's
    validated ids. Soil evaluation rows go into the soil queue ahead of the remaining training tiers (7-12), which
    keep collecting; a stage-2 retrain of G/H (and A3) on more tiers happens only if the collection finishes and
    time allows. Evaluation order: A primary (M1, M3a, A3, straight 6) and B primary (G and H on the Gator) ->
    M3b, M2 -> anchors (straight 6 on the Gator, H and straight 6 on the HMMWV) -> in-distribution.
