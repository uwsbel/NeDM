# NOTES S2: soil closed-loop evaluation, tasks A and B (2026-09-25, 14:18-)

Soil track, step 2 of the resumed workflow (PLAN 2.3, 3, 7.1-7.4, 7.7, 7.10). K3 = this folder, G3 =
`/work1/dannegrut/harry/experiments/arena_gator_20260925`. No existing script was edited, nothing in earlier artefact
folders or cluster roots was written, and nothing in `G3/source` was touched (new tools live in `G3/tools/s2`). Every
job and job step is in `G3/e3/submissions.tsv` and has a line in `LOG.md`. Results: `RESULTS_soil.md` and
`e6/analysis/*soil*`.

## 1. State found at 14:18

- No soil evaluation output existed (`e6/` held the rigid evaluation of the other track only).
- Soil stage 1 (NOTES_S1, VERIFY_S1 pass): task A models M1a, M1b, M2, M3a, M3b, A3 and task B's G deployed in
  `e5/deploy/*_soil`; H = the M1a soil ensemble (the H data file is byte-identical to the M1 file).
- Queue 34 of 50 array tasks. Every GPU partition was fully allocated. Our soil_v2 jobs (436080_12-17 ending ~15:15,
  436208, 436460_0-5, 436210 / 436215 ending ~17:50-18:30, 436353, 436461) were driving training tiers 7-12; pending on
  soil_v2: 436460_[6-7], 436490_[0-4], 436416_[0-2]. Other users' jobs with higher priority were queued ahead of ours on
  mi2101x and mi2104x (436416 was estimated to start 17:32).

## 2. Picks (local 5090, `scripts/ag_s2_picks.sh unseen|f104|heldout`)

`scripts/ag_picks.py --world crm --mode free` (CEM 4 x 64 from the case pose at rest; `ci_planner.py --family free
--world crm --arms B`) or `--mode straight6`, 6-8 processes in parallel, 14:22-14:34. Pick directories
`e6/picks/crm/<arena>/<model>_free` and `.../straight6`; models `e5/deploy/<M>_soil/<M>_soil_deploy_s*.pt`.

| set | groups | pick directories |
|---|---|---|
| unseen (task A) | 8 arenas x the declared 125-group soil subset (`suites/soil_unseen_subset.json`) | M1a, M1b, M3a, M3b, A3, M2 speed free + straight 6 m/s (56) |
| f104 800-group suite (task B + in distribution) | all 800 (`generalist_20260921/A_adapt/suite/cases`) | M1a (= H), G speed free + straight 6 m/s |
| f104 in distribution | the declared 200 (`suites/f104_indist_200.json`) | M3a, A3 speed free (M1a is the 800-group directory) |
| g203 / g228 held-out | 150 each | M1a, M3a, A3 speed free + straight 6 m/s |

- 69 directories, 11,000 picks, 0 failures; every directory's own lock plus the set lock
  `e6/picks/LOCK_crm.sha256` (ALL `2c766a52...`, `scripts/ag_e6b_lock.py`) written before any drive.
- Determinism: two directories re-planned in separate processes (G on the first 100 of the 800 f104 groups; M1b on
  the 125 g241 groups) gave byte-identical route files on 225 of 225 groups (M1b lock `4bda7a66` both times).

## 3. Drive rows and the task file soil_v3 (`scripts/ag_s2_build.sh`)

One `ag_eval_tasks.py build --world crm` per declared priority group, each with its own tier and every earlier build
as `--existing` (identical routes are one drive, in the earliest group; soil_v2 drives with the same group, vehicle,
case and route content are reused, e.g. the spread-arena straight 6 m/s headroom drives). crm_worker.py sorts every
worker's rows by tier only (random within a tier), so the tiers carry the declared order (PLAN 7.10):

| tier | build | arms (plain label) | new drives | reused / merged |
|---|---|---|---|---|
| -9 | `e6/tasks/soil_eval_p1.json` (`e4cc879b...`) | task A primary on the 8 unseen arenas: f104-only ensembles 1 and 2 (M1a, M1b), three arenas at the same total, ensemble 1 (M3a), three arenas with all data (A3), straight 6 m/s; task B primary on the 800 f104 groups: Gator-trained planner on the Gator (G), HMMWV-trained planner on the Gator (H) | 6,171 (4,572 HMMWV, 1,599 Gator) | 309 spread headroom drives reused, 98 identical routes merged |
| -8 | `soil_eval_p2.json` (`8ab0f468...`) | three arenas at the same total, ensemble 2 (M3b); two arenas at the same total (M2) | 1,902 | 14 + 67 reused, 2 merged |
| -7 | `soil_eval_p3.json` (`9d4bc015...`) | anchors on the 800 f104 groups: straight 6 m/s on the Gator, H on the HMMWV, straight 6 m/s on the HMMWV | 2,363 (778 Gator) | 22 reused, 15 merged |
| -6 | `soil_eval_p4.json` (`d42cb02b...`) | in distribution: M3a, A3 on the 200 f104 groups (M1a there is the H-on-HMMWV drive); M1a, M3a, A3 on the g203 / g228 held-out groups | 1,273 | 10 reused, 16 merged |
| -5 | `soil_eval_p5.json` (`36f4d632...`) | extra, not in the declared order: straight 6 m/s on the held-out groups | 291 | 9 reused |

- 12,000 new drives (9,623 HMMWV, 2,377 Gator). Rows: tier as above, `kind eval`, `extra ['--vehicle', hmmwv|gator]`
  (no runtime fingerprint: the frozen soil dispatcher refuses it), unique ids `<group>__<arm>` / `gator__<group>__<arm>`,
  unique episode seeds (md5 of the id, salted on a clash).
- Reading of "M1" in the A primary: the declared reference M1 = M1a + M1b (per-group mean), as in the family of the
  frozen rigid spec (`e6/analysis/spec_rigid_v1.json`: soil P1/P2 use M1 = mean of M1a and M1b); "M3a" and then "M3b"
  are the two halves of M3. So M1b is in the first tier and M3b in the second.
- `e3/tasks/soil_v3.json` (43,901 rows, `654ca61b...`, `scripts/ag_s2_soil_v3.py`) = soil_v2 unchanged as a prefix
  (checked row by row) + the 12,000 evaluation rows; `soil_v3_check.json` (6 rows of tier -9: 3 HMMWV, 3 Gator,
  `c857bd63...`). Both read-only in `G3/tasks`. All route and case files staged with `ag_eval_tasks.py stage --execute`
  (new paths only; every listed file hash-equal on G3), 0 missing paths over the 12,000 rows.

## 4. Analysis spec, frozen before any outcome

`e6/analysis/spec_soil_v1.json` sha256 `cf813e0a20f9110a...` (`scripts/ag_s2_spec.py`), frozen and logged at 14:40
before any soil evaluation outcome existed. Family = PLAN 7.3 with the rigid spec's definitions (P1 soil M3 vs M1,
P2 soil A3 vs M1 on "goal not reached", pooled over the 8 unseen arenas, M1 / M3 = per-group means of the two
ensembles; P3 / P4 rigid fixed 2 m/s from the rigid results by `scripts/ag_s2_family.py`, Holm over all four).
Secondary contrasts, gaps, dose, headroom, time ratios as listed in the spec file.
