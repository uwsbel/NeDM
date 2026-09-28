# REVIEW R1: adversarial review of PLAN.md (Polaris and M113 on f104 soil), 2026-09-27 23:45-00:00 CDT

Read: PLAN.md (sha256 e166e1fb... checked, unchanged), LOG.md, scout reports S1-S5 and CRITIC.md in full.
Wrote only this file and NOTES_R1.md. No Chrono run, no cluster command, no edits.

Ranking: **blocker** = if followed as written, the night misses its goals or a question gets answered for the wrong
reason; **should-fix** = could make an answer to Q1-Q3 wrong, unfair or hard to defend; **note** = caveat or wording.
Every item ends with amendment text for PLAN section 9.

## Checks I ran myself (the three most important claims, plus two cheap extras)

1. **The stock Polaris driveline gives more power than its engine map allows exactly when wheels spin on soil.**
   - Code: `chrono/src/chrono_vehicle/wheeled_vehicle/driveline/ChSimpleDriveline.cpp:106-116`. The speed sent to the
     gearbox is multiplied by the axle ratio 0.25, and the wheel torque is divided by it. So the engine sits near idle
     (at most 42 rad/s in S1's runs), and wheel torque does not fall as the wheels speed up.
   - Data: S1's four stock-driveline f104 soil drives (`scratch/S1/f104_soil/polaris_stock_cyl025/*/trajectory.npz`).
     True wheel power = 16 x recorded engine power. It peaked at **96 kW and 105 kW on the two 6 m/s routes**, against
     the map's 77 kW. It was above 60 kW on 23-27 % of their frames.
   - Wheel slip was large on every drive (soil wheel radius 0.25 m): median 0.19-0.34, 90th percentile 0.9-1.2.
   - So the plan's reason ("at 0-6 m/s it acts like a one-gear ~50 kW vehicle", PLAN 1.2) holds only without slip. On
     soil, the defect shows up in exactly the spinning, digging situations where the Gator and the HMMWV fail.
2. **The "same end state on >= 95 %" re-drive rule cannot see a broken Gator path.**
   - `scratch/S3/sample_A.json`: the stored Gator runs on sample A end as 134 blockage stops, 9 goals and
     1 breakthrough.
   - 95 % of 144 allows 7 disagreements. A Gator path that lost 7 of its 9 goals would still pass.
3. **Rows at a lower tier are always driven first, and the episode time limit is set per job, not per row.**
   - `scripts/crm_worker.py:79` sorts by tier, and workers claim the lowest tier first.
   - `:26` sets the episode timeout from the job's environment (default 2,400 s).
   - `:131` retires a GPU worker after 3 failures in a row.
   - PLAN 2.2 puts the M113 quick look in tier -3, ahead of the Polaris rows in tiers -2 and -1.
4. Extra: in the Gator study's 6,400 suite drives (`K3/e6/index/soil_eval_bfull.json`), "goal" and "goal reached
   safely" differ by at most 0.4 points for every wheeled arm. So using "safely" costs nothing for the Polaris.
5. Extra: the 800 suite pairs are 800 one-pair groups in only **9 terrain clusters** (same index file). A "group
   bootstrap" over them is really an independent-pairs bootstrap.

---

## Blockers

### B1. The M113 rows share the Polaris smoke pool, at a tier that is driven first
- **Where:** PLAN 1.3 lets the M113 in at 03:00, after the smoke starts at 01:30. PLAN 2.2 puts its quick look at
  tier -3, and its rest of sample A at tier -2 if the cost is at most 8x.
- **What happens:** the M113 rows come in through a larger task file. Every newly started job then drives them first,
  because their tier is lower than the Polaris rows still waiting (tiers -2 and -1).
- **Cost:** S2 measured 7.7x the HMMWV's cost (about 15 wall s per simulated s projected on MI350X). The rows last
  25-40 s, and up to 120 s when the vehicle creeps or slides.
  - Quick look: 2 arms x 48 = 96 drives, about 14 GPU-hours, i.e. about 2.3 wall hours on 6 MI350X.
  - Rest of sample A: about 190 more drives, about 4-5 wall hours.
  - The Polaris's sample B and sensitivity rows, and so the Q1 decision, slip by the same amount.
- **Other problems:**
  - The M113 needs a 3,600 s episode timeout and a claim margin of 3,000 s or more. Both are settings of the whole job
    (`crm_worker.py:26`; `ag_soil_launch.sh:25`), so they would also apply to the Polaris workers.
  - M113 failures count toward the 3-in-a-row rule that retires Polaris workers.
- **Amendment:** "M113 rows (m113, m113_g4 and their 0.5 ms Gator controls, see S7) go in their own task file. They
  run in their own jobs: at most 2 GPUs, devel, mi2101x or mi3501x, `CRM_EPISODE_TIMEOUT_S=3600`, claim margin
  3,000 s. They may write to the same output folder, with their own id prefixes. No M113 row is ever added to a task
  file read by the Polaris or collection jobs. Quick look = sample A tier 0 only (24 routes, one per group) per arm.
  The rest of sample A runs only if the quick look costs at most 8x and the M113 jobs fit the 50-task cap. The Polaris
  decision never waits for the M113."

### B2. Stage 0 uses fixed pass marks that can exclude a vehicle for reasons unrelated to soil, or exclude the references
- **Where:** PLAN 2.1 applies fixed pass marks: speed within 10 % after 4 s at 2, 4 and 6 m/s, and cross-track under
  0.5 m on the 0.10 /m half circle. "A vehicle failing stage 0 does not enter stage 1."
- **The M113 almost certainly fails for reasons unrelated to soil:**
  - its top speed on flat soil is about 4.1 m/s (S2 1.2), below 0.9 x 6 m/s;
  - its cross-track RMS is 0.91-1.05 m even on rigid ground with the frozen follower (S2 1.4).
  - So Q1 for the M113 would be answered by how well our follower tracks it, not by how it traverses soil.
- **The references may fail too:**
  - the HMMWV's RMS is 0.49-0.63 m on the test bends (S1 5.1);
  - the Gator has 14 kW and fails 89 % of its constant 6 m/s soil routes.
  - The rule does not exempt them. The Gator re-drive is the reference for Q1.
  - Not verified; it would be known only when stage 0 runs.
- **Amendment:** "Stage 0 never excludes the HMMWV or the Gator. A new vehicle is excluded from stage 1 only for any
  of these:
  - the launch check fails;
  - non-finite state, a crash, or a fall-through;
  - it cannot reach 2 m/s on flat soil;
  - a braked hold on 10 degrees slides more than 0.2 m, or faster than 0.1 m/s, in the 2 s after the settle.

  Speed tracking at 4 and 6 m/s and the half-circle cross-track are reported next to the HMMWV's values. They never
  exclude a vehicle."

---

## Should-fix

### S1. Q1 and Q3 rest on the stock driveline, and nothing says what happens if the driveline decides the result
- **Evidence:** check 1 above.
- **Gaps:**
  - PLAN 2.4 only attaches a "driveline-dependent" label when the arms differ by more than 10 points.
  - Nothing says what the collection does then. The critic had proposed letting the user decide before tiers 7-12.
  - Nothing tests whether the 90 % bar (Q3) survives a power-correct Polaris.
- **Amendment:**
  - "Q1 is reported as '**better (robust)**' only if `polaris_pc` also meets criteria 2 and 5 of 2.4 at the point
    estimate. Otherwise it is '**better with Chrono's shipped driveline only**'.
  - In either case the collection runs on the declared primary `polaris`. If the result is driveline-dependent, the
    report asks the user before tiers 7-12.
  - For Q3, the locked `polaris_grad` picks are also driven with `polaris_pc` (800 drives, about 2-3 billed), reported
    and not gating. If the bar result differs between the two, the Q3 answer says so."

### S2. The re-drive rule cannot detect a broken Gator path, but it decides whether stored Gator drives are reused
- **Evidence:** check 2 above.
- **Why it matters:** PLAN 2.3 makes array identity "reported, not gating". PLAN 4.2 reuses the stored `G_full` CEM
  drives (family test 4) and `straight6_gator` on the strength of this rule. S1 6.4.3 has the right check (same node,
  old against new dispatcher, identical arrays), but the plan never makes it a gate.
- **Amendment:**
  - "Before the smoke, one script on one GPU drives 2 HMMWV rows and 2 Gator rows through the old and the new
    dispatcher, back to back. All arrays must be identical, or the smoke does not start.
  - The `gatorctl` re-drive must reproduce at least 8 of the 9 stored Gator goals, and at least 95 % of the end states.
    If it does not: stop Q1, find the cause, and re-drive `G_full` CEM for family test 4 (about 2 billed) instead of
    reusing it.
  - Add Gator straight 6 m/s control rows on sample B, at least 16 and ideally all 96 (under 1 billed). Sample B's
    consistency threshold (83.3 %) and the reused `straight6_gator` are stored values."

### S3. The "planner works" criterion (a) mixes up "the planner adds value" with the user's bar
- **Where:** PLAN 4.3 requires `<v>_grad` to beat the straight route (Holm) for "the planner works".
- **Problem:**
  - The Polaris may already reach the goal on most straight 6 m/s routes (the ceiling rule of 2.4 expects this).
  - Then (a) can come out "inconclusive" while the planner meets 90 %. The report would then say "does not work" to a
    user whose question was the success rate.
- **Amendment:** "Q3 is answered by (b) and (c) alone. (a) is reported separately as 'the planner adds value over the
  straight route: yes / no / inconclusive'. If the straight route reaches 88 % or more on the 800 pairs, the report
  says: 'the bar is met by the straight route; the planner's gain is X points [interval]'."

### S4. The user's bar is "reaches the goal **safely**", but the plan tests "reaches the goal"
- **Where:** LOG.md:11 says "safely". PLAN Q3 (line 14) and 4.3(b) say only "reaches the goal".
- **Cost:** none for wheeled vehicles (check 4). It matters for anything that slides backwards, such as the M113 on
  slopes (S2 3).
- **Amendment:** "The bar is 'reached safely' = goal reached and not unsafe (`ga_analyze.safe_labels`), with
  belly-flagged drives also counted as failures. The plain goal-reached rate is reported beside it."

### S5. Which stage answers Q3 is not declared
- **Where:** PLAN 4.1 trains stage 1 (tiers 0-6) and stage 2 (all tiers), and both are evaluated on the 800 pairs.
- **Problem:** that gives two looks at the bar with no rule for choosing between them.
- **Amendment:** "Q3, and the Holm family of 4.4, are judged on stage 2 only: all 15,235 ids, the 'same amount of
  data'. Stage 1 is an interim readout. It is reported, but it never answers Q3 and never changes a decision."

### S6. The bootstrap unit for "clearly above" and for the family's one-sided p is ambiguous
- **Where:** PLAN 4.3 and 4.4 say "group bootstrap". Every suite group is one pair, so that is an independent-pairs
  bootstrap.
- **Problem:**
  - The Gator study's analysis resampled the 9 terrain clusters (`spec_soil_v1_Bfull.json`, `cluster_key` 'cluster')
    to get its one-sided p.
  - With only 9 clusters, the two methods can differ a lot.
- **Amendment:** "The one-sided p and 'clearly above' use the `ag_analyze` cluster bootstrap: key 'cluster',
  B = 4,000, seed 0, one-sided 95 %. The pair bootstrap and exact McNemar are reported beside it."

### S7. M113 fairness items (apply only if it enters the smoke)
- **(a) Time step.** The M113 runs at a 0.5 ms step and the Gator at 1 ms. For the HMMWV, that change flipped 3.5 %
  of outcomes.
  - Amendment: "Drive `gator05` rows, the Gator at 0.5 ms (per-row `config`), on the same M113 routes and in the same
    M113 jobs. The M113 is paired with `gator05`."
- **(b) Pad variants tuned to pass.** "Holds on 10 degrees with pads or a declared grouser variant" lets the pad be
  tuned until it passes.
  - Amendment: "Declare the pad list and order now: thin pad, thick pad, then thin pad with one 0.08 m cross ridge.
    Use the first that holds (the S2-B2 hold rule). Report every variant tried. The M113 answer says 'with a pad
    chosen by the hold test'."
- **(c) Breakthrough offset.** At rest the pads float 0.10-0.13 m above the surface (S2 3). With "the same 0.30 m" the
  M113 must dig about 0.1 m deeper than a calibrated wheel before the rule fires. A partly exempt vehicle wins partly
  for that reason (S3 2.3).
  - Amendment: "M113 sinkage is measured from each quarter's settled pad height at the end of the settle. The
    threshold is 0.30 m. The largest sinkage is reported for every vehicle."
- **(d) Reference point at the front sprocket, 2 m ahead of the centre.** "Goal reached" fires while the rear half of
  the M113 has not yet crossed the last 2 m of terrain.
  - Amendment: "Also report the M113's goal rate recomputed from the chassis centre of mass along the trajectory. If
    the two differ by more than 3 points, Q1 uses the centre-of-mass version."
- **(e) Wording.** If only `m113_g4` passes, the answer is "an M113 with 4x lower gearbox ratios". It is not
  "Chrono's M113".
- **(f) Cost gate.** The gate of 8x was set just above S2's measured 7.7x, while S3 had proposed 6x.
  - Amendment: state why 8x, or restore 6x.

### S8. Tier order delays the Q1 decision, and ready-made Gator and HMMWV work waits needlessly
- **Tier order:** sample B's straight 6 m/s rows, needed for the Q1 consistency check, sit in tier -1 with the
  432 sensitivity rows. The decision then waits for all of tier -1: about 1.3 wall hours on 6 MI350X after sample A
  (my estimate at S3's 0.444 simulated s per wall s per GPU).
- **Amendment:** "Tiers are:
  - -5: identity rows, plus 2 Polaris canary rows;
  - -4: the quick look;
  - -3: the rest of sample A;
  - -2: sample B straight 6 m/s for `polaris`, plus the Gator B controls;
  - -1: the three sensitivity arms and sample B's H_full picks.

  Q1 is decided once tier -2 is done."
- **Ready-made work:** the `G_full_grad` and `H_full_grad` picks need only existing models (about 35-50 min each on
  the 5090).
- **Amendment:** "Write the analysis spec now. Compute and lock these two pick sets tonight. Put their drives in the
  first collection task file, at a negative tier. Only the Polaris picks wait for its model."

### S9. The soil-wheel calibration is not re-checked on the cluster build
- **Why:** the 0.25 m cylinder was calibrated on the local build `a92c6f72`, and the cluster runs `f54254fa`. PLAN 2.1
  has no pass mark for it.
- **Amendment:** "Stage 0 passes the Polaris wheel only if, on the same node as the HMMWV settle:
  - the settled axle-mean sinkage is within +-0.01 m of the HMMWV's;
  - the launch height is 0.30-0.45 m;
  - the vertical speed at 0.8 s is under 0.05 m/s.

  If not, stop the Polaris and recalibrate (S1 6.2)."

---

## Notes

- **N1. The Q1 bar against the Gator is low, and the wheel-model checks lean one way.**
  - Criterion 5 (under 80.6 %) is nearly automatic for the Polaris.
  - `polaris_w08` only tests a more generous wheel.
  - The soil sees every wheel about 0.16 m wider than it is. That helps the narrow Polaris tyre most: x1.75, against
    x1.5 for the Gator (S5 3.4).
  - The rigid floor at 0.32 m hides the M113's low ground pressure (S5 1.3).
  - Report all of these as caveats. Report Polaris against the stored HMMWV on sample A (69.4 %) as a secondary. If the
    budget allows, add a less generous wheel arm on sample A: r = 0.22 m, which sinks about 3 cm deeper.
- **N2. Holm family size.** Fix it at 3 whether or not the M113 is admitted. Otherwise the Polaris's significance level
  depends on an event after launch. The p-values expected for the Polaris make this moot.
- **N3. Test pairs seen early.** Sample B is 96 of the 800 test pairs, and its outcomes are seen before the Q3 picks
  and analysis spec. Report the bar also on the 704 pairs outside B. Decide the optional wide-shape arms on time and
  budget only.
- **N4. The HMMWV gradient check has no rule.** Declare that the default planner is never switched after the fact. If
  `<v>_cem` reaches 90 % or more and `<v>_grad` does not, the report says so plainly.
- **N5. Similar id prefixes.** `polaris_pc__`, `polaris_4wd__`, `polaris_w08__` and `gatorctl__` look like `polaris__`
  and `gator__`. The dataset and evaluation builders must match the exact prefix and the exact vehicle-block name, and
  assert that no sensitivity or control row reaches a dataset.
- **N6. A failed Gator re-drive has no declared action.** Add: stop Q1 and do not compare (see S2).
- **N7. The corrected Polaris has never been built.** The re-framed + power-corrected combination has never run
  (CRITIC 3a.1). If `polaris_pc` fails its build, Q1 is reported with "driveline sensitivity not measured".
- **N8. Plan timestamp.** The plan header says "written 23:55", but the file time and the LOG hash are 23:42. Correct
  it in section 9 so the record shows the plan came before any drive.
- **N9. Reuse of sample B drives.** Sample B's straight 6 m/s drives count toward Q3 only if the dispatcher and the
  data-folder hashes are the same as for the collection.

## Feasibility tonight (my reading)
- **Polaris:** feasible if M1 freezes by about 01:00.
- **Smoke:** with B1 and S8, sample A plus the B straight-route rows take about 1.1 wall hours on 6 MI350X. A Q1
  decision near 03:00 is realistic.
- **Collection, tiers 0-6:** the 50-78 Gator-speed simulated hours fit by late morning only if mi2104x and mi2101x
  nodes join (S3 4.6).
- **M113:** realistic only as the separate, capped side job of B1.
