# NOTES R1 (plan review), 2026-09-28 00:00 CDT
The full review, with amendment text for each item, is in REVIEW_R1.md. Only REVIEW_R1.md and this file were written. PLAN.md is unchanged (hash checked); no runs, no jobs.

Checks I ran:
- The stock Polaris driveline delivers 96 and 105 kW at the wheels on S1's two 6 m/s f104 soil drives. The engine map peak is 77 kW, and slip p90 was 0.9-1.2 (ChSimpleDriveline.cpp:106-116).
- The stored Gator results on sample A are 9 goals, 134 blockage stops and 1 breakthrough. A 95 % end-state match rule therefore still passes with 7 of the 9 goals lost.
- crm_worker.py:79 drives the lowest tier first; :26 sets the episode timeout per job; :131 retires a worker after 3 failures in a row.
- In the Gator study, "goal" and "goal safely" differ by at most 0.4 points; the 800 suite pairs form only 9 terrain clusters.

Blockers:
- B1. The M113 rows go into the Polaris smoke task file at tier -3 and are added at 03:00. New jobs would then drive them first: about 14 GPU-hours for the quick look, and 4-5 wall hours more for the rest of sample A. The Q1 decision slips by the same amount, and the M113's 3,600 s timeout would have to be set for every worker. Fix: a separate M113 task file and jobs (at most 2 GPUs), a 24-route quick look, and the Polaris decision never waits for it.
- B2. Stage 0 has fixed pass marks (speed within 10 % at 6 m/s; cross-track under 0.5 m). The M113 is certain to fail them because of its top speed and how the follower tracks it, not because of soil. The HMMWV and Gator reference vehicles are not exempted. Fix: never exclude the references; exclude a new vehicle only if the launch check fails, the state goes non-finite, it cannot reach 2 m/s, or it fails the braked 10-degree hold; report the rest relative to the HMMWV.

Should-fix:
- S1. Driveline defect. Call Q1 "robust" only if the power-corrected Polaris also meets criteria 2 and 5. Also drive the locked Polaris gradient picks with the power-corrected Polaris (about 2-3 billed). Ask the user before tiers 7-12 if the result depends on the driveline.
- S2. Make the old-vs-new dispatcher array identity, on one GPU, a gate. Require at least 8 of the 9 stored Gator goals to reproduce. Otherwise re-drive the Gator CEM planner arm (G_full CEM). Add Gator straight-route control rows on sample B.
- S3. Answer Q3 by (b)+(c) only. Report (a), "beats the straight route", separately, because of the ceiling.
- S4. The bar is "reaches the goal safely" (LOG.md:11); the plan says only "reaches the goal".
- S5. Declare that stage 2 (all 15,235 ids) answers Q3 and carries the test family; stage 1 is an interim readout.
- S6. The "group bootstrap" of 4.3/4.4 is over single-pair groups. Use the Gator study's 9-cluster bootstrap for p and "clearly above".
- S7. M113, if it runs: a Gator control at 0.5 ms; a pad list fixed in advance and a defined hold rule; breakthrough measured from the settled pad height (pads float 0.10-0.13 m); goal also recomputed at the centre of mass (the reference point is 2 m ahead of it); "re-geared M113" wording; explain the 8x cost gate (S3 said 6x).
- S8. Renumber the tiers so sample B's straight-route rows come right after sample A (Q1 decided about 1.3 h sooner). Compute and lock the Gator and HMMWV gradient picks tonight.
- S9. Stage-0 pass mark for the wheel: on the cluster, settled Polaris sinkage within 0.01 m of the HMMWV's on the same node.

Notes (N1-N9): the Gator bar is low, and the wheel sensitivity only tests a more generous wheel; the soil sees narrow tyres as wider; fix the Holm family at 3; 96 of the 800 test pairs (sample B) are seen early; no rule for the HMMWV gradient check; exact-prefix matching; no declared action if the Gator re-drive fails; the power-corrected Polaris has never been built; plan timestamp 23:55 vs hash 23:42; reuse of sample B drives needs the same hashes.

For the orchestrator to decide: adopt B1, B2, S1-S6 and S8 as a section 9 amendment before the smoke is launched. S7 only if the M113 is admitted.
