# Synthesis of the diagnosis (D1, D2, W): which hypotheses to test first

Written 2026-09-22 09:40. Sources: PLAN.md, scout/D1_anatomy, D2_crossscore, W_window, s1/compare_s1_picks.json. New
counts come from read-only reads of K1 `A_adapt/a5/results_crm_A5.json` and `a3/results_crm_A0A3.json`
(`per_group[g][arm]['fail']`; paired group bootstrap, 4,000 resamples) and `scout/d1_cache/features.json`.

Baseline: H fails 129 of 800 soil groups at 3 s (83.9 %): 86 in all-fail groups, 43 in the 105 choice-dependent groups.

**Correction to a D1 caveat.** `a3/` and `a5/picks_crm_{H,T}.log` both load `train/deploy_v1/{H,T}_deploy_s*.pt`, so
for H and T the moving-vs-standing gap compares the same model. Only the S_crm pairing mixes ensembles.

## Ranking

| rank | hypothesis | removable (of H's 129) | evidence |
|---|---|---|---|
| 1 | H2: the approach commits the vehicle | about 79 net | H 129 moving vs 50 standing: +9.88 points [7.37, 12.50]; 99 groups fail only moving, 20 only standing. T 125 vs 31: +11.75 [9.38, 14.25]. The gap is 0 on flat approaches and +21 to +38 above 17 deg. Position AUC 0.855 vs vehicle condition 0.60. Distance before the decision: 6.2 m at 3 s, about 0.7 m at 1 s, 0.1 m at 0.5 s. |
| 2 | H3: training-distribution gap | 3 s: at most 43, likely 15-25 | Among soil-competent routes, within-group AUC is 0.643 and removes 23 % of random-pick regret. Regret 4.88 points [3.38, 6.38], plus about 1 point lost to search. Calibration: H's 8th decile predicts 0.2 % against 25.2 % realised. The standing-start ensemble has ECE 0.049 vs H's 0.098. |
| 3 | H1: planner/controller handover | 0-8 (about 0.9 points); at most 28 | Positive step within a group: odds ratio 0.90 [0.65, 1.27] with route grade in the model. Terrain-matched pairs: 13.8 vs 10.5 % fail; times the 25.4 % of H picks with a step above 1.5 m/s, that is about 0.9 points. H's choice-group picks fail 35 % with a step above 1.5 m/s vs 43 % at matched speed. In all-fail groups, 20/20 matched-start routes failed. Bound: 28 groups stall within 2 s, 81 of those 85 runs in all-fail groups. |
| 4 | H4: data balance | about 0 | Soil-only S'_crm vs H: 83.0 vs 83.9 % closed loop; within-group 0.814 vs 0.836 (0.573 vs 0.643 on soil-competent routes). The training rows are already balanced (58k rigid / 57k soil). |
| 5 | H5: architecture | about 0 | Night 2: CNN-GRU tied every transformer. Within group: H 0.836, H masked 0.834, P 0.828, so the context holds no within-group signal to extract. |

## Ceiling implied by the all-fail groups

- **3 s protocol:** choosing among the soil arms' routes cannot beat 714/800 = 89.25 % (89.75 % with the rigid
  specialist's 4 rescues): +5.4 to +5.9 points over H. With the first steep cell 2.0 m ahead at frame 60 and 20/20
  matched starts failed, new route shapes are unlikely to go further.
- **Decision on the start pad:** of the 86, the standing start completes 68 (S_crm), 66 (T), 64 (H), 76 under any of the
  three; 10 (1.25 points) fail in both protocols. The three standing arms together fail 14 of 800 (98.25 %); single arms
  93.8-96.1 %.

## Order

1. **Now, in parallel:**
   - **E1 (H2):** pass 1 at 1 s (and 0.5 s), both worlds, from `a5data/tasks`; pass 2 with T/free (existing ensemble, no
     history, no training) on 800 groups. H2 predicts near 31 failures, the null near 125. About 1.7-2.0 billed
     node-hours per length (K1 timings: pass 1 1.65 GPU-h, pass 2 about 13 GPU-h). Minimal version: 1 s only.
   - **E2 (H1):** drive the existing H/cont picks (lock `31c22e71`) from the frozen frame-60 states, with a same-job
     re-drive of K1 H/free as control (about 2 billed node-hours each). Expected 0 to +1 point. The rule needs about +2,
     so a pass would contradict D1 and D2.
   - **Optional H3 probe in the E2 job:** K1 `train/branch_v2` hist_aux ensemble (752 soil branch anchors x 3); 800
     drives, no training.
   - **Locally:** train H_short on mixed_reanchor + short_anchor (+ k = 60 rows of anchor_k40_60_80 minus the 38,997
     duplicates). Measure risk-model AUC by window, on val groups.
2. **S2:** H_short/free at the length E1 favours, plus a standing-start arm of the same ensemble in the same job. The
   adaptive claim is H_short vs T and vs the standing start. Beating 83.9 % passes on the decision point alone.
3. **S3 after E1/E2 only**, sized to the winner (15-18 billed node-hours per length; short length only if S2 wins).

## Drop or shrink

- **S1 heading-capped arms (H and S'_crm):** drop. Rescuing routes turned further (+3.9 deg [+1.2, +6.6]). Standing-start
  successes in all-fail groups were 34.6 deg off the line. The cap removes the above-30-deg bin (31.5 % of H's picks).
  The model rates the capped picks riskier: mean P 4.0 % to 14.2 %, higher in 69 % of groups.
- **H4 data-fraction curves:** drop. They are confounded by step count (VERIFY_ci_train issue 3) and predicted null.
  Keep at most one offline arm at CRM batch fraction 0.75.
- **H5:** one offline txjoint-vs-gru run (3 seeds, val). Closed loop only if it gains more than 0.01 within-group AUC.
- **W gate:** replace "domain AUC >= 0.95" before E1 returns. It passes at 0.1 s, before the vehicle moves (engine speed
  alone 0.990). Use risk-model AUC by window and a domain check that leaves out the start.
