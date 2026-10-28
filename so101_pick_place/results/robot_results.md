# SO101 robot results (from the raw logs)

Source: `~/so101_r2s_claude/robot_logs/<session>/session.json` (typed outcome) and `ep_NN/episode_live.{json,npz}`. Built by `scripts/build_results.py`. Place = the person at the robot typed `p` (cube on the plate).

## Students

| Label | Bundle on the robot computer | Checkpoint (local copy of the bundle) | student.pt sha256 (first 16) |
|---|---|---|---|
| tb10 | so101_hw_v13_tb10 | ~/so101_r2s_claude/ckpt/v12/students/v12_tb10_s1210/it_01000.pt | 6458a4f791d10521 |
| D | so101_hw_v8_roll801 | ~/so101_r2s_claude/ckpt/v8/v8_roll_s801/it_02000.pt | d9a35bfc23abde45 |
| visR500 | so101_hw_v13_visR500 | ~/so101_r2s_claude/ckpt/v12/students/v12_visR_s1231/it_00500.pt | 516eb659f6720066 |
| visR_s1304_it750 | so101_hw_v13q_visR_s1304_it750 | ~/so101_r2s_claude/ckpt/v13/quick/v13_visR_s1304/it_00750.pt | 7485a7088b3be705 |
| visRtb05_s1301_it500 | so101_hw_v13q_visRtb05_s1301_it500 | ~/so101_r2s_claude/ckpt/v13/quick/v13_visRtb05_s1301/it_00500.pt | 3886e4b90e18d24a |
| tb05 | so101_hw_v13_tb05 | ~/so101_r2s_claude/ckpt/v12/students/v12_tb05_s1205/it_01500.pt | 2dea35124ca7a701 |
| visR0_250 | so101_hw_v13_visR0_250 | ~/so101_r2s_claude/ckpt/v13vis/visR0_s1231/it_00250.pt | 85a1983beb384021 |

## Totals

| Student | Test | Places | Rate | 95 % (Wilson) |
|---|---|---|---|---|
| tb10 | stress 10-27 (6 sessions) | 36/50 | 72 % | 58-83 % |
| tb10 | stress, R->L | 19/23 | 83 % | 63-93 % |
| tb10 | stress, L->R | 17/27 | 63 % | 44-78 % |
| tb10 | all 10-27 (screening + stress) | 40/54 | 74 % | 61-84 % |
| D | all 10-24..10-27 (7 sessions) | 19/28 | 68 % | 49-82 % |
| D | day 20261024 | 6/8 | 75 % | 41-93 % |
| D | day 20261025 | 6/8 | 75 % | 41-93 % |
| D | day 20261026 | 2/4 | 50 % | 15-85 % |
| D | day 20261027 | 5/8 | 62 % | 31-86 % |
| D | screening 10-27 | 5/8 | 62 % | 31-86 % |
| tb10 | screening 10-27 | 4/4 | 100 % | 51-100 % |
| visR500 | screening 10-27 | 2/4 | 50 % | 15-85 % |
| visR_s1304_it750 | screening 10-27 | 3/4 | 75 % | 30-95 % |
| visRtb05_s1301_it500 | screening 10-27 | 2/4 | 50 % | 15-85 % |
| tb05 | screening 10-27 | 3/4 | 75 % | 30-95 % |
| visR0_250 | screening 10-27 | 3/4 | 75 % | 30-95 % |
| all 7 | screening 10-27 | 22/32 | 69 % | 51-82 % |

## Independent recount of the stress test

- session.json place flags, 6 sessions: **36/50** (report claim 36/50; agrees: True).
- Per session: 145520 8/10, 150916 2/2, 151136 5/8, 151953 7/10, 153045 8/10, 153943 6/10
- Typed code letters agree with the flags: True; ep_NN/episode_live.json outcome agrees: True; stress_episodes.json placed/typed agree: True; same 50 episode ids: True.
- Direction label (stress_episodes.json `dir`) = sign of the arm-frame lateral offset plate y - cube y (y = r sin(deg)): 50/50 agree; |offset| < 15 mm (near ambiguous): eval_20261027_145520/ep_05, eval_20261027_151953/ep_02.
- Excluded: eval_20261027_150916/ep_03 (aborted before the student loop (holding at HOME); not in session.json episodes[]; no policy step was run).

## Episodes

| Session | Ep | Group | Student | Typed | Place | Dir | Cube r mm / deg | End reason | Duration s |
|---|---|---|---|---|---|---|---|---|---|
| eval_20261027_145520 | ep_01 | stress_tb10 | tb10 | GLP NICE | 1 | R->L | 184 / -18 | user stop (key b' ') | 17.6 |
| eval_20261027_145520 | ep_02 | stress_tb10 | tb10 | GLP NICE | 1 | R->L | 231 / -19 | user stop (key b' ') | 17.1 |
| eval_20261027_145520 | ep_03 | stress_tb10 | tb10 | GLP NIE | 1 | R->L | 226 / -34 | user stop (key b' ') | 21.3 |
| eval_20261027_145520 | ep_04 | stress_tb10 | tb10 | GLP NICE | 1 | R->L | 258 / +8 | user stop (key b' ') | 17.2 |
| eval_20261027_145520 | ep_05 | stress_tb10 | tb10 | - FAILED GRASP, MIGHT BE OUTSIDE RANGE | 0 | L->R | 278 / +15 | user stop (key b' ') | 28.9 |
| eval_20261027_145520 | ep_06 | stress_tb10 | tb10 | - FAILED GRASP | 0 | L->R | 260 / +28 | time limit | 40.0 |
| eval_20261027_145520 | ep_07 | stress_tb10 | tb10 | GLP NICE | 1 | R->L | 248 / -45 | user stop (key b' ') | 25.4 |
| eval_20261027_145520 | ep_08 | stress_tb10 | tb10 | GLP BIT OFF | 1 | R->L | 258 / +7 | user stop (key b' ') | 17.0 |
| eval_20261027_145520 | ep_09 | stress_tb10 | tb10 | GLP BIT OFF | 1 | L->R | 228 / +37 | user stop (key b' ') | 35.9 |
| eval_20261027_145520 | ep_10 | stress_tb10 | tb10 | GLP BIT OFF | 1 | R->L | 243 / -17 | user stop (key b' ') | 17.8 |
| eval_20261027_150916 | ep_01 | stress_tb10 | tb10 | GLP NICE | 1 | R->L | 212 / -19 | user stop (key b' ') | 18.0 |
| eval_20261027_150916 | ep_02 | stress_tb10 | tb10 | GLP NICE | 1 | R->L | 208 / -9 | user stop (key b' ') | 14.5 |
| eval_20261027_151136 | ep_01 | stress_tb10 | tb10 | GLP BIT OFF | 1 | R->L | 208 / +8 | user stop (key b' ') | 15.9 |
| eval_20261027_151136 | ep_02 | stress_tb10 | tb10 | GLP BIT OFF | 1 | R->L | 258 / +9 | user stop (key b' ') | 17.6 |
| eval_20261027_151136 | ep_03 | stress_tb10 | tb10 | GLP | 1 | R->L | 215 / -24 | user stop (key b' ') | 18.4 |
| eval_20261027_151136 | ep_04 | stress_tb10 | tb10 | GLP | 1 | R->L | 207 / -23 | user stop (key b' ') | 15.7 |
| eval_20261027_151136 | ep_05 | stress_tb10 | tb10 | - FAILED GRASP | 0 | R->L | 214 / -12 | time limit | 40.0 |
| eval_20261027_151136 | ep_06 | stress_tb10 | tb10 | GLP | 1 | R->L | 208 / -10 | user stop (key b' ') | 15.2 |
| eval_20261027_151136 | ep_07 | stress_tb10 | tb10 | - FAILED GRASP | 0 | R->L | 189 / -32 | time limit | 40.0 |
| eval_20261027_151136 | ep_08 | stress_tb10 | tb10 | - FAILED GRASP | 0 | R->L | 218 / -33 | user stop (key b' ') | 39.0 |
| eval_20261027_151953 | ep_01 | stress_tb10 | tb10 | GL PLACE IS SHORT | 0 | L->R | 183 / +26 | user stop (key b' ') | 19.3 |
| eval_20261027_151953 | ep_02 | stress_tb10 | tb10 | GLP BIT OFF | 1 | L->R | 250 / -11 | user stop (key b' ') | 17.3 |
| eval_20261027_151953 | ep_03 | stress_tb10 | tb10 | GLP OFF | 1 | L->R | 207 / +12 | user stop (key b' ') | 18.2 |
| eval_20261027_151953 | ep_04 | stress_tb10 | tb10 | GL | 0 | R->L | 185 / -45 | time limit | 40.0 |
| eval_20261027_151953 | ep_05 | stress_tb10 | tb10 | GLP | 1 | L->R | 233 / +20 | user stop (key b' ') | 19.7 |
| eval_20261027_151953 | ep_06 | stress_tb10 | tb10 | GL FAILED GRASP IN THE BEGINING AND TIME OUT DURING CARRY | 0 | L->R | 206 / +27 | time limit | 40.0 |
| eval_20261027_151953 | ep_07 | stress_tb10 | tb10 | GLP | 1 | L->R | 235 / +25 | user stop (key b' ') | 18.0 |
| eval_20261027_151953 | ep_08 | stress_tb10 | tb10 | GLP NICE | 1 | L->R | 265 / +11 | time limit | 40.0 |
| eval_20261027_151953 | ep_09 | stress_tb10 | tb10 | GLP OFF | 1 | L->R | 215 / +34 | user stop (key b' ') | 29.9 |
| eval_20261027_151953 | ep_10 | stress_tb10 | tb10 | GLP NICE | 1 | R->L | 289 / -27 | user stop (key b' ') | 33.9 |
| eval_20261027_153045 | ep_01 | stress_tb10 | tb10 | GLP BIT OFF | 1 | L->R | 219 / +17 | user stop (key b' ') | 18.6 |
| eval_20261027_153045 | ep_02 | stress_tb10 | tb10 | GLP NICE | 1 | L->R | 194 / +9 | user stop (key b' ') | 17.0 |
| eval_20261027_153045 | ep_03 | stress_tb10 | tb10 | GLP BIT OFF | 1 | L->R | 202 / -8 | user stop (key b' ') | 14.4 |
| eval_20261027_153045 | ep_04 | stress_tb10 | tb10 | GL SHORT | 0 | L->R | 181 / +33 | user stop (key b' ') | 24.0 |
| eval_20261027_153045 | ep_05 | stress_tb10 | tb10 | GL NO DROP | 0 | L->R | 184 / +22 | time limit | 40.0 |
| eval_20261027_153045 | ep_06 | stress_tb10 | tb10 | GLP | 1 | R->L | 205 / -47 | user stop (key b' ') | 18.7 |
| eval_20261027_153045 | ep_07 | stress_tb10 | tb10 | GLP | 1 | L->R | 256 / +4 | user stop (key b' ') | 21.4 |
| eval_20261027_153045 | ep_08 | stress_tb10 | tb10 | GLP NICE | 1 | L->R | 268 / -10 | user stop (key b' ') | 34.4 |
| eval_20261027_153045 | ep_09 | stress_tb10 | tb10 | GLP BIT OFF | 1 | L->R | 240 / +26 | user stop (key b' ') | 20.9 |
| eval_20261027_153045 | ep_10 | stress_tb10 | tb10 | GLP BIT OFF | 1 | L->R | 248 / -9 | user stop (key b' ') | 17.8 |
| eval_20261027_153943 | ep_01 | stress_tb10 | tb10 | GLP | 1 | L->R | 213 / +20 | user stop (key b' ') | 17.2 |
| eval_20261027_153943 | ep_02 | stress_tb10 | tb10 | GL SHORT | 0 | L->R | 249 / -1 | time limit | 40.0 |
| eval_20261027_153943 | ep_03 | stress_tb10 | tb10 | GL STOP | 0 | L->R | 191 / +21 | time limit | 40.0 |
| eval_20261027_153943 | ep_04 | stress_tb10 | tb10 | GL NO DROP | 0 | L->R | 218 / +40 | time limit | 40.0 |
| eval_20261027_153943 | ep_05 | stress_tb10 | tb10 | GLP BIT OFF | 1 | L->R | 221 / +14 | user stop (key b' ') | 22.1 |
| eval_20261027_153943 | ep_06 | stress_tb10 | tb10 | GLP OFF | 1 | R->L | 237 / -35 | user stop (key b' ') | 27.4 |
| eval_20261027_153943 | ep_07 | stress_tb10 | tb10 | GLP BIT OFF | 1 | L->R | 212 / +16 | user stop (key b' ') | 19.2 |
| eval_20261027_153943 | ep_08 | stress_tb10 | tb10 | GL NO DROP | 0 | L->R | 177 / +22 | time limit | 40.0 |
| eval_20261027_153943 | ep_09 | stress_tb10 | tb10 | GLP | 1 | R->L | 201 / -21 | user stop (key b' ') | 16.6 |
| eval_20261027_153943 | ep_10 | stress_tb10 | tb10 | GLP BIT OFF | 1 | R->L | 228 / -12 | user stop (key b' ') | 17.3 |
| eval_20261027_133039 | ep_01 | screening_20261027 | D | gl place off mat | 0 |  |  | time limit | 40.0 |
| eval_20261027_133039 | ep_02 | screening_20261027 | D | GLP PLACE OFF MAT | 1 |  |  | user stop (key b' ') | 30.3 |
| eval_20261027_133039 | ep_03 | screening_20261027 | D | GL PLACE SHORT | 0 |  |  | user stop (key b' ') | 24.4 |
| eval_20261027_133039 | ep_04 | screening_20261027 | D | GLP NICE | 1 |  |  | user stop (key b' ') | 26.6 |
| eval_20261027_133521 | ep_01 | screening_20261027 | visR500 | GLP NICE | 1 |  |  | user stop (key b' ') | 23.2 |
| eval_20261027_133521 | ep_02 | screening_20261027 | visR500 | GL FAILED DROP THE CUBE ABOVE MAT | 0 |  |  | time limit | 40.0 |
| eval_20261027_133521 | ep_03 | screening_20261027 | visR500 | GL PLACE BUT OFF FROM MAT | 0 |  |  | user stop (key b' ') | 34.4 |
| eval_20261027_133521 | ep_04 | screening_20261027 | visR500 | GLP NICE | 1 |  |  | user stop (key b' ') | 24.3 |
| eval_20261027_133953 | ep_01 | screening_20261027 | visR_s1304_it750 | GLP NICE | 1 |  |  | user stop (key b' ') | 33.0 |
| eval_20261027_133953 | ep_02 | screening_20261027 | visR_s1304_it750 | GLP BIT OFF | 1 |  |  | user stop (key b' ') | 31.7 |
| eval_20261027_133953 | ep_03 | screening_20261027 | visR_s1304_it750 | GL PLACE IS OFF FROM MAT | 0 |  |  | user stop (key b' ') | 28.7 |
| eval_20261027_133953 | ep_04 | screening_20261027 | visR_s1304_it750 | GLP NICE | 1 |  |  | user stop (key b' ') | 25.1 |
| eval_20261027_134411 | ep_01 | screening_20261027 | visRtb05_s1301_it500 | GLP BIT OFF | 1 |  |  | user stop (key b' ') | 15.7 |
| eval_20261027_134411 | ep_02 | screening_20261027 | visRtb05_s1301_it500 | GL STOP | 0 |  |  | time limit | 40.0 |
| eval_20261027_134411 | ep_03 | screening_20261027 | visRtb05_s1301_it500 | GLP BIT OFF BUT GOOD | 1 |  |  | user stop (key b' ') | 22.1 |
| eval_20261027_134411 | ep_04 | screening_20261027 | visRtb05_s1301_it500 | GL PLACE OFF | 0 |  |  | user stop (key b' ') | 39.5 |
| eval_20261027_134903 | ep_01 | screening_20261027 | tb05 | glp bit off | 1 |  |  | user stop (key b' ') | 20.3 |
| eval_20261027_134903 | ep_02 | screening_20261027 | tb05 | GLP NICE | 1 |  |  | user stop (key b' ') | 21.1 |
| eval_20261027_134903 | ep_03 | screening_20261027 | tb05 | GL SHORT | 0 |  |  | user stop (key b' ') | 30.4 |
| eval_20261027_134903 | ep_04 | screening_20261027 | tb05 | GLP BIT OFF | 1 |  |  | user stop (key b' ') | 31.1 |
| eval_20261027_135259 | ep_01 | screening_20261027 | D | - FAILED GRASP | 0 |  |  | user stop (key b' ') | 26.6 |
| eval_20261027_135259 | ep_02 | screening_20261027 | D | GLP BIT OFF | 1 |  |  | user stop (key b' ') | 24.9 |
| eval_20261027_135259 | ep_03 | screening_20261027 | D | GLP NICE | 1 |  |  | user stop (key b' ') | 26.4 |
| eval_20261027_135259 | ep_04 | screening_20261027 | D | GLP NICE | 1 |  |  | user stop (key b' ') | 28.5 |
| eval_20261027_135657 | ep_01 | screening_20261027 | tb10 | GLP NICE | 1 |  |  | user stop (key b' ') | 15.4 |
| eval_20261027_135657 | ep_02 | screening_20261027 | tb10 | GLP BIT OFF BUT GOOD | 1 |  |  | user stop (key b' ') | 17.6 |
| eval_20261027_135657 | ep_03 | screening_20261027 | tb10 | GLP NO STOP BUT OFF | 1 |  |  | user stop (key b' ') | 19.5 |
| eval_20261027_135657 | ep_04 | screening_20261027 | tb10 | GLP NICE | 1 |  |  | user stop (key b' ') | 24.9 |
| eval_20261027_140036 | ep_01 | screening_20261027 | visR0_250 | GLP OFF CENTER | 1 |  |  | user stop (key b' ') | 24.1 |
| eval_20261027_140036 | ep_02 | screening_20261027 | visR0_250 | - FAILED GRASP | 0 |  |  | user stop (key b' ') | 28.3 |
| eval_20261027_140036 | ep_03 | screening_20261027 | visR0_250 | GLP NICE | 1 |  |  | user stop (key b' ') | 21.6 |
| eval_20261027_140036 | ep_04 | screening_20261027 | visR0_250 | GLP OFF FROM CENTER | 1 |  |  | user stop (key b' ') | 39.8 |
| eval_20261024_202249 | ep_01 | D_20261024_26 | D | GLP NICE | 1 |  |  | user stop (key b' ') | 23.1 |
| eval_20261024_202249 | ep_02 | D_20261024_26 | D | GLP NICE | 1 |  |  | user stop (key b' ') | 27.9 |
| eval_20261024_202249 | ep_03 | D_20261024_26 | D | GLP BUT BIT OFF FROM CENTER OF MAT | 1 |  |  | user stop (key b' ') | 36.4 |
| eval_20261024_202249 | ep_04 | D_20261024_26 | D | GL SOMEHOW STUCK DURING TRANSPORTING TO MAT | 0 |  |  | time limit | 40.0 |
| eval_20261024_205852 | ep_01 | D_20261024_26 | D | GLP NICE | 1 |  |  | user stop (key b' ') | 23.4 |
| eval_20261024_205852 | ep_02 | D_20261024_26 | D | GLP BUT PLACE IN THE WRONG SPOT | 1 |  |  | user stop (key b' ') | 34.3 |
| eval_20261024_205852 | ep_03 | D_20261024_26 | D | GLP NICE | 1 |  |  | user stop (key b' ') | 25.3 |
| eval_20261024_205852 | ep_04 | D_20261024_26 | D | - FAILED GRASP | 0 |  |  | time limit | 40.0 |
| eval_20261025_192402 | ep_01 | D_20261024_26 | D | GLP NICE | 1 |  |  | user stop (key b' ') | 26.1 |
| eval_20261025_192402 | ep_02 | D_20261024_26 | D | GLP NICE | 1 |  |  | user stop (key b' ') | 27.4 |
| eval_20261025_192402 | ep_03 | D_20261024_26 | D | GLP PASS | 1 |  |  | user stop (key b' ') | 35.7 |
| eval_20261025_192402 | ep_04 | D_20261024_26 | D | GLP NICE | 1 |  |  | user stop (key b' ') | 26.8 |
| eval_20261025_194447 | ep_01 | D_20261024_26 | D | GL STOP | 0 |  |  | user stop (key b' ') | 25.8 |
| eval_20261025_194447 | ep_02 | D_20261024_26 | D | GL STOP | 0 |  |  | user stop (key b' ') | 22.4 |
| eval_20261025_194447 | ep_03 | D_20261024_26 | D | GLP NICE | 1 |  |  | user stop (key b' ') | 26.3 |
| eval_20261025_194447 | ep_04 | D_20261024_26 | D | GLP SLIGHTLY OFF | 1 |  |  | user stop (key b' ') | 26.8 |
| eval_20261026_131348 | ep_01 | D_20261024_26 | D | GLP NICE | 1 |  |  | user stop (key b' ') | 26.9 |
| eval_20261026_131348 | ep_02 | D_20261024_26 | D | GLP BIT OFF | 1 |  |  | user stop (key b' ') | 26.6 |
| eval_20261026_131348 | ep_03 | D_20261024_26 | D | GL STOP | 0 |  |  | user stop (key b' ') | 28.4 |
| eval_20261026_131348 | ep_04 | D_20261024_26 | D | - MISS GRASP | 0 |  |  | time limit | 40.0 |
