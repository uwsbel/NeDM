# Verification of module ga_branch_anchors (PLAN A4 anchor selection, stage `select`)

Adversarial check run 2026-09-21 18:11-18:25 on luffy, CPU only (6 worker processes, `CUDA_VISIBLE_DEVICES=` empty), no
cluster submissions, no Chrono/CRM runs. No repository file was modified; scratch outputs went to `/tmp/verify_anchors/`
and `/tmp/verify_anchors_never/`. Files reviewed: `scripts/ga_branch_anchors.py`, `A_adapt/NOTES_ga_branch_anchors.md`,
`A_adapt/a4/anchors/*`, against `PLAN.md` (conventions, A4) and `scout/plan_review.md` (findings 11, 14, 15, 16).

## Verdict: pass (with notes)

The full run reproduces byte-for-byte from a fresh scan, every one of the 1,600 anchor records agrees with my own
recomputation from the raw recordings, and the plan's conventions (split rules, blacklist, history window, anchor
classes, quotas, pairing) are met. The notes below are about how one plan phrase was read (the CRM sinkage screen), a
wording point in the survivor counts, and things the next stage (continuations, pass-1 replay) must take into account.
None changes a selected anchor.

## What I re-ran and checked

| check | method | result |
|---|---|---|
| Reproduction | `--stage select --out /tmp/verify_anchors --workers 6` (fresh scan, no cache) | 30 s wall; `anchors_crm.json`, `anchors_rigid.json` byte-identical; both `hist_*.npz` array-identical; `summary.json` identical except `written`/`wall_s`; `scan_*.json` identical as dicts (rows written in worker-completion order, so bytes differ); 0 scan errors on 15,235 CRM + 24,000 rigid recordings |
| `--lp-same-frame never` variant | re-run with the cached scans | same 800 episodes and classes in both worlds, clean-moving F unchanged; low-progress at onset-10: CRM 302/320, rigid 319/320 (as claimed) |
| Structure | own code over `anchors_*.json` + the B-cache manifest | 800 per world; train 420 clean-moving + 280 low-progress, val 30 + 20, test 30 + 20; 800 distinct groups per world (max 1 per group); same episode list in the same order in both worlds, class agrees on all 800; same F on 480/480 clean-moving and 65/320 low-progress, `same_frame` flag consistent; split, group and status of every anchor equal the manifest; no planner-suite id or group; every group matches `f104_v2_group_*`; `hist_index` matches `anchor_id` in the npz |
| Every anchor record (1,600) | my own onset finder (cumulative-sum window scan), my own brute-force polyline projection, step-by-step history window, sha256, route json vs `command_reference.npz`, `case.json` goal/id/split, `outcome.json` status, `interval_start_s[F] == F x 0.05` | 0 discrepancies: onset, rollback, pose/vx/throttle/brake/steer at F, station/lateral deviation/remaining, class rules (clean-moving: F in {40,80,120}, vx > 1, dev < 1 m, >= 12 m left, not parked, no stall onset or rollback before F; low-progress: F in [onset-20, onset+10], not parked, >= 5 m left, CRM drop < 0.1 m), route path/sha/content, recorded rigid `route_sha256`, goal and goal distance, status |
| History convention | rebuilt `hist[t] = [state[F-39+t][cols 0-6, 11-15], action[F-40+t]]`, mask = action frame >= 0 | identical to both npz files; layout equals `ga_build_mixed.py` (`cut_episode`) and the PLAN text; all 800 windows fully valid per world (min F = 40; no low-progress candidate in the whole scan has onset < 50, so partial windows never arose) |
| Onset vs B-cache `stalled` | first `stalled` frame of `B_tracker/cache_v1/<episode>@<world>.npz` for all 640 low-progress anchors | equal on 317/320 CRM and 311/320 rigid; the 12 exceptions all have the onset at frame >= 1,209, past the cache cut at 1,200 (the note checked only 23 rigid anchors and reported 3 exceptions in total) |
| Survivor / margin statistics | own scan of all 15,235 CRM recordings | with onset: train 9,339 / val 518 / test 518 (equal); onset s p10/50/90 5.65/10.4/25.0 (equal); onset-to-end s 4.15/8.05/22.95 (equal); screen-only pass at onset-10: 7,915 train (module 7,858) and any F 9,181 (module 9,131): the module's "survivor" also applies the 5 m remaining / not-parked / bounds rules (57 and 50 episodes), see note 2 |
| Why a cut is not at onset-10 | per anchor | CRM 65: 22 shared frame with rigid, 26 shared frame and onset-10 infeasible, 17 onset-10 infeasible (screen or remaining); rigid 55: 54 shared frame, 1 with < 5 m of route left at onset-10 (5.04 m at F) |
| Replay cost hint | sum of `t_F_s` | CRM 6,932.9 s (low-progress 5,012.9 s, max 88.4 s), rigid 9,716.6 s (max 100.9 s), as reported |
| Scope | grep | the script has no grid, grade, torch, ssh, sbatch or subprocess reference: the "grade on the v2 grid" and route-contract conventions apply to the continuation stage, which is not implemented here (`--stage continuations` exits with a message) |
| Tracked files | `git diff --stat` | only the pre-existing nav_v1 edits from before this module; nothing under the specialist artefact roots changed |

## Notes (none blocking)

1. **CRM sinkage screen is a weak reading of the plan phrase.** PLAN A4 / review finding 15 say "sinkage increase at
   the cut < 0.1 m" on `spindle_z - bmp_ground_z`. The module measures the drop of the mean spindle height over the 20
   frames before F. For the preferred cut F = onset-10 that window is [onset-30, onset-10], i.e. entirely before the
   stall run, so the screen mostly removes anchors that were already sinking while still moving (84 % pass). Two other
   readings, computed on the 320 chosen CRM low-progress anchors: the further drop in the second after F is
   p50/p90 0.049/0.091 m with 29/320 (9 %) >= 0.1 m; the drop since the first second of the episode is >= 0.1 m for
   15/320 (5 %). The collector's own breakthrough quantity (tyre radius minus spindle height over the ground under each
   wheel, `crm_collect.py:285`) is not stored per frame (`bmp_ground_z_m` is at the chassis reference), so the absolute
   per-wheel sinkage at F cannot be recomputed offline; a chassis-centre proxy is useless on this arena (it would put
   233/420 clearly moving clean-moving anchors "0.1 m below ground"). Suggestion for the continuation stage: carry the
   two extra numbers per anchor and decide before launch whether the 29 anchors that sink >= 0.1 m in the second after
   the cut are wanted (they are the ones with the least room to differ between continuations).
2. **Survivor wording.** The note's "7,858 pass the screen at onset-10" combines the screen with the not-parked, 5 m
   remaining and frame-bound rules; the screen alone passes 7,915 (train). Cosmetic.
3. **Rigid low-progress anchors are often rolling back or transient.** At F, 47/320 rigid low-progress anchors have
   vx < -0.3 m/s and 78/320 have vx < 0 (CRM: 0 and 3); 115/320 rigid low-progress episodes later reached the goal
   (transient stalls the PID drove out of). Both follow from the plan's onset-only definition, but the branch collector
   will start continuations while the vehicle rolls backwards, and the labeller's rollback flag (secondary label) will
   fire in the grace window for those anchors unless it is evaluated after the grace window as the plan says.
4. **Pass-1 class check cannot see the stall at horizon F.** With the cut 0.5 s before the stall run, a replay that
   stops at F observes 0/320 anchors inside a stall run (the module's `in_stall_run_at_F` is 0 in both worlds; only
   70 CRM / 94 rigid have |vx| < 0.3 at F). The plan's "class recurs" check for A4 must replay to at least onset+10
   (F+20 for the preferred cut, up to F+30 for cuts at onset-20) and compare onsets, or it is vacuous. The module's
   note already says this; I am adding the horizon.
5. **Cache-cut anchors.** 3 CRM + 9 rigid low-progress anchors have onsets past frame 1,200 (the B-cache cut), so the
   `stalled` mask cannot corroborate them; they are also the longest prefixes (up to 88 s CRM, 101 s rigid). If a
   `--max-F` cap is added later it must be applied before selection (the seed-locked draw changes).
6. **Held-out allocation** 50 val / 50 test and the 5 m minimum remaining route are the implementer's choices (task text
   left them open); both are compatible with PLAN A4 ("100 from held-out groups, kept in their split").
7. `hist_*.npz` stores float32 while `mixed_reanchor.npz` stores float16; harmless for the consumers.

## Commands

```
PYTHONPATH=src:scripts CUDA_VISIBLE_DEVICES= python scripts/ga_branch_anchors.py --stage select --out /tmp/verify_anchors --workers 6
PYTHONPATH=src:scripts CUDA_VISIBLE_DEVICES= python scripts/ga_branch_anchors.py --stage select --out /tmp/verify_anchors_never --workers 6 --lp-same-frame never
CUDA_VISIBLE_DEVICES= python /tmp/verify_anchors/recompute.py     # own recomputation of all 1,600 anchors + full CRM survivor scan (28 s)
```
