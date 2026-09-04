# Session brief — 2026-09-04 (afternoon): tracker + planner rollout inside the NRD

**Goal set by the user:** reach a planner rollout inside the NRD model that uses
both the physical state and the camera representation, learning from the earlier
state-only HMMWV tracking study; launch cluster jobs in parallel.

**References:** `wp3_implementation_notes.md` (tracker, scorer, all numbers),
`wp2_implementation_notes.md` addenda (pose drift, two-stage prediction,
static scene), study plan v1.4 §9.5 / §10 / §20.

## Delivered

1. **Closed the WP2 open items** (cluster jobs 405002–405007, all exit 0):
   pose drift explains ~40 % of the 5 s shortfall; a *fair* test of predicting
   the camera window forward loses to simply holding it within 0.5 s, so the
   "index, don't predict" design now rests on evidence; the scene is static
   within an episode (0.996 map agreement vs 0.80 floor).
2. **Tracker (WP3) built and trained in imagination**, reusing the earlier
   study's transferable choices (bounded positive reward, hard steering-rate
   clamp in training, loose failure bound, real recorded start windows, same PPO
   block). Held-out 3 s fragments: 0.15 m mean cross-track vs 0.245 m for a
   scripted pure-pursuit controller; two seeds agree; a state-history
   observation buys nothing.
3. **Planner rollout scorer (the §9.5 mechanism) built and calibrated.**
   Oracle candidates are driven by the tracker inside the frozen model from each
   layout's real start context and scored on time, energy, tracking, safety and
   collision. On 32 held-out layouts the imagined time-to-goal on the recorded
   route matches the recorded one with correlation 0.997 (MAE 1.1 s); all 32
   reach the end; candidates separate cleanly on time vs energy.

## What is not done

- No Chrono run: the tracker's transfer (G6) and the scorer's absolute accuracy
  are unverified in the simulator.
- Energy from the power head is biased ~25–30 % low; relative ranking only.
- Candidates come from the privileged oracle, not yet from the camera (costmap head).

## Files (all uncommitted, on top of the morning's twelve)

`src/nedm/traverse/nrd_model.py`, `tracker_env.py`; `scripts/traverse_wp3_routes.py`,
`traverse_wp3_train_tracker.py`, `traverse_wp3_eval_tracker.py`,
`traverse_wp4_score_candidates.py`, `traverse_wp2_static_check.py`;
`traverse_wp2_train_map.py` (two-stage / eval-only flags); plan v1.4; notes.
