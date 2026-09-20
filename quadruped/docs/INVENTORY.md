# Inventory of the old tree

Surveyed 2026-09-20 against `kyle/locomotion`. 154 `.py` + 35 `.sh` under `scripts/`, 39
of them Go2-named. This is what survives the rebuild and what does not.

## Reuse unchanged

| file | lines | why |
|---|---|---|
| `src/nedm/quadruped/terrain.py` | 212 | `build_crm` is the whole terrain patch and SPH parameterisation, path-free, duck-typed on `args`. Already contains an assert-it-took-effect helper for the SPH knobs. |
| `src/nedm/quadruped/robot.py` | 135 | URDF load, actuation, `apply_pd`. |
| `src/nedm/quadruped/constants.py` | 140 | soil presets, stand pose, joint tables; every constant carries the measurement that fixed it. |
| `src/nedm/training/dataset.py` | 187 | **needs no change for segmentation** -- see below. |
| `scripts/evaluation/crm_verdict.py` | 142 | pure function of two JSON files, stdlib only. |
| `scripts/collection/derive_contact_flags.py` | 92 | stdlib only, idempotent, dry-run by default. |
| `scripts/collection/add_gravity_channels.py` | 75 | one import to repoint at `params/transforms.py`. |

## Reuse in part

**`src/nedm/quadruped/imported_policy.py`** -- the class goes with the old checkpoint, but
three things stay:

- `_projected_gravity(q)`, exact from the quaternion. Moves to `params/transforms.py`;
  `add_gravity_channels.py` depends on it.
- **The command-family scheduler, L84-181.** `PARAM_RANGES`, `FAMILY_PARAMS` for all eight
  families, `_sched()`, and `stratified_params()`. Pure math with no policy dependency,
  and it is the plan's "command variety" excitation already written. Moves to
  `params/excitation.yaml` plus a small scheduler module.
- `family_seed()` uses **crc32, not `hash()`**, because `PYTHONHASHSEED` salting once made
  a whole collection non-reproducible. Keep the crc32.

**`scripts/evaluation/score_crm_tracking.py`** -- reuse with a one-line retarget. It never
imports the policy; it shells out to the collector with `--imported-ckpt`, so a memoryless
MLP needs no change here. Retarget the subprocess path to `quadruped/collect.py` and make
it absolute, drop `--payload-kg`.

**`src/nedm/quadruped/dataset.py`** (611) -- keep `capture_row` and the field plumbing,
move the channel lists out of module globals into `params/presets.yaml`.

## Rewrite

**`scripts/collection/collect_go2_smoke.py`** (1006) -- `run_episode()` is a single
~640-line function fusing scene construction, policy, excitation, gates and IO with no
seam. Its scene setup is already one call into `terrain.py`, so almost nothing physical is
lost. Salvage the spawn-on-bed assertion, the divergence checks, and the sidecar schema.

**`src/nedm/training/preprocess.py`** (713) -- carries HMMWV/arm/dpend/frames branches the
Go2 path does not need, and segmentation touches its central allocation logic.

## Delete

Roughly 60-80 of 189 files under `scripts/`. The pattern is one script per experiment,
named after the experiment, never run again.

- `scripts/ablations/` -- all 20; the `l8` prefix and machine names date them to one study
- `scripts/analysis/` -- ~8 of 10, **after mining them**: this is where the 4%/rank-2
  figure, the command-realisation deadband and the 83% push contamination were computed.
  The findings belong in `LESSONS.md`; the code does not need to survive.
- `scripts/evaluation/` -- ~35 of 54, retired case studies plus Go2 one-offs
- `scripts/collection/` -- ~20 of 34, HMMWV collectors, per-corpus prepare scripts, repairs

Keep out of the cull: `crm_verdict.py`, `score_crm_tracking.py`, `check_action_roundtrip.py`,
`backfill_chrono_provenance.py`, `compare_score_files.py`.

Also delete `scripts/quadruped_go2_crm.py` (492, duplicates the collector for video),
`drive_go2_collection.py` (261, superseded by `collect.py` + `machines.yaml`), and
`collect_go2_excitation.py` (535) -- but mine its docstring first, which holds the measured
excitation numbers, and note it injects i.i.d. per-step noise, which the plan rules out.

## The segmentation finding

**The hardest-sounding requirement in the plan is the cheapest item here.**

`dataset.py` needs **zero changes**. Windowing keys off `(episode_starts, episode_lengths)`
and never assumes a pair is a whole episode -- only that its rows are contiguous. Emit one
entry per SEGMENT and windowing is correct automatically.

The work is confined to `preprocess.py` emitting per-segment entries, and to keeping two
passes per-segment rather than per-file:

- **circular unwrap** (L358) runs `np.unwrap` per episode before slicing; across a dropped
  push window it would reintroduce the jump it exists to remove
- **the contact-mode Schmitt trigger** (L391) carries temporal state across samples for the
  same reason

Window arithmetic checks out against the plan: 1,475 rows at S=128 gives 1,348 windows;
seven 190-row segments give 441. The superlinearity warning is quantitatively right.
