# Cases for the arena study (module E1), declared 2026-09-25 before generation

All case sets are made with `scripts/gen_cases.py` (12 designed routes per group, 90/5/5 hash split written into
each case) from the repo root with `PYTHONPATH=src:scripts` and the nedm python; on-policy routes with
`scripts/f104_n2_onpolicy.py --n 8` (map-free: it reads only `route_00` and the start pose of each case and checks
the routes against the 40 m arena bound, 6 m/s and 0.125 /m curvature). Declared seeds (not changed afterwards):

| set | folder | arena | groups | strata | id prefix | seed | avoid |
|---|---|---|---|---|---|---|---|
| training pool | `train_g203/cases` | g203 | 1,200 | all | `g203_v2_group` | 2026092501203 | - |
| training pool | `train_g228/cases` | g228 | 1,200 | all | `g228_v2_group` | 2026092501228 | - |
| on-policy routes | `train_g203_onpolicy` | g203 | 8 per group | - | (group ids above) | 2026092505203 | - |
| on-policy routes | `train_g228_onpolicy` | g228 | 8 per group | - | (group ids above) | 2026092505228 | - |
| in-distribution held-out | `heldout_g203/cases` | g203 | 150 | feature | `g203_heldout_group` | 2026092502203 | `train_g203/cases`, 2 m |
| in-distribution held-out | `heldout_g228/cases` | g228 | 150 | feature | `g228_heldout_group` | 2026092502228 | `train_g228/cases`, 2 m |
| dev | `dev_g217/cases` | g217 | 150 | feature | `g217_dev_group` | 2026092503217 | - |
| unseen test | `test_g260/cases` | g260 | 250 | feature | `g260_test_group` | 2026092504260 | - |
| unseen test | `test_g271/cases` | g271 | 250 | feature | `g271_test_group` | 2026092504271 | - |
| unseen test | `test_g251/cases` | g251 | 250 | feature | `g251_test_group` | 2026092504251 | - |
| unseen test | `test_g247/cases` | g247 | 250 | feature | `g247_test_group` | 2026092504247 | - |

Training waves use `--wave arena_gator_train`; suites use `--wave arena_gator_suite`. The four test arenas are the
four closest of seeds 241-280 to f104 (`../arenas/selection.json`). Suite ids are blacklisted by name pattern in
`scripts/ci_train.py` (SUITE_GROUPS), `scripts/ga_build_mixed.py` (BLACKLIST) and `scripts/ci_a5data.py`
(SUITE_PATTERNS); lock files for every suite are in `../suites/`.

Note: `gen_cases.py` writes `"campaign": "fdm_f104_50h_20260909/gen_v1"` into every `cases.json` (a constant label in
that script, not a statement about where the cases belong) and every case's `family` is `f104_terrain_family`.

## Spread test arenas (module E1b, PLAN section 7 item 1), declared 2026-09-25 02:30 before generation

Four more unseen test arenas, chosen by `scripts/ag_spread_select.py` (`../arenas/selection_spread.json`: ranks 5-40 of
the E1 ranking in quarters, smallest BMP sha256 in each). Same command, strata, wave and id pattern as the near test
arenas above; seeds follow the same scheme (`2026092504` + arena number). No avoid set (no training pairs on these
arenas).

| set | folder | arena | groups | strata | id prefix | seed | avoid |
|---|---|---|---|---|---|---|---|
| unseen test (spread) | `test_g258/cases` | g258 | 250 | feature | `g258_test_group` | 2026092504258 | - |
| unseen test (spread) | `test_g268/cases` | g268 | 250 | feature | `g268_test_group` | 2026092504268 | - |
| unseen test (spread) | `test_g263/cases` | g263 | 250 | feature | `g263_test_group` | 2026092504263 | - |
| unseen test (spread) | `test_g241/cases` | g241 | 250 | feature | `g241_test_group` | 2026092504241 | - |

Declared subsets (E1b, from the ids only, by md5 of the group id as a hex string, lowest first):
- soil unseen subset: the 125 lowest-md5 groups of each of the 8 unseen test arenas (g260 g271 g251 g247 g258 g268
  g263 g241) -> `../suites/soil_unseen_subset.json`;
- f104 in-distribution reference: the 200 lowest-md5 hill/crater groups of `f104_pair_group_*` ->
  `../suites/f104_indist_200.json`;
- spread headroom scan: the 75 lowest-md5 groups of each spread arena's suite -> `../e3/tasks/spread_headroom_rows.json`.
