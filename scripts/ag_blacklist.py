#!/usr/bin/env python3
"""All suite-id patterns of the arena study in one place, and a leak check (arena_gator_20260925, E1b).

E1 appended its 7 suite patterns to the three builder lists (ci_train.SUITE_GROUPS, ga_build_mixed.BLACKLIST,
ci_a5data.SUITE_PATTERNS). Those patterns name each arena (g260_test_group_* ...), so the 4 spread test suites added
in E1b (g258, g268, g263, g241) are NOT caught by the builders; existing scripts are not edited tonight. Until the
three lists are extended, anything that builds training data should either
  - import this module and call patch_builders() BEFORE using ci_train / ga_build_mixed / ci_a5data (in-process: the
    lists are extended in place, ci_a5data's equality check still holds), or
  - run assert_clean(ids) on every id / group it writes.

  python scripts/ag_blacklist.py                     report which patterns each builder list lacks
  python scripts/ag_blacklist.py --ids FILE.json     check a task file / list of ids (exit 1 on any suite id)
"""
import argparse, fnmatch, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

F104 = ['f104_crm_eval_group_*', 'f104_g1_test_group_*', 'f104_pair_group_*']
E1 = ['g260_test_group_*', 'g271_test_group_*', 'g251_test_group_*', 'g247_test_group_*', 'g203_heldout_group_*',
      'g228_heldout_group_*', 'g217_dev_group_*']
E1B = ['g258_test_group_*', 'g268_test_group_*', 'g263_test_group_*', 'g241_test_group_*']
ALL = F104 + E1 + E1B


def is_suite(s):
    return any(fnmatch.fnmatch(s, p) for p in ALL)


def assert_clean(ids):
    bad = [s for s in ids if is_suite(s)]
    assert not bad, f'{len(bad)} suite ids, e.g. {bad[:3]}'


def builder_lists():
    import ci_train, ga_build_mixed, ci_a5data
    return {'ci_train.SUITE_GROUPS': ci_train.SUITE_GROUPS, 'ga_build_mixed.BLACKLIST': ga_build_mixed.BLACKLIST,
            'ci_a5data.SUITE_PATTERNS': ci_a5data.SUITE_PATTERNS}


def missing_from_builders():
    return {k: [p for p in ALL if p not in v] for k, v in builder_lists().items()}


def patch_builders():
    """Extend the three builder lists in place (this process only) with the patterns they lack."""
    import ci_train, ga_build_mixed, ci_a5data
    for lst in (ci_train.SUITE_GROUPS, ga_build_mixed.BLACKLIST):
        lst += [p for p in ALL if p not in lst]
    ci_a5data.SUITE_PATTERNS = tuple(ci_a5data.SUITE_PATTERNS) + tuple(p for p in ALL if p not in ci_a5data.SUITE_PATTERNS)
    assert all(not v for v in missing_from_builders().values())
    return builder_lists()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ids', type=Path)
    a = ap.parse_args()
    print(json.dumps(missing_from_builders(), indent=1))
    if a.ids:
        d = json.loads(a.ids.read_text())
        ids = [x if isinstance(x, str) else x.get('group', x.get('id')) for x in d]
        bad = [s for s in ids if is_suite(s)]
        print(f'{len(ids)} ids, {len(bad)} suite ids' + (f', e.g. {bad[:3]}' if bad else ''))
        sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
