"""Tests of nedm.traversing.evaluation.suites (stdlib unittest only).

CI: subset grammar, md5 pair blocks, lock recomputation and tamper detection on synthetic locks, the Task's portable
round trip (paths under Env.data rebased on another machine's Env), hashing and file re-checks, make_case ->
custom:<dir> and its refusals (a custom route_00 is endpoint-checked) on a synthetic arena.
Release (NEDM_DATA = the release restore base): every released suite loads with its locks recomputed from the file
contents (SUITE_LOCKED 4327c110..., ALL_LOCKED c2d0022c... + E1b e2909946..., SUBSETS 05039510..., tracking_suite
a53b0eb6..., tasks_main ea670e3e...); the md5 selections equal the released and recorded lists (soil_unseen_subset.json,
f104_indist_200.json, the PR3 risk and tracker checks); approach routes have equal content in the a5 and a5data folders
(A12); md5 blocks of 8 equal the recorded rigid shards of rigid_eval_unseen.json. smoke144 also needs the release
index cache (its cases and routes are tar members).

    NEDM_DATA=/home/harry/NeDM-traverse_mppi PYTHONPATH=src python -m unittest discover -s tests/traversing/evaluation \\
        -p test_suites.py -v
"""
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from nedm.traversing.evaluation import suites as S
from nedm.traversing.evaluation.config import ConfigError, Env, ReleaseError
from nedm.traversing.evaluation.routes import route_sha256
from nedm.traversing.evaluation.suites import Task, blocks, load_suite, make_case, md5hex, select

try:
    from .common import CACHE, DATA
except ImportError:                     # discover -s tests/traversing/evaluation
    from common import CACHE, DATA

HERE = Path(__file__).resolve().parent
GOLD = HERE / 'goldens' / 'suites'
T = 'artifacts/traverse/'


def fake(n=12, arenas=('a', 'b')):
    return [Task(f'g_{i:03d}', arenas[i % len(arenas)], Path(f'/c/{i}.json'), sha=dict(case='0' * 64),
                 meta=dict(stratum='fresh' if i < 8 else 'reused', k=str(i % 3))) for i in range(n)]


def lock_text(entries, head_note='x'):
    h = hashlib.sha256()
    for n in sorted(entries):
        h.update(n.encode())
        h.update(bytes.fromhex(entries[n]))
    return f'{h.hexdigest()}  {head_note}\n' + ''.join(f'{d}  {n}\n' for n, d in entries.items())


def recompute(root, lines):
    """Independent recomputation from the file CONTENTS: sha256 over (name + sha256 digest of the file), sorted."""
    h = hashlib.sha256()
    for n in sorted(ln.split('  ', 1)[1] for ln in lines[1:]):
        h.update(n.encode())
        h.update(hashlib.sha256((root / n).read_bytes()).digest())
    return h.hexdigest()


class TestSelectBlocks(unittest.TestCase):
    def test_select(self):
        ts = fake()
        self.assertEqual(select(ts, None), ts)
        self.assertEqual([t.id for t in select(ts, 'stratum=reused')], ['g_008', 'g_009', 'g_010', 'g_011'])
        self.assertEqual({t.id for t in select(ts, 'k=0|2')}, {t.id for t in ts if t.meta['k'] != '1'})
        low = select(ts, 'stratum=fresh,lowest_md5=3')
        self.assertEqual({t.id for t in low}, set(sorted((t.id for t in ts[:8]), key=md5hex)[:3]))
        self.assertEqual([t.id for t in low], [t.id for t in ts if t in low])                   # suite order kept
        per = select(ts, 'lowest_md5_per_arena=2')
        self.assertEqual(Counter(t.arena for t in per), {'a': 2, 'b': 2})
        self.assertEqual({t.id for t in per if t.arena == 'a'},
                         set(sorted((t.id for t in ts if t.arena == 'a'), key=md5hex)[:2]))
        self.assertEqual([t.id for t in select(ts, 'arena=b,id=g_003|g_005')], ['g_003', 'g_005'])
        for bad in ('stratum=nope', 'colour=red', 'lowest_md5=0', 'lowest_md5=13', 'lowest_md5=x', 'stratum=',
                    'lowest_md5_per_arena=7', 'arena=a|zz'):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                select(ts, bad)

    def test_blocks(self):
        ts = fake(19)
        bs = blocks(ts, 8)
        self.assertEqual([len(b) for b in bs], [8, 8, 3])
        self.assertEqual([t.id for b in bs for t in b], sorted((t.id for t in ts), key=md5hex))
        for bad in (dict(tasks=ts, size=0), dict(tasks=ts + ts[:1], size=4)):
            with self.assertRaises(ConfigError):
                blocks(**bad)

    def test_task_roundtrip(self):
        here, there = Env(Path('/data/a')), Env(Path('/scratch/b'))
        t = Task('m', 'g203', Path('/data/a/x/c.json'), goals=((1.0, 2.0), (3.0, 4.0)), map=Path('/elsewhere/m'),
                 meta={'n_goals': 2}, sha=dict(case='1' * 64, map='2' * 64))
        d = json.loads(json.dumps(t.to_dict(here)))
        self.assertEqual((d['case'], d['map']), ('data:x/c.json', '/elsewhere/m'))       # restore path | absolute
        self.assertEqual(Task.from_dict(d, here), t)
        moved = Task.from_dict(d, there)
        self.assertEqual((moved.case, moved.map, moved.goals, moved.sha, moved.meta),
                         (Path('/scratch/b/x/c.json'), Path('/elsewhere/m'), t.goals, t.sha, t.meta))
        self.assertEqual((moved.kind, fake(1)[0].kind), ('mission', 'single_goal'))
        self.assertEqual(len({t, Task.from_dict(d, here)} | set(fake())), 13)               # hashable (meta, sha aside)
        with self.assertRaisesRegex(ValueError, r"no sha256 for \['case', 'route'\]"):
            Task('x', 'f104', Path('/c.json'), route=Path('/r.json'))


class TestLocks(unittest.TestCase):
    def test_recompute_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            files = {'b.json': b'1', 'a.json': b'22', 'dir/c.json': b'333'}
            entries = {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}
            text = lock_text(entries)
            (d / 'L.sha256').write_text(text)
            head = text.split()[0]
            self.assertEqual(S._lock(d / 'L.sha256', head=head), entries)
            self.assertEqual(S._lock(d / 'L.sha256', sha=hashlib.sha256(text.encode()).hexdigest()), entries)
            for bad, kw in ((text, dict(head='0' * 64)), (text, dict(sha='0' * 64)),
                            (text.replace(entries['a.json'], '0' * 64), {}),        # a digest changed, head kept
                            (text + f'{"1" * 64}  extra.json\n', {}),                 # a file added
                            (text.replace('a.json', 'A.json'), {})):                  # a name changed
                (d / 'L.sha256').write_text(bad)
                with self.subTest(kw=kw), self.assertRaises(ReleaseError):
                    S._lock(d / 'L.sha256', **kw)
            (d / 'x.json').write_bytes(b'{"a": 1}')
            self.assertEqual(S._json(d / 'x.json', hashlib.sha256(b'{"a": 1}').hexdigest()), {'a': 1})
            with self.assertRaisesRegex(ReleaseError, 'is not the locked'):
                S._read(d / 'x.json', '0' * 64)


class TestCustom(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.env = Env(d / 'data')
        for name, size in (('zz1', 80.0), ('zz2', 60.0)):
            a = d / f'data/assets/traverse/arena_{name}'
            a.mkdir(parents=True)
            (a / 'arena_000.bmp').write_bytes(b'BM' + name.encode())
            (a / 'arena_meta.json').write_text(json.dumps({'size_m': size}))
        self.out = d / 'cases'
        self.route = dict(waypoints=[[-10.0, 0.0], [0.0, 0.1], [10.1, 0.0]], speeds=[2.0, 2.0, 0.0],
                          stations=[0.0, 10.0, 20.1], headings=[0.0, 0.0, 0.0])

    def tearDown(self):
        self.tmp.cleanup()

    def test_make_case_and_load(self):
        p = make_case(self.out, 'my_case_1', 'zz1', (-10.0, 0.0), 0.0, (10.0, 0.0), route=self.route, env=self.env)
        make_case(self.out, 'my_case_2', 'zz1', (0.0, -20.0), 1.5, (0.0, 20.0), split='val', env=self.env)
        case = json.loads(p.read_text())
        self.assertEqual((case['id'], case['split'], case['arena'], case['layout']['assets'], case['goal_radius_m']),
                         ('my_case_1', 'test', 'assets/traverse/arena_zz1', [], 2.5))
        ts = load_suite(f'custom:{self.out}', env=self.env)
        self.assertEqual([(t.id, t.arena, t.map, t.kind) for t in ts],
                         [('my_case_1', 'zz1', None, 'single_goal'), ('my_case_2', 'zz1', None, 'single_goal')])
        self.assertEqual((ts[0].route, ts[1].route, ts[0].route00),
                         (self.out / 'routes/my_case_1/route.json', None, None))
        self.assertEqual([t.id for t in load_suite(f'custom:{self.out}', 'id=my_case_2', env=self.env)], ['my_case_2'])
        self.assertEqual((ts[0].read_case(), ts[0].read_route('route')['waypoints'].tolist()),
                         (case, self.route['waypoints']))
        p.write_text(p.read_text().replace('"horizon_s": 120.0', '"horizon_s": 60.0'))       # changed after loading
        with self.assertRaisesRegex(ReleaseError, 'is not the locked'):
            ts[0].read_case()

    def test_make_case_refusals(self):
        far = dict(self.route, waypoints=[[-9.0, 0.0], [0.0, 0.0], [10.0, 0.0]])
        for args, kw, err, msg in (
                (('a', 'zz1', (0, 0), 0.0, (1.0, 1.0)), {}, ConfigError, 'goal lies within'),
                (('a', 'zz1', (0, 0), 0.0, (45.0, 0.0)), {}, ConfigError, 'inside +-40 m'),
                (('a', 'zz1', (0, 0), float('nan'), (20.0, 0.0)), {}, ConfigError, 'yaw be finite'),
                (('a', 'zz1', (float('nan'), 0), 0.0, (20.0, 0.0)), {}, ConfigError, 'inside +-40 m'),
                (('a/b', 'zz1', (0, 0), 0.0, (20.0, 0.0)), {}, ConfigError, "id 'a/b'"),
                (('a', '../x', (0, 0), 0.0, (20.0, 0.0)), {}, ConfigError, "arena '../x'"),
                (('a', 'zz1', (0, 0), 0.0, (20.0, 0.0)), dict(split='eval'), ConfigError, "split 'eval'"),
                (('a', 'zz2', (0, 0), 0.0, (20.0, 0.0)), {}, ReleaseError, 'size_m must be 80'),
                (('a', 'zz3', (0, 0), 0.0, (20.0, 0.0)), {}, ReleaseError, 'no arena_000.bmp'),
                (('a', 'f104', (0, 0), 0.0, (20.0, 0.0)), {}, ReleaseError, 'release file missing'),
                (('a', 'zz1', (-10.0, 0.0), 0.0, (10.0, 0.0)), dict(route=far), ReleaseError, 'within 0.25 m')):
            with self.subTest(msg=msg), self.assertRaisesRegex(err, msg.replace('+', r'\+').replace('(', r'\(')):
                make_case(self.out, *args, env=self.env, **kw)
        make_case(self.out, 'a', 'zz1', (0, 0), 0.0, (20.0, 0.0), env=self.env)
        with self.assertRaisesRegex(ConfigError, 'exists'):
            make_case(self.out, 'a', 'zz1', (0, 0), 0.0, (20.0, 0.0), env=self.env)

    def test_custom_refusals(self):
        with self.assertRaisesRegex(ConfigError, 'no case files'):
            load_suite(f'custom:{self.out}', env=self.env)
        p = make_case(self.out, 'a', 'zz1', (0, 0), 0.0, (20.0, 0.0), env=self.env)
        case = json.loads(p.read_text())
        for bad, msg in ((dict(case, id='b'), "id 'b' != 'a'"), (dict(case, split='x'), "split 'x'"),
                         (dict(case, layout=dict(case['layout'], assets=[{}])), 'layout.assets must be []')):
            p.write_text(json.dumps(bad))
            with self.subTest(msg=msg), self.assertRaises(ReleaseError) as e:
                load_suite(f'custom:{self.out}', env=self.env)
            self.assertIn(msg, str(e.exception))
        with self.assertRaisesRegex(ConfigError, "suite 'f104'"):
            load_suite('f104', env=self.env)
        p.write_text(json.dumps(case))                                  # a custom route_00 is endpoint-checked too
        (self.out / 'routes/a').mkdir(parents=True)
        (self.out / 'routes/a/route_00.json').write_text(json.dumps(dict(self.route, waypoints=[[0.0, 0.0], [5.0, 0.1],
                                                                                                 [19.0, 0.0]])))
        with self.assertRaisesRegex(ReleaseError, 'route_00.json: needs .* within 0.25 m'):
            load_suite(f'custom:{self.out}', env=self.env)


@unittest.skipUnless(DATA, 'NEDM_DATA (the release restore base) is not set')
class TestReleasedSuites(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = Env.from_environ()
        cls.s = {n: load_suite(n, env=cls.env) for n in S.SUITES if n != 'smoke144'}

    def test_shapes(self):
        f, us, ur, tr, ms = (self.s[n] for n in S.SUITES[:5])
        self.assertEqual((len(f), len(us), len(ur), len(tr), len(ms)), (800, 1000, 2000, 423, 30))
        self.assertEqual(Counter(t.meta['stratum'] for t in f), {'fresh': 600, 'reused': 200})
        self.assertEqual(Counter(t.arena for t in us), {a: 125 for a in S.UNSEEN})
        self.assertEqual(Counter(t.arena for t in ur), {a: 250 for a in S.UNSEEN})
        self.assertEqual(Counter(t.meta['stratum'] for t in tr), {'feasible': 141, 'infeasible': 282})
        self.assertEqual(Counter(t.arena for t in ms), dict(f104=4, g203=2, g216=2, g217=2, g228=2, g231=2, g204=4,
                                                            g213=4, g223=4, g234=4))
        self.assertTrue(all(t.kind == 'mission' and 5 <= len(t.goals) <= 8 for t in ms))
        self.assertEqual({t.id for t in ms if t.map is None}, {t.id for t in ms if t.arena not in S.MAPS})
        for t in f:
            self.assertTrue(t.route00.is_file() and t.approach.is_file() and t.route is None and t.map == f[0].map)
        self.assertTrue(all(t.route and t.route00 is None for t in tr))
        self.assertTrue(all(t.route00.name == 'route_00.json' and t.map.name == f'arena_{t.arena}' for t in ur))

    def test_locks_recompute_from_contents(self):
        base = DATA / T / 'generalist_20260921/A_adapt/suite'
        self.assertEqual(recompute(base, (base / 'SUITE_LOCKED.sha256').read_text().splitlines()), S.LOCKS['f104_800'])
        root = DATA / T / 'arena_gator_20260925/suites'
        for name, key in (('ALL_LOCKED.sha256', 'E1'), ('ALL_LOCKED_E1b.sha256', 'E1b'),
                          ('SUBSETS_LOCKED_E1b.sha256', 'subsets')):
            self.assertEqual(recompute(DATA, (root / name).read_text().splitlines()), S.LOCKS[key], name)
        for arena in S.UNSEEN:                  # every case and designed route of the 8 unseen suites (26,008 files)
            lines = (root / f'test_{arena}.SUITE_LOCKED.sha256').read_text().splitlines()
            self.assertEqual(recompute(DATA, lines), lines[0].split()[0], arena)
            self.assertEqual(len(lines) - 1, 3251)
        for rel, key in ((T + 'generalist_20260921/B_tracker/suite/tracking_suite.json', 'tracker423'),
                         (T + 'fdm_f104_50h_20260909/nav_v1/local_luffy/tasks_main.json', 'missions30'),
                         (T + 'crm_improve_20260922/a5data/approach_suite_index.json', 'approach')):
            self.assertEqual(hashlib.sha256((DATA / rel).read_bytes()).hexdigest(), S.LOCKS[key], rel)

    def test_md5_selections_equal_the_recorded_lists(self):
        g = json.loads((GOLD / 'selections.json').read_text())
        for k in ('pr3_risk_groups', 'pr3_tracker_routes'):
            ids = [t.id for t in select(self.s[g[k]['suite']], g[k]['subset'])]
            self.assertEqual(sorted(ids, key=md5hex), g[k]['ids'], k)
        root = DATA / T / 'arena_gator_20260925/suites'
        subsets = S._lock(root / 'SUBSETS_LOCKED_E1b.sha256', head=S.LOCKS['subsets'])
        soil = S._json(root / 'soil_unseen_subset.json', subsets[T + 'arena_gator_20260925/suites/soil_unseen_subset.json'])
        self.assertEqual([t.id for t in self.s['unseen_soil1000']],
                         [t.id for t in select(self.s['unseen_rigid2000'], 'lowest_md5_per_arena=125')])
        self.assertEqual(sorted(t.id for t in self.s['unseen_soil1000']), sorted(soil['groups']))
        indist = S._json(root / 'f104_indist_200.json', subsets[T + 'arena_gator_20260925/suites/f104_indist_200.json'])
        hills = 'hill_cross_slope|hill_entry_cross_exit|crater_cross_slope|crater_entry_cross_exit'
        ids = [t.id for t in select(self.s['f104_800'], f'stratum=fresh,evaluation_stratum={hills},lowest_md5=200')]
        self.assertEqual(sorted(ids), sorted(indist['groups']))

    def test_approach_routes_equal_across_folders(self):
        """A12 / F17: the 3 s (a5) and 0.5 s (a5data) approach folders hold the same route content for all 800."""
        n = 0
        for t in self.s['f104_800']:
            a5 = json.loads((DATA / T / f'generalist_20260921/A_adapt/a5/approach/{t.id}.json').read_text())
            self.assertEqual(route_sha256(a5), route_sha256(json.loads(t.approach.read_text())), t.id)
            n += 1
        self.assertEqual(n, 800)

    def test_blocks_equal_recorded_rigid_shards(self):
        rows = json.loads(self.env.file(f'data:{T}arena_gator_20260925/e6/tasks/rigid_eval_unseen.json').read_text())
        shard = {}
        for r in rows:
            self.assertEqual(shard.setdefault(r['group'], r['shard']), r['shard'])
        bs = blocks(self.s['unseen_rigid2000'], 8)
        self.assertEqual(len(bs), 250)
        self.assertEqual({t.id: 3000 + i for i, b in enumerate(bs) for t in b}, shard)

    def test_tamper_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            base = d / T / 'generalist_20260921/A_adapt/suite'
            base.mkdir(parents=True)
            src = DATA / T / 'generalist_20260921/A_adapt/suite'
            for n in ('SUITE_LOCKED.sha256', 'suite.json'):
                (base / n).write_bytes((src / n).read_bytes())
            (base / 'suite.json').write_bytes((src / 'suite.json').read_bytes().replace(b'"fresh"', b'"frsh "', 1))
            with self.assertRaisesRegex(ReleaseError, 'suite.json: sha256 .* is not the locked'):
                S._f104('f104_800', Env(d))

    @unittest.skipUnless(CACHE, 'the release index cache (NEDM_RELEASE_CACHE) is not available')
    def test_smoke144(self):
        ts = load_suite('smoke144', env=self.env)
        sample = json.loads(self.env.file(f'data:{T}offroad_vehicles_20260927/scratch/S3/sample_A.json').read_text())
        self.assertEqual([t.id for t in ts], [r['pair_id'] for r in sample['rows']])
        self.assertEqual(Counter(t.meta['kind'] for t in ts), {'designed': 86, 'on_policy': 58})
        self.assertEqual(Counter(t.meta['stratum'] for t in ts).most_common(1)[0][1], 24)
        self.assertTrue(all(t.route.is_file() and t.arena == 'f104' for t in ts))


if __name__ == '__main__':
    unittest.main()
