"""Tests of nedm.traversing.evaluation.config (stdlib unittest only).

CI: every refusal and off-matrix rule, collect-all, non-finite and mistyped values, tags and seeds against the goldens
written by the ORIGINAL 901d6c9 code (eval_class_staging/goldens/config_suites_goldens.py: planner_arms.arm_specs with
the deployed-tag line of ci_planner.py:634, checked there against 333 released pick summaries; nav_runner.py:145 seed
key), derived F / external, TOML round trip and sha, arm-file refusals, Env and the release resolver on a synthetic
manifest (import boundaries: test_imports.py).
Release (NEDM_DATA = the release restore base): every headline arm file validates against its suite with every input
checked against the pinned manifest and the checkpoints' kind and training ground (planner.model_info); input refusals
(world, decision frame, picks made for another arm or planner, soil step, models of another ground).

    NEDM_DATA=/home/harry/NeDM-traverse_mppi PYTHONPATH=src python -m unittest discover -s tests/traversing/evaluation \\
        -p test_config.py -v
"""
import csv
import functools
import gzip
import hashlib
import io
import json
import tempfile
import tomllib
import unittest
from dataclasses import replace
from pathlib import Path

from nedm.traversing.evaluation import config as C
from nedm.traversing.evaluation.config import ConfigError, Env, EvalConfig, Release, ReleaseError, load_arms

try:
    from .common import CACHE, DATA
except ImportError:                     # discover -s tests/traversing/evaluation
    from common import CACHE, DATA

HERE = Path(__file__).resolve().parent
GOLD = HERE / 'goldens'
ARMS = C.REPO_ROOT / 'configs/traversing/evaluation'
SUITE_OF = {'m1_navigation': ['missions30'], 'm2_shared_risk': ['f104_800'], 'm3_tracker': ['tracker423'],
            'm4a_unseen': ['unseen_soil1000', 'unseen_rigid2000'], 'm4b_vehicles': ['f104_800', 'unseen_soil1000'],
            'smoke': ['smoke144']}
BASE = dict(name='arm', ground='soil', soil_config='crm_main', planner='cem', models='data:m/x_s*.pt')


def cfg(**kw):
    return EvalConfig(**{**BASE, **kw})


def every_arm(path):
    """All arms of a file, one ground at a time (a name may be used on both grounds)."""
    doc = tomllib.loads(Path(path).read_text())
    grounds = {a.get('ground', doc.get('defaults', {}).get('ground', 'rigid')) for a in doc['arm']}
    return [a for g in sorted(grounds) for a in load_arms(path, ground=g)]


def toml_dumps(arms):
    """A minimal TOML writer for the round trip (strings, numbers, booleans, flat tables; None = key omitted)."""
    def val(v):
        if isinstance(v, bool):
            return 'true' if v else 'false'
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, str):
            return json.dumps(v)
        if isinstance(v, list):
            return '[' + ', '.join(map(val, v)) + ']'
        return '{' + ', '.join(f'{k} = {val(x)}' for k, x in v.items()) + '}'
    return ''.join('[[arm]]\n' + ''.join(f'{k} = {val(v)}\n' for k, v in a.to_dict().items() if v is not None) + '\n'
                   for a in arms)


class TestMatrix(unittest.TestCase):
    def test_headline_like_configs_are_clean(self):
        for c in (cfg(), cfg(planner='cem_grad'), cfg(approach_s=0.5), cfg(ground='rigid', soil_config=None, speed=2.0),
                  cfg(planner='straight', models=None, speed=6.0), cfg(ground='rigid', soil_config=None,
                                                                       planner='straight', models=None, speed=2.0),
                  cfg(planner='given', models=None, label='tracker', controller='tracker', actor=C.ACTOR_V2),
                  cfg(planner='given', models=None, label='tracker', controller='pid_held'),
                  cfg(vehicle='polaris', planner='cem_grad', label='rollback_belly'),
                  cfg(vehicle='gator', planner='given', models=None, label='goal_belly'),
                  cfg(ground='rigid', soil_config=None, planner='live', replan='waypoint', controller='nav_pid',
                      label='mission', build_lock='x.json'),
                  cfg(ground='rigid', soil_config=None, planner='live', replan=1.0, controller='nav_pid',
                      label='mission', build_lock='x.json', latency_replay='data:r/{task}/decisions.json')):
            self.assertEqual(c.validate(), [], c)
            self.assertTrue(c.validated, c)

    def test_every_refusal(self):
        live = dict(ground='rigid', soil_config=None, planner='live', replan='waypoint', controller='nav_pid',
                    label='mission', build_lock='x.json')
        cases = [
            (dict(vehicle='m113'), 'smoke-only tracked stand-in'),
            (dict(vehicle='tank'), "vehicle 'tank' is not one of"),
            (dict(ground='mud'), "ground 'mud' is not one of"),
            (dict(planner='mppi'), "planner 'mppi' is not one of"),
            (dict(update='adam'), "update 'adam' is not one of"),
            (dict(controller='lqr'), "controller 'lqr' is not one of"),
            (dict(label='fail'), "label 'fail' is not one of"),
            (dict(name='a__b'), 'single underscores'),
            (dict(vehicle='polaris', ground='rigid', soil_config=None, label='rollback_belly'), 'no rigid Polaris path'),
            (dict(soil_config=None), 'soil needs soil_config'),
            (dict(ground='rigid'), 'soil_config is for soil only'),
            (dict(soil_config='crm_m113'), '"crm_main" or a JSON path'),
            (dict(approach_s=0.52), 'a multiple of 0.05 s in [0, 120)'),
            (dict(approach_s=120.0), 'a multiple of 0.05 s in [0, 120)'),
            (dict(planner='straight', models=None, approach_s=0.5), 'the two-pass protocol plans at F'),
            (dict(planner='given', models=None, label='goal_belly', vehicle='gator', approach_s=0.5), 'two-pass'),
            (dict(decisions='data:x/poses_crm_all.json'), 'decisions are moving-start states'),
            (dict(models=None), 'planner cem needs models'),
            (dict(planner='straight'), 'models are not used by planner straight'),
            (dict(planner='cem_grad', speed=2.0), 'speed with planner cem_grad'),
            (dict(speed=7.0), 'route speeds are clipped to [0.5, 6]'),
            (dict(planner='straight', models=None, speed=3.0), 'offset-0 anchor at one of (2.0, 4.0, 6.0)'),
            (dict(rounds=0), 'rounds must be >= 2 and samples >= 1'),
            (dict(rounds=1, samples=256), 'one-shot pools, arms A and D, are dropped'),
            (dict(samples=0), 'rounds must be >= 2 and samples >= 1'),
            (dict(planner='straight', models=None, rounds=8), 'are not used by planner straight'),
            (dict(grad={'steps': 10}), 'grad overrides need planner cem_grad'),
            (dict(planner='cem_grad', grad={'stepz': 10}), "grad ['stepz']: not keys of GRAD"),
            (dict(planner='cem_grad', grad={'steps': '60'}), "grad ['steps']: not keys of GRAD or not of their types"),
            (dict(planner='cem_grad', grad={'betas': [0.9], 'lr_a': float('inf'), 'keep': 1}),
             "grad ['betas', 'lr_a', 'keep']"),
            (dict(planner='given', models=None, picks='data:p', label='goal_belly', vehicle='gator'), 'locked picks'),
            (dict(approach_s=0.5, picks='data:p'), 'locked picks after an approach need decisions'),
            (dict(live, vehicle='gator'), 'M1 ran the HMMWV on rigid ground only'),
            (dict(live, controller='pid'), 'built around nav_pid'),
            (dict(live, build_lock=None), 'live needs build_lock'),
            (dict(live, replan=None), 'live needs replan'),
            (dict(replan=2.0), 'replan is for planner live only'),
            (dict(latency_replay='data:x'), 'latency_replay is for planner live only'),
            (dict(live, replan=0.07), 'a positive multiple of 0.05 s'),
            (dict(live, replan='sometimes'), '"waypoint" or a positive multiple'),
            (dict(live, latency_replay='data:x'), 'latency_replay needs a periodic replan'),
            (dict(controller='nav_pid'), 'nav_pid is the M1 mission controller'),
            (dict(vehicle='polaris', planner='given', models=None, label='goal_belly', controller='tracker',
                  actor=C.ACTOR_V2), 'tracker on the polaris: only driven on the HMMWV'),
            (dict(planner='given', models=None, label='tracker', controller='tracker'), 'the tracker needs actor'),
            (dict(actor=C.ACTOR_V2), 'actor is for the tracker only'),
            (dict(label='rollback_belly'), 'rollback_belly on the hmmwv: it has no belly points'),
            (dict(label='mission'), 'label mission <=> planner live'),
            (dict(live, label='rollback'), 'label mission <=> planner live'),
            (dict(rounds='4'), 'rounds has type str'),
            (dict(allow_unvalidated=1), 'allow_unvalidated has type int'),
            (dict(rounds=True), 'rounds has type bool'),
            (dict(live, replan=float('inf')), 'replan = inf is not finite'),
            (dict(approach_s=float('nan')), 'approach_s = nan is not finite'),
        ]
        for kw, msg in cases:
            with self.subTest(kw=kw):
                problems = cfg(**kw).validate()
                self.assertTrue(any(msg in p for p in problems), (msg, problems))
                self.assertFalse(any(p.startswith('outside the validated matrix') and msg in p for p in problems), msg)
                allowed = replace(cfg(**kw), allow_unvalidated=True) if 'type' not in msg else None
                if allowed is not None:                      # a refusal is never lifted by the escape hatch
                    self.assertTrue(any(msg in p for p in allowed.validate()), msg)

    def test_off_matrix_needs_allow_unvalidated(self):
        cases = [(dict(vehicle='gator', ground='rigid', soil_config=None), 'gator on rigid ground'),
                 (dict(vehicle='polaris_4wd', planner='given', models=None, label='goal_belly'), 'smoke-only variant'),
                 (dict(soil_config='/tmp/other.json'), 'soil config /tmp/other.json'),
                 (dict(approach_s=2.0), 'approach_s 2.0 (recorded: 0.5, 1, 3 s)'),
                 (dict(vehicle='gator', approach_s=0.5), 'an approach on the gator'),
                 (dict(rounds=8), '8 x 64 sampling'), (dict(rounds=2, samples=128), '2 x 128 sampling'),
                 (dict(update='mppi'), 'update mppi (every headline'),
                 (dict(seed_tag='mine'), 'seed_tag or grad overrides'),
                 (dict(planner='cem_grad', grad={'steps': 5}), 'seed_tag or grad overrides'),
                 (dict(speed=2.0), 'fixed-speed family at 2.0 m/s on soil'),
                 (dict(planner='straight', models=None, speed=2.0), 'straight at 2.0 m/s on soil'),
                 (dict(planner='straight', models=None, speed=4.0), 'straight at 4.0 m/s on soil'),
                 (dict(ground='rigid', soil_config=None, planner='live', replan=0.5, controller='nav_pid',
                       label='mission', build_lock='x'), 'replan 0.5 (recorded'),
                 (dict(ground='rigid', soil_config=None, planner='live', replan=2.0, controller='nav_pid',
                       label='mission', build_lock='x', latency_replay='data:x'), 'latency replay with a period'),
                 (dict(controller='pid_held'), 'pid_held on planned routes'),
                 (dict(planner='given', models=None, label='tracker', controller='tracker', actor='data:mine.npz'),
                  'an actor other than the released'),
                 (dict(planner='given', models=None), 'label rollback with planner given on the hmmwv'),
                 (dict(vehicle='polaris', label='rollback'), 'label rollback with planner cem on the polaris')]
        for kw, msg in cases:
            with self.subTest(kw=kw):
                c = cfg(**kw)
                self.assertFalse(c.validated)
                problems = c.validate()
                self.assertEqual(len(problems), 1, problems)
                self.assertTrue(problems[0].startswith('outside the validated matrix') and msg in problems[0], problems)
                self.assertEqual(replace(c, allow_unvalidated=True).validate(), [])
                self.assertFalse(replace(c, allow_unvalidated=True).validated)

    def test_collect_all(self):
        c = EvalConfig(name='bad name', vehicle='polaris', ground='rigid', planner='cem_grad', speed=9.0,
                       controller='tracker', label='mission', approach_s=0.33, replan=2.0)
        problems = c.validate()
        for msg in ('single underscores', 'no rigid Polaris path', 'a multiple of 0.05 s', 'needs models',
                    'speed with planner cem_grad', 'clipped to [0.5, 6]', 'replan is for planner live only',
                    'tracker on the polaris', 'the tracker needs actor', 'label mission <=> planner live',
                    'outside the validated matrix'):
            self.assertTrue(any(msg in p for p in problems), (msg, problems))
        self.assertGreaterEqual(len(problems), 12)
        with self.assertRaises(ConfigError) as e:
            raise ConfigError(problems)
        self.assertEqual(e.exception.problems, problems)
        self.assertEqual(str(e.exception).count('\n  - '), len(problems))

    def test_types_first(self):
        problems = EvalConfig(name='a', rounds=4.0, speed='2', grad=[1]).validate()
        self.assertEqual(sorted(p.split()[0] for p in problems), ['grad', 'rounds', 'speed'])
        problems = EvalConfig(name='a', speed=float('-inf'), approach_s=float('nan')).validate()
        self.assertEqual(problems, ['approach_s = nan is not finite', 'speed = -inf is not finite'])
        with self.assertRaisesRegex(ConfigError, 'grad'):                  # not JSON: no sha, reported at once
            EvalConfig(name='a', planner='cem_grad', grad={'steps': object()})


class TestDerived(unittest.TestCase):
    def test_tags_equal_original(self):
        """planner_arms.arm_specs (+ the ci_planner.py:634 deployed tag) for arms B and C, both worlds, free and fixed2
        (the one-shot arms A and D are dropped); tag and seed md5(id + tag) per arm."""
        g = json.loads((GOLD / 'config/tags.json').read_text())
        self.assertEqual(sorted({s['arm'] for s in g['sampling']}), ['B', 'C'])
        self.assertEqual(len(g['sampling']), 8)
        self.assertGreater(len(g['recorded']), 300)
        for s in g['sampling']:
            c = cfg(ground=s['ground'], soil_config='crm_main' if s['ground'] == 'soil' else None, speed=s['speed'],
                    rounds=s['rounds'], samples=s['samples'])
            with self.subTest(s=s):
                self.assertEqual(c.tag, s['tag'])
                self.assertEqual(replace(c, planner='cem_grad', speed=None).tag if s['speed'] is None else c.tag,
                                 s['tag'])
                self.assertEqual({gid: c.seed(gid) for gid in s['seeds']}, s['seeds'])
        for s in g['live']:
            c = cfg(ground='rigid', soil_config=None, planner='live', replan=s['replan'], controller='nav_pid',
                    label='mission', build_lock='x')
            self.assertEqual((c.tag, c.seed('f104_nav_000')), (s['key'], s['seed_f104_nav_000']))
        self.assertEqual(replace(cfg(ground='rigid', soil_config=None, planner='live', controller='nav_pid',
                                     label='mission', build_lock='x'), replan=2).tag, 'periodic2.0')   # int -> 2.0
        self.assertEqual(cfg(ground='rigid', soil_config=None, speed=2).tag, 'n2iter_cem4x64_fixed2')  # int -> 2.0
        self.assertEqual(cfg(seed_tag='mine').tag, 'mine')
        self.assertIsNone(cfg(planner='straight', models=None).tag)
        self.assertIsNone(cfg(planner='given', models=None).tag)
        with self.assertRaisesRegex(ValueError, 'uses no rng'):
            cfg(planner='given', models=None).seed('g')

    def test_frame_and_external(self):
        for a, f in ((0.0, 0), (0.5, 10), (1.0, 20), (3.0, 60), (3, 60)):
            self.assertEqual(cfg(approach_s=a).F, f)
        self.assertEqual([cfg(controller=k).external for k in C.CONTROLLERS], [False, True, True, False])

    def test_sha_and_roundtrip(self):
        self.assertEqual(cfg(approach_s=3).sha, cfg(approach_s=3.0).sha)
        self.assertEqual(cfg(grad={'betas': (0.9, 0.99)}).sha, cfg(grad={'betas': [0.9, 0.99]}).sha)
        self.assertNotEqual(cfg().sha, cfg(samples=65).sha)
        for f in sorted(ARMS.glob('*.toml')):
            arms = every_arm(f)
            with tempfile.TemporaryDirectory() as d:
                (Path(d) / 'a.toml').write_text(toml_dumps(arms))
                again = every_arm(Path(d) / 'a.toml')
            self.assertEqual([a.sha for a in again], [a.sha for a in arms], f.name)
            self.assertEqual([EvalConfig.from_dict(json.loads(json.dumps(a.to_dict()))).sha for a in arms],
                             [a.sha for a in arms])

    def test_headline_files_offline(self):
        want = {'m1_navigation': 4, 'm2_shared_risk': 6, 'm3_tracker': 6, 'm4a_unseen': 4, 'm4b_vehicles': 5, 'smoke': 5}
        self.assertEqual(sorted(p.stem for p in ARMS.glob('*.toml')), sorted(want))
        for stem, n in want.items():
            arms = every_arm(ARMS / f'{stem}.toml')
            self.assertEqual(len(arms), n, stem)
            for a in arms:
                with self.subTest(arm=(stem, a.name, a.ground)):
                    self.assertEqual(a.validate(), [])
                    self.assertEqual(a.validated, not a.allow_unvalidated)
        m2 = {(a.name, a.ground): a for a in every_arm(ARMS / 'm2_shared_risk.toml')}
        self.assertEqual((m2['shared_hist_early_rows_0p5s_grad', 'soil'].F, m2['shared_hist_3s', 'rigid'].F), (10, 60))
        self.assertEqual([a.ground for a in load_arms(ARMS / 'm3_tracker.toml', ground='soil')], ['soil'] * 3)


class TestArmFiles(unittest.TestCase):
    def load(self, text, **kw):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'a.toml').write_text(text)
            return load_arms(Path(d) / 'a.toml', **kw)

    def test_defaults_merge(self):
        arms = self.load('[defaults]\nground = "rigid"\nplanner = "straight"\n'
                         '[[arm]]\nname = "a"\n[[arm]]\nname = "b"\nplanner = "given"\n')
        self.assertEqual([(a.name, a.ground, a.planner) for a in arms], [('a', 'rigid', 'straight'), ('b', 'rigid', 'given')])

    def test_refusals(self):
        for text, msg in (('[[arm]]\nname = "a"\nvehicel = "gator"\n', "unknown key 'vehicel'"),
                          ('[[arm]]\nground = "soil"\n', 'name is required'),
                          ('[[arm]]\nname = "a"\n[[arm]]\nname = "a"\n', "arm ('a', 'rigid') appears 2 times"),
                          ('[arms]\nname = "a"\n', "unknown top-level key 'arms'"),
                          ('[defaults]\nground = "rigid"\n', 'no [[arm]] tables'),
                          ('defaults = 3\n[[arm]]\nname = "a"\n', '[defaults] must be a table'),
                          ('arm = [1]\n', '[defaults] must be a table and the arms [[arm]] tables'),
                          ('[[arm]]\nname = "a"\nplanner = "cem_grad"\ngrad = {steps = 1979-05-27}\n',
                           "arm 0: grad {'steps'"),
                          ('[[arm]]\nname = "a"\n[[arm]]\nname = "a"\nground = "soil"\n',
                           "arms ['a'] exist on both grounds: pass ground")):
            with self.subTest(msg=msg), self.assertRaises(ConfigError) as e:
                self.load(text)
            self.assertTrue(any(msg in p for p in e.exception.problems), e.exception.problems)
        with self.assertRaises(ConfigError):
            self.load('[[arm]]\nname = "a"\n', ground='mud')
        self.assertEqual([a.ground for a in self.load('[[arm]]\nname = "a"\n[[arm]]\nname = "a"\nground = "soil"\n',
                                                      ground='soil')], ['soil'])


class TestEnvRelease(unittest.TestCase):
    def test_from_environ(self):
        with tempfile.TemporaryDirectory() as d:
            e = Env.from_environ({'NEDM_DATA': d, 'NEDM_DEVICE': 'cpu'})
            self.assertEqual((e.data, e.device, e.chrono_data), (Path(d), 'cpu', None))
            for env, msg in (({}, 'NEDM_DATA (the release restore base) is not set'),
                             ({'NEDM_DATA': d, 'NEDM_VEHICLE': 'gator'}, 'NEDM_VEHICLE is set'),
                             ({'NEDM_DATA': d + '/nope'}, 'is not a directory'),
                             ({'NEDM_DATA': d, 'NEDM_CHRONO_DATA': d + '/nope'}, 'NEDM_CHRONO_DATA')):
                with self.subTest(env=env), self.assertRaises(ConfigError) as ex:
                    Env.from_environ(env)
                self.assertTrue(any(msg in p for p in ex.exception.problems), ex.exception.problems)
            self.assertEqual(Env.expand('data:x/{arena}/{task}', type('T', (), {'arena': 'g260', 'id': 'g_1'})),
                             'data:x/g260/g_1')
            with self.assertRaises(ConfigError):
                Env.expand('data:x/{arena}')

    def test_resolver_on_a_synthetic_release(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            data, cache = d / 'data', d / 'cache'
            files = {'art/m/x_s0.pt': b'zero', 'art/m/x_s1.pt': b'one', 'art/p.json': b'{}'}
            members = {'art/t/a.json': b'[1]', 'art/t/b.json': b'[2]'}
            for rel, b in {**files, **members}.items():
                (data / rel).parent.mkdir(parents=True, exist_ok=True)
                (data / rel).write_bytes(b)
            idx = io.StringIO()
            w = csv.writer(idx)
            w.writerow(['path', 'bytes', 'sha256', 'shard'])
            for rel, b in members.items():
                w.writerow([rel, len(b), hashlib.sha256(b).hexdigest(), 'part-00000.tar.gz'])
            hub = 'traversing/evaluation/tars/index.csv.gz'
            (cache / hub).parent.mkdir(parents=True)
            (cache / hub).write_bytes(gzip.compress(idx.getvalue().encode()))
            man = {'path_remap': {'prefixes': {'/work1/x/': 'art/'}}, 'items': {
                'plain': {'bundle': 'models', 'restore_root': 'art', 'files': [f'traversing/models/plain/{r[4:]}'
                                                                                 for r in files]},
                'tars': {'bundle': 'evaluation', 'restore_root': 'art/t', 'files': [hub]}},
                'files': {**{f'traversing/models/plain/{r[4:]}': dict(kind='file', sha256=hashlib.sha256(b).hexdigest(),
                                                                     bytes=len(b)) for r, b in files.items()},
                          hub: dict(kind='index', sha256=hashlib.sha256((cache / hub).read_bytes()).hexdigest())}}
            (d / 'm.json').write_text(json.dumps(man))
            rel = Release(data, cache, d / 'm.json')
            self.assertEqual(rel.file('art/p.json'), data / 'art/p.json')
            self.assertEqual(rel.file('art/t/b.json'), data / 'art/t/b.json')             # tar member via the index
            self.assertEqual(rel.glob('art/m/x_s*.pt'), [data / 'art/m/x_s0.pt', data / 'art/m/x_s1.pt'])
            self.assertEqual(rel.remap('/work1/x/t/a.json'), 'art/t/a.json')
            (data / 'art/m/x_s2.pt').write_bytes(b'extra')                                  # not released
            with self.assertRaisesRegex(ReleaseError, 'x_s2.pt is not a file of the pinned release'):
                rel.glob('art/m/x_s*.pt')
            (data / 'art/m/x_s2.pt').unlink()
            (data / 'art/t/a.json').write_bytes(b'[9]')                                    # tampered
            (data / 'art/m/x_s1.pt').rename(data / 'art/m/gone')
            for call, msg in ((lambda: rel.glob('art/m/x_s*.pt'), 'x_s1.pt: release file missing'),
                              (lambda: rel.file('art/t/a.json'), 'content differs'),
                              (lambda: rel.file('art/m/x_s1.pt'), 'release file missing'),
                              (lambda: rel.file('art/../art/p.json'), 'not a normalised restore path'),
                              (lambda: rel.file('art/q.json'), 'not a file of the pinned release'),
                              (lambda: rel.glob('art/*/x.pt'), 'wildcards only in the last path component'),
                              (lambda: Release(data, d / 'nocache', d / 'm.json').file('art/t/b.json'),
                               'set NEDM_RELEASE_CACHE')):
                with self.subTest(msg=msg), self.assertRaises(ReleaseError) as e:
                    call()
                self.assertIn(msg, str(e.exception))
            (cache / hub).write_bytes(gzip.compress(b'path,bytes,sha256,shard\n'))
            with self.assertRaises(ReleaseError) as e:
                Release(data, cache, d / 'm.json').file('art/t/b.json')
            self.assertIn('differs from the pinned manifest', str(e.exception))


@unittest.skipUnless(DATA, 'NEDM_DATA (the release restore base) is not set')
@unittest.skipUnless(CACHE, 'the release index cache (NEDM_RELEASE_CACHE) is not available: tar members cannot be checked')
class TestReleaseInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nedm.traversing.evaluation.planner import model_info
        from nedm.traversing.evaluation.suites import load_suite
        cls.env, cls.info = Env.from_environ(), staticmethod(functools.lru_cache(maxsize=None)(model_info))
        cls.tasks = {s: load_suite(s, env=cls.env) for s in {s for v in SUITE_OF.values() for s in v}}

    def test_headline_arms_validate_with_their_inputs(self):
        n = 0
        for stem, suites in SUITE_OF.items():
            for a in every_arm(ARMS / f'{stem}.toml'):
                for s in suites:
                    if (stem == 'm4a_unseen' and (s == 'unseen_soil1000') != (a.ground == 'soil')) \
                            or (stem == 'm4b_vehicles' and (s == 'unseen_soil1000') != (
                                a.name.endswith('_unseen') or a.name == 'polaris_straight_6mps')):
                        continue
                    with self.subTest(arm=(stem, a.name, a.ground, s)):
                        want = ['build lock'] if stem == 'm1_navigation' else []   # written by fingerprint.py later
                        got = a.validate(self.env, tasks=self.tasks[s], model_info=self.info)
                        self.assertEqual([p.split()[0] + ' ' + p.split()[1] for p in got], want, got)
                        n += 1
        self.assertEqual(n, 30)

    def test_input_refusals(self):
        m2 = {(a.name, a.ground): a for a in every_arm(ARMS / 'm2_shared_risk.toml')}
        soil, f104 = m2['shared_hist_early_rows_0p5s_grad', 'soil'], self.tasks['f104_800']
        rigid, hn = m2['shared_hist_early_rows_0p5s_grad', 'rigid'], m2['shared_hist_early_rows_0p5s', 'soil']
        with tempfile.NamedTemporaryFile('w', suffix='.json') as bad:
            json.dump({'step_s': 0.003}, bad)
            bad.flush()
            for c, msg in ((replace(soil, decisions=rigid.decisions), 'not a poses_crm[_all].json file'),
                           (replace(soil, approach_s=1.0), 'needs F = 20'),
                           (replace(soil, picks=rigid.picks), 'were made for (world, models, mode, moving start, arms)'),
                           (replace(m2['shared_hist_3s', 'soil'], approach_s=0.0, decisions=None),
                            "'artifacts/traverse/generalist_20260921/A_adapt/train/deploy_v1/H_deploy_s*.pt', None, True"),
                           (replace(soil, picks=m2['shared_hist_3s', 'soil'].picks), 'were made for'),
                           (replace(soil, planner='cem'), "True, ('B', 'G')), not this arm"),      # G folder, B arm
                           (replace(hn, planner='cem_grad'), "True, ('B',)), not this arm"),       # B folder, G arm
                           (replace(soil, models=soil.models.replace('s*.pt', 't*.pt')), 'matches no release file'),
                           (replace(soil, picks='data:artifacts/traverse/{arena}/x'), 'PICKS_LOCKED.sha256'),
                           (replace(soil, soil_config=bad.name, allow_unvalidated=True), 'must be given and divide')):
                with self.subTest(msg=msg):
                    problems = c.validate(self.env, tasks=f104, model_info=self.info)
                    self.assertTrue(any(msg in p for p in problems), problems)
        self.assertIn('no state for 1 tasks', ' '.join(soil.validate(
            self.env, tasks=[replace(f104[0], id='f104_pair_group_9999')], model_info=self.info)))
        self.assertIn('is templated: validate with the suite tasks',
                      ' '.join(replace(soil, picks='data:x/{arena}/p').validate(self.env, model_info=False)))
        self.assertEqual(soil.validate(self.env, tasks=f104, model_info=lambda p: ('ci_train', 'both')), [])
        self.assertEqual(soil.validate(self.env, tasks=f104, model_info=False), [])       # a drive worker, explicit
        self.assertIn('kind and training ground not checked', ' '.join(soil.validate(self.env, tasks=f104)))
        self.assertIn('cem_grad needs a ci_train ensemble', ' '.join(soil.validate(
            self.env, tasks=f104, model_info=lambda p: ('ga_train', 'both'))))
        self.assertIn('need one of', ' '.join(soil.validate(self.env, tasks=f104, model_info=lambda p: (p.name[-4], None))))

    def test_training_ground(self):
        """A model trained on soil does not plan on rigid ground (ag_picks.py:191-192), not even unvalidated; the shared
        models ('both') and legacy models (no domain_filter) plan on either."""
        soil_models = 'data:artifacts/traverse/arena_gator_20260925/e5/deploy/A3_soil/A3_soil_deploy_s*.pt'
        c = EvalConfig(name='x', planner='cem', models=soil_models)
        want = [f"models {soil_models} were trained on ['crm'] (domain_filter), not for rigid"]
        self.assertEqual(c.validate(self.env, model_info=self.info), want)
        self.assertEqual(replace(c, allow_unvalidated=True).validate(self.env, model_info=self.info), want)
        self.assertEqual(replace(c, ground='soil', soil_config='crm_main').validate(self.env, model_info=self.info), [])
        for models in ('crm_improve_20260922/deploy_v1/deploy_a1_haux_gru_s*.pt', 'crm_f104_v1/train_v1/deploy/CRM_N2_s*.pt'):
            with self.subTest(models):
                self.assertEqual(replace(c, models='data:artifacts/traverse/' + models).validate(
                    self.env, model_info=self.info), [])


if __name__ == '__main__':
    unittest.main()
