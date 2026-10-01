"""The run folder without Chrono (FINAL_DESIGN 4.4-4.5, 8.1 test 6): every drive is a fake drive process (the real
spawn with a patched subprocess.run that writes the record a drive would) and the node is patched. No partial run is
counted and DONE is written last and re-checked; a resume on the pinned node re-drives only unfinished arms, a resume
on another node re-drives every pair with an unfinished arm whole (pair-atomic) keeping the picks no pass 1 made; the
plan stage neither pins nor moves; the pairing invariant, the refusals (build lock, code, GPUs, walltime guard) and a
drive's deterministic input problems refused before any Chrono object (exit 3); the two-pass episode (pass 1 at
horizon L, the decision at F, the branch, the early-ended prefix, no route); the missing-drive policy on crash cells.

    PYTHONPATH=src python -m unittest discover -s tests/traversing/evaluation -p test_runner.py -v
"""
import hashlib
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from nedm.traversing.evaluation import runner as R
from nedm.traversing.evaluation.config import ConfigError, Env, EvalConfig
from nedm.traversing.evaluation.planner import Pick
from nedm.traversing.evaluation.routes import route_json, route_sha256
from nedm.traversing.evaluation.suites import Task, md5hex

from . import tables as TB
from .test_episode import route

GIVEN = dict(planner='given', label='tracker', allow_unvalidated=True)          # no build_lock: unvalidated
ARMS = (EvalConfig(name='pid_native', **GIVEN), EvalConfig(name='pid_held_50ms', controller='pid_held', **GIVEN))


class Killed(Exception):
    """The job ends in the middle of a drive."""


class FakeDrive:
    """A drive process: per (task, arm, 'pass1' | 'drive') an outcome: a status (n frames), 'fail' (exit 1 after a
    partial file) or 'kill'. Default: goal_reached after 30 frames, a pass 1 times out at its horizon."""

    def __init__(self, **plan):
        self.plan, self.calls = plan, []

    def __call__(self, cmd, **kw):
        j, out = json.loads(Path(cmd[cmd.index('--input') + 1]).read_text()), Path(cmd[cmd.index('--out') + 1])
        key = (j['task']['id'], j['arm']['name'], 'pass1' if j['horizon_s'] < 120. else 'drive')
        self.calls.append(key)
        status, n = self.plan.get('/'.join(key), ('timeout', round(j['horizon_s'] / .05)) if key[2] == 'pass1'
                                  else ('goal_reached', 30))
        if status == 'kill':
            raise Killed(key)
        if status == 'fail':
            (out / 'trajectory.npz').write_bytes(b'partial')
            return SimpleNamespace(returncode=1)
        st = np.zeros((n, 17), np.float32)
        st[:, 0] = 2.
        a = dict(state=st, action=np.zeros((n, 3), np.float32), pose=np.c_[np.linspace(0., 19., n), np.zeros((n, 2))],
                 parked=np.zeros(n, bool), desired_speed_mps=np.full(n, 2.), terminal_pose=np.array([1., 0., 0.]),
                 terminal_state=st[-1])
        R.Record(key[0], key[1], status, True, frames=n, elapsed_s=n * .05, positive_work_kj=1.,
                 route_sha256=route_sha256(j['route']), provenance=dict(host='fake'), arrays=a).save(
            out, [('route.json', j['route'])])
        return SimpleNamespace(returncode=0)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def node(host='nodeA', build='b1', code=None):
    return lambda env: dict(host=host, job_id=None, build_sha=build, fingerprint={}, code_sha=code or R.code_sha())


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.env = Env(self.tmp / 'data', self.tmp / 'chrono', 'cpu')
        self.out = self.tmp / 'run'
        self.tasks = [self.task(f'g{k:04d}') for k in range(5)]

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp)

    def task(self, tid):
        case = dict(id=tid, split='test', layout=dict(start_xy=[0., 0.], start_yaw=0., assets=[]), goal_xy=[20., 0.],
                    goal_radius_m=2.5)
        (cp := self.tmp / f'{tid}.json').write_text(json.dumps(case))
        (rp := self.tmp / f'{tid}_route.json').write_text(json.dumps({k: np.asarray(v).tolist()
                                                                       for k, v in route(0, 1.).items()}))
        return Task(tid, 'f104', cp, route=rp, approach=rp, sha={k: sha(p) for k, p in dict(case=cp, route=rp,
                                                                                             approach=rp).items()})

    def run_arms(self, fake, arms=ARMS, host='nodeA', **kw):
        evals = [R.TraversalEval(c, self.out, self.env) for c in arms]
        with mock.patch.object(R.subprocess, 'run', side_effect=fake), mock.patch.object(R, 'node_info', node(host)):
            return R.run_arms(evals, self.tasks, **kw)

    def block(self, fake, host='nodeA', **kw):
        with mock.patch.object(R.subprocess, 'run', side_effect=fake), mock.patch.object(R, 'node_info', node(host, **{
                k: kw.pop(k) for k in ('build', 'code') if k in kw})):
            return R.run_block(self.out, 0, env=self.env, **kw)

    def done(self, t, arm):
        return (self.out / 'runs' / t / arm / 'DONE').exists()


class TestRunFolder(Base):
    def test_done_last_no_partial_and_same_node_resume(self):
        """A crash (2 failed attempts) is DONE with status crash; a killed drive leaves no DONE and is not counted;
        DONE is the last file and lists every other file; the same node then re-drives only the unfinished arm."""
        t1, t2 = self.tasks[1].id, self.tasks[2].id
        fake = FakeDrive(**{f'{t1}/pid_held_50ms/drive': ('fail', 0), f'{t2}/pid_held_50ms/drive': ('kill', 0)})
        with self.assertRaisesRegex(RuntimeError, '1 pairs failed'):
            self.run_arms(fake, workers=2)
        cells = R.cells(self.out, self.env)
        self.assertEqual((cells[t1]['pid_held_50ms'], set(cells[t2])), ('crash', {'pid_native'}))
        self.assertEqual(json.loads((self.out / f'runs/{t1}/pid_held_50ms/attempts.jsonl').read_text().splitlines()[-1])
                         ['attempt'], 2)
        for t in self.tasks:
            for arm in ('pid_native', 'pid_held_50ms'):
                d = self.out / 'runs' / t.id / arm
                if not (d / 'DONE').exists():
                    self.assertEqual((t.id, arm, (d / 'record.json').exists()), (t2, 'pid_held_50ms', False))
                    continue
                files = {p.relative_to(d).as_posix(): p for p in d.rglob('*') if p.is_file() and p.name != 'DONE'}
                done = json.loads((d / 'DONE').read_text())
                self.assertEqual(done['files'], {k: sha(p) for k, p in files.items()})
                self.assertGreaterEqual((d / 'DONE').stat().st_mtime_ns, max(p.stat().st_mtime_ns
                                                                      for p in files.values()))
                self.assertEqual(json.loads((d / 'record.json').read_text())['provenance']['run']['host'], 'nodeA')
        again = FakeDrive()
        recs = self.block(again, workers=2)
        self.assertEqual((again.calls, len(recs)), ([(t2, 'pid_held_50ms', 'drive')], 10))
        self.assertTrue(all(self.done(t.id, a.name) for t in self.tasks for a in ARMS))
        self.assertEqual(TB.code(self.out / f'runs/{t2}/pid_held_50ms', 'tracker'), 'S')

    def test_pair_atomic_resume_on_another_node(self):
        """On another node the arms of the unfinished pair move to superseded/ and the pair is driven again whole with
        the picks no pass 1 made; finished pairs stay; the pin records the new node with the old one in its history."""
        t2 = self.tasks[2].id
        with self.assertRaises(RuntimeError):
            self.run_arms(FakeDrive(**{f'{t2}/pid_held_50ms/drive': ('kill', 0)}))
        first = (self.out / f'runs/{t2}/pid_native/DONE').read_bytes()
        again = FakeDrive()
        self.block(again, host='nodeB')
        self.assertEqual(sorted(again.calls), sorted([(t2, 'pid_native', 'drive'), (t2, 'pid_held_50ms', 'drive')]))
        moved = {p.name.split('__')[0]: p for p in (self.out / 'superseded' / t2).iterdir()}   # the partial one too
        self.assertEqual((sorted(moved), (moved['pid_native'] / 'DONE').read_bytes()), (['pid_held_50ms', 'pid_native'],
                                                                                        first))
        for arm in ('pid_native', 'pid_held_50ms'):                      # the standing picks were kept, not re-planned
            self.assertEqual((self.out / f'runs/{t2}/{arm}/pick.json').read_bytes(), (moved[arm] / 'pick.json').read_bytes())
        pin = json.loads((self.out / 'blocks/0.pin.json').read_text())
        self.assertEqual((pin['host'], [h['host'] for h in pin['history']]), ('nodeB', ['nodeA']))
        nothing = FakeDrive()
        self.block(nothing, host='nodeB', build='b2')                    # another build: a node change too
        self.assertEqual(nothing.calls, [])                              # every pair finished: nothing moves

    def test_pairing_blocks_and_refusals(self):
        """md5-sorted blocks of whole pairs; a run folder holds one evaluation; code, GPU and guard refusals."""
        evals = [R.TraversalEval(c, self.out, self.env) for c in ARMS]
        bl = R.prepare(self.out, evals, self.tasks, block_size=2)
        order = sorted((t.id for t in self.tasks), key=md5hex)
        self.assertEqual(bl, [order[:2], order[2:4], order[4:]])
        self.assertEqual(json.loads((self.out / 'blocks.json').read_text()), dict(arms=[c.name for c in ARMS],
                                                                                   blocks=bl))
        self.assertEqual(R.prepare(self.out, evals, self.tasks, block_size=2), bl)            # the same: accepted
        for arms, tasks, size in ((evals[:1], self.tasks, 2), (evals, self.tasks[:4], 2), (evals, self.tasks, 3)):
            with self.assertRaisesRegex(ConfigError, 'one evaluation'):
                R.prepare(self.out, arms, tasks, block_size=size)
        for bad in ([evals[0], R.TraversalEval(ARMS[0], self.out, self.env)],
                    [evals[0], R.TraversalEval(EvalConfig(name='soil_pid', ground='soil', soil_config='crm_main',
                                                          **GIVEN), self.out, self.env)]):
            with self.assertRaisesRegex(ConfigError, 'one ground'):
                R.prepare(self.out, bad, self.tasks)
        late = FakeDrive()
        with self.assertRaisesRegex(RuntimeError, 'walltime guard: 2 pairs left'):  # within 2 x GUARD_S: no pair starts
            self.block(late, deadline=time.time() + 1.5 * R.GUARD_S['rigid'])
        self.assertEqual((late.calls, list((self.out / 'runs').glob('*/*/DONE'))), ([], []))
        with self.assertRaisesRegex(ConfigError, 'accept-code-change'):
            self.block(FakeDrive(), code='0' * 64)
        self.block(FakeDrive(), code='0' * 64, accept_code_change=True)
        (self.out / 'job').mkdir()
        (self.out / 'job/code_sha.json').write_text(json.dumps(dict(code_sha='1' * 64)))
        with self.assertRaisesRegex(ConfigError, 'job/code_sha.json'):
            self.block(FakeDrive())
        soil = R.TraversalEval(EvalConfig(name='soil_pid', ground='soil', soil_config='crm_main', **GIVEN), self.out,
                               self.env)
        with mock.patch.object(R, 'load_run', return_value=([soil], {}, [[]])), \
                self.assertRaisesRegex(ConfigError, 'one drive process per GPU'):
            self.block(FakeDrive(), workers=2, gpus=[0])
        locked = R.TraversalEval(EvalConfig(name='pid_native', planner='given', label='tracker',
                                            build_lock='configs/traversing/evaluation/locks/chrono-build.json'), self.out,
                                 self.env)
        with mock.patch.object(R, 'load_run', return_value=([locked], {}, [[]])), \
                self.assertRaisesRegex(ConfigError, 'this build differs from the lock'):
            self.block(FakeDrive())                                      # the patched node has another fingerprint

    def test_plan_stage_neither_pins_nor_moves_and_done_is_rechecked(self):
        """--stage plan (on the planner's machine) writes the picks only; the drive stage on another node then pins
        and drives them. A run file changed after DONE stops the read-out."""
        self.run_arms(FakeDrive(), stage='plan', host='luffy')
        self.assertFalse((self.out / 'blocks/0.pin.json').exists())
        self.assertEqual(len(list((self.out / 'runs').glob('*/*/pick.json'))), 10)
        drives = FakeDrive()
        self.block(drives, host='k007')
        self.assertEqual((len(drives.calls), (self.out / 'superseded').exists()), (10, False))
        self.assertEqual(json.loads((self.out / 'blocks/0.pin.json').read_text())['host'], 'k007')
        (self.out / f'runs/{self.tasks[3].id}/pid_native/route.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'changed after DONE'):
            R.cells(self.out, self.env)

    def test_drive_refuses_bad_inputs_before_chrono(self):
        """Route ends, a soil config without a key and an unreadable input refuse (ConfigError; the child exits 3, never
        retried into a crash record) before any Chrono object."""
        task, bad = self.tasks[0], route(0, 1.)
        bad['waypoints'][-1] = [25., 0.]
        soil = EvalConfig(name='s', ground='soil', soil_config=str(self.tmp / 'soil.json'), **GIVEN)
        (self.tmp / 'soil.json').write_text(json.dumps(dict(step_s=0.001)))
        with mock.patch.object(R.Sim, 'build') as build:
            for cfg, r, ground, msg in ((ARMS[0], bad, 'rigid', 'must start and end within 0.25 m'),
                                        (soil, route(0, 1.), 'soil', 'missing .*tire_mesh')):
                with self.subTest(msg), mock.patch.dict(os.environ, R.process_env(ground, 0)), \
                        self.assertRaisesRegex(ConfigError, msg):
                    R.drive(cfg, task, r, env=self.env)
            build.assert_not_called()
        for p in (self.env.data, self.env.chrono_data):
            p.mkdir(exist_ok=True)
        inp = dict(task=task.to_dict(self.env), arm=ARMS[0].to_dict(), route=route_json(bad), horizon_s=120., branch=None)
        (self.tmp / 'input.json').write_text(json.dumps(inp))
        env = dict(R.process_env('rigid'), NEDM_DATA=str(self.env.data), NEDM_CHRONO_DATA=str(self.env.chrono_data))
        with mock.patch.dict(os.environ, env), mock.patch('sys.stdout', new=io.StringIO()) as out, \
                self.assertRaises(SystemExit) as ex:
            R.main(['--input', str(self.tmp / 'input.json'), '--out', str(self.tmp / 'o')])
        self.assertEqual((ex.exception.code, out.getvalue()[:10]), (R.REFUSED, '# refused:'))

    def test_missing_drive_policy(self):
        """A crashed arm is the 'crash' cell; M2 tables drop the pair, the unseen Polaris table counts it U, the
        others leave it out ('-'); an arm that never finished is absent ('-')."""
        arms = [EvalConfig(name=n, label='rollback', allow_unvalidated=True, planner='given')
                for n in ('shared_hist_3s', 'shared_hist_0p5s')]
        t0 = self.tasks[0].id
        self.run_arms(FakeDrive(**{f'{t0}/shared_hist_0p5s/drive': ('fail', 0)}), arms=arms)
        cells = R.cells(self.out, self.env)
        meta = dict(group_id='g', arena='f104', world='crm_soil', suite_stratum='fresh', terrain_stratum='x')
        rows = TB.table_rows('m2_shared_risk_soil', [({**meta, 'group_id': t}, c) for t, c in cells.items()])
        head = rows[0]
        got = {r[0]: {a: r[head.index(a)] for a in ('shared_hist_3s', 'shared_hist_0p5s', 'pooled_3s')}
               for r in rows[1:]}
        self.assertEqual(got[t0], dict(shared_hist_3s='-', shared_hist_0p5s='-', pooled_3s='-'))
        self.assertEqual(got[self.tasks[1].id], dict(shared_hist_3s='S', shared_hist_0p5s='S', pooled_3s='-'))
        crash = cells[t0]['shared_hist_0p5s']
        for table, arm, want in (('m4_polaris_unseen_soil', 'polaris_straight_6mps', 'U'),
                                 ('m4_unseen_arenas_hmmwv_soil', 'straight_route_6mps', '-')):
            t = TB.TABLES[table]
            row = TB.table_rows(table, [({c: 'x' for c in t.ids}, {arm: crash})])[1]
            self.assertEqual(row[list(t.columns).index(arm)], want)


class TestTwoPass(Base):
    ARM = EvalConfig(name='early', planner='cem', models='m_s*.pt', approach_s=0.5, allow_unvalidated=True)

    def episode(self, fake, pick):
        seen = []
        ev = R.TraversalEval(self.ARM, self.out, self.env)
        with mock.patch.object(R.subprocess, 'run', side_effect=fake), \
                mock.patch.object(R.TraversalEval, 'plan', lambda s, task, dec: seen.append(dec) or pick):
            return ev.episode(self.tasks[0]), seen

    def test_pass1_decision_branch(self):
        """Pass 1 at horizon L; the decision at F = 10 from its terminal state; pass 2 = the approach with a branch to
        the pick at F; the record carries the pick and the decision source."""
        fake, br = FakeDrive(), route(1, 2.)
        rec, seen = self.episode(fake, Pick.of(br, 'G'))
        t = self.tasks[0].id
        self.assertEqual(fake.calls, [(t, 'early', 'pass1'), (t, 'early', 'drive')])
        self.assertEqual((seen[0].frame, seen[0].pose.tolist(), seen[0].source[:6]), (10, [1., 0., 0.], 'pass1:'))
        inp = json.loads((self.out / f'runs/{t}/early/input.json').read_text())
        self.assertEqual((inp['horizon_s'], inp['branch'][0], route_sha256(inp['branch'][1])),
                         (120., 10, route_sha256(br)))
        self.assertEqual(json.loads((self.out / f'runs/{t}/early/pass1/input.json').read_text())['horizon_s'], .5)
        self.assertEqual((rec.status, rec.pick['arm'], rec.pick['decision'][:6]), ('goal_reached', 'G', 'pass1:'))
        self.assertTrue(self.done(t, 'early'))

    def test_prefix_ends_early_and_no_route(self):
        """A pass 1 that stops before F is the run (no decision, no plan); a pick without a route is 'no_route' (U)."""
        t = self.tasks[0].id
        rec, seen = self.episode(FakeDrive(**{f'{t}/early/pass1': ('rollover', 4)}), Pick.of(None, 'B'))
        self.assertEqual((rec.status, rec.frames, rec.pick, seen), ('rollover', 4, dict(decided=False,
                                                                                         pass1_status='rollover'), []))
        self.assertEqual(TB.code(self.out / f'runs/{t}/early', 'rollback'), 'U')
        self.out = self.tmp / 'run2'
        rec, seen = self.episode(FakeDrive(), Pick.of(None, 'B'))
        self.assertEqual((rec.status, len(seen), TB.code(self.out / f'runs/{t}/early', 'rollback')),
                         ('no_route', 1, 'U'))


if __name__ == '__main__':
    unittest.main()
