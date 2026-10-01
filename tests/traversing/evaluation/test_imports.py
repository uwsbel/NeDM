"""Import boundaries of nedm.traversing.evaluation, one fresh interpreter per module (stdlib
unittest only). The package (its lazy exports included), config, suites, labels, routes and the Chrono half (sim,
vehicles, controllers, episode, runner: pychrono inside functions only) load no torch; no module loads pychrono, scipy,
yaml, the experiment branch (nedm.traverse) or another study's code (nedm.core) at import.

    PYTHONPATH=src python -m unittest discover -s tests/traversing/evaluation -p test_imports.py -v
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / 'src'
BANNED = ('pychrono', 'scipy', 'yaml', 'nedm.traverse', 'nedm.core')
TORCH_FREE = ('', '.config', '.suites', '.labels', '.routes', '.sim', '.vehicles', '.controllers', '.episode', '.runner')
RULES = {**{m: BANNED + ('torch',) for m in TORCH_FREE}, '.planner': BANNED, '.refine': BANNED, '.nav': BANNED}
CODE = '''import importlib, sys
m = importlib.import_module({mod!r})
[getattr(m, a) for a in {attrs!r}]
print(','.join(sorted(k for k in sys.modules if any(k == b or k.startswith(b + '.') for b in {banned!r}))))'''


def loaded(mod, banned, attrs=()):
    out = subprocess.run([sys.executable, '-c', CODE.format(mod=mod, banned=banned, attrs=attrs)], capture_output=True,
                         text=True, check=True, env={**os.environ, 'PYTHONPATH': str(SRC)})
    return out.stdout.strip()


class TestImports(unittest.TestCase):
    def test_boundaries(self):
        for sub, banned in RULES.items():
            mod = 'nedm.traversing.evaluation' + sub
            with self.subTest(mod):
                attrs = ('EvalConfig', 'Task', 'load_suite') if not sub else ()      # lazy exports stay torch-free
                self.assertEqual(loaded(mod, banned, attrs), '')
        self.assertIn('torch', loaded('nedm.traversing.evaluation', ('torch',), ('Pick',)))    # the check can fail


if __name__ == '__main__':
    unittest.main()
