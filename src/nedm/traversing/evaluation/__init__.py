"""Traversing evaluation: one arm (``EvalConfig``) evaluated on released or custom tasks (FINAL_DESIGN.md).

Exports resolve lazily, so importing a torch-free submodule (config, suites, routes, labels, the drive workers) never
loads the planner's torch (FINAL_DESIGN 1.5). runner.py (TraversalEval, Record) plugs in here once built.
"""

import importlib

_EXPORTS = {**dict.fromkeys(('ConfigError', 'Env', 'EvalConfig', 'ReleaseError', 'load_arms'), 'config'),
            **dict.fromkeys(('SUITES', 'Task', 'blocks', 'load_suite', 'make_case', 'select'), 'suites'),
            **dict.fromkeys(('Decision', 'Pick'), 'planner')}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    return getattr(importlib.import_module(f'.{_EXPORTS[name]}', __name__), name)
