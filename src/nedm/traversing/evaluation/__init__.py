"""Traversing evaluation: one arm (``EvalConfig``) on released or custom tasks (README.md). Exports resolve lazily:
importing a torch-free module never loads the planner's torch."""
import importlib

_EXPORTS = {**dict.fromkeys(('ConfigError', 'Env', 'EvalConfig', 'ReleaseError', 'load_arms'), 'config'),
            **dict.fromkeys(('SUITES', 'Task', 'blocks', 'load_suite', 'make_case', 'select'), 'suites'),
            **dict.fromkeys(('Decision', 'Pick'), 'planner'),
            **dict.fromkeys(('Record', 'TraversalEval', 'run_arms'), 'runner')}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    return getattr(importlib.import_module(f'.{_EXPORTS[name]}', __name__), name)
