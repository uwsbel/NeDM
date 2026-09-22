#!/usr/bin/env python3
"""Catch unbound local names before they cost a GPU-minute.

`py_compile` only proves a file parses. It happily accepts a function that reads a local
name nothing assigns, and on this project that is not hypothetical: replacing the spawn
logic in `collect.py` removed the assignment of `plo_x` and left a use of it 130 lines
later, which surfaced as a NameError AFTER episode 0 had finished simulating. Three
minutes of GPU to learn something a static pass knows instantly.

No linter is installed here and this is not worth a dependency.

SCOPE-AWARENESS IS THE WHOLE JOB. A first version walked each function with `ast.walk`,
which descends into nested `def`s, so a closure's own parameters read inside it were
charged to the enclosing function: nine false positives against one real finding. A
checker that cries wolf gets ignored, and being ignored is worse than not existing. So
this one stops at function boundaries and builds a real scope chain -- a nested function's
defaults and decorators evaluate in the ENCLOSING scope, while its body does not.
"""
from __future__ import annotations

import ast
import builtins
import sys
from pathlib import Path

# Module-level dunders every module has but dir(builtins) does not list; __file__ was
# reported unbound in corpus_check.py, a false positive.
BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__spec__", "__package__"}
FUNC = (ast.FunctionDef, ast.AsyncFunctionDef)
SCOPED = FUNC + (ast.Lambda, ast.ClassDef)


def _own(node):
    """Nodes belonging to this scope: the body, but not nested scopes' bodies.

    A nested function's defaults and decorators DO belong here -- they are evaluated
    where the `def` appears, not where the function runs.
    """
    out = []
    stack = list(ast.iter_child_nodes(node))
    while stack:
        n = stack.pop()
        if isinstance(n, SCOPED):
            out.append(n)                                  # the name it binds
            for d in getattr(n, "decorator_list", []):
                stack.extend(ast.walk(d))
            args = getattr(n, "args", None)
            if args is not None:
                for d in list(args.defaults) + [x for x in args.kw_defaults if x]:
                    stack.extend(ast.walk(d))
            continue                                        # do NOT descend into its body
        out.append(n)
        stack.extend(ast.iter_child_nodes(n))
    return out


def binds(node):
    out = set()
    args = getattr(node, "args", None)
    if args is not None:
        for a in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
            out.add(a.arg)
        if args.vararg:
            out.add(args.vararg.arg)
        if args.kwarg:
            out.add(args.kwarg.arg)
    for n in _own(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, SCOPED):
            out.add(getattr(n, "name", ""))
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                out.add((al.asname or al.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            out.update(n.names)
    out.discard("")
    return out


def reads(node):
    return {n.id for n in _own(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def check(path):
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    problems = []

    def walk(node, chain):
        """chain: names visible from enclosing scopes, outermost first."""
        here = binds(node)
        visible = set().union(*chain) if chain else set()
        if isinstance(node, FUNC):
            for name in sorted(reads(node) - here - visible - BUILTINS):
                problems.append((node.name, node.lineno, name))
        # Class bodies do not create a scope their nested functions can see.
        passed = chain + [here] if not isinstance(node, ast.ClassDef) else chain
        for child in ast.walk(node):
            if child is node or not isinstance(child, SCOPED):
                continue
            # only direct children of this scope
            if any(child is c for c in _own(node)):
                walk(child, passed)

    walk(tree, [])
    return problems


def main() -> int:
    bad = 0
    for p in sys.argv[1:]:
        for fname, line, name in check(p):
            print(f"{p}:{line}: {fname}() reads unbound name {name!r}")
            bad += 1
    print("namecheck: clean" if not bad else f"namecheck: {bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
