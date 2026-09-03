#!/usr/bin/env python3
"""Every `args.X` read must have a matching `add_argument` (or subparser dest).

Written after the SECOND occurrence of the same defect in
`scripts/quadruped_go2_crm.py`: a commit that adds the *use* of an argument and
never adds the `add_argument` that creates it. Both instances passed every check
we had, because `args.cam_follow` is an attribute access on a bound name -- the
static unbound-name pass added after the first occurrence sees nothing wrong,
and Python itself is happy until the line executes.

It fails LATE, which is what makes it expensive: the run builds an
886,611-particle CRM terrain, attaches the sensor, prints its startup banner and
then dies, so roughly a minute of setup burns before the AttributeError.
"""
import ast
import pathlib
import sys

ROOTS = ("scripts", "src", "tools")


def check(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(), str(path))
    declared: set[str] = set()
    used: dict[str, int] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            # add_subparsers(dest="cmd") also defines an attribute.
            if node.func.attr in ("add_argument", "add_subparsers"):
                for kw in node.keywords:
                    if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
                        declared.add(kw.value.value)
                        break
                else:
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            if arg.value.startswith("--"):
                                declared.add(arg.value[2:].replace("-", "_"))
                            elif not arg.value.startswith("-"):
                                declared.add(arg.value.replace("-", "_"))
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "args" and isinstance(node.ctx, ast.Load)):
            used.setdefault(node.attr, node.lineno)

    if not declared:          # not an argparse module; nothing to say
        return []
    return [f"{path}:{line}: args.{name} has no add_argument"
            for name, line in sorted(used.items()) if name not in declared]


def main() -> int:
    findings = []
    for root in ROOTS:
        for path in sorted(pathlib.Path(root).rglob("*.py")):
            findings += check(path)
    print("\n".join(findings) if findings
          else f"OK: every args.X across {'/'.join(ROOTS)} has a matching add_argument")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
