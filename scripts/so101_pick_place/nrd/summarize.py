"""Table of the task metrics (data.evaluate reports) of the SO101 NRDs: validation reports of the training runs and
reports of origin/main's `python -m nedm.contact_nrd.test` (validation re-score and sealed test).

    python -I summarize.py OUT_PREFIX LABEL=FILE[:MODEL][@GROUP] ...

FILE: a training validation.json (MODEL empty) or a test.py --output JSON (MODEL = the --model name).
GROUP: selected (default), all, or a family name. Writes OUT_PREFIX.json and OUT_PREFIX.md.

Release check of the SO101 pick-and-place study (so101_pick_place/README.md, "The NRD"; reproduce.sh steps
nrd_val, nrd_strip).
"""
import json
import math
import sys

ROWS = [   # label, path in the report, scale, format
    ("episodes", ("episodes",), 1, "{:.0f}"),
    ("1 s windows: cube position p50 (mm)", ("windows", "1.0", "cube", "pos_m", "median"), 1e3, "{:.2f}"),
    ("1 s windows: cube position p95 (mm)", ("windows", "1.0", "cube", "pos_m", "p95"), 1e3, "{:.2f}"),
    ("1 s windows: cube angle p50 (deg)", ("windows", "1.0", "cube", "angle_rad", "median"), 180 / math.pi, "{:.2f}"),
    ("1 s windows: cube angle p95 (deg)", ("windows", "1.0", "cube", "angle_rad", "p95"), 180 / math.pi, "{:.2f}"),
    ("1 s windows: arm joint RMS p95 (mrad)", ("windows", "1.0", "arm", "joint_rms_rad", "p95"), 1e3, "{:.2f}"),
    ("1 s windows (count)", ("windows", "1.0", "windows"), 1, "{:.0f}"),
    ("4 s from the first state: cube position p50 (mm)", ("free", "target", "cube", "pos_m", "median"), 1e3, "{:.2f}"),
    ("4 s from the first state: cube position p95 (mm)", ("free", "target", "cube", "pos_m", "p95"), 1e3, "{:.2f}"),
    ("4 s from the first state: cube angle p95 (deg)", ("free", "target", "cube", "angle_rad", "p95"), 180 / math.pi, "{:.2f}"),
    ("4 s path RMS of the cube p95 (mm)", ("free", "path_rmse_m", "p95"), 1e3, "{:.2f}"),
    ("contact events: wrong per recorded event", ("free", "contacts", "all", "event_error_rate"), 1, "{:.3f}"),
    ("contact switch accuracy", ("free", "contacts", "all", "gate_accuracy"), 1, "{:.4f}"),
    ("lifted: disagreement (%)", ("free", "flags", "lifted", "disagreement"), 100, "{:.2f}"),
    ("dropped: disagreement (%)", ("free", "flags", "dropped", "disagreement"), 100, "{:.2f}"),
    ("finite rollouts", ("finite_fraction",), 1, "{:.4f}"),
    ("selection score (lower is better)", ("selection_score",), 1, "{:.3f}"),
]


def dig(x, path):
    for p in path:
        if not isinstance(x, dict) or p not in x:
            return None
        x = x[p]
    return x


def group_of(report, group):
    if group == "all":
        return report
    if group == "selected":
        return report.get("selected")
    return (report.get("families") or {}).get(group)


def main():
    out = sys.argv[1]
    cols, cache = [], {}
    for item in sys.argv[2:]:
        label, spec = item.split("=", 1)
        spec, group = (spec.split("@", 1) + ["selected"])[:2]
        path, model = (spec.split(":", 1) + [""])[:2]
        if path not in cache:
            cache[path] = json.load(open(path))
        rep = cache[path]["models"][model] if model else cache[path]
        g = group_of(rep, group)
        cols.append((label, path, model, group, g))
    table = {}
    for name, path, scale, fmt in ROWS:
        table[name] = {}
        for label, _, _, _, g in cols:
            v = dig(g, path) if g is not None else None
            table[name][label] = None if v is None else v * scale
    families = {}
    for label, path, model, group, _ in cols:
        rep = cache[path]["models"][model] if model else cache[path]
        for fam, block in (rep.get("families") or {}).items():
            v = dig(block, ("windows", "1.0", "cube", "pos_m", "p95"))
            families.setdefault(fam, {})[label] = None if v is None else 1e3 * v
    lines = ["| Metric | " + " | ".join(c[0] for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for name, path, scale, fmt in ROWS:
        lines.append(f"| {name} | " + " | ".join("-" if table[name][c[0]] is None else fmt.format(table[name][c[0]])
                                                  for c in cols) + " |")
    lines += ["", "1 s windows, cube position p95 (mm) per family (whole report of each column):", "",
              "| Family | " + " | ".join(c[0] for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for fam, vals in families.items():
        lines.append(f"| {fam} | " + " | ".join("-" if vals.get(c[0]) is None else f"{vals[c[0]]:.2f}" for c in cols) + " |")
    open(out + ".md", "w").write("\n".join(lines) + "\n")
    json.dump(dict(columns=[dict(label=c[0], file=c[1], model=c[2], group=c[3]) for c in cols], metrics=table,
                   family_cube_pos_1s_p95_mm=families), open(out + ".json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
