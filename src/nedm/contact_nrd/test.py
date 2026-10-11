"""Score trained contact NRD models on a test directory (every episode in it).

From the repo root (after ``python -m nedm.contact_nrd.download --part test models``):

    PYTHONPATH=src python -m nedm.contact_nrd.test --data artifacts/contact_nrd/pool/test \\
        --model seed61=artifacts/contact_nrd/pool/models/seed61.pt seed62=artifacts/contact_nrd/pool/models/seed62.pt

PATH is a checkpoint file or a run directory with best.pt. Each episode is a free rollout from its first recorded
state with its recorded commands (data.free_metrics). One line per model, each error as median / p95: the target
body's position error at the target time, its orientation error (only if it has a quaternion) and its path RMS error
up to the target time; then the fraction of episodes whose contact events match the recorded ones and the fraction
of finite rollouts.

Data with an ``evaluation`` block in system.json (configs/contact_nrd/evaluation/README.md) or with episode families
are scored with data.evaluate instead: the block's task metrics (first-state rollouts and/or rollouts from recorded
histories at fixed horizons, contact events per pair, selection score) for all episodes, for each family and for the
families with select: true. One block of lines per model and group: the score and the fraction of episodes whose
rollouts in the scored modes are finite (report-only modes: see their lines), then per rollout mode and horizon the
error of each moving body (median / p95; in brackets the windows and the rollouts that are not finite up to that
step, which count as an error of 1000, the cap of every error), then the contact events (model convention and
physical) and the outcome flags (recorded against predicted); for all episodes also the window errors at the longest
horizon that has windows, by the contacts at the window start. A horizon, mode or contact group without any rollout
(e.g. a horizon longer than every episode of a family) has no line; a family score that needs such a statistic is
"not available". Episodes of families with "use": "core_only" (training data for the core only) are not scored and
not loaded (load_data core_only="drop"): one line gives their number, and the output JSON has it as
core_only_left_out.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from nedm.contact_nrd.data import (channel_index, evaluate, file_rows, file_sha256, free_metrics, load_data,
                                   step_average_rates, add_derived, add_relative)
from nedm.contact_nrd.model import load_model

UNITS = {"pos_m": ("position", 1000.0, "mm"), "angle_rad": ("orientation", 180 / math.pi, "deg"),
         "joint_rms_rad": ("joint RMS", 1.0, "rad"), "vel_mps": ("velocity", 1.0, "m/s")}


def _errors(block):
    parts = []
    for body, errs in block.items():
        if not isinstance(errs, dict):
            continue
        for m, q in errs.items():
            what, k, unit = UNITS[m]
            parts.append(f"{body} {what} {k * q['median']:.3f} / {k * q['p95']:.3f} {unit}")
    return ", ".join(parts)


def _line(text, block, count=None):
    """One line of errors: text (rollouts, non-finite ones): body errors."""
    note = ([] if count is None else [str(count)]) + ([f"{block['nonfinite']} non-finite"] if block["nonfinite"] else [])
    return f"  {text}{' (' + ', '.join(note) + ')' if note else ''}: {_errors(block)}"


def describe(name, group, report, detail=False):
    """Text lines of one task-metrics report (detail: also the window errors by the contacts at the start). Lines
    without any rollout are left out."""
    terms = [t for t in report["score_terms"] if t["value"] is not None]
    missing = len(report["score_terms"]) - len(terms)
    score = (f"score {report['selection_score']:.4f}" if report["selection_score"] is not None else
             f"score not available ({missing} scored statistic{'s' if missing > 1 else ''} without rollouts)")
    lines = [f"{name}, {group} ({report['episodes']} episodes): {score}, finite in scored modes "
             f"{report['finite_fraction']:.4f}; "
             + ", ".join(f"{t['stat']} {t['value']:.4g}" for t in terms)]
    modes = [(mode, label) for mode, label in (("free", "first state"), ("windows", "windows"),
                                                ("long_windows", "long windows"), ("event_windows", "event windows"))
             if mode in report and report[mode].get("windows", 1)]
    for mode, label in modes:
        if mode == "free":
            lines += [_line(f"{label}, at {at}", report["free"][at]) for at in ("target", "final")]
        else:
            lines += [_line(f"{label} {h} s", v, v["windows"]) for h, v in report[mode].items()
                      if isinstance(v, dict) and v.get("windows")]
    for mode, label in modes:
        c, p = report[mode]["contacts"]["all"], report[mode]["contacts_physical"]["all"]
        def events(x):
            out = x["onset"]["timing_left_out"] + x["offset"]["timing_left_out"]
            return (f"{x['onset']['true']} onsets, {x['offset']['true']} offsets, {x['onset']['missed'] + x['offset']['missed']} "
                    f"missed, {x['onset']['false'] + x['offset']['false']} false, timing p95 {1000 * x['timing_s']['p95']:.0f} ms"
                    + (f" ({out} matched events left out of the timing)" if out else ""))
        lines.append(f"  {label} contacts: {events(c)}, switch accuracy {c['gate_accuracy']:.4f}; physical: {events(p)}")
        flags = report[mode].get("flags", {})
        if flags:
            first = next(iter(flags.values()))
            nonfinite = f" ({first['nonfinite']} non-finite)" if first["nonfinite"] else ""
            lines.append(f"  {label} outcomes of {first['rollouts']} rollouts{nonfinite}, recorded / predicted "
                         "(disagreement): " + ", ".join(f"{k} {f['recorded']} / {f['predicted']} ({f['disagreement']:.4f})"
                                                        for k, f in flags.items()))
    if detail and "windows" in report:
        for start, block in report["windows"]["by_start_contact"].items():
            reach = [h for h, v in block.items() if v["windows"]]
            if reach:
                h = max(reach, key=float)
                lines.append(_line(f"windows {h} s from contacts {start}", block[h], block[h]["windows"]))
    return lines


def task_main(args, models, data):
    """Data with an evaluation block or families: data.evaluate per model, all episodes and per family."""
    system = data["system"]
    if "evaluation" not in system and any(m.config.get("derived") or m.config.get("relative") for _, m in models.values()):
        raise ValueError("models with derived or relative channels need data with an evaluation block")
    if "evaluation" in system:
        kinds = {tuple(sorted(m.config["integrate"]["kinds"])) for _, m in models.values()}
        if len(kinds) != 1:
            raise ValueError("the models integrate different body kinds; score them separately")
        step_average_rates(data, set(kinds.pop()))
        derived = {json.dumps(m.config.get("derived"), sort_keys=True) for _, m in models.values()}
        if len(derived) != 1:
            raise ValueError("the models use different derived channels; score them separately")
        derived = json.loads(derived.pop())
        add_derived(data, derived)
        relative = {json.dumps(m.config.get("relative"), sort_keys=True) for _, m in models.values()}
        if len(relative) != 1:
            raise ValueError("the models use different relative channels; score them separately")
        add_relative(data, json.loads(relative.pop()))
    ids = torch.arange(len(data["splits"]), device=args.device)
    left_out = data.get("core_only_left_out", 0)   # not loaded
    if left_out:
        names = [f["name"] for f in system["families"] if f.get("use") == "core_only"]
        print(f"{left_out} episodes of core-only families ({', '.join(names)}) are not scored", flush=True)
    results = {}
    for name, (path, model) in models.items():
        m = file_rows(evaluate(model, data, ids, by_family=True), data)
        groups = [("all", m)] + [(f"family {k}", v) for k, v in m.get("families", {}).items()]
        groups += [("selected families", m["selected"])] if "selected" in m else []
        for group, r in groups:
            if "evaluation" in system:
                print("\n".join(describe(name, group, r, detail=group == "all")), flush=True)
            else:
                print(f"{name}, {group} ({r['episodes']} episodes): target position {1000 * r['target_pos_m']['median']:.3f} / "
                      f"{1000 * r['target_pos_m']['p95']:.3f} mm, events ok {r['event_ok_fraction']:.4f}, "
                      f"finite {r['finite_fraction']:.4f}, score {r['selection_score']:.4f}", flush=True)
        results[name] = dict(checkpoint=str(path), checkpoint_sha256=file_sha256(path), **m)
    return results, left_out


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="test directory (system.json, unified_data.npz)")
    parser.add_argument("--model", nargs="+", required=True, metavar="NAME=PATH", help="checkpoint or run directory")
    parser.add_argument("--output", type=Path, help="JSON file with the summary and the per-episode results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    models = {}
    for item in args.model:
        name, path = item.split("=", 1)
        path = Path(path) / "best.pt" if Path(path).is_dir() else Path(path)
        models[name] = (path, load_model(path, args.device)[0])
    data = load_data(args.data, args.device, model_step_s=next(iter(models.values()))[1].dt, core_only="drop")
    system = data["system"]
    if "evaluation" in system or data["family"] is not None:
        results, left_out = task_main(args, models, data)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(dict(data=str(args.data), data_sha256=file_sha256(args.data / "unified_data.npz"),
                                                   target_step=data["target_step"], dt_s=data["dt"], models=results,
                                                   **({"core_only_left_out": left_out} if left_out else {})), indent=1))
        return
    has_quat = bool(channel_index(system, system["bodies"][system["target_body"]]["name"], "quaternion"))
    ids = torch.arange(len(data["splits"]), device=args.device)
    results = {}
    for name, (path, model) in models.items():
        m = free_metrics(model, data, ids)
        line = (f"{name}: target position {1000 * m['target_pos_m']['median']:.3f} / {1000 * m['target_pos_m']['p95']:.3f} mm")
        if has_quat:
            line += (f", target angle {math.degrees(m['target_angle_rad']['median']):.3f} / "
                     f"{math.degrees(m['target_angle_rad']['p95']):.3f} deg")
        line += (f", path RMS {1000 * m['path_rmse_m']['median']:.3f} / {1000 * m['path_rmse_m']['p95']:.3f} mm"
                 f" (median / p95), events ok {m['event_ok_fraction']:.4f}, finite {m['finite_fraction']:.4f}"
                 f" ({m['episodes']} episodes)")
        print(line, flush=True)
        results[name] = dict(checkpoint=str(path), checkpoint_sha256=file_sha256(path),
                             **{k: m[k] for k in ("target_pos_m", "target_angle_rad", "path_rmse_m", "event_ok_fraction",
                                                  "finite_fraction", "episodes", "per_episode")})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(data=str(args.data), data_sha256=file_sha256(args.data / "unified_data.npz"),
                                               target_step=data["target_step"], dt_s=data["dt"], models=results), indent=1))


if __name__ == "__main__":
    main()
