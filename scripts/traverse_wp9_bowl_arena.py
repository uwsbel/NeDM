#!/usr/bin/env python
"""Write the controlled-bowl arm set for the crater-learnability diagnostic (plan §B1).

Three arms from ONE bowl field on ONE shared 8-bit height range, so the plain, the approach
corridor and the roughness are bit-identical outside the rim and only the depression differs:

    deep     47 deg wall, 3.2 m, 4.5 m flat bottom, 15 deg entry ramp  -- the trap
    shallow  same rim, scaled to 0.9 m (about 17 deg)                  -- driveable control
    flat     no depression                                             -- driveable control

The defaults are the setting the B1 geometry probe selected on newton: at commanded 2/4/6
m/s under sustained full throttle the vehicle does not escape, at 8 m/s it pitches over
(that arm is excluded from the claim), and chassis-terrain contact peaks at 49-99 kN
against a 25 kN vehicle weight. The shallow and flat controls completed the crossing at all
four speeds. Constant-slope walls of 36/39/42 deg were all escaped, momentum-driven, and
50 deg pitched the vehicle over at every speed.

The BMP orientation is Chrono's convention as calibrated on arena_v1 (copied, as the f-family does).

  PYTHONPATH=src python scripts/traverse_wp9_bowl_arena.py
  PYTHONPATH=src python scripts/traverse_wp9_bowl_arena.py --wall-deg 44 --depth-m 4.2 --prefix arena_bowl44
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nedm.traverse.bowl import (  # noqa: E402
    ArmSpec,
    BowlSpec,
    grey_difference_outside_rim,
    measure_arena,
    write_bowl_arms,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="assets/traverse")
    ap.add_argument("--prefix", default="arena_bowl", help="arms are written to <root>/<prefix>_<arm>")
    ap.add_argument("--orientation-from", default="assets/traverse/arena_v1")
    ap.add_argument("--wall-deg", type=float, default=47.0)
    ap.add_argument("--depth-m", type=float, default=3.2)
    ap.add_argument("--bottom-radius-m", type=float, default=4.5)
    ap.add_argument("--entry-deg", type=float, default=15.0)
    ap.add_argument("--entry-azimuth-deg", type=float, default=180.0)
    ap.add_argument("--entry-halfwidth-deg", type=float, default=28.0)
    ap.add_argument("--centre-m", type=float, nargs=2, default=(0.0, 0.0), metavar=("X", "Y"),
                    help="bowl centre in the SIMULATED frame (what TerrainMap.features returns)")
    ap.add_argument("--round-m", type=float, default=0.25, help="quantisation-safety rounding, >= 1.5 px")
    ap.add_argument("--roughness-m", type=float, default=0.03)
    ap.add_argument("--shallow-depth-m", type=float, default=0.9)
    ap.add_argument("--shallow-mode", choices=("scale", "clip"), default="scale")
    ap.add_argument("--seed", type=int, default=901)
    ap.add_argument("--measure-json", default="", help="also write the measurement table here")
    args = ap.parse_args()

    orient = json.loads((Path(args.orientation_from) / "arena_meta.json").read_text())["orientation"]
    orient = {**orient, "note": f"Chrono's BMP convention, copied from {Path(args.orientation_from).name} (calibrated there)"}

    spec = BowlSpec(
        wall_deg=args.wall_deg,
        depth_m=args.depth_m,
        bottom_radius_m=args.bottom_radius_m,
        entry_deg=args.entry_deg,
        entry_azimuth_deg=args.entry_azimuth_deg,
        entry_halfwidth_deg=args.entry_halfwidth_deg,
        centre_x_m=float(args.centre_m[0]),
        centre_y_m=float(args.centre_m[1]),
        round_m=args.round_m,
        roughness_amplitude_m=args.roughness_m,
        seed=args.seed,
    )
    arms = (
        ArmSpec("deep", None, "scale"),
        ArmSpec("shallow", args.shallow_depth_m, args.shallow_mode),
        ArmSpec("flat", 0.0, "scale"),
    )
    written = write_bowl_arms(spec, Path(args.root), orient, arms=arms, prefix=args.prefix)

    rows = {tag: measure_arena(d) for tag, d in written.items()}
    hdr = (f"{'arm':<9s}{'D_req':>7s}{'D_bmp':>7s}{'wall_req':>9s}{'wall_bmp':>9s}{'wall_max':>9s}"
           f"{'chord':>7s}{'relief':>8s}{'step_cm':>8s}{'steps/D':>8s}{'plain_max':>10s}{'R_top':>7s}")
    print(hdr)
    for tag in written:
        m = rows[tag]
        print(f"{tag:<9s}{m['requested_depth_m']:7.2f}{m['measured_depth_m']:7.2f}"
              f"{m['requested_wall_deg']:9.2f}{m['measured_wall_deg']:9.2f}{m['measured_wall_max_deg']:9.2f}"
              f"{m['chord_grade_deg']:7.1f}{m['feature_relief_m']:8.2f}{m['quant_step_m']*100:8.2f}"
              f"{m['quant_steps_over_depth']:8.0f}{m['plain_max_slope_deg']:10.2f}{m['R_top_m']:7.2f}")

    tags = list(written)
    diffs = {}
    for i, a in enumerate(tags):
        for b in tags[i + 1:]:
            d = grey_difference_outside_rim(written[a], written[b])
            diffs[f"{a}-{b}"] = d
            print(f"outside the rim {a} vs {b}: max grey difference {d['max_grey_diff']} "
                  f"over {d['pixels_compared']} px ({d['n_differing']} differ)")
    for tag, d in written.items():
        print(f"wrote {d}")

    if args.measure_json:
        Path(args.measure_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.measure_json).write_text(json.dumps(
            {"spec": {k: v for k, v in vars(args).items()}, "arms": rows, "grey_diff": diffs}, indent=2))


if __name__ == "__main__":
    main()
