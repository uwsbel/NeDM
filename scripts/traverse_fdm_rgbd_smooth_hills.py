#!/usr/bin/env python3
"""Bounded smooth-grade physics probes; never trains or chooses from NN scores."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import traverse_fdm_rgbd_chrono as runner

SPECS = [
    ("gaussian_h3p5_s3p5", "gaussian", 3.5, 3.5),
    ("gaussian_h4p5_s3p5", "gaussian", 4.5, 3.5),
    ("gaussian_h5p5_s3p5", "gaussian", 5.5, 3.5),
    ("gaussian_h6p5_s3p5", "gaussian", 6.5, 3.5),
    ("gaussian_h6_s4p5", "gaussian", 6., 4.5),
    ("gaussian_h8_s4p5", "gaussian", 8., 4.5),
    ("quintic_h4_r8", "quintic", 4., 8.),
    ("quintic_h5_r8", "quintic", 5., 8.),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def author(arena, kind, height, extent, center=(-8., 0.)):
    orientation = json.loads((ROOT / "assets/traverse/arena_v1/arena_meta.json").read_text())["orientation"]
    assert orientation["rot90"] == 0 and orientation["flipud"]
    resolution = 80./512
    coordinates = -40.+(np.arange(512)+.5)*resolution
    xx, yy = np.meshgrid(coordinates, coordinates)
    radius = np.hypot(xx-center[0], yy-center[1])
    if kind == "gaussian":
        heights = height*np.exp(-.5*(radius/extent)**2)
        max_grade = height*math.exp(-.5)/extent
        formula = "H exp(-0.5 (r/sigma)^2); infinitely differentiable rounded toe and crest"
    else:
        t = np.clip(radius/extent, 0., 1.)
        heights = height*(1.-(6*t**5-15*t**4+10*t**3))
        max_grade = 1.875*height/extent
        formula = "H [1-(6t^5-15t^4+10t^3)], t=clip(r/R,0,1); zero first/second derivative at toe and crest"
    low, high = -.1, height+.1
    gray = np.rint(255*(heights-low)/(high-low)).clip(0,255).astype(np.uint8)
    quantized = low+gray.astype(float)*(high-low)/255.
    gy, gx = np.gradient(quantized, resolution)
    grade = np.hypot(gx, gy)
    steps = max(float(np.abs(np.diff(quantized, axis=0)).max()), float(np.abs(np.diff(quantized, axis=1)).max()))
    arena.mkdir(parents=True, exist_ok=False)
    Image.fromarray(gray[::-1]).save(arena / "arena_000.bmp")
    metadata = {"size_m":80., "pixels":512, "resolution_m_per_px":resolution,
        "height_min_m":low, "height_max_m":high, "quantization_step_m":(high-low)/255.,
        "bmp":"arena_000.bmp", "seed":2026090901, "orientation":orientation, "features":[],
        "orientation_calibration_scope":"Inherited coordinate-transform calibration from arena_v1; its fit statistics are not a fresh height accuracy measurement on this hill",
        "smooth_hill":{"kind":kind,"height_m":height,"extent_m":extent,"center_xy":list(center),
            "formula":formula,"analytic_max_grade_degrees":math.degrees(math.atan(max_grade)),
            "quantized_center_difference_max_grade_degrees":math.degrees(math.atan(float(grade.max()))),
            "quantized_max_neighbor_height_difference_m":steps,
            "quantization_note":"Chrono uses the continuous triangulated8-bit heightfield; report measured physical grade as well as analytic design"}}
    runner.dump(arena / "arena_meta.json", metadata)
    return metadata["smooth_hill"]


def make_cases(out, shifted=False, fallback=False, refine=False):
    from nedm.traverse.fdm_rgbd_planner import propose_route_families
    from nedm.traverse.fdm_mppi import MPPIConfig, validate_reference

    template = json.loads((ROOT / "artifacts/traverse/fdm_rgbd_fourway_20260909/cases/fourway_v1.json").read_text())
    records = []
    specs = [(*spec, (-8.,0.)) for spec in SPECS]
    if shifted:
        specs = [("gaussian_h5p5_s3p5_cxm4", "gaussian", 5.5, 3.5, (-4.,0.)),
                 ("gaussian_h5p5_s3p5_cx0", "gaussian", 5.5, 3.5, (0.,0.)),
                 ("gaussian_h6_s4p5_cx0", "gaussian", 6., 4.5, (0.,0.))]
    if fallback:
        specs = [("gaussian_h6p5_s4p5_cx0", "gaussian", 6.5, 4.5, (0.,0.)),
                 ("gaussian_h7_s4p5_cx0", "gaussian", 7., 4.5, (0.,0.))]
    if refine:
        specs = [(f"gaussian_h6p{digit}_s4p5_cx0", "gaussian", 6.+digit/10., 4.5, (0.,0.))
                 for digit in (6,7,8,9)]
    for suffix, kind, height, extent, center in specs:
        case_id = ("smooth_refine_" if refine else "smooth_strength_" if fallback else "smooth_shift_" if shifted else "smooth_probe_")+suffix
        arena = ROOT / "assets/traverse" / ("arena_fdm_"+case_id)
        metadata = author(arena, kind, height, extent, center)
        case = copy.deepcopy(template)
        case.update(id=case_id, split="geometry_probe", arena=str(arena.relative_to(ROOT)),
            role="Geometry-only physical screening before final hill and NN prediction freeze; no training; fixed friction/power/PID")
        case["layout"]["episode_id"] = case_id
        routes = propose_route_families([-18.,0.,0.],[18.,0.],speeds=(4.,6.),offsets=(0.,-8.,8.))
        case["settle_reference"] = routes[0]
        case["smooth_hill"] = metadata
        case_path = out / f"{case_id}.json"
        runner.dump(case_path, case)
        paths = []
        for index, route in enumerate(routes):
            validity = validate_reference(route, [], MPPIConfig(max_speed_mps=6.), [-18.,0.,0.])
            assert validity["valid"], (case_id, index, validity)
            path = out / "routes" / case_id / f"family_{index:02d}.json"
            runner.dump(path, route)
            paths.append(path)
        chosen = [1,3,5] if shifted and not fallback else [1]
        records.append({"scene_id":case_id,"split":"geometry_probe","case":case_path.name,
            "routes":[str(paths[i].relative_to(out)) for i in chosen],"case_sha256":sha(case_path),
            "route_sha256":[sha(paths[i]) for i in chosen],"smooth_hill":metadata,
            "future_safe_detour":str(paths[5].relative_to(out)),"arena_bmp_sha256":sha(arena / "arena_000.bmp")})
    runner.dump(out / "cases.json", {"schema":1,"records":records,
        "selection":("One bounded heightrefinement H6.6/6.7/6.8/6.9 after6.5passed and7rolledback; sigma4.5,center0,straight6m/s.25s diagnostic horizonoverride declared separately. No NNscore selection." if refine else "Predeclared straight-only fallback: center0 restored launch andH6cleared; increaseheightto6.5/7withsigma4.5 andallsettingsunchanged. No NNscore selection." if fallback else "After original broad hills blocked every family, move three declared profiles later along the same path to develop lateral clearance; fixedstraight/right/left6m/s. No NNscore selection." if shifted else "Eight profiles fixed before physical outcomes. Straight6m/s initial screening; diagnostics and unchanged+8m geometricdetour on promising profiles. NN weights/friction/PID unchanged."),
        "source_case":"fdm_rgbd_fourway_20260909/cases/fourway_v1.json",
        "prohibited":"No training and no replacement of old mesa scenario or results"})
    print(json.dumps({"out":str(out),"cases":len(records),"routes":sum(len(r["routes"]) for r in records)}))


def probe(args):
    from nedm.traverse.fdm_slope_probe import SmoothHillDiagnostics
    if (Path(args.out) / "outcome.json").exists():
        raise FileExistsError("Existing physical results must not be overwritten")
    args.command = "collect"
    args.backend, args.depth_ray_scale = "Vulkan_RT_lavapipe", 1.
    args.record_rgbd_stride, args.path_height_source = 0, "truth"
    args.frame_observer = SmoothHillDiagnostics(args.out, args.case)
    runner.run_chrono(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make")
    make.add_argument("--out", type=Path, required=True)
    make.add_argument("--shifted", action="store_true")
    make.add_argument("--fallback", action="store_true")
    make.add_argument("--refine", action="store_true")
    run = sub.add_parser("probe")
    run.add_argument("--case", required=True)
    run.add_argument("--route", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--chrono-data", required=True)
    run.add_argument("--horizon-s", type=float)
    args = parser.parse_args()
    make_cases(args.out, args.shifted, args.fallback, args.refine) if args.command == "make" else probe(args)


if __name__ == "__main__":
    main()
