#!/usr/bin/env python3
"""Deterministic large heightfield scenes for paired RGB-D FDM data collection.

Geometry only: this script does not run physics, train, label failures, or score
routes. All candidate references are geometric and shared by construction;
terrain/asset metadata is restricted to simulator construction and auditing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FAMILIES = ("rolling_hills", "ridge_passes", "cross_slopes", "valley_network", "rough_mosaic", "mixed_obstacles")
SPLIT_COUNTS = {"train": 4, "val": 1, "test": 1}
START = np.array([-100., -55.])
GOAL = -START
OFFSETS = (0., -22., 22., -44., 44.)
SPEEDS = (2., 4., 6.)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def encode(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(type(obj).__name__)
    path.write_text(json.dumps(value, indent=2, default=encode, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smoothstep(x):
    x = np.clip(x, 0., 1.)
    return x*x*x*(10.+x*(-15.+6.*x))


def gauss(x, y, cx, cy, sx, sy=None, angle=0.):
    sy = sx if sy is None else sy
    c, s = math.cos(angle), math.sin(angle)
    dx, dy = x-cx, y-cy
    u, v = c*dx+s*dy, -s*dx+c*dy
    return np.exp(-.5*((u/sx)**2+(v/sy)**2))


def coherent_noise(shape, rng, res, corr):
    # Fourier-filtered periodic field; no extra dependency or per-pixel loops.
    ky = np.fft.fftfreq(shape[0], d=res)[:, None]
    kx = np.fft.rfftfreq(shape[1], d=res)[None, :]
    filt = np.exp(-2.*np.pi**2*corr**2*(kx*kx+ky*ky))
    field = np.fft.irfft2(np.fft.rfft2(rng.standard_normal(shape))*filt, s=shape)
    return field/max(float(field.std()), 1e-10)


def canonical_route_geometry(x, y, offset):
    direction = (GOAL-START)/np.linalg.norm(GOAL-START)
    normal = np.array([-direction[1], direction[0]])
    dx, dy = x-START[0], y-START[1]
    t = np.clip((dx*direction[0]+dy*direction[1])/np.linalg.norm(GOAL-START), 0., 1.)
    lateral = dx*normal[0]+dy*normal[1]
    return t, lateral-float(offset)*np.sin(np.pi*t)**2


def terrain_field(x, y, rng, family, res, composition_ood=False):
    height = np.zeros_like(x)
    features = []
    def hill(cx, cy, amplitude, sigma, sy=None, angle=0., kind="smooth_hill"):
        nonlocal height
        height += amplitude*gauss(x, y, cx, cy, sigma, sy, angle)
        features.append({"kind":kind, "center_xy_canonical_m":[cx,cy], "amplitude_m":amplitude,
                         "sigma_x_m":sigma, "sigma_y_m":sigma if sy is None else sy, "yaw_rad":angle})
    # Broad, low background relief keeps empty sectors from being identical.
    hill(float(rng.uniform(-30.,30.)), float(rng.uniform(-30.,30.)), float(rng.uniform(1.,2.)), 48.)
    if family == "rolling_hills":
        for _ in range(11):
            hill(float(rng.uniform(-78.,78.)),float(rng.uniform(-85.,85.)),float(rng.uniform(3.,9.)),float(rng.uniform(7.,17.)))
        hill(float(rng.uniform(-20.,20.)),float(rng.uniform(-15.,15.)),7.0,5.3,kind="steep_smooth_hill")
        rough, corr = .12, 2.5
    elif family == "ridge_passes":
        for ridge_index, cx in enumerate((-47., 17., 66.)):
            cy = float(rng.uniform(-55.,55.))
            amplitude = float(rng.uniform(8.0,11.5))
            sigma = float(rng.uniform(5.5,8.0))
            ridge = amplitude*np.exp(-.5*((x-cx)/sigma)**2)*np.exp(-.5*(y/95.)**6)
            gap = 1.-.94*np.exp(-.5*((y-cy)/float(rng.uniform(12.,18.)))**2)
            height += ridge*gap
            features.append({"kind":"ridge_with_pass", "center_x_canonical_m":cx, "pass_y_canonical_m":cy,
                             "height_m":amplitude,"sigma_x_m":sigma,"ridge_index":ridge_index})
        rough, corr = .10, 2.8
    elif family == "cross_slopes":
        for cx, cy in ((-48.,-22.),(14.,17.),(66.,38.)):
            hill(cx+float(rng.uniform(-10.,10.)),cy+float(rng.uniform(-12.,12.)),float(rng.uniform(8.,12.)),
                 float(rng.uniform(7.,11.)),float(rng.uniform(25.,40.)),angle=float(rng.uniform(-.65,.65)),kind="banked_elongated_hill")
        rough, corr = .15, 2.0
    elif family == "valley_network":
        for cy in (-45.,20.,69.):
            amplitude = float(rng.uniform(-7.,-4.))
            centerline = cy+13.*np.sin((x+float(rng.uniform(-20.,20.)))/32.)
            height += amplitude*np.exp(-.5*((y-centerline)/float(rng.uniform(8.,13.)))**2)*np.exp(-.5*(x/100.)**6)
            features.append({"kind":"meandering_valley", "baseline_y_canonical_m":cy,"depth_m":-amplitude,
                             "meander_amplitude_m":13.,"meander_scale_m":32.})
        for _ in range(5):
            hill(float(rng.uniform(-65.,65.)),float(rng.uniform(-70.,70.)),float(rng.uniform(3.,7.)),float(rng.uniform(7.,15.)))
        rough, corr = .18, 2.5
    elif family == "rough_mosaic":
        for _ in range(8):
            hill(float(rng.uniform(-80.,80.)),float(rng.uniform(-80.,80.)),float(rng.uniform(-3.,7.)),float(rng.uniform(6.,16.)))
        rough, corr = .48, 1.5
        rough_mask = .2+.8*(.5+.5*np.sin(x/25.)*np.cos(y/31.))
        height += coherent_noise(x.shape,rng,res,.9)*.10*rough_mask
    else:
        for cx, cy, amplitude, sigma in ((-45.,-8.,7.,5.2),(18.,18.,9.,8.),(62.,-15.,7.5,6.)):
            hill(cx+float(rng.uniform(-9.,9.)),cy+float(rng.uniform(-10.,10.)),amplitude*float(rng.uniform(.9,1.1)),sigma)
        hill(30.,-48.,-5.,12.,24.,.2,kind="smooth_basin")
        rough, corr = .24, 1.8
    noise = coherent_noise(x.shape,rng,res,corr)*rough
    height += noise
    features.append({"kind":"coherent_roughness","rms_design_m":rough,"correlation_m":corr})
    if composition_ood:
        # A composition absent from train/validation: two oblique ridge lines
        # intersect a curved trough. Their parameters are fixed, not tuned to
        # any observed outcome or score. All sealed outcomes remain unopened.
        for angle, cx in ((-.58,-18.),(.72,38.)):
            hill(cx,-5.,5.5,5.5,43.,angle,kind="heldout_oblique_ridge")
        trough = y+24.*np.sin((x-10.)/38.)
        height -= 3.*np.exp(-.5*(trough/8.)**2)*np.exp(-.5*(x/90.)**6)
        features.append({"kind":"heldout_curved_trough","depth_m":3.,"width_sigma_m":8.})
    # One gentle route corridor gives each scene a geometric positive-control
    # candidate. Its side and relief vary; physics still determines feasibility.
    # Candidate generation itself sees no such label and filters no terrain.
    control_offset = float(rng.choice([-44.,44.]))
    t, distance = canonical_route_geometry(x,y,control_offset)
    corridor_weight = 1.-smoothstep((np.abs(distance)-6.)/10.)
    control_relief = float(rng.uniform(1.5,4.5))
    control_ground = control_relief*(np.sin(np.pi*t)**2+.22*np.sin(3.*np.pi*t)**2)
    height = (1.-corridor_weight)*height+corridor_weight*control_ground
    features.append({"kind":"gentle_control_corridor","offset_m":control_offset,
                     "core_half_width_m":6.,"blend_width_m":10.,"relief_m":control_relief,
                     "semantics":"Geometry control only; no claim of Chrono passability"})
    # Exactly flat launch and arrival pads, with zero first/second derivatives
    # at each end of the 12 m quintic transition. No sharp launch lip.
    for center in (START,GOAL):
        radius = np.hypot(x-center[0],y-center[1])
        height *= smoothstep((radius-10.)/12.)
    return height,features,control_offset


def rotate_xy(points, angle):
    c,s = math.cos(angle),math.sin(angle)
    p = np.asarray(points,float)
    return np.stack((c*p[...,0]-s*p[...,1],s*p[...,0]+c*p[...,1]),axis=-1)


def make_assets(rng, family, control_offset, angle, tmap):
    assets = []
    offsets = [o for o in OFFSETS if o != control_offset]
    count = 3 if family in ("rolling_hills","cross_slopes") else 6 if family != "mixed_obstacles" else 11
    direction = (GOAL-START)/np.linalg.norm(GOAL-START)
    normal = np.array([-direction[1],direction[0]])
    for index in range(count):
        accepted = None
        for attempt in range(200):
            t = float(rng.uniform(.16,.86))
            offset = float(offsets[index%len(offsets)])
            p = START+t*(GOAL-START)+(offset*np.sin(np.pi*t)**2+float(rng.uniform(-1.2,1.2)))*normal
            _, control_dist = canonical_route_geometry(p[0],p[1],control_offset)
            if abs(control_dist) < 12. or min(np.linalg.norm(p-START),np.linalg.norm(p-GOAL)) < 24.:
                continue
            world = rotate_xy(p,angle)
            if any(np.linalg.norm(world-np.array([a["x_m"],a["y_m"]])) < 8. for a in assets):
                continue
            if float(tmap.slope(*world)) > math.tan(math.radians(22.)):
                continue
            accepted = world
            break
        if accepted is None:
            continue
        if family == "mixed_obstacles" and index%3 == 2:
            radius = float(rng.uniform(.25,.42))
            assets.append({"kind":"tree","x_m":float(accepted[0]),"y_m":float(accepted[1]),
                           "yaw_rad":float(rng.uniform(-math.pi,math.pi)),"footprint_radius_m":radius,
                           "dims":{"trunk_radius_m":radius,"trunk_height_m":3.4,"canopy_radius_m":1.6}})
        else:
            edge = float(rng.uniform(1.2,2.8))
            assets.append({"kind":"rock","x_m":float(accepted[0]),"y_m":float(accepted[1]),
                           "yaw_rad":float(rng.uniform(-math.pi,math.pi)),"footprint_radius_m":edge/math.sqrt(2.),
                           "dims":{"edge_m":edge,"height_m":float(rng.uniform(.8,1.9))}})
    return assets


def geometry_stats(tmap, start, goal):
    from nedm.traverse.terrain import _slope_stats
    grid = tmap.height_grid
    native = _slope_stats(grid,tmap.res)
    span_cells = max(1,round(.5/tmap.res))
    dx = (grid[:,2*span_cells:]-grid[:,:-2*span_cells])/(2*span_cells*tmap.res)
    dy = (grid[2*span_cells:,:]-grid[:-2*span_cells,:])/(2*span_cells*tmap.res)
    grade = np.hypot(dx[span_cells:-span_cells],dy[:,span_cells:-span_cells])
    pad_reports = []
    theta = np.linspace(0.,2.*np.pi,80,endpoint=False)
    for name,pad in (("launch",start),("arrival",goal)):
        radii = np.linspace(0.,8.,17)
        xy = pad+np.stack((radii[:,None]*np.cos(theta),radii[:,None]*np.sin(theta)),axis=-1).reshape(-1,2)
        heights = tmap.height(xy[:,0],xy[:,1])
        slopes = tmap.slope(xy[:,0],xy[:,1])
        report = {"name":name,"center_xy_m":pad.tolist(),"verified_radius_m":8.,"sample_count":len(xy),
                  "height_range_m":float(np.ptp(heights)),"max_slope_deg":math.degrees(math.atan(float(slopes.max()))),
                  "height_m":float(np.median(heights))}
        if report["height_range_m"] > 1e-10 or report["max_slope_deg"] > 1e-8:
            raise AssertionError(("Nonflat quantized launch/arrival pad",report))
        pad_reports.append(report)
    return {"height_min_m":float(grid.min()),"height_max_m":float(grid.max()),"relief_m":float(np.ptp(grid)),
            "quantization_step_m":tmap.meta["quantization_step_m"],"resolution_m_per_px":tmap.res,
            "native_slope_stats":native,"grade_measurement_span_m":2*span_cells*tmap.res,
            "max_grade_deg":math.degrees(math.atan(float(grade.max()))),
            "p99_grade_deg":math.degrees(math.atan(float(np.quantile(grade,.99)))),
            "fraction_over_30deg":float((grade>math.tan(math.radians(30.))).mean()),
            "fraction_over_40deg":float((grade>math.tan(math.radians(40.))).mean()),
            "flat_pads":pad_reports}


def create_scene(root, arena_root, family_index, split, index, size, pixels, horizon):
    from nedm.traverse.terrain import TerrainMap
    from nedm.traverse.fdm_rgbd_planner import propose_route_families,check_reference_contract
    from nedm.traverse.fdm_mppi import MPPIConfig,validate_reference
    split_offset = {"train":10000,"val":20000,"test":30000}[split]
    seed = 2026090900+split_offset+family_index*100+index
    rng = np.random.default_rng(seed)
    family = FAMILIES[family_index]
    scene_id = f"diverse_v1_{split}_{family}_{index:02d}"
    arena = arena_root/f"arena_fdm_{scene_id}"
    arena.mkdir(parents=True,exist_ok=False)
    angle = float(rng.integers(0,4))*math.pi/2.
    start,goal = rotate_xy(np.stack((START,GOAL)),angle)
    res = size/pixels
    coordinate = -size/2.+(np.arange(pixels)+.5)*res
    xx,yy = np.meshgrid(coordinate,coordinate)
    c,s = math.cos(angle),math.sin(angle)
    canonical_x,canonical_y = c*xx+s*yy,-s*xx+c*yy
    composition_ood = split == "test" and family_index%2 == 1
    heights,features,control_offset = terrain_field(canonical_x,canonical_y,rng,family,res,composition_ood)
    low = math.floor(float(heights.min())*20.)/20.-.05
    high = math.ceil(float(heights.max())*20.)/20.+.05
    gray = np.rint(255.*(heights-low)/(high-low)).clip(0,255).astype(np.uint8)
    Image.fromarray(gray[::-1]).save(arena/"arena_000.bmp")
    orientation = {"rot90":0,"flipud":True,"calibrated":True,
                   "calibration_source":"assets/traverse/arena_v1/arena_meta.json",
                   "scope":"Inherited BMP coordinate transform only; no new Chrono height residual claim"}
    metadata = {"size_m":size,"pixels":pixels,"resolution_m_per_px":res,"height_min_m":low,"height_max_m":high,
                "quantization_step_m":(high-low)/255.,"bmp":"arena_000.bmp","seed":seed,"orientation":orientation,
                "features":[],"authoring_features":features,"family":family,"scene_rotation_rad":angle,
                "canonical_to_world":"world = R(scene_rotation_rad) @ canonical",
                "authoring_feature_scope":"Simulator construction only; never model input",
                "launch_pads":{"canonical_centers_xy_m":[START.tolist(),GOAL.tolist()],"flat_radius_m":10.,"transition_width_m":12.,"profile":"quintic smootherstep"},
                "split":split,"evaluation_stratum":"composition_ood" if composition_ood else "unseen_layout",
                "heightfield_convention":"World grid h[iy,ix], +x columns and +y rows, pixel centers at -size/2+(i+0.5)*size/pixels. BMP rows are vertically reversed. TerrainMap reverses them back; BMP remains truth."}
    dump(arena/"arena_meta.json",metadata)
    tmap = TerrainMap.from_dir(arena)
    if not np.array_equal(tmap.height_grid,low+gray.astype(float)/255.*(high-low)):
        raise AssertionError("BMP orientation/quantization mismatch")
    stats = geometry_stats(tmap,start,goal)
    metadata["geometry_audit"] = stats
    dump(arena/"arena_meta.json",metadata)
    yaw = float(math.atan2(goal[1]-start[1],goal[0]-start[0]))
    routes = propose_route_families([*start,yaw],goal,speeds=SPEEDS,offsets=OFFSETS,step_m=.5)
    assets = make_assets(rng,family,control_offset,angle,tmap)
    case = {"id":scene_id,"split":split,"arena":str(arena.relative_to(ROOT)),"family":family,
            "evaluation_stratum":metadata["evaluation_stratum"],
            "layout":{"episode_id":scene_id,"seed":seed,"assets":assets,"house_xy":goal.tolist(),"house_yaw":yaw,
                      "start_xy":start.tolist(),"start_yaw":yaw},"goal_xy":goal.tolist(),"goal_radius_m":3.,"horizon_s":horizon,
            "family_parameters":{"speeds":list(SPEEDS),"offsets":list(OFFSETS)},"settle_reference":routes[0],
            "arena_half_extent_m":size/2.,"collection_contract":{"maximum_duration_s":horizon,"goal_completion_is_success":True,
              "horizon_timeout_is_not_goal_success":True,"outcomes_from_physics_only":True,
              "record_sustained_failure_until_horizon":True,"initial_observation_requires_large_arena_camera":True},
            "role":"Independent whole-scene split; geometric references are not filtered using terrain or obstacle truth",
            "protected_test_policy":"No training, checkpoint selection, cost tuning or outcome inspection on test before freeze"}
    case_path = root/"cases"/f"{scene_id}.json"
    dump(case_path,case)
    route_paths,checks = [],[]
    for route_index,route in enumerate(routes):
        route["meta"].update(scene_id=scene_id,family=family,route_index=route_index,
            route_generation="Start/goal and geometric offset/speed only; no terrain or obstacle query",
            nominal_cruise_time_s=float(route["stations"][-1]/route["meta"]["cruise_speed_mps"]),
            maximum_collection_duration_s=horizon)
        check_reference_contract(route)
        check = validate_reference(route,[],MPPIConfig(arena_half_extent_m=size/2.,max_speed_mps=6.),[*start,yaw])
        if not check["valid"]:
            raise AssertionError((scene_id,route_index,check))
        path = root/"cases"/"routes"/scene_id/f"family_{route_index:02d}.json"
        dump(path,route)
        route_paths.append(str(path.relative_to(root/"cases")))
        checks.append(check)
    record = {"scene_id":scene_id,"split":split,"family":family,"seed":seed,
              "evaluation_stratum":metadata["evaluation_stratum"],"case":case_path.name,"routes":route_paths,
              "case_sha256":sha(case_path),"route_sha256":[sha(root/"cases"/p) for p in route_paths],
              "arena":str(arena.relative_to(ROOT)),"arena_bmp_sha256":sha(arena/"arena_000.bmp"),
              "arena_meta_sha256":sha(arena/"arena_meta.json"),"geometry_stats":stats,"asset_count":len(assets),
              "geometry_control_offset_m":control_offset,"route_lengths_m":[float(r["stations"][-1]) for r in routes],
              "reference_validation":checks,"straight_start_goal_distance_m":float(np.linalg.norm(goal-start))}
    return record


def make_preview(root,records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource,Normalize
    from nedm.traverse.terrain import TerrainMap
    selected = [next(r for r in records if r["split"] == "train" and r["family"] == family) for family in FAMILIES]
    fig,axes = plt.subplots(2,3,figsize=(16,11),layout="constrained")
    for ax,record in zip(axes.flat,selected):
        case = json.loads((root/"cases"/record["case"]).read_text())
        tmap = TerrainMap.from_dir(ROOT/record["arena"])
        rgb = LightSource(azdeg=305,altdeg=42).shade(tmap.height_grid,cmap=plt.get_cmap("terrain"),
              norm=Normalize(tmap.height_grid.min(),tmap.height_grid.max()),vert_exag=1.,dx=tmap.res,dy=tmap.res)
        ax.imshow(rgb,origin="lower",extent=(-120,120,-120,120))
        for path in record["routes"][::3]:
            route = json.loads((root/"cases"/path).read_text())
            xy = np.asarray(route["waypoints"])
            ax.plot(xy[:,0],xy[:,1],color="#092d53",lw=.9,alpha=.7)
        for asset in case["layout"]["assets"]:
            color = "#263c25" if asset["kind"] == "tree" else "#a82f28"
            ax.add_patch(plt.Circle((asset["x_m"],asset["y_m"]),asset["footprint_radius_m"]+1.2,color=color))
        start,goal = np.array(case["layout"]["start_xy"]),np.array(case["goal_xy"])
        for point in (start,goal):
            ax.add_patch(plt.Circle(point,10.,fill=False,ec="white",lw=1.4))
        ax.scatter(*start,s=55,c="white",edgecolor="#073045",marker="o",zorder=5)
        ax.scatter(*goal,s=100,c="#ffe856",edgecolor="#073045",marker="*",zorder=5)
        st = record["geometry_stats"]
        ax.set_title(f"{record['family'].replace('_',' ').title()}\nRelief {st['relief_m']:.1f} m | p99 grade {st['p99_grade_deg']:.0f}° | {record['asset_count']} assets",fontsize=11)
        ax.set_xlabel("World x [m]");ax.set_ylabel("World y [m]")
        ax.set_aspect("equal");ax.set_xlim(-120,120);ax.set_ylim(-120,120)
    fig.suptitle("240 m terrain families — training geometry only\nFive geometric paths × three speeds per scene; circles are flat pads, star is goal",fontsize=16)
    preview = root/"geometry"
    preview.mkdir(parents=True,exist_ok=True)
    fig.savefig(preview/"terrain_families_preview.png",dpi=150)
    fig.savefig(preview/"terrain_families_preview.pdf")
    plt.close(fig)
    fig,axes = plt.subplots(2,3,figsize=(16,8),layout="constrained")
    for ax,record in zip(axes.flat,selected):
        tmap = TerrainMap.from_dir(ROOT/record["arena"])
        for path in record["routes"][::3]:
            route = json.loads((root/"cases"/path).read_text())
            xy = np.asarray(route["waypoints"])
            offset = route["meta"]["lateral_offset_m"]
            control = offset == record["geometry_control_offset_m"]
            ax.plot(route["stations"],tmap.height(xy[:,0],xy[:,1]),lw=2.5 if control else 1.2,
                    label=f"{offset:+.0f} m"+(" gentle corridor" if control else ""))
        ax.set_title(record["family"].replace("_"," ").title())
        ax.set_xlabel("Reference station [m]");ax.set_ylabel("Quantized height [m]")
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle("Profiles sampled from the actual BMP — candidate commands remain terrain independent",fontsize=15)
    fig.savefig(preview/"route_height_profiles.png",dpi=150)
    fig.savefig(preview/"route_height_profiles.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"artifacts/traverse/fdm_diverse_v1_20260909")
    parser.add_argument("--arena-root",type=Path,default=ROOT/"assets/traverse")
    parser.add_argument("--pixels",type=int,default=512)
    parser.add_argument("--size-m",type=float,default=240.)
    parser.add_argument("--horizon-s",type=float,default=180.)
    args = parser.parse_args()
    if args.size_m != 240. or args.pixels < 512:
        raise ValueError("This declared scene layout requires 240 m arenas and at least512 pixels")
    root = args.out.resolve()
    if (root/"cases"/"cases.json").exists():
        raise FileExistsError("Preserve existing manifests; choose a fresh output/arena prefix for revisions")
    records=[]
    for split,count in SPLIT_COUNTS.items():
        for family_index in range(len(FAMILIES)):
            for index in range(count):
                record = create_scene(root,args.arena_root.resolve(),family_index,split,index,args.size_m,args.pixels,args.horizon_s)
                records.append(record)
                print(f"{record['scene_id']}: relief {record['geometry_stats']['relief_m']:.2f} m, peakgrade {record['geometry_stats']['max_grade_deg']:.1f} deg, {len(record['routes'])} routes",flush=True)
    hashes=[r["arena_bmp_sha256"] for r in records]
    assert len(set(hashes)) == len(hashes)
    manifest = {"schema":1,"campaign":"fdm_diverse_v1_20260909","records":records,"cases":[r["case"] for r in records],
                "source_sha256":sha(__file__),"split_unit":"Whole independent arena; all route siblings and every time window inherit scene split",
                "counts":{"train":24,"val":6,"test":6,"total_routes":540},
                "test_strata":{"unseen_layout":3,"composition_ood":3},
                "test_policy":"Freeze every model, normalization, cost weight and threshold before opening test outcomes; no test-derived selection",
                "geometry_only":True,"physics_outcomes_not_yet_measured":True,
                "scene_construction_note":"Six base families with independent seeds. Three test scenes add the fixed oblique-ridge plus curved-trough composition absent from train/val. No terrain/obstacle truth is an input to model/scorer.",
                "reference_note":"Five purely geometric offsets ×2/4/6m/s; authored gentle corridor is a construction control, not a feasibility label. Candidate references are never geometry filtered.",
                "camera_requirement":"Current 80m camera cannot be reused. This campaign requires declared registered global RGB-D geometry covering240m; rendering protocol is supplied by collector.",
                "termination_note":"180s cap. A horizon timeout is not goal completion; retain effort/slip/signed-progress telemetry and both outcomes."}
    dump(root/"cases"/"cases.json",manifest)
    for split in SPLIT_COUNTS:
        subset=[r for r in records if r["split"] == split]
        dump(root/"cases"/f"{split}_manifest.json",{**manifest,"records":subset,"cases":[r["case"] for r in subset]})
    pilot=[next(r for r in records if r["split"] == split and r["family"] == family)
           for split,family in (("train","rolling_hills"),("train","ridge_passes"),("train","rough_mosaic"),("val","mixed_obstacles"))]
    dump(root/"cases"/"pilot_manifest.json",{**manifest,"records":pilot,"cases":[r["case"] for r in pilot],
         "pilot_only":True,"pilot_horizon_override_s":60.,"pilot_note":"Throughput and non-test data plumbing smoke; does not redefine full180s cases"})
    make_preview(root,records)
    dump(root/"geometry"/"geometry_audit.json",{"source_sha256":sha(__file__),"scenes":len(records),"routes":540,
         "unique_bmp_sha256":len(set(hashes)),"all_72_pad_checks_passed":True,"all_540_reference_checks_passed":True,
         "stats":[{"scene_id":r["scene_id"],"split":r["split"],**r["geometry_stats"]} for r in records],
         "limits":"Geometry validation is not a Chrono execution or calibrated physical slope/normal check. Pixel coordinates use existing TerrainMap convention; inherited transform needs runtime spot-check at new scale."})
    print(json.dumps({"manifest":str(root/"cases"/"cases.json"),"scenes":len(records),"routes":540}))


if __name__ == "__main__":
    main()
