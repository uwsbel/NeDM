#!/usr/bin/env python3
"""Construct training-side terrain probes; physics decides their outcomes.

Terrain truth is used solely to author simulator scenes. Every reference is a
geometric start-goal family independent of heights, and no probe is a reserved
evaluation scene. Probe outcomes may inform later training-scene parameters.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v)+"\n")


def gaussian_arena(destination, amplitude=2.8, sigma=1.7, center=(0., 0.), seed=2026090811):
    orientation = json.loads((ROOT / "assets/traverse/arena_v1/arena_meta.json").read_text())["orientation"]
    coordinates = -40.+(np.arange(512)+.5)*(80./512)
    xx, yy = np.meshgrid(coordinates, coordinates)
    height = amplitude*np.exp(-.5*((xx-center[0])**2+(yy-center[1])**2)/sigma**2)
    lo, hi = -.1, amplitude+.1
    # World rows increase in +Y; inverse of the empirically calibrated flip.
    assert orientation["rot90"] == 0 and orientation["flipud"]
    gray = np.rint(255.*(height-lo)/(hi-lo)).clip(0,255).astype(np.uint8)[::-1]
    destination.mkdir(parents=True, exist_ok=True)
    Image.fromarray(gray).save(destination / "arena_000.bmp")
    meta = {"size_m":80., "pixels":512, "resolution_m_per_px":80./512,
            "height_min_m":lo, "height_max_m":hi, "quantization_step_m":(hi-lo)/255.,
            "bmp":"arena_000.bmp", "seed":seed, "orientation":orientation,
            "features":[{"kind":"hill", "x_m":center[0], "y_m":-center[1],
                         "sigma_m":sigma, "amplitude_m":amplitude}],
            "authored_training_probe":{"kind":"Gaussian mound", "amplitude_m":amplitude, "sigma_m":sigma}}
    dump(destination / "arena_meta.json", meta)


def mesa_arena(destination, amplitude=1.8, core_radius=1.4, wall_width=.7, center=(0.,0.), seed=2026090812):
    """A rounded steep berm; it remains a legal continuous rigid heightfield."""
    orientation=json.loads((ROOT / "assets/traverse/arena_v1/arena_meta.json").read_text())["orientation"]
    coordinates=-40.+(np.arange(512)+.5)*(80./512)
    xx,yy=np.meshgrid(coordinates,coordinates)
    radius=np.hypot(xx-center[0],yy-center[1])
    t=np.clip((radius-core_radius)/wall_width,0.,1.)
    height=amplitude*(1.-t*t*(3.-2.*t))
    lo,hi=-.1,amplitude+.1
    assert orientation["rot90"]==0 and orientation["flipud"]
    gray=np.rint(255.*(height-lo)/(hi-lo)).clip(0,255).astype(np.uint8)[::-1]
    destination.mkdir(parents=True,exist_ok=True)
    Image.fromarray(gray).save(destination / "arena_000.bmp")
    dump(destination / "arena_meta.json",{"size_m":80.,"pixels":512,"resolution_m_per_px":80./512,
        "height_min_m":lo,"height_max_m":hi,"quantization_step_m":(hi-lo)/255.,
        "bmp":"arena_000.bmp","seed":seed,"orientation":orientation,"features":[],
        "authored_training_probe":{"kind":"rounded steep mesa","amplitude_m":amplitude,
            "core_radius_m":core_radius,"wall_width_m":wall_width,"center_xy":list(center)}})


def make_case(case_id, arena, start, goal, destination, split="train", role=None):
    from nedm.traverse.fdm_rgbd_planner import propose_route_families
    from nedm.traverse.fdm_mppi import MPPIConfig, validate_reference
    yaw = float(np.arctan2(goal[1]-start[1], goal[0]-start[0]))
    case = {"id":case_id, "split":split, "arena":str(arena.relative_to(ROOT)),
            "layout":{"episode_id":case_id, "seed":2026090811, "assets":[], "house_xy":list(goal),
                      "house_yaw":yaw, "start_xy":list(start), "start_yaw":yaw},
            "goal_xy":list(goal), "goal_radius_m":2.5, "horizon_s":10.,
            "family_parameters":{"speeds":[4.,6.], "offsets":[0.,-8.,8.]},
            "role":role or "training-side physical probe; not a reserved demonstration or test"}
    case_file = destination / f"{case_id}.json"
    dump(case_file,case)
    routes = propose_route_families([*start,yaw],goal,**case["family_parameters"])
    case["settle_reference"]=routes[0]
    dump(case_file,case)
    paths, hashes = [], []
    for i, route in enumerate(routes):
        check = validate_reference(route,[],MPPIConfig(),[*start,yaw])
        if not check["valid"]:
            raise ValueError((case_id,i,check))
        path = destination / "routes" / case_id / f"family_{i:02d}.json"
        dump(path,route)
        paths.append(str(path.relative_to(destination)))
        hashes.append(sha256(path))
    return {"scene_id":case_id, "split":split, "case":case_file.name, "routes":paths,
            "case_sha256":sha256(case_file), "route_sha256":hashes}


def focused_cohort(destination):
    """Freeze varied training/validation layouts before their physics outcomes."""
    rng=np.random.default_rng(2026090817)
    records=[]
    for split,count in (("train",8),("val",4)):
        for index in range(count):
            case_id=f"focus_terrain_{split}_{index:03d}"
            base=rng.uniform(-5.,5.,2)
            yaw=float(rng.uniform(-np.pi,np.pi))
            tangent=np.array([np.cos(yaw),np.sin(yaw)])
            normal=np.array([-tangent[1],tangent[0]])
            lateral=0. if index<count//2 else (-4. if index<count*3//4 else 4.)
            center=base+lateral*normal
            ahead=float(rng.uniform(9.5,10.5))
            amplitude=float(rng.uniform(1.65,2.05))
            core=float(rng.uniform(1.3,1.7))
            width=float(rng.uniform(.65,1.05))
            arena=ROOT / "assets/traverse" / f"arena_fdm_{case_id}"
            mesa_arena(arena,amplitude,core,width,center,seed=2026090817+index)
            record=make_case(case_id,arena,base-ahead*tangent,base+(36.-ahead)*tangent,
                destination,split=split,role="New focused terrain cohort; parameters informed only by training-side physical probes")
            record["construction_parameters"]={"height_m":amplitude,"core_radius_m":core,
                "wall_width_m":width,"feature_lateral_m":lateral,"feature_center_xy":center.tolist(),
                "ahead_m":ahead,"yaw_rad":yaw}
            records.append(record)
    dump(destination / "cases.json",{"schema":1,"records":records,"cases":[r["case"] for r in records],
        "split_unit":"whole scene identity, shared by all six sibling references",
        "selection":"8 train and4 validation geometric layouts fixed before their physics outcomes; half centerline and quarter on each detour side",
        "parameter_evidence":"terrain_probes_mesa_v1: all straight routes blocked, all detours feasible; instantaneous low-vx stall proxy is reported independently"})
    # A separate diagnostic is never included in the training/validation manifest.
    diag=ROOT / "artifacts/traverse/fdm_rgbd_stall_demo_v1/cases"
    base=np.array([11.,-7.]);yaw=.6
    tangent=np.array([np.cos(yaw),np.sin(yaw)])
    arena=ROOT / "assets/traverse/arena_fdm_stall_demo_v1"
    mesa_arena(arena,1.9,1.55,.9,base,seed=2026090818)
    record=make_case("terrain_blockage_diagnostic_v1",arena,base-10.*tangent,base+26.*tangent,
        diag,split="diagnostic",role="Predeclared held-out development diagnostic; excluded from every training and validation cohort")
    dump(diag / "cases.json",{"schema":1,"records":[record],"cases":[record["case"]],
         "selection":"Declared after training-side physics probes, before model scores on this layout; no later geometry tuning"})
    print(json.dumps({"out":str(destination),"scenes":12,"routes":72,"diagnostic":str(diag)}))


def main():
    from nedm.traverse.bowl import BowlSpec, ArmSpec, write_bowl_arms
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("artifacts/traverse/fdm_rgbd_terrain_probes_v1/cases"))
    parser.add_argument("--kind",choices=("initial","mesa","focused"),default="initial")
    args=parser.parse_args()
    if args.kind=="focused":
        focused_cohort(args.out)
        return
    orientation=json.loads((ROOT / "assets/traverse/arena_v1/arena_meta.json").read_text())["orientation"]
    arena_root=ROOT / "assets/traverse"
    if args.kind=="initial":
        gauss=arena_root / "arena_fdm_probe_gaussian"
        gaussian_arena(gauss)
        records=[make_case("terrain_probe_gaussian",gauss,(-10.,0.),(26.,0.),args.out)]
        for name,depth,wall in (("compact47",2.5,47.),("compact52",3.,52.)):
            spec=BowlSpec(wall_deg=wall,depth_m=depth,bottom_radius_m=2.,entry_deg=15.,roughness_amplitude_m=.02,seed=2026090811)
            paths=write_bowl_arms(spec,arena_root,orientation,arms=(ArmSpec("deep",None,"scale"),),prefix=f"arena_fdm_probe_{name}")
            records.append(make_case(f"terrain_probe_{name}",paths["deep"],(-12.,0.),(24.,0.),args.out))
    else:
        records=[]
        for name,amplitude,core,width in (("mesa12",1.2,1.4,.5),("mesa18",1.8,1.4,.7),("mesa24",2.4,2.,.8)):
            arena=arena_root / f"arena_fdm_probe_{name}"
            mesa_arena(arena,amplitude,core,width)
            records.append(make_case(f"terrain_probe_{name}",arena,(-10.,0.),(26.,0.),args.out))
    dump(args.out / "cases.json",{"schema":1,"role":"training-only probe parameters fixed before physical outcomes",
         "records":records,"cases":[r["case"] for r in records]})
    print(json.dumps({"out":str(args.out),"scenes":len(records),"routes":6*len(records)}))


if __name__ == "__main__":
    main()
