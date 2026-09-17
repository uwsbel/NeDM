#!/usr/bin/env python3
"""Read-only AMD validation of large-arena height and recorded RGB-D geometry.

Constructs only a Chrono terrain patch: no vehicle rollout, new rendering,
training, or calibration mutation. Simulator truth appears only in audit
artifacts, never in the saved observation consumed by the model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def stats(error):
    error=np.asarray(error,float)
    if not len(error):
        return {"count":0,"median_abs_m":None,"p95_abs_m":None,"rmse_m":None,"max_abs_m":None,"signed_mean_m":None}
    return {"count":int(len(error)),"median_abs_m":float(np.median(np.abs(error))),
            "p95_abs_m":float(np.quantile(np.abs(error),.95)),"rmse_m":float(np.sqrt(np.mean(error**2))),
            "max_abs_m":float(np.max(np.abs(error))),"signed_mean_m":float(np.mean(error))}


def construct_terrain(arena,tmap):
    import pychrono as chrono
    import pychrono.vehicle as veh
    system=chrono.ChSystemSMC()
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    terrain=veh.RigidTerrain(system)
    material=chrono.ChContactMaterialSMC()
    material.SetFriction(.9)
    meta=tmap.meta
    terrain.AddPatch(material,chrono.CSYSNORM,str(arena/meta["bmp"]),float(meta["size_m"]),float(meta["size_m"]),
                     float(meta["height_min_m"]),float(meta["height_max_m"]))
    terrain.Initialize()
    system.GetCollisionSystem().BindAll()
    height=float(meta["height_max_m"])+10.
    def query(points):
        return np.asarray([terrain.GetHeight(chrono.ChVector3d(float(x),float(y),height)) for x,y in points],float)
    # Retain system and terrain so the captured query function owns live C++ objects.
    return system,terrain,query


def check_pads(tmap,query,case):
    reports=[]
    for name,center in (("launch",case["layout"]["start_xy"]),("arrival",case["goal_xy"])):
        center=np.asarray(center,float)
        rr,tt=np.meshgrid(np.linspace(0.,8.,9),np.linspace(0.,2.*np.pi,32,endpoint=False))
        points=center+np.column_stack((rr.ravel()*np.cos(tt.ravel()),rr.ravel()*np.sin(tt.ravel())))
        actual=query(points)
        target=tmap.height(points[:,0],points[:,1])
        # Fit a plane as a robust measured grade over the whole protected pad.
        plane=np.linalg.lstsq(np.c_[points-center,np.ones(len(points))],actual,rcond=None)[0]
        reports.append({"name":name,"center_xy_m":center.tolist(),"radius_m":8.,"height_range_m":float(np.ptp(actual)),
                        "fitted_grade_deg":math.degrees(math.atan(float(np.linalg.norm(plane[:2])))),
                        "height_vs_bmp":stats(actual-target),"sample_count":len(points)})
    return reports


def camera_metadata(observation_dir,override):
    source=Path(override) if override else observation_dir/"observation.json"
    value=json.loads(source.read_text())
    camera=value.get("camera",value)
    for key in ("width","height","hfov_rad","cam_height_m","depth_ray_scale","max_depth_m"):
        if key not in camera:
            raise ValueError(f"Missing declared camera field: {key}")
    return camera,source


def marker_check(rgb,camera,truth_path):
    from nedm.traverse.camera import CameraModel
    if not truth_path.exists():
        return {"available":False,"reason":"No separate sensor_validation_truth.json; no independent RGB registration claim"}
    truth=json.loads(truth_path.read_text())
    xyz=truth.get("roof_marker_world_xyz_m",truth.get("marker_world_xyz_m"))
    if xyz is None:
        return {"available":False,"truth_path":str(truth_path),"reason":"Truth artifact has no roof_marker_world_xyz_m"}
    camera_model=CameraModel(width=int(camera["width"]),height=int(camera["height"]),
                            hfov_rad=float(camera["hfov_rad"]),cam_height_m=float(camera["cam_height_m"]))
    u,v=camera_model.world_to_pixel(*np.asarray(xyz,float))
    radius=max(8,int(round(12.*int(camera["width"])/1024.)))
    x0,x1=max(0,int(round(u))-radius),min(rgb.shape[1],int(round(u))+radius+1)
    y0,y1=max(0,int(round(v))-radius),min(rgb.shape[0],int(round(v))+radius+1)
    patch=rgb[y0:y1,x0:x1].astype(float)
    mask=(patch[...,2]>110.)&(patch[...,2]>1.35*patch[...,0]+10.)&(patch[...,2]>1.2*patch[...,1]+10.)
    result={"available":True,"truth_sha256":sha(truth_path),"expected_pixel_uv":[float(u),float(v)],
            "blue_marker_pixels":int(mask.sum()),"search_radius_px":radius,
            "scope":"One roof marker checks RGB world projection; terrain depth residuals separately check range/FOV across the arena"}
    if mask.sum()<2:
        result.update(detected=False,pixel_error=None)
    else:
        rows,cols=np.nonzero(mask)
        found=[float(cols.mean()+x0),float(rows.mean()+y0)]
        result.update(detected=True,detected_pixel_uv=found,pixel_error=float(math.hypot(found[0]-u,found[1]-v)))
    return result


def check_observation(case,tmap,query,observation_dir,args):
    from nedm.traverse.camera import CameraModel
    from nedm.traverse.fdm_diverse_data import encode_global_rgbd
    camera,camera_source=camera_metadata(observation_dir,args.camera_json)
    observation_path=observation_dir/"observation.npz"
    with np.load(observation_path) as obs:
        rgb,depth=np.asarray(obs["rgb"]),np.asarray(obs["depth_m"],float)
        pose=np.asarray(obs["pose"],float)
        encoded=np.asarray(obs["rgbd"],float)
    if depth.shape!=(int(camera["height"]),int(camera["width"])) or rgb.shape!=(*depth.shape,3):
        raise ValueError("Raw RGB/depth dimensions disagree with the declared camera")
    model=CameraModel(width=int(camera["width"]),height=int(camera["height"]),
                      hfov_rad=float(camera["hfov_rad"]),cam_height_m=float(camera["cam_height_m"]))
    world_x,world_y,world_z=model.depth_to_world(depth,convention="ray",ray_scale=float(camera["depth_ray_scale"]))
    finite=np.isfinite(depth)&(depth>0.)&(depth<float(camera["max_depth_m"]))
    interior=finite&(np.abs(world_x)<tmap.half-3.)&(np.abs(world_y)<tmap.half-3.)
    clear=interior.copy()
    for asset in case["layout"]["assets"]:
        radius=max(float(asset["footprint_radius_m"]),float(asset["dims"].get("canopy_radius_m",0.)))+4.
        clear &= (world_x-float(asset["x_m"]))**2+(world_y-float(asset["y_m"]))**2>radius**2
    clear &= (world_x-pose[0])**2+(world_y-pose[1])**2>7.**2
    flat_indices=np.flatnonzero(clear)
    if len(flat_indices)<min(100,args.depth_samples):
        raise ValueError("Too few unobstructed interior terrain pixels for a camera check")
    rng=np.random.default_rng(args.seed+1)
    chosen=np.sort(rng.choice(flat_indices,size=min(args.depth_samples,len(flat_indices)),replace=False))
    rows,cols=np.unravel_index(chosen,depth.shape)
    points=np.column_stack((world_x.ravel()[chosen],world_y.ravel()[chosen]))
    measured_z=world_z.ravel()[chosen]
    actual_height=query(points)
    bmp_height=tmap.height(points[:,0],points[:,1])
    error=measured_z-actual_height
    groups={"all":stats(error),"inner_arena":stats(error[np.max(np.abs(points),axis=1)<.5*tmap.half]),
            "outer_arena":stats(error[np.max(np.abs(points),axis=1)>.75*tmap.half])}
    for sx,sy,label in ((-1,-1,"southwest"),(-1,1,"northwest"),(1,-1,"southeast"),(1,1,"northeast")):
        groups[label]=stats(error[(points[:,0]*sx>0)&(points[:,1]*sy>0)])
    # Verify all physical arena corners project inside the raw frame over the
    # declared terrain elevation range, without using pixel-hit errors to mask.
    corners=np.array([[x,y,z] for x in (-tmap.half,tmap.half) for y in (-tmap.half,tmap.half)
                      for z in (tmap.meta["height_min_m"],tmap.meta["height_max_m"])])
    uu,vv=model.world_to_pixel(corners[:,0],corners[:,1],corners[:,2])
    margin=float(np.min(np.r_[uu,vv,model.width-1-uu,model.height-1-vv]))
    recomputed=encode_global_rgbd(rgb,depth,camera)
    if recomputed.shape!=encoded.shape:
        raise ValueError("Saved model input shape disagrees with the current declared encoding")
    encode_error=float(np.max(np.abs(recomputed-encoded)))
    marker=marker_check(rgb,camera,observation_dir/"sensor_validation_truth.json")
    result={"observation_sha256":sha(observation_path),"camera_metadata_sha256":sha(camera_source),"camera":camera,
            "raw_shape":list(rgb.shape),"model_rgbd_shape":list(encoded.shape),"metric_depth_dtype":"float64 for audit; source dtype retained in observation",
            "valid_raw_depth_fraction":float(finite.mean()),"unobstructed_interior_pixels":int(clear.sum()),
            "excluded_asset_margin_m":4.,"excluded_vehicle_radius_m":7.,"interior_edge_margin_m":3.,
            "arena_corner_min_image_margin_px":margin,"all_arena_corners_visible":margin>=0.,
            "coverage_width_at_zero_elevation_m":2.*model.cam_height_m*math.tan(model.hfov_rad/2.),
            "depth_elevation_vs_chrono_m":groups,"depth_elevation_vs_bmp_m":stats(measured_z-bmp_height),
            "same_ray_xy_bmp_vs_chrono_m":stats(bmp_height-actual_height),
            "model_encoding_max_abs_error":encode_error,
            "model_invalid_fraction":float((encoded[3]==-2.).mean()),
            "valid_model_elevation_fraction_at_clip_limit":float((np.abs(encoded[3][encoded[3]!=-2.])>=1.).mean()),
            "rgb_marker_alignment":marker,
            "exclusion_rule":"Masks depend only on bounds, finite range and declared asset/vehicle footprints; never exclude pixels according to measured error",
            "limits":"Terrain residual is geometric sensor calibration, not learned accuracy. One RGB marker cannot prove every object boundary is registered."}
    samples={"camera_pixel_rc":np.column_stack((rows,cols)),"camera_hit_xy_m":points,
             "camera_hit_elevation_m":measured_z,"camera_hit_chrono_height_m":actual_height,"camera_hit_bmp_height_m":bmp_height}
    return result,samples


def main():
    from nedm.traverse.terrain import TerrainMap
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",type=Path,required=True)
    parser.add_argument("--observation-dir",type=Path)
    parser.add_argument("--camera-json",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--world-samples",type=int,default=4000)
    parser.add_argument("--depth-samples",type=int,default=4096)
    parser.add_argument("--seed",type=int,default=2026090917)
    parser.add_argument("--strict",action="store_true",help="Exit2 if declared audit gates fail; report is always retained")
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve previous audit results; use a new output directory")
    if args.world_samples<100 or args.depth_samples<100:
        raise ValueError("Use at least100 deterministic samples per diagnostic")
    case=json.loads(args.case.read_text())
    if case["split"]=="test":
        raise ValueError("Runtime development camera calibration cannot open a sealed test case")
    arena=(ROOT/case["arena"]).resolve()
    meta_path=arena/"arena_meta.json"
    initial_meta_sha=sha(meta_path)
    tmap=TerrainMap.from_dir(arena)
    started=time.monotonic()
    system,terrain,query=construct_terrain(arena,tmap)
    rng=np.random.default_rng(args.seed)
    points=rng.uniform(-tmap.half+3.,tmap.half-3.,size=(args.world_samples,2))
    actual=query(points)
    expected=tmap.height(points[:,0],points[:,1])
    world_stats=stats(actual-expected)
    pad_stats=check_pads(tmap,query,case)
    report={"schema":1,"scene_id":case["id"],"split":case["split"],"source_sha256":sha(__file__),
            "case_sha256":sha(args.case),"arena_bmp_sha256":sha(arena/tmap.meta["bmp"]),"arena_meta_sha256":initial_meta_sha,
            "world_height_chrono_minus_bmp_m":world_stats,"flat_pads":pad_stats,"query_ray_origin_z_m":tmap.meta["height_max_m"]+10.,
            "geometry_only_no_simulation":True,"orientation_modified":False}
    arrays={"world_xy_m":points,"world_chrono_height_m":actual,"world_bmp_height_m":expected}
    q=float(tmap.meta["quantization_step_m"])
    gates={"world_median_abs_m":max(.08,2.*q),"world_p95_abs_m":max(.35,5.*q),"pad_height_range_m":.02,
           "pad_fitted_grade_deg":.2,"camera_median_abs_m":.10,"camera_p95_abs_m":.35,
           "model_encoding_max_abs_error":1e-6,"rgb_marker_max_pixel_error":4.}
    checks={"world_height_median":world_stats["median_abs_m"]<=gates["world_median_abs_m"],
            "world_height_p95":world_stats["p95_abs_m"]<=gates["world_p95_abs_m"],
            "flat_pads":all(p["height_range_m"]<=gates["pad_height_range_m"] and p["fitted_grade_deg"]<=gates["pad_fitted_grade_deg"] for p in pad_stats)}
    if args.observation_dir:
        observation,samples=check_observation(case,tmap,query,args.observation_dir,args)
        report["observation"]=observation;arrays.update(samples)
        d=observation["depth_elevation_vs_chrono_m"]["all"]
        checks.update(camera_coverage=observation["all_arena_corners_visible"],
                      camera_depth_median=d["median_abs_m"]<=gates["camera_median_abs_m"],
                      camera_depth_p95=d["p95_abs_m"]<=gates["camera_p95_abs_m"],
                      model_encoding=observation["model_encoding_max_abs_error"]<=gates["model_encoding_max_abs_error"])
        marker=observation["rgb_marker_alignment"]
        if marker["available"]:
            checks["rgb_marker_alignment"]=bool(marker.get("detected") and marker["pixel_error"]<=gates["rgb_marker_max_pixel_error"])
    report.update(gates=gates,checks=checks,all_available_checks_passed=all(checks.values()),wall_s=time.monotonic()-started)
    if sha(meta_path)!=initial_meta_sha:
        raise RuntimeError("Audit unexpectedly modified frozen arena metadata")
    args.out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(args.out/"geometry_samples.npz",**arrays)
    dump(args.out/"geometry_validation.json",report)
    print(json.dumps({"out":str(args.out),"checks":checks,"world_height":world_stats,
                      "camera_depth":report.get("observation",{}).get("depth_elevation_vs_chrono_m",{}).get("all"),
                      "wall_s":report["wall_s"]},indent=2))
    return 0 if all(checks.values()) or not args.strict else 2


if __name__=="__main__":
    raise SystemExit(main())
