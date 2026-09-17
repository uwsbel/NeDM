"""Attach actual current RGB-D to the frozen FDM identities and physical targets.

All old privileged candidate fields are discarded. No TerrainMap is opened.
Only train/validation source episodes listed in the frozen pack are accessed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from nedm.traverse.fdm_rgbd_data import CAMERA, COMMAND_FIELDS, GLOBAL_FIELDS, prepare_observation, build_reference_features
from nedm.traverse.fdm_data import load_episode, tracked_route_indices, DT
from nedm.traverse.storage import EpisodeReader


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root",type=Path,default=Path("/home/harry/NeDM"))
    ap.add_argument("--base-pack",type=Path,default=ROOT/"artifacts/traverse/fdm_fast_data_v2")
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--workers",type=int,default=4)
    args=ap.parse_args()
    source=args.source_root.resolve()
    if args.out.resolve().is_relative_to(source):raise ValueError("Output must be outside source checkout")
    args.out.mkdir(parents=True,exist_ok=True)
    if any(args.out.iterdir()):raise ValueError("Use a new empty output directory")
    base=json.loads((args.base_pack/"manifest.json").read_text())
    norm={}
    manifest={"schema":1,"observation":"Actual current RGB-D, calibrated registration; no BMP or asset geometry input",
        "camera":CAMERA,"image_encoding":{"shape":[4,128,128],"dtype":"float16","rgb":"box-resized uint8 /255",
            "depth":"observed ray depth -> RGB-registered elevation, fixed clip(z/10,-1,1); invalid=-2; conservative validity in box resize"},
        "commands_fields":COMMAND_FIELDS,"global_fields":GLOBAL_FIELDS,
        "localization":"Measured current world XY/yaw supplies path/image correspondence; not an end-to-end localization claim",
        "reference_confound":"Archived reference speeds may already contain authored slope/curvature limits. Commands are known actions, but matched evaluation must include uniform/non-terrain-authored speed proposals and image ablations.",
        "controller_domain":base["controller_domain"],"history_fields":base["history_fields"],
        "startup_padding":base["startup_padding"],"event_names":base["event_names"],
        "horizon_s":4.,"output_dt_s":.2,"history_frames":16,
        "base_pack_manifest_sha256":sha(args.base_pack/"manifest.json"),
        "split_rule":base["split_rule"],"selection":base["selection"],
        "code_sha256":{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/"src/nedm/traverse/fdm_rgbd_data.py",ROOT/"src/nedm/traverse/fdm_data.py",ROOT/"src/nedm/traverse/storage.py"]},
        "splits":{}}
    keep=("history","trajectory","work","events","trajectory_mask","event_mask","episode_index","anchor")
    for split in ("train","val"):
        basefile=args.base_pack/(split+".npz")
        if sha(basefile)!=base["splits"][split]["sha256"]:raise ValueError("Frozen base pack changed")
        with np.load(basefile) as d:arrays={k:d[k].copy() for k in keep}
        records=json.loads((args.base_pack/(split+"_episodes.json")).read_text())
        n=len(arrays["history"])
        imagepath=args.out/(split+"_rgbd.npy")
        images=np.lib.format.open_memmap(imagepath,mode="w+",dtype=np.float16,shape=(n,4,128,128))
        arrays.update(commands=np.empty((n,20,5),np.float32),nominal_pose=np.empty((n,20,4),np.float32),global_features=np.empty((n,8),np.float32))

        def one(pair):
            epi,rec=pair
            path=source/rec["source"]
            for name,key in (("meta.json","meta_sha256"),("states.npz","states_sha256")):
                if sha(path/name)!=rec[key]:raise ValueError("Frozen physical source changed: "+str(path))
            store=path.parent
            cam=json.loads((store/"manifest.json").read_text())["camera"]
            for key in ("width","height","hfov_deg","cam_height_m","depth_ray_scale"):
                if not np.isclose(cam[key],CAMERA[key]):raise ValueError("Camera contract changed: "+key)
            ep=load_episode(path);route=ep["meta"]["route"]
            indices=tracked_route_indices(route,ep["poses"])
            rows=np.flatnonzero(arrays["episode_index"]==epi)
            reader=EpisodeReader(path,cache_chunks=1)
            try:
                for row in rows:
                    anchor=int(arrays["anchor"][row])
                    obs=reader.read_window(anchor,1)
                    image=prepare_observation(obs["rgb"][0],obs["depth_mm"][0])
                    if not np.isfinite(image).all():raise ValueError("Nonfinite RGBD")
                    images[row]=image
                    feats=build_reference_features(route,ep["poses"][anchor],station=route["stations"][indices[anchor]],elapsed_s=anchor*DT)
                    for k,v in feats.items():arrays[k][row]=v
            finally:reader.close()
            return dict(rec,rgb_sha256=sha(path/"rgb.bin"),depth_sha256=sha(path/"depth.bin"))

        new_records=[]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i,rec in enumerate(pool.map(one,enumerate(records))):
                new_records.append(rec)
                if (i+1)%50==0:print(split,i+1,"/",len(records),flush=True)
        images.flush();del images
        if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError("Nonfinite model features")
        target=args.out/(split+".npz");np.savez(target,**arrays)
        (args.out/(split+"_episodes.json")).write_text(json.dumps(new_records,indent=1)+"\n")
        if split=="train":
            for k in ("history","commands","global_features"):
                a=arrays[k].reshape(-1,arrays[k].shape[-1])
                norm[k]={"mean":a.mean(0,dtype=np.float64).tolist(),"std":np.maximum(a.std(0,dtype=np.float64),.01).tolist()}
        manifest["splits"][split]={k:v for k,v in base["splits"][split].items() if k not in ("shapes","sha256")}
        manifest["splits"][split].update(shapes={k:list(v.shape) for k,v in arrays.items()},sha256=sha(target),rgbd_file=imagepath.name,rgbd_shape=[n,4,128,128],rgbd_sha256=sha(imagepath))
        print(split,"complete",n,"current observations",flush=True)
    (args.out/"normalization.json").write_text(json.dumps(norm,indent=2)+"\n")
    manifest["normalization_sha256"]=sha(args.out/"normalization.json")
    (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__=="__main__":main()
