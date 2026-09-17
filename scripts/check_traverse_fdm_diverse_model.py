#!/usr/bin/env python3
"""CPU inference-only checks of diverse FDM geometry, masks and serialization.

No optimizer is created and no training update is performed. Synthetic affine
RGB and constant-height ray depth give a known physical patch projection target.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"scripts"))

from nedm.traverse.fdm_diverse_data import encode_global_rgbd,build_command_features
from nedm.traverse.fdm_diverse_model import (RGBDFDMConfig,RGBDFiniteHorizonFDM,rotate_world_batch,
    fdm_loss,load_rgbd_checkpoint,pose_to_body_twist,integrate_body_twist)


def synthetic_observation():
    camera={"width":1024,"height":1024,"hfov_rad":math.radians(47.),"cam_height_m":400.,
            "depth_ray_scale":1.,"max_depth_m":600.,"model_image_size":512,"elevation_scale_m":40.}
    u,v=np.meshgrid(np.arange(1024),np.arange(1024))
    f=512./math.tan(camera["hfov_rad"]/2.)
    tx,ty=(u-511.5)/f,-(v-511.5)/f
    z=10.
    depth=(400.-z)*np.sqrt(1.+tx*tx+ty*ty)
    world_x,world_y=tx*(400.-z),ty*(400.-z)
    # Float affine channels are constructed after the uint8 encoding check so
    # projection accuracy is not obscured by 8-bit color quantization.
    rgb=np.rint(np.clip(np.stack(((world_x+200.)/400.,(world_y+200.)/400.,np.ones_like(world_x)*.3),axis=-1),0.,1.)*255).astype(np.uint8)
    encoded=encode_global_rgbd(rgb,depth,camera)
    assert np.max(np.abs(encoded[3]-.25))<1e-6
    uu,vv=np.meshgrid(np.arange(512),np.arange(512))
    f512=256./math.tan(camera["hfov_rad"]/2.)
    x=(uu-255.5)/f512*390.;y=-(vv-255.5)/f512*390.
    encoded[0]=(x+200.)/400.;encoded[1]=(y+200.)/400.;encoded[2]=.3
    return torch.from_numpy(encoded)[None],camera


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a fresh report path")
    torch.set_num_threads(2)
    torch.manual_seed(911)
    image,camera=synthetic_observation()
    cfg=RGBDFDMConfig(candidate_patches=True,horizon=60,dt=.2)
    model=RGBDFiniteHorizonFDM(cfg).eval()
    global_features=torch.tensor([[120.,100.,20.,25.,60./40.,-40./40.,math.sin(.7),math.cos(.7)]])
    relative=torch.tensor([[[5.,2.,math.sin(.4),math.cos(.4)]]])
    with torch.inference_mode():
        patch=model.candidate_patch_pixels(image,relative,global_features)
    center_x=60.+math.cos(.7)*5.-math.sin(.7)*2.
    center_y=-40.+math.sin(.7)*5.+math.cos(.7)*2.
    world_x=center_x+math.cos(1.1)*model.patch_forward_m-math.sin(1.1)*model.patch_left_m
    world_y=center_y+math.sin(1.1)*model.patch_forward_m+math.cos(1.1)*model.patch_left_m
    expected=torch.stack((world_x/200.,world_y/200.,torch.full_like(world_x,-.4),torch.full_like(world_x,.25)))
    projection_error=float((patch[0,0]-expected).abs().max())
    assert projection_error<2e-6,projection_error
    rotation_errors=[]
    for turns in (1,2,3):
        rotated=rotate_world_batch({"rgbd":image,"global_features":global_features},torch.tensor([turns]))
        with torch.inference_mode():
            observed=model.candidate_patch_pixels(rotated["rgbd"],relative,rotated["global_features"])
        error=float((patch-observed).abs().max());rotation_errors.append(error)
        assert error<2e-6,(turns,error)
    # Modality interventions must remove depth even from crop positioning.
    rgb_model=RGBDFiniteHorizonFDM(RGBDFDMConfig(arm="rgb_only",candidate_patches=True)).eval()
    poisoned=image.clone();poisoned[:,3]=torch.randn_like(poisoned[:,3])*.3
    with torch.inference_mode():
        rgb_first=rgb_model.candidate_patch_pixels(image,relative,global_features)
        rgb_second=rgb_model.candidate_patch_pixels(poisoned,relative,global_features)
        blank=model.candidate_patch_pixels(image,relative,global_features,control="blank")
    assert torch.equal(rgb_first,rgb_second)
    assert torch.count_nonzero(blank)==0
    # Twelve-second command sequence, physical rotations, and twist inversion.
    route={"waypoints":[[0.,0.],[200.,0.]],"stations":[0.,200.],"speeds":[4.,4.],"headings":[0.,0.]}
    command=build_command_features(route,[0.,0.,0.])
    assert command["commands"].shape==(60,5)
    assert abs(float(command["commands"][-1,0])-48.)<1e-4
    assert np.max(np.abs(command["commands"][:,1]))==0.
    pose=torch.from_numpy(command["nominal_pose"])[None]
    roundtrip=integrate_body_twist(pose_to_body_twist(pose,.2),.2)
    twist_error=float((roundtrip-pose).abs().max());assert twist_error<1e-5
    batch={"rgbd":image,"history":torch.zeros(1,16,24),"commands":torch.from_numpy(command["commands"])[None],
           "global_features":torch.from_numpy(command["global_features"])[None],"nominal_pose":pose,
           "trajectory":pose.clone(),"work":torch.zeros(1,60,1),"events":torch.zeros(1,60,3),
           "trajectory_mask":torch.ones(1,60,1),"event_mask":torch.ones(1,60,3),
           "attitude":torch.zeros(1,60,2),"attitude_mask":torch.ones(1,60,2)}
    with torch.inference_mode():
        output=model(batch)
        poisoned_targets={**batch,"trajectory":torch.randn_like(pose)*1e4,"events":torch.ones_like(batch["events"]),
                          "attitude":torch.ones_like(batch["attitude"])*1e4}
        independent_output=model(poisoned_targets)
    assert all(torch.equal(output[key],independent_output[key]) for key in output)
    assert output["trajectory"].shape==(1,60,4) and output["attitude"].shape==(1,60,2)
    assert torch.all(output["work"][:,1:]>=output["work"][:,:-1]) and torch.all(output["work"]>=0.)
    loss_weights={"xy":1.,"yaw":.5,"work":.2,"events":1.,"attitude":.5}
    masked={**batch,"attitude_mask":torch.zeros_like(batch["attitude_mask"])}
    with torch.inference_mode():
        loss1,parts1=fdm_loss(model,output,masked,loss_weights)
        loss2,parts2=fdm_loss(model,output,{**masked,"attitude":torch.ones_like(batch["attitude"])*1e4},loss_weights)
    assert torch.equal(loss1,loss2) and parts1["attitude"]==0. and parts2["attitude"]==0.
    # Loading patch buffers was a historic non-contiguous-state_dict failure.
    with tempfile.TemporaryDirectory(prefix="fdm_diverse_cpu_check_") as temporary:
        checkpoint=Path(temporary)/"synthetic.pt"
        torch.save({"model_family":"diverse_rgbd_reference_gru","model_config":asdict(cfg),
                    "normalization":{},"model_state":model.state_dict()},checkpoint)
        loaded,_=load_rgbd_checkpoint(checkpoint)
        with torch.inference_mode():
            restored=loaded(batch)
        serialization_error=max(float((restored[key]-output[key]).abs().max()) for key in output)
        assert serialization_error==0.
    # A rollover attitude beyond90 degrees must be representable; angular loss
    # must treat +pi and -pi as neighboring orientations.
    attitude_probe=RGBDFiniteHorizonFDM(RGBDFDMConfig(horizon=2,candidate_patches=False)).eval()
    with torch.no_grad():
        attitude_probe.output_head.weight.zero_()
        attitude_probe.output_head.bias.zero_()
        attitude_probe.output_head.bias.reshape(2,9)[:,7]=math.atanh(.75)
    probe_batch={**batch,"commands":batch["commands"][:,:2],"nominal_pose":batch["nominal_pose"][:,:2]}
    with torch.inference_mode():
        probe_output=attitude_probe(probe_batch)
    beyond_ninety=float(probe_output["attitude"][0,0,0])
    assert abs(beyond_ninety-3.*math.pi/4.)<1e-6
    wrap_target=torch.zeros_like(batch["attitude"]);wrap_target[...,0]=-math.pi+.01
    wrap_prediction=torch.zeros_like(batch["attitude"]);wrap_prediction[...,0]=math.pi-.01
    roll_mask=torch.zeros_like(batch["attitude_mask"]);roll_mask[...,0]=1.
    with torch.inference_mode():
        _,wrapped_parts=fdm_loss(model,{**output,"attitude":wrap_prediction},
            {**batch,"attitude":wrap_target,"attitude_mask":roll_mask},loss_weights)
    wrapped_loss=float(wrapped_parts["attitude"])
    assert abs(wrapped_loss-.0008)<2e-7,wrapped_loss
    # A shared static image must be shuffled by identity, not by window row.
    from traverse_fdm_rgbd_diverse_train import evaluate_rgbd
    arrays={key:np.repeat(value.numpy(),4,axis=0) for key,value in batch.items() if key!="rgbd"}
    class ImageFixture:
        def __init__(self,identities):
            self.index=np.asarray(identities)
            self.requested=[]
        def batch(self,rows):
            ids=self.index[rows];self.requested.extend(ids.tolist())
            return image.expand(len(rows),-1,-1,-1).clone()
    images=ImageFixture([0,0,1,1])
    shuffled,_=evaluate_rgbd(model,arrays,images,2,torch.device("cpu"),loss_weights,"shuffle")
    assert images.requested==[1,1,0,0],images.requested
    assert shuffled["shuffle"]["same_image_windows"]==0
    try:
        evaluate_rgbd(model,arrays,ImageFixture([0,0,0,0]),2,torch.device("cpu"),loss_weights,"shuffle")
    except ValueError as error:
        assert "fewer than two" in str(error)
    else:
        raise AssertionError("One-image validation shuffle must be unavailable")
    report={"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "reviewed_source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                ROOT/"src/nedm/traverse/fdm_diverse_model.py",ROOT/"src/nedm/traverse/fdm_diverse_data.py",ROOT/"scripts/traverse_fdm_rgbd_diverse_train.py")},
            "optimizer_created":False,"training_updates":0,"device":"cpu","passed":True,
            "physical_patch_projection_max_abs_error":projection_error,"quarter_turn_crop_max_abs_errors":rotation_errors,
            "nominal_twist_roundtrip_max_abs_error":twist_error,"checkpoint_prediction_max_abs_error":serialization_error,
            "roll_beyond90deg_prediction_deg":math.degrees(beyond_ninety),"wrapped_roll_loss":wrapped_loss,
            "shuffle_same_image_windows":shuffled["shuffle"]["same_image_windows"],
            "checks":["400m metric ray depth and40m elevation scale", "512px candidate perspective and left/forward axes",
                      "three quarter-turn coordinate augmentations", "RGB-only crop geometry excludes depth", "blank patch intervention",
                      "60×0.2s command horizon", "body-twist integration", "targets do not enter model inference",
                      "positive monotone work in declared units", "attitude mask excludes finite masked targets", "patch checkpoint round-trip",
                      "roll beyond90 degrees", "wrapped roll error around±pi", "shuffle deranges image identities",
                      "one-image shuffle declared unavailable"]}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
