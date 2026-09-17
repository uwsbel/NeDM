#!/usr/bin/env python3
"""CPU contracts for RGB-D route scoring; no optimizer updates or simulation."""
from pathlib import Path
import json
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nedm.traverse.fdm_rgbd_data import build_command_features
from nedm.traverse.fdm_rgbd_model import RGBDFDMConfig, RGBDFiniteHorizonFDM
from nedm.traverse.fdm_rgbd_planner import (RGBDCostConfig, RGBDReferenceScorer, check_reference_contract,
    plan_rgbd_routes, propose_route_families)
from nedm.traverse.fdm_mppi import MPPIConfig, validate_reference


def main():
    torch.set_num_threads(2)
    torch.manual_seed(19)
    model = RGBDFiniteHorizonFDM(RGBDFDMConfig(cnn_channels=4, visual_hidden=8,
        history_hidden=8, command_hidden=8, forward_hidden=8, decoder_hidden=16)).eval()
    rng = np.random.default_rng(19)
    image = rng.random((4,128,128), dtype=np.float32)
    history = rng.random((16,24), dtype=np.float32)
    anchor, goal = np.array([0.,-15.,np.pi/2]), np.array([0.,15.])
    routes = propose_route_families(anchor, goal, speeds=(2.,4.), offsets=(0.,-4.,4.))
    cfg = MPPIConfig(samples=8, iterations=1)
    assert len(routes) == 6
    assert all(validate_reference(r, [], cfg, anchor)["valid"] for r in routes)
    assert len({r["waypoints"].tobytes() for r in routes}) == 3
    reversed_route = propose_route_families([0.,0.,np.pi],[30.,0.],speeds=(4.,),offsets=(0.,))[0]
    inconsistent = {**routes[0], "stations":routes[0]["stations"]*10.}
    for invalid in (reversed_route, inconsistent):
        try:
            check_reference_contract(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed/cusped reference accepted")
    scorer = RGBDReferenceScorer(model,image,history,anchor,goal,batch_size=2)
    prediction = scorer.predict(routes)
    features = [build_command_features(r,anchor,station=0.) for r in routes]
    batch = {k:torch.from_numpy(np.stack([f[k] for f in features]))
             for k in ("commands","global_features","nominal_pose")}
    batch.update(rgbd=torch.from_numpy(image)[None].expand(6,-1,-1,-1),
                 history=torch.from_numpy(history)[None].expand(6,-1,-1))
    with torch.no_grad():
        direct = model(batch)
    np.testing.assert_allclose(prediction["trajectory"],direct["trajectory"].numpy(),atol=1e-5,rtol=1e-5)
    np.testing.assert_allclose(prediction["event_probability"],torch.sigmoid(direct["event_logits"]).numpy(),atol=1e-6)
    patch_model = RGBDFiniteHorizonFDM(RGBDFDMConfig(cnn_channels=4, visual_hidden=8,
        history_hidden=8, command_hidden=8, forward_hidden=8, decoder_hidden=16,
        candidate_patches=True, patch_hidden=8, progress_event_definition="sustained_stall")).eval()
    patch_scorer = RGBDReferenceScorer(patch_model,image,history,anchor,goal,batch_size=2)
    patch_cached = patch_scorer.predict(routes)
    with torch.no_grad():
        patch_direct = patch_model(batch)
    np.testing.assert_allclose(patch_cached["trajectory"],patch_direct["trajectory"].numpy(),atol=1e-5,rtol=1e-5)
    np.testing.assert_allclose(patch_cached["event_probability"],torch.sigmoid(patch_direct["event_logits"]).numpy(),atol=1e-6)
    assert patch_scorer.progress_event_definition == "sustained_stall"

    # Higher goal progress improves the time/progress objective, but high
    # collision or eligible low-progress risk can outweigh that advantage.
    xy = np.zeros((2,20,4)); xy[:,:,3] = 1.
    xy[0,:,0] = np.arange(1,21)*.2*2.
    xy[1,:,0] = np.arange(1,21)*.2*4.
    output = {"trajectory":xy,"event_probability":np.zeros((2,20,3)),"work":np.zeros((2,20,1))}
    safe = scorer.cost_breakdown(output)
    assert safe["cost"][1] < safe["cost"][0]
    output["event_probability"][1,0,0] = .9
    risky = scorer.cost_breakdown(output)
    assert risky["cost"][1] > risky["cost"][0]
    output["event_probability"].fill(0.)
    output["event_probability"][1,:9,2] = 1.
    np.testing.assert_array_equal(scorer.cost_breakdown(output)["cost"],safe["cost"])
    output["event_probability"][1,9,2] = .9
    assert scorer.cost_breakdown(output)["cost"][1] > safe["cost"][0]
    near_goal = RGBDReferenceScorer(model,image,history,anchor,np.array([0.,-10.]))
    output["event_probability"].fill(0.)
    arrived = near_goal.cost_breakdown(output)
    output["event_probability"][:,9:,2] = 1.
    np.testing.assert_array_equal(near_goal.cost_breakdown(output)["cost"],arrived["cost"])
    boundary_goal = RGBDReferenceScorer(model,image,history,anchor,np.array([0.,-8.]))
    output["event_probability"].fill(0.)
    boundary_cost = boundary_goal.cost_breakdown(output)["cost"].copy()
    output["event_probability"][0,9,2] = 1.  # Arrival at t=2.0 s for the 2 m/s reference.
    np.testing.assert_array_equal(boundary_goal.cost_breakdown(output)["cost"],boundary_cost)

    permissive = RGBDCostConfig(refine_families=2,max_contact_probability=1.,max_low_progress_probability=1.)
    result = plan_rgbd_routes(model,image,history,anchor,goal,families=routes,
                              mppi_config=cfg,cost_config=permissive)
    assert not result["abstained"] and result["route"] is not None
    assert len(result["refinements"]) == 2
    assert validate_reference(result["route"],[],cfg,anchor)["valid"]
    again = plan_rgbd_routes(model,image,history,anchor,goal,families=routes,
                              mppi_config=cfg,cost_config=permissive)
    assert result["cost"] == again["cost"]
    np.testing.assert_array_equal(result["route"]["waypoints"],again["route"]["waypoints"])
    blocked = plan_rgbd_routes(model,image,history,anchor,goal,families=routes,
        mppi_config=cfg,cost_config=RGBDCostConfig(max_contact_probability=0.))
    assert blocked["abstained"] and blocked["route"] is None
    print(json.dumps({"training_performed":False,"cached_vs_direct_inference":True,
        "multiple_kinematic_route_families":True,"time_progress_risk_tradeoff":True,
        "unsupervised_early_low_progress_excluded":True,"seeded_mppi_deterministic":True,
        "all_high_risk_abstention":True,"post_arrival_parking_excluded":True,
        "cusp_and_station_consistency":True,"candidate_patch_cached_parity":True,
        "authored_obstacles_or_terrain_inputs":False,
        "model_evaluations":result["model_evaluations"]},indent=2))


if __name__ == "__main__":
    main()
