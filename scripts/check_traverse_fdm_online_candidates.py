#!/usr/bin/env python3
"""Independent causal candidate-retention tests, without simulation or training."""
import hashlib
import copy
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nedm.traverse.fdm_online_candidates import online_route_families, locate_reference, reference_fingerprint, POLICY
from nedm.traverse.fdm_diverse_planner import propose_route_families, check_reference_contract, RGBDReferenceScorer
from nedm.traverse.fdm_diverse_data import build_command_features


def line(y=2.):
    xy=np.column_stack((np.linspace(-100.,100.,401),np.full(401,y)))
    return {'waypoints':xy,'stations':np.arange(401)*.5,'speeds':np.full(401,4.),'headings':np.zeros(401),'meta':{}}


def main():
    original=propose_route_families([-100.,0.,0.],[100.,0.]);active=original[13]
    frozen={k:np.asarray(active[k]).copy() for k in ('waypoints','stations','speeds','headings')}
    index=120;pose=np.r_[active['waypoints'][index],active['headings'][index]]
    retained=online_route_families(pose,[100.,0.],active,original)
    assert retained[0]['meta']['candidate_origin']=='active_reference'
    for key,value in frozen.items():
        assert np.array_equal(retained[0][key],value) and np.array_equal(active[key],value)
    assert retained[0]['meta']['fdm_station']==active['stations'][index]
    assert retained[0]['meta']['current_reference_distance_m']==0.
    fingerprints=[reference_fingerprint(r) for r in retained]
    assert len(fingerprints)==len(set(fingerprints))
    expected=build_command_features(active,pose,station=active['stations'][index],elapsed_s=10.)
    actual=build_command_features(retained[0],pose,station=retained[0]['meta']['fdm_station'],elapsed_s=10.)
    assert all(np.array_equal(expected[k],actual[k]) for k in expected)
    # A no-motion replan preserves exact geometry/speed identity and station.
    repeated=online_route_families(pose,[100.,0.],retained[0],original)
    assert reference_fingerprint(repeated[0])==fingerprints[0]
    assert repeated[0]['meta']['fdm_station']==retained[0]['meta']['fdm_station']
    # Boundaries use nearest discrete waypoint and wrapped heading differences.
    goal=[100.,2.];reference=line();active_line=propose_route_families([0.,0.,0.],goal,speeds=(4.,),offsets=(0.,))[0]
    near=online_route_families([0.,0.,0.],goal,active_line,[reference],speeds=(4.,),offsets=(0.,))
    assert any(r['meta']['candidate_origin']=='original_family_00' for r in near)
    far=online_route_families([0.,-.00001,0.],goal,active_line,[reference],speeds=(4.,),offsets=(0.,))
    assert not any(r['meta']['candidate_origin']=='original_family_00' for r in far)
    for degrees,allowed in ((14.999,True),(15.001,False),(360.001,True)):
        rows=online_route_families([0.,0.,np.deg2rad(degrees)],goal,active_line,[reference],speeds=(4.,),offsets=(0.,))
        assert any(r['meta']['candidate_origin']=='original_family_00' for r in rows)==allowed
    # Retained active route is not silently dropped by the nearby-family rule.
    off=online_route_families([0.,-10.,0.],goal,reference,[],speeds=(4.,),offsets=(0.,))
    assert off[0]['meta']['candidate_origin']=='active_reference' and off[0]['meta']['current_reference_distance_m']>2.
    # Future truth-like metadata never changes the model's command inputs.
    poisoned={**active,'meta':{'future_contact':1,'terrain_cost':1e10,'fdm_station':-999.}}
    alternate=locate_reference(poisoned,pose,'active_reference')
    inputs=build_command_features(alternate,pose,station=alternate['meta']['fdm_station'],elapsed_s=10.)
    assert all(np.array_equal(expected[k],inputs[k]) for k in expected)
    # Actual failure trigger from grid412098: the vehicle points away from the
    # goal, so the straight fresh Hermite family folds through a cusp. Preserve
    # the valid active route and reject only the three affected speed variants.
    reverse_pose=np.array([-100.,0.,np.pi]);reverse_goal=np.array([100.,0.])
    straight=propose_route_families([-100.,0.,0.],reverse_goal,speeds=(4.,),offsets=(0.,))[0]
    straight_before=copy.deepcopy(straight)
    rejected=[]
    reverse_families=online_route_families(reverse_pose,reverse_goal,straight,[],rejected=rejected)
    assert len(reverse_families)==13 and len(rejected)==3
    assert [row['candidate_origin'] for row in rejected]==[f'fresh_family_{i:02d}' for i in range(3)]
    assert all(row['reason']=='reference_contract' and 'cusp' in row['detail'] for row in rejected)
    assert all(len(row['reference_sha256'])==64 for row in rejected)
    json.dumps(rejected,allow_nan=False)
    assert reverse_families[0]['meta']['candidate_origin']=='active_reference'
    context=object.__new__(RGBDReferenceScorer)
    context.anchor_pose=reverse_pose;context.goal_xy=reverse_goal
    for candidate in reverse_families:
        check_reference_contract(candidate)
        context.check_context(candidate)
    again_rejected=[]
    again=online_route_families(reverse_pose,reverse_goal,straight,[],rejected=again_rejected)
    assert rejected==again_rejected
    for first,second in zip(reverse_families,again,strict=True):
        assert first['meta']==second['meta']
        for key in ('waypoints','stations','speeds','headings'):
            np.testing.assert_array_equal(first[key],second[key])
            np.testing.assert_array_equal(straight[key],straight_before[key])
    # The executed route and frozen original family are strict inputs. Their
    # corruption must not become an apparently successful proposal rejection.
    malformed=[]
    for label,change in (
        ('speed_shape',lambda route: route.update(speeds=route['speeds'][:-1])),
        ('speed_nan',lambda route: route['speeds'].__setitem__(0,np.nan)),
        ('speed_negative',lambda route: route['speeds'].__setitem__(0,-.1)),
        ('speed_above_support',lambda route: route['speeds'].__setitem__(0,6.1)),
        ('station_corruption',lambda route: route['stations'].__setitem__(1,0.)),
        ('zero_length_segment',lambda route: route['waypoints'].__setitem__(1,route['waypoints'][0])),
    ):
        bad=copy.deepcopy(straight);change(bad)
        for origin,active_arg,original_arg in (('active',bad,[]),('original',straight,[bad])):
            try:
                online_route_families(reverse_pose,reverse_goal,active_arg,original_arg,rejected=[])
            except ValueError:
                malformed.append(origin+'/'+label)
            else:
                raise AssertionError(f'Corrupt {origin}/{label} was silently accepted or skipped')
    try:
        online_route_families(reverse_pose,[101.,0.],straight,[],rejected=[])
    except ValueError as error:
        assert 'goal' in str(error)
    else:
        raise AssertionError('Wrong-goal active reference was not rejected strictly')
    report={'passed':True,'policy':POLICY,'native_reference_preserved':True,'input_routes_unmodified':True,
            'active_station_and_command_features_exact':True,'exact_geometry_speed_duplicates_removed':True,
            'nearby_distance_heading_boundaries_and_angle_wrap_verified':True,
            'active_route_retained_even_when_far_with_model_risk_check_still_required':True,
            'future_metadata_excluded_from_model_commands':True,
            'reversed_heading_regression':{'pose':reverse_pose.tolist(),'goal':reverse_goal.tolist(),
                'accepted':len(reverse_families),'rejected':rejected,'all_accepted_geometry_speed_goal_station_contracts_pass':True,
                'replay_arrays_metadata_and_rejections_exact':True,'input_reference_unmodified':True},
            'corrupt_active_and_frozen_original_raise':malformed,
            'wrong_goal_active_raises':True,
            'source_sha256':hashlib.sha256((ROOT/'src/nedm/traverse/fdm_online_candidates.py').read_bytes()).hexdigest()}
    if len(sys.argv)>1:Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
