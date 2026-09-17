"""Report existing frozen-model diagnostic arrays; never runs inference."""
from pathlib import Path
import json,hashlib,csv,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
audit=json.loads((ROOT/'audit.json').read_text());trials=audit['trials']
by_name={t['trial_id']:t for t in trials}
ordered=[by_name[f'diverse_v1_test_{scene}_00_selected_rgbd_{arm}'] for scene in ('rolling_hills','rough_mosaic','cross_slopes') for arm in ('time','energy')]
def short(t):return t['scene_id'].removeprefix('diverse_v1_test_').removesuffix('_00')+' / '+('time + work' if t['trial_id'].endswith('_energy') else 'time')
def fmt(x,d=2):return 'none' if x is None else f'{x:.{d}f}'
def secs(x):return '—' if x is None else f'{x:.2f} s'
def pct(x):return f'{100*x:.1f}%'
def p(label,path):return f'[{label}]({Path(path).resolve()})'
def similar_safe(t):
 return [x for x in t['launch_candidates'][1:] if x['nearest_previous_fixed_reference'] and x['nearest_previous_fixed_reference']['safe_goal'] and max(x['nearest_previous_fixed_reference']['reference_max_abs_differences'].values())<.01]
fig,axes=plt.subplots(3,2,figsize=(14,11),layout='constrained')
for ax,t in zip(axes.flat,ordered):
 rows=t['anchors'];ts=np.array([x['time_s'] for x in rows]);contact=np.array([x['final_reference_cost']['contact_probability'] for x in rows]);bounded=np.array([x['final_reference_cost']['low_progress_probability'] for x in rows])
 ax.plot(ts,contact,'o-',ms=3,lw=1.5,label='Predicted contact');ax.plot(ts,bounded,'s-',ms=3,lw=1.5,label='Predicted bounded motion (eligible)')
 ax.axhline(.35,color='C0',ls='--',lw=1,alpha=.7,label='Contact cap 0.35');ax.axhline(.5,color='C1',ls='--',lw=1,alpha=.7,label='Bounded-motion cap 0.50')
 tc=np.array([bool(x['truth_horizon_event'][0]) for x in rows]);tb=np.array([bool(x['truth_horizon_event'][2]) for x in rows])
 ax.scatter(ts[tc],np.full(tc.sum(),1.06),marker='|',s=70,color='darkgreen',label='Measured contact in next 12 s')
 ax.scatter(ts[tb],np.full(tb.sum(),1.13),marker='|',s=70,color='darkred',label='Measured bounded motion in next 12 s')
 if t['first_contact_s'] is not None:
  ax.axvline(t['first_contact_s'],color='black',lw=1.2,label='First measured contact')
  if t['first_contact_s']>=12:ax.axvline(t['first_contact_s']-12,color='gray',ls=':',lw=1.2,label='Contact minus 12 s')
 ax.set_ylim(-.03,1.19);ax.set_xlabel('Measured anchor time (s)');ax.set_ylabel('Probability');ax.set_title(short(t));ax.grid(alpha=.15)
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=9)
fig.suptitle('POST-HOC: unchanged executed reference, frozen FDM at causal measured anchors\nOnly the launch decision was deployed; later curves are diagnostic re-evaluations',fontsize=13)
fig.savefig(ROOT/'causal_risk.png',dpi=170);fig.savefig(ROOT/'causal_risk.pdf');plt.close(fig)
fig,axes=plt.subplots(3,2,figsize=(14,10),layout='constrained')
for ax,t in zip(axes.flat,ordered):
 rows=t['anchors'];ts=[x['time_s'] for x in rows]
 ax.plot(ts,[x['forecast_xy_fde_m'] for x in rows],'o-',ms=3,color='C3',label='FDM error at last measured forecast endpoint')
 ax.plot(ts,[x['forecast_xy_ade_m'] for x in rows],color='C1',label='Mean FDM position error across horizon')
 ax.plot(ts,[x['actual_future_tracking_max_m'] for x in rows],color='C0',label='Max actual distance from reference, next 12 s')
 if t['first_contact_s'] is not None:ax.axvline(t['first_contact_s'],color='black',ls='--',lw=1,label='First measured contact')
 ax.set_title(short(t));ax.set_xlabel('Measured anchor time (s)');ax.set_ylabel('Distance (m)');ax.grid(alpha=.15)
h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=2,fontsize=9)
fig.suptitle('POST-HOC: finite-horizon position prediction and physical reference tracking\nTracking distance is to the commanded waypoint polyline, not a reconstructed PID Bezier curve',fontsize=13)
fig.savefig(ROOT/'forecast_and_tracking_error.png',dpi=170);fig.savefig(ROOT/'forecast_and_tracking_error.pdf');plt.close(fig)
summary=[];candidate_table=[]
for t in ordered:
 launch=t['anchors'][0];final=launch['final_reference_cost'];parent=launch['parent_reference_cost'];ss=similar_safe(t);best=min(ss,key=lambda x:x['cost']['unfiltered_cost']) if ss else None
 first_reject=next((x for x in t['anchors'] if not x['final_reference_cost']['allowed_by_predicted_risk']),None)
 true_reject=next((x for x in t['anchors'] if not x['final_reference_cost']['allowed_by_predicted_risk'] and x['hazard_within_12s'] and not x['contact_already_seen']),None)
 falseaccept=[x for x in t['anchors'] if x['final_reference_cost']['allowed_by_predicted_risk'] and x['hazard_within_12s'] and not x['contact_already_seen']]
 row={'trial_id':t['trial_id'],'first_contact_s':t['first_contact_s'],'launch_contact_probability':final['contact_probability'],'launch_bounded_probability':final['low_progress_probability'],
  'launch_measured_contact_in_horizon':bool(launch['truth_horizon_event'][0]),'launch_fde_m':launch['forecast_xy_fde_m'],
  'first_audited_rejection_s':None if first_reject is None else first_reject['time_s'],'first_audited_rejection_with_horizon_hazard_s':None if true_reject is None else true_reject['time_s'],
  'precontact_false_accept_anchors_s':[x['time_s'] for x in falseaccept],
  'parent_cost':parent['cost'],'refined_cost':final['cost'],'refinement_cost_change':final['cost']-parent['cost'],
  'best_prior_safe_similar_sibling_index':None if best is None else best['family_index'],'best_prior_safe_similar_sibling_cost':None if best is None else best['cost']['unfiltered_cost'],
  'first_audited_rejection_is_before_contact_enters_horizon':bool(first_reject and t['first_contact_s'] and first_reject['time_s']+12.<t['first_contact_s']),
  'full_goal_safe':t['outcome']['schema_safe_goal_reached']}
 summary.append(row)
 for x in t['launch_candidates']:
  c=x['cost'];m=x['nearest_previous_fixed_reference'];meta=x['meta']
  candidate_table.append({'trial_id':t['trial_id'],'role':x['role'],'family_index':x['family_index'],'offset_m':meta.get('lateral_offset_m'),'cruise_mps':meta.get('cruise_speed_mps'),
   **{key:c[key] for key in ('allowed_by_predicted_risk','cost','unfiltered_cost','estimated_time_to_goal_s','predicted_horizon_work_kj','estimated_route_work_kj','energy_cost_s','contact_probability','low_progress_probability','rollover_probability','predicted_peak_roll_deg','predicted_peak_pitch_deg','attitude_cost_s','predicted_goal_progress_m')},
   'contact_cost_s':60*c['contact_probability'],'bounded_cost_s':60*c['low_progress_probability'],'rollover_cost_s':120*c['rollover_probability'],'progress_cost_s':-.1*c['predicted_goal_progress_m'],
   'prior_fixed_reference':None if m is None else m['family'],'prior_safe_goal':None if m is None else m['safe_goal'],'reference_arrays_exact':None if m is None else m['reference_arrays_exact'],
   'reference_waypoint_max_abs_difference_m':None if m is None else m['reference_max_abs_differences']['waypoints'],'prior_outcome_is_same_policy_counterfactual':False})
for name,rows in (('summary.csv',summary),('launch_cost_decomposition.csv',candidate_table)):
 with (ROOT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
report=['**Post-hoc diagnostic of frozen RGB-D FDM / MPPI decisions**','',
 'The four failed protected trials contain two different problems. Rolling-hills contact occurs about30 seconds after launch, beyond the12-second forecast; the initial finite forecast cannot establish safety of the full reference. Rough mosaic also has a genuine near-term false acceptance: the time arm assigns11.6% contact risk despite measured chassis contact11.70 seconds later. The energy arm accepts again at2 seconds while its later collision is inside the horizon.','',
 'Later evaluations of the unchanged reference often reject it before contact. These judgments were never executed in plan-once mode, and they do not prove that a particular replanning or stopping policy would succeed. The earliest rolling-hills rejection is actually too early for the measured contact label: its12-second future contains no contact or bounded-motion event.','',
 '| Scene / arm | First contact | Launch contact / bounded probability | Contact within launch12s | First audited rejection | First audited rejection with actual hazard in12s |','|---|---:|---:|---|---:|---:|']
for t,r in zip(ordered,summary):report.append(f"| {short(t)} | {secs(r['first_contact_s'])} | {pct(r['launch_contact_probability'])} / {pct(r['launch_bounded_probability'])} | {r['launch_measured_contact_in_horizon']} | {secs(r['first_audited_rejection_s'])} | {secs(r['first_audited_rejection_with_horizon_hazard_s'])} |")
report += ['',p('Causal risk curves',ROOT/'causal_risk.png')+' · '+p('Position and reference-tracking errors',ROOT/'forecast_and_tracking_error.png')+' · '+p('Complete numerical audit',ROOT/'audit.json'),'',
 'The rough-mosaic energy arm has a transient rejection at0.50 seconds, then accepts at2 seconds and rejects again later. A single early rejection is therefore not a monotonic or calibrated warning. Successful cross-slope controls remain accepted at every audited anchor through32 seconds. All supported risk heads are present. The deployed caps are contact0.35 and bounded motion0.50; rollout/attitude masks and limits come from each frozen protocol.','',
 '**What MPPI changed.** Final weighted means are re-scored through the same frozen eligibility gate; these failures are not caused by an unchecked mean. The reported final-reference forecasts are distinct from saved pre-refinement family scores.','',
 '| Scene / arm | Parent model cost | Refined model cost | Best sibling with a similar previously safe fixed route | Maximum waypoint / speed change |','|---|---:|---:|---:|---:|']
for t,r in zip(ordered,summary):report.append(f"| {short(t)} | {r['parent_cost']:.3f} | {r['refined_cost']:.3f} | {fmt(r['best_prior_safe_similar_sibling_cost'],3)} | {t['final_max_lateral_vs_parent_m']:.3f}m / {t['final_max_speed_vs_parent_mps']:.3f}m/s |")
report += ['',
 'Matched physical replays now establish one avoidable ranking failure in rough-mosaic/time. The exact original best unrefined family12 (−44m at6m/s; model cost47.608) safely reaches the goal in41.30s. MPPI instead refines parent family6 (cost48.721) into a route with lower predicted cost46.475, making it the winner; that route contacts within12s, blocks and times out at180s. The best-unrefined intervention was selected by the original model-cost argmin, before its physical outcome was known. This is a harmful ranking change in this case, not evidence that removing MPPI improves general performance.','',
 'All nine controlled trials passed artifact/source/runtime/input checks, and all four selected-reference controls matched every one of241 stored physical arrays exactly. All four selected parent references were already unsafe: the rough-mosaic parent reaches the goal at120.25s after contact and blockage, while its refinements time out; rolling-hills parents and refinements both time out after contact/blockage. Thus refinement worsens the rough-mosaic outcome but does not create its initial unsafe classification, and small reference changes are not necessary for every failure. These interventions change the executed reference, including geometry and speed; they do not isolate lateral deformation alone.','',
 p('Verified physical counterfactual report',ROOT.parent/'refinement_summary/report.md')+' · '+p('Counterfactual cost and route figure',ROOT.parent/'refinement_summary/ranking_counterfactual.png')+' · '+p('Physical checks and results',ROOT.parent/'refinement_summary/report.json'),'',
 'Previously collected safe routes exist at both±44m on rolling hills,−44m on rough mosaic, and+44m on cross slopes. They are contextual feasibility evidence: their waypoints differ from the actual online siblings by roughly millimeters, and their controller/collector provenance differs. No existing sibling outcome is labeled an exact counterfactual for a later measured state. The audit retains exact array comparisons, launch differences, prior source hashes and a separate2m/15-degree proximity flag for conditional sibling queries.','',
 p('Every launch candidate and cost component',ROOT/'launch_cost_decomposition.csv')+' · '+p('Compact table',ROOT/'summary.csv'),'',
 '**Reconstruction and limits.** We use the unchanged protected LAST5000 checkpoint and all archived online_v9 source bytes. Frame-zero measured pose/state and reconstructed history match the recorded decision exactly; replayed saved base-family scores agree within0.001 (the observed differences are recorded). CPU runtime versions are pinned in provenance. The decision re-evaluation performs no training, model selection or physics stepping. The separately linked counterfactuals add controlled physical measurements without changing the original protected evaluation.','',
 'Inputs are the same single pre-drive RGB-D snapshot, measured pose and causal17-state/action history, supplied goal, and frozen references. Outputs are compared only with actual future measurements along the unchanged plan-once reference. Every terminal/censored mask is retained. Bounded-motion supervision requires a full2-second window, so outputs before2 seconds are ineligible. The exact frozen scorer also removes predicted post-goal parking. Raw early probabilities are saved only as diagnostics.','',
 'Anchor selection is post-hoc:0,2,4,… seconds, measured hazard-minus12s and onset boundaries±0.05s, then extra measured frames around the first audited accept/reject transition. It is not an exhaustive20Hz alarm scan; “first audited” must not be read as the earliest possible crossing. Future outcomes select diagnostic sample times and labels, but never enter model features. Later sibling queries are conditional predictions, not new physical counterfactuals.','',
 'Native-spline inspection isolates interpolation from actual vehicle deviation. Exact frozen3-D spline knots and native ChBezierCurve were reconstructed in memory with no vehicle or simulation step. Rough-mosaic/time stays within0.079m of this curve before contact (0.005m at contact), whereas rough-mosaic/energy reaches0.902m at firstcontact. Spline-to-command-polyline differences are only0.0043m and0.0029m. The planner uses a1.3m footprint half-width; small centerline tracking error does not establish footprint clearance. Rolling-hills trajectories deviate about4.38m/4.16m before returning within0.034m of the reference at contact. Cross controls stay within0.168m. The native tracking measurements are separate from decoder endpoint error, which can be tens of meters.', '', p('Native spline and tracking audit',ROOT/'native_curve_audit.json')+' · '+p('Read-only native geometry source',ROOT/'audit_native_curve.py'), '',
 'Each trial folder contains anchor_records.json, launch_candidates.json, summary.json and causal_forecasts.npz. The NPZ retains all forecast/target masks and selected commands, nominal_pose and global_features for independent perception/support auditing. Forecast errors can grow well before contact while physical reference tracking remains comparatively close; the plotted waypoint-polyline distance is not a full tire-terrain or PID-path diagnosis.','',
 p('Reconstruction source',ROOT/'audit_decisions.py')+' · '+p('Reporting source',ROOT/'report_decisions.py')+' · '+p('Source, checkpoint and runtime provenance',ROOT/'provenance.json')+' · '+p('Downloaded raw artifact hashes',ROOT/'inputs/download_manifest.json')]
def readable(line):
 parts=re.split(r'(\[[^\]]+\]\([^)]+\))',line)
 for i in range(0,len(parts),2):
  text=parts[i]
  text=re.sub(r'\b([A-Za-z]+)(?=\d)',r'\1 ',text)
  text=re.sub(r'(?<=\d)(m/s|m|s)\b',r' \1',text)
  text=text.replace('firstcontact','first contact').replace('the−','the −').replace('both±','both ±').replace('separate2','separate 2').replace('post-hoc:0,2,4,…','post-hoc: 0, 2, 4, …')
  parts[i]=text
 return ''.join(parts)
(ROOT/'report.md').write_text('\n'.join(map(readable,report))+'\n')
manifest={'scope':'POST-HOC DIAGNOSTIC','physical_counterfactual_sha256':{name:hashlib.sha256((ROOT.parent/'refinement_summary'/name).read_bytes()).hexdigest() for name in ('report.md','report.json','ranking_counterfactual.png')},'inference_provenance_sha256':hashlib.sha256((ROOT/'provenance.json').read_bytes()).hexdigest(),'report_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'artifacts':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/name for name in ('report.md','audit.json','causal_risk.png','causal_risk.pdf','forecast_and_tracking_error.png','forecast_and_tracking_error.pdf','summary.csv','launch_cost_decomposition.csv','native_curve_audit.json','audit_native_curve.py')]}}
(ROOT/'report_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'trials':len(ordered),'anchors':sum(t['audited_anchors'] for t in ordered),'report':str(ROOT/'report.md')},indent=2))
