"""Assemble OUT/sysid_v8.json from the part results (table.json, timing.json, lag_fit.json, appearance.json,
servo_check.json, roll.json). The recommended training ranges are written here (rules in the comments).

    PYTHONPATH=<wt>/src ~/miniconda3/envs/so101_vision/bin/python make_json.py
"""
from __future__ import annotations

import json

import numpy as np

import sysid_lib as L


def ld(name):
    return json.loads((L.OUT / name).read_text())


FLOOR_M = 0.0015                # joint_env.FLOOR: lowest commanded gripper point above the table (sim frame)
ANNULUS = dict(r=(0.15, 0.27), ang_deg=75.0)     # task workspace (task.py: r about the pan axis, angle +-75 deg)


def annulus_plane_mm(coef, n=121):
    """Calibrated plane (mm) over the task annulus about the pan axis -> (min, max)."""
    r = np.linspace(*ANNULUS["r"], n)
    a = np.radians(np.linspace(-ANNULUS["ang_deg"], ANNULUS["ang_deg"], n))
    R, A = np.meshgrid(r, a)
    x, y = L.PAN_XY[0] + R * np.cos(A), L.PAN_XY[1] + R * np.sin(A)
    z = (coef[0] + coef[1] * x + coef[2] * y) * 1e3
    return float(z.min()), float(z.max())


def lag_mixture(fracs, k_max=3, p_min=0.05):
    """Equal-weight mixture of per-query whole-step lag distributions (dicts 'k' -> share; >= k_max pooled into
    k_max), each of 0 .. k_max at least p_min, renormalised, rounded to 0.01 (largest remainder, sum 1)."""
    m = np.zeros(k_max + 1)
    for f in fracs:
        for k, v in f.items():
            m[min(int(k), k_max)] += v / len(fracs)
    m = m / m.sum()
    low = np.zeros(len(m), bool)
    while True:                                   # raise small shares to p_min, scale the others to fill the rest
        low |= m < p_min - 1e-12
        m = np.where(low, p_min, m * (1.0 - p_min * low.sum()) / m[~low].sum())
        if not (m[~low] < p_min - 1e-12).any():
            break
    u = np.floor(m * 100).astype(int)
    for i in np.argsort(-(m * 100 - u))[:100 - u.sum()]:
        u[i] += 1
    return {str(k): float(u[k] / 100) for k in range(k_max + 1)}


def main():
    T, TM, LG, AP, SV, RO = (ld(n) for n in ("table.json", "timing.json", "lag_fit.json", "appearance.json",
                                             "servo_check.json", "roll.json"))
    cal = L.calib()
    tp = cal["scene"]["table_plane"]
    out = dict(version="sysid_v8", date="2026-10-23",
               frames="sim frame = arm base frame of calib_v1 (joint map + FK), z up; sim table / mat top z = 0",
               data=dict(robot_episodes=T["n_robot_episodes"], robot_steps=TM["n_steps"], robot_queries=TM["n_queries"],
                         robot_minutes=round(sum(r["T_s"] for r in TM["per_episode"]) / 60, 1), sessions_with_first_frames=len(AP["first_frames"]),
                         teleop_episodes=T["teleop"]["n"], teleop_frames=T["floor_edge"]["teleop"]["n"],
                         robot_logs=["~/so101_r2s_claude/robot_logs/{eval_*/ep_*,live_*}",
                                     "~/so101_r2s_claude/robot_logs_lab/so101_hw_*/eval_*/ep_*"],
                         teleop="~/so101_r2s_claude/raw (LeRobot v3, 30 fps)"))
    # ---- (1) table / mat ------------------------------------------------------------------------------------------
    fe = T["floor_edge"]
    out["table_plane"] = dict(
        meaning="real mat top (the surface the cube rests on) z = c0 + c1 x + c2 y (m, sim frame)",
        coef=tp["coef"], tilt_deg=tp["tilt_deg"], source="calib_v1 scene.table_plane (teleop bundle adjustment)",
        verdict=f"supported within about +-3 mm where the gripper reached the mat (robot: "
                f"{fe['robot_near_plane']['n_episodes']} episodes at single positions, near_plane; teleop: all frames, "
                f"lowest {fe['teleop']['min']:.1f} mm); the image check agrees in the median only; the robot data cannot "
                "test the tilt",
        checks=dict(
            robot_floor_edge=dict(note="measured lowest gripper point - plane, every robot step (mm). The steps within "
                                       "+-3 mm come from the few episodes in near_plane only (one position each; the "
                                       "floor rule kept the gripper higher elsewhere): local support of the plane there, "
                                       "not a check of the whole plane", n=fe["robot"]["n"],
                                  min=fe["robot"]["min"], p0_01=fe["robot"]["p0_01"], n_below_0=fe["robot"]["n_below_0"],
                                  n_below_minus2=fe["robot"]["n_below_m2"], n_within_3=fe["robot"]["n_within_3"],
                                  near_plane={k: v for k, v in fe["robot_near_plane"].items() if k != "note"}),
            teleop_floor_edge=dict(note="same, every teleop frame (no floor rule)", n=fe["teleop"]["n"],
                                   min=fe["teleop"]["min"], p0_1=fe["teleop"]["p0_1"], n_below_0=fe["teleop"]["n_below_0"],
                                   n_below_minus2=fe["teleop"]["n_below_m2"]),
            teleop_lowest_point_near_grasp_minus_plane_mm=T["teleop"]["low_grasp_min_minus_plane_mm"],
            robot_image_cube_height=dict(
                note="CONSISTENCY CHECK OF THE MEDIAN ONLY, not independent: the external camera pose was fitted "
                     "together with the plane (cube centre assumed at plane + 15 mm). External-camera ray through the "
                     "resting cube's blob, at the FK cube xy of the hold; height - 15 mm - plane (mm); per hold spread "
                     "p5-p95 about -10 .. +12 mm", minus_plane_mm_session_turn_corrected=T["holds"]["img_rotcorr_minus_plane_minus15_mm"],
                minus_plane_mm=T["holds"]["img_minus_plane_minus15_mm"], plane_fit=T["robot_img_minus15"]),
            robot_hold_fk_upper_bound=dict(
                note="grasp point at the hold start - 15 mm: an UPPER bound of the mat (the floor rule keeps the jaw "
                     "high, the jaw holds the cube above its centre)", minus_plane_mm=T["robot_gp_minus15"]["minus_calib_plane_mm"],
                plane_fit=T["robot_gp_minus15"])),
        at_cube_positions_mm=T["plane_at_holds_mm"], at_plate_positions_mm=T["plane_at_plates_mm"],
        at_cubes_and_plates_mm=T["plane_at_cubes_and_plates_mm"],
        cube_xy=dict(x=T["holds"]["cube_x"], y=T["holds"]["cube_y"]), plate_xy=T["plate_xy"],
        sim="flat, z = 0 (NRD worlds and training renders); geometry.table_plane null in scenes v2 / v3")
    # range: the plane over every observed cube / plate position (p5-p95 -15.7 .. -10.5 mm, min / max -17.5 / -8.3)
    # widened by the calibration tolerance (+-3 mm, floor-edge checks) -> U(-21, -5) mm, centre -12.8 mm
    a = T["plane_at_cubes_and_plates_mm"]
    lo = float(np.floor(a["min"] - 3.0))
    hi = float(np.ceil(a["max"] + 3.0))
    coef = np.asarray(tp["coef"], np.float64)
    an_lo, an_hi = annulus_plane_mm(coef)
    tz_const = float(np.ceil(a["max"])) / 1e3               # plane max over the observed positions, rounded up (m)
    clear = dict(at_observed_min_mm=FLOOR_M * 1e3 + tz_const * 1e3 - a["max"],
                 at_observed_p50_mm=FLOOR_M * 1e3 + tz_const * 1e3 - a["p50"],
                 at_observed_max_mm=FLOOR_M * 1e3 + tz_const * 1e3 - a["min"])
    out["table_dz_range_m"] = dict(
        center=round(a["p50"] / 1e3, 4), lo=lo / 1e3, hi=hi / 1e3, nominal_spread_p5_p95=[a["p5"] / 1e3, a["p95"] / 1e3],
        rule="per world: mat top = sim z 0 + dz (constant), dz ~ U(lo, hi); lo / hi = min / max of the measured plane "
             "over all 184 observed cube and plate positions -+ 3 mm calibration tolerance",
        plane_option=dict(note="if the training table can tilt: c0 = coef[0] +- 0.004, c1 = coef[1] +- 0.015, "
                               "c2 = coef[2] +- 0.015 (uniform)", coef=tp["coef"], c0_pm=0.004, c1_pm=0.015, c2_pm=0.015),
        coupling_rule=(
            "the action-layer floor must move with the mat, in training AND on the robot, else the lower table has no "
            "effect. Training: floor = FLOOR + dz in each world (joint_env.floor_guard floor=FLOOR + dz, or plane "
            "(dz, 0, 0)), so the floor stays 1.5 mm above that world's mat. Robot: the floor is FLOOR + table_z "
            "(hw_lib.ActionLayer, run_student_real --table-z); with table_z = 0 (111 of 115 episodes) it holds the "
            "commanded lowest point 10-19 mm above the real mat, and a student trained to go ~13 mm lower is clipped "
            "there. The robot guards hard_floor_m / cmd_floor_m are relative to table_z and move with it. The floor "
            "the teacher and student train with must equal the robot's (same clearance above the mat)."),
        deployment=dict(
            table_z_constant_m=tz_const,
            table_z_constant_rule="plane maximum over the observed cube and plate positions, rounded up to whole mm "
                                  "(toward the sim table): a constant floor never below the mat where the cubes and "
                                  "plates were",
            floor_clearance_above_real_mat_mm=clear,
            plane_on_task_annulus_mm=dict(min=an_lo, max=an_hi, note="calibrated plane over the task workspace "
                                          "(r 0.15-0.27 m about the pan axis, +-75 deg): at the +y edge the mat is "
                                          f"higher than at any observed position; there the constant floor would be "
                                          f"{an_hi - tz_const * 1e3 - FLOOR_M * 1e3:.1f} mm below the mat (no cube or "
                                          "plate was seen there)"),
            unsafe=f"a constant table_z at the centre ({a['p50']:.1f} mm) puts the floor up to "
                   f"{a['max'] - a['p50']:.1f} mm below the mat at the highest observed position (plane "
                   f"{a['max']:.1f} mm) and up to {an_hi - a['p50']:.1f} mm on the task annulus (tilt 36 mm/m)",
            tilted_floor=dict(coef=tp["coef"], note="matched option: floor = FLOOR + plane(x, y), clearance 1.5 mm "
                              "everywhere (+- the 3 mm calibration tolerance). joint_env.floor_guard already takes "
                              "plane=; hw_lib.ActionLayer / Guard take only a constant table_z (code change needed)"),
            recommendation="robot: tilted floor from the plane coefficients when the code exists; until then "
                           f"table_z = {tz_const:.3f} m (clearance {clear['at_observed_min_mm']:.1f}-"
                           f"{clear['at_observed_max_mm']:.1f} mm, p50 {clear['at_observed_p50_mm']:.1f} mm above the "
                           "real mat, so some descent is still clipped at the low positions); training: floor FLOOR + dz"),
        sim_now=0.0)
    ho, fo, lh = T["held_cube_offset_mm"], T["floor_override"], T["holds"]["low_minus_plane_mm"]
    npl = T["floor_edge"]["robot_near_plane"]
    n_tz0 = sum(1 for e in L.list_episodes() if e["table_z"] == 0.0)
    out["mat"] = dict(
        cube_rest_surface="= table_plane (calib: cube centre 15 mm above it)",
        mat_thickness_m=dict(real="not resolved (calib assumes 0.001; the cube-rest plane is what matters)", sim=0.001),
        plate_thickness_m=dict(real=cal["scene"]["plate_thickness"], sim=0.001,
                               source="calib_v1 plate_top_z (teleop releases)"),
        plate_top_at_plates_mm=T["plate_top_at_plates_mm"],
        teleop_plate_top_calib_mm=T["teleop"]["plate_top_calib_mm"],
        held_cube_offset_mm=T["held_cube_offset_mm"],
        held_cube_note=f"the real jaw holds the cube {ho['p50']:.1f} mm (p5-p95 {ho['p5']:.1f}-{ho['p95']:.1f}) above its "
                       "centre at the hold start; sim teacher about 2.5 mm. Cause: the action-layer floor holds the "
                       "lowest COMMANDED gripper point at 1.5 mm + table_z (sim frame) unless the measured pose is "
                       "already lower (then the measured lowest point is the limit, and the command can follow it "
                       f"down: more than 2 mm below the floor in {fo['n_below_floor_minus_2mm']} of {fo['n_episodes']} "
                       f"episodes, more than 5 mm in {fo['n_below_floor_minus_5mm']}). At the hold start the measured "
                       f"lowest gripper point is {lh['p50']:.1f} mm (p5-p95 {lh['p5']:.1f}-{lh['p95']:.1f}) above the "
                       f"real mat; the gripper reached the mat (+-3 mm) in {npl['n_episodes']} episodes only "
                       f"(near_plane). table_z was 0 in {n_tz0} / {T['n_robot_episodes']} episodes. The offset changes "
                       "only if BOTH the training mat / floor (dz) and the robot floor (table_z or a tilted floor) are "
                       "changed (table_dz_range_m.coupling_rule).",
        floor_override=T["floor_override"],
        hold_low_minus_plane_mm=T["holds"]["low_minus_plane_mm"], n_holds=T["n_holds"])
    # ---- (2) image lag -----------------------------------------------------------------------------------------------
    tl, tl2 = LG["total_lag"], LG["total_lag_calib_minus_mean_teleop_age"]
    fc, idn = LG["free_cube_robot_wrist"], LG["identifiability"]
    ident_err = {k: v["error_ms"] for k, v in idn.items() if isinstance(v, dict) and "error_ms" in v}
    rng_steps = {cam: sorted([tl[cam]["central"]["lag_steps"]["p50"], tl2[cam]["lag_steps"]["p50"]])
                 for cam in ("wrist", "external")}
    fracs = [tl[c]["central"]["steps_rounded_frac"] for c in ("wrist", "external")] + \
            [tl2[c]["steps_rounded_frac"] for c in ("wrist", "external")]
    rec = lag_mixture(fracs)
    rec_mean = sum(int(k) * v for k, v in rec.items())
    out["image_lag_steps"] = dict(
        meaning="age of the image content relative to the joint reading of the same query, in 20 ms sim steps "
                "(sim renders the current state: 0)",
        frame_age_ms=TM["frame_age_ms"], grab_minus_joint_ms=TM["grab_minus_joint_ms"],
        plausible_range_p50_steps=dict(
            wrist=rng_steps["wrist"], external=rng_steps["external"],
            note="the two pipeline estimates below bound the plausible range; neither is preferred"),
        estimates=dict(
            direct_fit=dict(
                note="wrist camera, resting cube, cube position free per episode (lag_fit.py). WEAK on the robot: the "
                     "wrist cube moves slowly in the image (feature speed p50 below), the cost curve is flat "
                     "(cost_at_ms), and blocks / students disagree by more than their bootstrap ranges "
                     f"({min(v['tau_ms'] for v in list(LG['free_cube_by_block'].values()) + list(LG['free_cube_by_student'].values())):.0f} to "
                     f"{max(v['tau_ms'] for v in list(LG['free_cube_by_block'].values()) + list(LG['free_cube_by_student'].values())):.0f} ms: "
                     "an error the bootstrap does not see). On synthetic blobs with the logged motion it recovers "
                     f"tau within {min(ident_err.values()):+.1f} .. {max(ident_err.values()):+.1f} ms, also with a 1 deg "
                     "camera turn or joint offset (identifiability_error_ms), so the spread comes from something not "
                     "modelled. The teleop check only shows the method works with fast arm motion.",
                pipeline_ms_wrist=fc["tau_pipeline_ms"], boot_p5_p50_p95=fc["tau_boot_p5_p50_p95"],
                pipeline_ms_external=tl["external"]["central"]["pipeline_ms"],
                cost_at_ms=fc["cost_at_ms"], cost_rise_at_ms=fc["cost_rise_at_ms"],
                feature_speed_px_per_10ms=fc["feature_speed_px_per_10ms"],
                by_block_ms={k: [v["tau_ms"], v["tau_boot_p5_p50_p95"][0], v["tau_boot_p5_p50_p95"][2]]
                             for k, v in LG["free_cube_by_block"].items()},
                by_block_note="free-cube fit per time block (lag_fit.py): tau, bootstrap p5, p95 (ms)",
                block_spread=LG["free_cube_block_spread"],
                by_student_ms={k: [v["tau_ms"], v["tau_boot_p5_p50_p95"][0], v["tau_boot_p5_p50_p95"][2]]
                               for k, v in LG["free_cube_by_student"].items()},
                fast_samples=dict(tau_ms=LG["free_cube_fast_samples"]["tau_ms"],
                                  boot_p5_p50_p95=LG["free_cube_fast_samples"]["tau_boot_p5_p50_p95"],
                                  speed_threshold_px_per_10ms=LG["free_cube_fast_samples"]["speed_threshold_px_per_10ms"],
                                  tau_change_ms=LG["free_cube_fast_samples"]["tau_change_ms"]),
                frame_age_check=LG["frame_age_check"],
                identifiability_error_ms=ident_err,
                cube_shift_check_bias_removal=LG["bias_removal_cube_shift_check"],
                teleop_check_wrist_ms=LG["free_cube_teleop_wrist"]["tau_ms"],
                teleop_check_calib_ms=LG["free_cube_teleop_wrist"]["calib_time_offset_ms"],
                external="robot wrist + 5.2 ms (calib external - wrist offset); not fitted directly (the held cube "
                         "moves too slowly in the external view)",
                n_samples=fc["n_samples"], n_episodes=fc["n_episodes"],
                lag_ms=dict(wrist=tl["wrist"]["central"]["lag_ms"], external=tl["external"]["central"]["lag_ms"]),
                lag_steps=dict(wrist=tl["wrist"]["central"]["lag_steps"], external=tl["external"]["central"]["lag_steps"],
                               wrist_rounded_frac=tl["wrist"]["central"]["steps_rounded_frac"],
                               external_rounded_frac=tl["external"]["central"]["steps_rounded_frac"])),
            calib_minus_mean_teleop_age=dict(
                note="LeRobot timing (get_observation reads the motors first, then the non-blocking read_latest): the "
                     "calibration offset contains the teleop frame age at the read; mean 1/2 frame = 16.7 ms removed. "
                     "Exact only if the camera and the record loop drift in phase (unknown)",
                pipeline_ms=dict(wrist=tl2["wrist"]["pipeline_ms"], external=tl2["external"]["pipeline_ms"]),
                lag_ms=dict(wrist=tl2["wrist"]["lag_ms"], external=tl2["external"]["lag_ms"]),
                lag_steps=dict(wrist=tl2["wrist"]["lag_steps"], external=tl2["external"]["lag_steps"],
                               wrist_rounded_frac=tl2["wrist"]["steps_rounded_frac"],
                               external_rounded_frac=tl2["external"]["steps_rounded_frac"])),
            upper_bound_task_assumption=dict(
                note="calib offset (42-48 ms) + frame age: counts the teleop frame age twice; the teleop check "
                     "contradicts it", lag_steps=TM["image_lag"]["pipeline_calib"]["external"]["lag_steps"])),
        dr_recommendation=dict(
            steps_probs=rec, mean_steps=rec_mean, mean_ms=rec_mean * 20.0,
            rule="whole-step lag of the rendered image (render the state h steps before the query read; both cameras, "
                 "drawn per query): equal-weight mixture of the per-query rounded lags of the two estimates for both "
                 "cameras, each of 0-3 steps at least 0.05 (renormalised). Covers the plausible p50 range "
                 f"(wrist {rng_steps['wrist'][0]:.2f}-{rng_steps['wrist'][1]:.2f}, external "
                 f"{rng_steps['external'][0]:.2f}-{rng_steps['external'][1]:.2f} steps)"))
    # ---- (3) loop ---------------------------------------------------------------------------------------------------
    lp = TM["loop"]
    cps = lp["chunk_period_steps"]["pct"]["p50"]
    p2 = float(np.clip(cps - 6.0, 0.0, 1.0))
    hold_extra = {"1": round(1.0 - p2, 2), "2": round(p2, 2)}
    war = lp["target_timeline_ms"]["write_after_read"]
    out["loop_period_ms"] = dict(
        all_steps=lp["step_period_ms_all"], query_steps=lp["step_period_ms_query_steps"],
        other_steps=lp["step_period_ms_other_steps"], chunk_5_steps=lp["chunk_period_ms"], chunk_sim=lp["chunk_period_sim_ms"],
        rate_hz_per_episode=lp["rate_hz_per_episode"], overruns_per_query=lp["overruns_per_query"],
        frac_steps_over_25ms=lp["frac_steps_over_25ms"], hist=lp["period_hist_ms"], model_ms=lp["model_ms"],
        command_delay_ms=dict(query_steps=lp["cmd_delay_ms_query_steps"], other_steps=lp["cmd_delay_ms_other_steps"],
                              note="joint read -> goal write; sim: 0"),
        chunk_period_steps=lp["chunk_period_steps"], target_timeline_ms=lp["target_timeline_ms"],
        note="the query step lasts 2.3 sim steps (model 30 ms): the previous chunk's last target is held ~54 ms, then "
             "the loop resyncs (no catch-up) when the step is > 40 ms late; chunks with a query step < 40 ms catch up "
             "(steps < 15 ms)",
        dr_recommendation=dict(
            robot_dyn=dict(d_lo=2, d_hi=2, jitter=0.4, delay_mode="queue"),
            rule=f"goal write of the first target of a new chunk {war['0']['p50']:.0f} ms (p5-p95 {war['0']['p5']:.0f}-"
                 f"{war['0']['p95']:.0f} ms) = {war['0']['p50'] / 20:.1f} steps after the joint reading; targets 0-4 at "
                 + " / ".join(f"{war[str(k)]['p50']:.0f}" for k in range(5)) + " ms (steps "
                 + " / ".join(f"{war[str(k)]['p50'] / 20:.1f}" for k in range(5)) + "): the queued d = 2 runs them at "
                 "steps 2-6. The servo's own delay comes on top and is already in RobotDynConfig.servo (delay_ms, "
                 "measured from the write: do not add it to d). All 5 targets run.",
            hold_extra_steps=hold_extra,
            hold_extra_rule="the real chunk lasts 127 ms (p50; 6.35 steps) against 100 ms in the sim: a 27 % longer "
                            "decision. Fixed-step form: the last target of each chunk holds h extra 20 ms steps before "
                            "the next query, h in {1, 2} with P(2) = p50 chunk steps - 6, so the mean decision lasts "
                            "6.35 steps; with the queued d = 2 the last target then holds 2-3 steps (real 54 ms). "
                            "Proposal only: a RobotDynConfig field hold_extra_steps (off by default = 5-step "
                            "decisions bit for bit) needs the env to step the world 5 + h times per decision (episode "
                            "length in decisions changes for the same seconds)."))
    # ---- (4) appearance ---------------------------------------------------------------------------------------------
    out["brightness"] = dict(
        session_exposure_factor=AP["session_exposure_factor"], episode_exposure_factor=AP["episode_exposure_factor"],
        within_episode_exposure_factor=AP["within_episode_exposure_factor"],
        day_night_first_query=AP["day_night_first_query"], sim_nominal_vs_real=AP["sim_nominal_vs_real"],
        all_queries_gray=AP["all_queries_gray"],
        first_frame_transients=[k for k, v in AP["first_frames"].items() if v["ext"]["gray"] > 120],
        dr_now=dict(exposure={"external": [0.8, 1.35], "wrist": [0.7, 1.7]}, gamma=[0.85, 1.25]),
        verdict="first-query linear brightness between sessions: external 0.95-1.08, wrist 0.94-1.07 of the median; "
                "within an episode (wrist auto exposure and view): p5-p95 0.89-1.10, worst 0.65-1.36; day and night "
                "differ by less than 1 gray level (tent light); real / nominal sim linear brightness: external 1.05, "
                "wrist 0.99; the scene exposure ranges (external 0.8-1.35, wrist 0.7-1.7) cover all of this. Real "
                "wrist thumbs clip 9.9 % of the pixels (nominal sim 1.4 %).")
    out["extcam"] = dict(
        per_session=AP["ext_pose_per_session"], ref_vs_teleop=AP["ext_pose_ref_vs_teleop"], summary=AP["ext_pose_summary"],
        verdict="the external camera turned slowly by 0.25 -> 0.91 deg against the calibration (teleop) frames "
                "(10-22 14:00 -> 10-23 21:56), mostly about the optical axis and the camera y axis; student crop-centre shift <= 3.2 raw px; "
                "the last 3 sessions fit a pure rotation only to 2.1-2.6 px (something else moved too). DR cam_rot "
                "+-1.49 deg covers it.")
    # ---- (5) servo --------------------------------------------------------------------------------------------------
    cm = SV["rms_deg_contact_mask_2deg"]
    dec_nc = cm["new_eps_without_contact_eps"]["decision"]
    dec_fit = SV["rms_deg"]["fitted_eval_setting"]["fit_eps"]["decision"]
    out["servo_check"] = dict(
        n_episodes=SV["n_episodes"], n_fit=SV["n_fit"], n_new=SV["n_new"], n_steps=SV.get("n_steps"),
        verdict=f"the fitted servo still agrees: on the {cm['n_new_without']} new episodes without the outliers the "
                f"decision-horizon RMS is {min(dec_nc):.2f}-{max(dec_nc):.2f} deg for each joint (contact mask 2 deg; "
                f"fit episodes {min(dec_fit):.2f}-{max(dec_fit):.2f} deg). Outliers by rule ({cm['outlier_rule']}): "
                f"{len(cm['contact_episodes'])} episodes; seen by eye in img/servo_outlier_views.jpg as open-jaw "
                "hovers with the fixed jaw tip on the cube, the shoulder_lift 2-3.6 deg from its command with the "
                "integral correction at its 3 deg cap (servo_per_episode.json): contact, not a servo change. The grid "
                "around the setting finds no better centre for 4 of 6 joints; lag x 1.5 and delay -20 ms make "
                "shoulder_lift / elbow 0.02-0.07 deg better (decision horizon).",
        rms_deg=SV["rms_deg"], rms_deg_contact_mask_2deg=SV.get("rms_deg_contact_mask_2deg"),
        grid_center_new=SV["grid_center_new"], grid_best_new_arm_mean=SV["grid_best_new_arm_mean"],
        grid_best_per_joint_new=SV["grid_best_per_joint_new"], grid_marginals_new=SV["grid_marginals_new"],
        joints=["pan", "shoulder_lift", "elbow", "wrist_flex", "wrist_roll", "jaw"])
    # ---- (6) roll ---------------------------------------------------------------------------------------------------
    out["roll"] = {k: RO[k] for k in RO if k != "rows"}
    (L.OUT / "sysid_v8.json").write_text(json.dumps(L.r3(out, 4), indent=1))
    print("wrote", L.OUT / "sysid_v8.json")


def session_table():
    """Markdown table: one row per session (robot thumbs, first frames, camera pose, timing)."""
    TM, AP = ld("timing.json"), ld("appearance.json")
    eps = L.list_episodes()
    stud = {}
    for e in eps:
        stud.setdefault(e["session"], set()).add(e["student"])
    rows = ["| Session | Day / night | Episodes | Students | Ext. gray (first query) | Wrist gray (first query) | "
            "Dark area / floor gray | Ext. camera turn vs calibration (deg) | Fit residual (px) | Crop-centre shift (raw px) "
            "| Frame age p50 ext / wrist (ms) | Loop rate p50 (Hz) |", "|" + "---|" * 12]
    S, EP, BS = AP["sessions_thumbs"], AP["ext_pose_per_session"], TM["by_session"]
    for name in sorted(set(S) | set(EP), key=lambda n: n.split("_", 1)[1]):
        s, p, b = S.get(name), EP.get(name), BS.get(name)
        rows.append("| " + " | ".join([
            name, (s or p or {}).get("dn", ""), str(s["n_ep"]) if s else "0 (no loop)",
            ", ".join(sorted(stud.get(name, []))) or "-",
            f"{s['ext_first_gray']:.1f}" if s else "-", f"{s['wr_first_gray']:.1f}" if s else "-",
            f"{s['ext_dark']:.1f} / {s['ext_floor']:.1f}" if s else "-",
            f"{p['rot_vs_teleop_norm_deg']:.2f}" if p else "-", f"{p['resid_px_p50']:.2f}" if p else "-",
            f"{np.hypot(*p['crop_centre_shift_px']):.2f}" if p else "-",
            f"{b['age_ext_p50']:.1f} / {b['age_wr_p50']:.1f}" if b else "-", f"{b['rate_hz_p50']:.1f}" if b else "-"]) + " |")
    (L.OUT / "session_table.md").write_text("\n".join(rows) + "\n")
    print("wrote", L.OUT / "session_table.md")


if __name__ == "__main__":
    main()
    session_table()
