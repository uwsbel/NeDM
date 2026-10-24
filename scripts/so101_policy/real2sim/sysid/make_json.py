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


def main():
    T, TM, LG, AP, SV, RO = (ld(n) for n in ("table.json", "timing.json", "lag_fit.json", "appearance.json",
                                             "servo_check.json", "roll.json"))
    cal = L.calib()
    tp = cal["scene"]["table_plane"]
    out = dict(version="sysid_v8", date="2026-10-23",
               frames="sim frame = arm base frame of calib_v1 (joint map + FK), z up; sim table / mat top z = 0",
               data=dict(robot_episodes=T["n_robot_episodes"], robot_steps=TM["n_steps"], robot_queries=TM["n_queries"],
                         robot_minutes=round(TM["n_steps"] * 0.0255 / 60, 1), sessions_with_first_frames=len(AP["first_frames"]),
                         teleop_episodes=T["teleop"]["n"], teleop_frames=T["floor_edge"]["teleop"]["n"],
                         robot_logs=["~/so101_r2s_claude/robot_logs/{eval_*/ep_*,live_*}",
                                     "~/so101_r2s_claude/robot_logs_lab/so101_hw_*/eval_*/ep_*"],
                         teleop="~/so101_r2s_claude/raw (LeRobot v3, 30 fps)"))
    # ---- (1) table / mat ------------------------------------------------------------------------------------------
    fe = T["floor_edge"]
    out["table_plane"] = dict(
        meaning="real mat top (the surface the cube rests on) z = c0 + c1 x + c2 y (m, sim frame)",
        coef=tp["coef"], tilt_deg=tp["tilt_deg"], source="calib_v1 scene.table_plane (teleop bundle adjustment)",
        verdict="confirmed within about +-3 mm; no better plane found in the robot data",
        checks=dict(
            robot_floor_edge=dict(note="measured lowest gripper point - plane, every robot step (mm)", n=fe["robot"]["n"],
                                  min=fe["robot"]["min"], p0_01=fe["robot"]["p0_01"], n_below_0=fe["robot"]["n_below_0"],
                                  n_below_minus2=fe["robot"]["n_below_m2"], n_within_3=fe["robot"]["n_within_3"]),
            teleop_floor_edge=dict(note="same, every teleop frame (no floor rule)", n=fe["teleop"]["n"],
                                   min=fe["teleop"]["min"], p0_1=fe["teleop"]["p0_1"], n_below_0=fe["teleop"]["n_below_0"],
                                   n_below_minus2=fe["teleop"]["n_below_m2"]),
            teleop_lowest_point_near_grasp_minus_plane_mm=T["teleop"]["low_grasp_min_minus_plane_mm"],
            robot_image_cube_height=dict(
                note="external-camera ray through the resting cube's blob, at the FK cube xy of the hold; height - 15 mm "
                     "- plane (mm)", minus_plane_mm=T["holds"]["img_minus_plane_minus15_mm"],
                plane_fit=T["robot_img_minus15"]),
            robot_hold_fk_upper_bound=dict(
                note="grasp point at the hold start - 15 mm: an UPPER bound of the mat (the floor rule keeps the jaw "
                     "high, the jaw holds the cube above its centre)", minus_plane_mm=T["robot_gp_minus15"]["minus_calib_plane_mm"],
                plane_fit=T["robot_gp_minus15"])),
        at_cube_positions_mm=T["plane_at_holds_mm"], at_plate_positions_mm=T["plane_at_plates_mm"],
        at_cubes_and_plates_mm=T["plane_at_cubes_and_plates_mm"],
        cube_xy=dict(x=T["holds"]["cube_x"], y=T["holds"]["cube_y"]), plate_xy=T["plate_xy"],
        sim="flat, z = 0 (NRD worlds and training renders); geometry.table_plane null in scenes v2 / v3")
    # range: the plane over every observed cube / plate position (p5-p95 -15.7 .. -10.5 mm, min / max -17.5 / -8.3)
    # widened by the calibration tolerance (+-3 mm, floor-edge checks) -> U(-20, -6) mm, centre -13 mm
    a = T["plane_at_cubes_and_plates_mm"]
    lo = float(np.floor(a["min"] - 3.0))
    hi = float(np.ceil(a["max"] + 3.0))
    out["table_dz_range_m"] = dict(
        center=round(a["p50"] / 1e3, 4), lo=lo / 1e3, hi=hi / 1e3, nominal_spread_p5_p95=[a["p5"] / 1e3, a["p95"] / 1e3],
        rule="per world: mat top = sim z 0 + dz (constant), dz ~ U(lo, hi); lo / hi = min / max of the measured plane "
             "over all 184 observed cube and plate positions -+ 3 mm calibration tolerance",
        plane_option=dict(note="if the training table can tilt: c0 = coef[0] +- 0.004, c1 = coef[1] +- 0.015, "
                               "c2 = coef[2] +- 0.015 (uniform)", coef=tp["coef"], c0_pm=0.004, c1_pm=0.015, c2_pm=0.015),
        sim_now=0.0)
    out["mat"] = dict(
        cube_rest_surface="= table_plane (calib: cube centre 15 mm above it)",
        mat_thickness_m=dict(real="not resolved (calib assumes 0.001; the cube-rest plane is what matters)", sim=0.001),
        plate_thickness_m=dict(real=cal["scene"]["plate_thickness"], sim=0.001,
                               source="calib_v1 plate_top_z (teleop releases)"),
        plate_top_at_plates_mm=T["plate_top_at_plates_mm"],
        teleop_plate_top_calib_mm=T["teleop"]["plate_top_calib_mm"],
        held_cube_offset_mm=T["held_cube_offset_mm"],
        held_cube_note="the real jaw holds the cube 10 mm (p5-p95 4-15) above its centre at the hold start; sim teacher "
                       "about 2.5 mm. Cause: the action-layer floor (lowest COMMANDED gripper point >= 1.5 mm + table_z, "
                       "sim frame) stops the descent 10-15 mm above the real mat; table_z was 0 in 111 / 115 episodes "
                       "(-5 mm in 4).",
        hold_low_minus_plane_mm=T["holds"]["low_minus_plane_mm"], n_holds=T["n_holds"])
    # ---- (2) image lag -----------------------------------------------------------------------------------------------
    tl = LG["total_lag"]
    out["image_lag_steps"] = dict(
        meaning="age of the image content relative to the joint reading of the same query, in 20 ms sim steps "
                "(sim renders the current state: 0)",
        frame_age_ms=TM["frame_age_ms"], grab_minus_joint_ms=TM["grab_minus_joint_ms"],
        capture_pipeline_ms=dict(
            robot_wrist_free_cube=LG["free_cube_robot_wrist"]["tau_pipeline_ms"],
            robot_wrist_free_cube_boot_p5_p95=[LG["free_cube_robot_wrist"]["tau_boot_p5_p50_p95"][0],
                                               LG["free_cube_robot_wrist"]["tau_boot_p5_p50_p95"][2]],
            robot_wrist_bias_removal=LG["bias_removal_robot_wrist"]["tau_pipeline_ms"],
            by_block_ms={"10-22 day": [18.9, 10.0, 27.9], "10-22 night": [12.0, 6.7, 18.1], "10-23 day": [10.5, 1.8, 16.9],
                         "10-23 night": [3.9, -1.0, 9.2]},
            by_block_note="free-cube fit per block: tau, bootstrap p5, p95 (ms)",
            teleop_check_wrist=LG["free_cube_teleop_wrist"]["tau_ms"],
            teleop_check_calib=LG["free_cube_teleop_wrist"]["calib_time_offset_ms"],
            external="robot wrist + 5.2 ms (calib external - wrist offset); not fitted directly (the held cube moves "
                     "too slowly in the external view)",
            n_samples=LG["free_cube_robot_wrist"]["n_samples"], n_episodes=LG["free_cube_robot_wrist"]["n_episodes"]),
        lag_ms=dict(wrist=tl["wrist"]["central"]["lag_ms"], external=tl["external"]["central"]["lag_ms"]),
        lag_steps=dict(wrist=tl["wrist"]["central"]["lag_steps"], external=tl["external"]["central"]["lag_steps"],
                       wrist_rounded_frac=tl["wrist"]["central"]["steps_rounded_frac"],
                       external_rounded_frac=tl["external"]["central"]["steps_rounded_frac"]),
        upper_bound_task_assumption=dict(note="calib offset (42-48 ms) + frame age: double counts the teleop frame age; "
                                              "contradicted by the teleop check", lag_steps=TM["image_lag"]["pipeline_calib"]["external"]["lag_steps"]),
        dr_recommendation=dict(steps_probs={"0": 0.1, "1": 0.45, "2": 0.35, "3": 0.1},
                               continuous_ms=[0.0, 60.0], center_ms=27.0,
                               rule="whole-step lag of the rendered image (both cameras, per episode, + per-query jitter "
                                    "of one step); covers the measured p5-p95 (0.5-2.3 steps) and the pipeline "
                                    "uncertainty (0-28 ms by block); 3 steps keeps the calib-based upper reading"))
    # ---- (3) loop ---------------------------------------------------------------------------------------------------
    lp = TM["loop"]
    out["loop_period_ms"] = dict(
        all_steps=lp["step_period_ms_all"], query_steps=lp["step_period_ms_query_steps"],
        other_steps=lp["step_period_ms_other_steps"], chunk_5_steps=lp["chunk_period_ms"], chunk_sim=lp["chunk_period_sim_ms"],
        rate_hz_per_episode=lp["rate_hz_per_episode"], overruns_per_query=lp["overruns_per_query"],
        frac_steps_over_25ms=lp["frac_steps_over_25ms"], hist=lp["period_hist_ms"], model_ms=lp["model_ms"],
        command_delay_ms=dict(query_steps=lp["cmd_delay_ms_query_steps"], other_steps=lp["cmd_delay_ms_other_steps"],
                              note="joint read -> goal write; sim: 0"),
        note="the query step lasts 2.3 sim steps (model 30 ms): the previous chunk's last target is held ~47 ms, then "
             "the loop resyncs (no catch-up) when the step is > 40 ms late; chunks with a query step < 40 ms catch up "
             "(steps < 15 ms)",
        dr_recommendation=dict(
            command_delay_steps=2, jitter=0.4, delay_mode="queue",
            rule="the first target of a new chunk reaches the servo 46 ms (p5-p95 37-48 ms) = 2.3 steps after the "
                 "joint reading; all 5 targets run; the last target of a chunk holds ~55 ms. RobotDynConfig d 2 "
                 "(jitter 1-3) in queue mode is the closest fixed-step form; the 27 % longer real chunk (127 vs "
                 "100 ms) has no fixed-step form"))
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
    out["servo_check"] = dict(
        n_episodes=SV["n_episodes"], n_fit=SV["n_fit"], n_new=SV["n_new"], n_steps=SV.get("n_steps"),
        verdict="the fitted servo still agrees: on the 99 new episodes without jaw-on-cube contact the decision-horizon "
                "RMS is 0.08-0.20 deg for each joint (fit episodes 0.08-0.26 deg; perfect tracking 0.52-1.80 deg). "
                "4 new episodes (open-jaw hovers with the fixed jaw tip on the cube) hold the shoulder_lift 2-3.6 deg "
                "from its command with the integral correction at its 3 deg cap: contact, not a servo change. The "
                "grid around the setting finds no better centre for 4 of 6 joints; lag x 1.5 and delay -20 ms make "
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
