"""Add the evidence texts, alternatives, colours and scene notes to calib_v1.json (after calib_export.py).

Usage: python calib_finalize.py --calib calib/calib_v1.json --colors calib/colors.json --notilt fits/notilt2_full.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calib_common import CALIB, load_json, save_json  # noqa: E402
from calib_joint import Data, Joint, load_x  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=str(CALIB / "calib_v1.json"))
    ap.add_argument("--colors", default=str(CALIB / "colors.json"))
    ap.add_argument("--notilt", default=str(CALIB / "fits/notilt2_full.npy"))
    a = ap.parse_args()
    cal = load_json(a.calib)
    jm = cal["joint_map"]
    off = jm["offset_deg"]
    jm["evidence"] = (
        "LeRobot 0.6.2 so_follower (use_degrees=True): arm joints in DEGREES mode = (raw - (range_min+range_max)/2) "
        "* 360/4095 (no clamping); gripper RANGE_0_100 = (raw - 2033)/(3551 - 2033)*100 (drive_mode 0, 1518 ticks = "
        "133.4 servo deg, 1.334 deg/unit). Signs: pinhole PnP on the 30 resting-cube points for all 32 sign "
        "combinations: (pan, lift, elbow, wrist_flex) = (+,+,+,+) rms 3.2-3.3 px; flipping pan puts the camera below "
        "the table (z -0.48 m, rms 3.9-4.1); every lift/elbow/wrist_flex flip gives rms >= 16.7 px. wrist_roll sign: "
        "full joint fit with roll sign -1 keeps 30 % of the wrist cube and 7 % of the wrist plate observations within "
        "8 px (vs 91 % / 76 % with +1). Offsets: joint bundle adjustment of both cameras with the arm kinematics "
        f"(lift {off[1]:.1f}, elbow {off[2]:.1f}, wrist_flex {off[3]:.1f}, wrist_roll {off[4]:.1f} deg; pan fixed 0 = "
        "gauge with the scene rotation about the pan axis). Rest pose: the overlay of the mapped folded pose matches "
        "the real external images (overlay_external_*). wrist_roll offset ~ -76 deg: with 0 the jaws in the external "
        "view open sideways while the real jaws are seen edge-on (evidence_roll_offset_external.png), and the wrist "
        "views cannot be fit (wrist cube median error 67 px with the MJCF mount, 2 px with the fitted offset); the "
        "calibration homing pose of the roll was probably not the diagram pose. Gripper: the moving-jaw tip in the "
        "wrist image (4291 frames) vs gripper value gives 1.30 deg/unit (per-frame fit; 1.32 deg/unit in the joint "
        "fit) - close to the 1:1 servo value 1.334; holding plateau g = 13.8 maps to sim jaw 7.2 deg, while the MJCF "
        "pads are 30 mm apart at ~9.5 deg (2.3 deg / ~4 mm gap mismatch of the jaw geometry).")
    # alternatives
    alt = {}
    try:
        d = Data()
        J = Joint(d)
        x = load_x(J, a.notilt)
        alt["flat_table"] = dict(
            note=("same fit with the table tilt held at 0 (only the height free): total robust cost +14 % (43,313 vs "
                  "37,866 on ~24,170 residuals); the tilt and the shoulder_lift offset are partly interchangeable"),
            offset_deg=[0.0, *np.degrees(J.g(x, "off")).tolist(), float(np.degrees(J.g(x, "offr")[0])), 0.0],
            table_plane_coef=J.g(x, "tbl").tolist())
    except Exception as ex:  # pragma: no cover
        alt["flat_table"] = dict(error=str(ex))
    alt["gripper_lin_hold_width"] = dict(
        gripper_lin=[float(jm["gripper_lin"][0]), float(np.radians(9.5) - jm["gripper_lin"][0] * 13.8)],
        note="same slope, offset chosen so that the real holding value g = 13.8 maps to the MJCF 30 mm pad gap (9.5 deg)")
    alt["gripper_lin_servo_slope"] = dict(
        gripper_lin=[float(np.radians(1.334)), float(jm["gripper_lin"][1] + (jm["gripper_lin"][0] - np.radians(1.334)) * 25)],
        note="1:1 servo slope (1.334 deg/unit), offset matched at g = 25 (the middle of the observed range)")
    jm["alternatives"] = alt
    cal["scene"]["colors_srgb"] = load_json(a.colors)
    cal["scene"]["notes"] = (
        "World = arm base frame (so101.xml). The table / tent floor is NOT at z = 0: fitted plane z = c0 + c1 x + c2 y "
        "(scene.table_plane; -4.7 mm at the base, about -16 mm at x = 0.3 m; tilt 2.1 deg down along +x, 1.1 deg up "
        "along +y). The black mat lies on it (rectangle 0.308 x 0.460 m, centre (0.273, -0.004), yaw -3.9 deg). Tent "
        "floor (white) edges on the table plane: table edge behind the robot at x ~ -0.04 (the base is clamped there, "
        "nothing beyond it), back wall bottom at y ~ +0.46, front edge at y ~ -0.45, right wall bottom at x ~ 0.59 "
        "(tent_edges_on_table); walls white, an LED strip along the top of the far (+x) side, the room (dark, a yellow "
        "wall) visible beyond the open table edge. All tent / mat edges are rotated by about -3.5 deg about z "
        "relative to the base axes (either the base is turned on the table or the pan zero is off by that much). "
        "Cube 30 mm (resting centre 15 mm above the local table), plate 50 x 50 mm, thickness held at 2 mm "
        "(prior; not resolved by the data). The real wrist camera is NOT at the MJCF wrist_cam pose: fitted pose in "
        "camera_mount is ~4.7 cm further back along the gripper axis and ~1.7 cm further out than the MJCF camera "
        "(the external view shows the camera board on a taller bracket). Image time lags the joint state by "
        "time_offset_frames (about 1.3-1.4 frames = 42-48 ms) in both cameras.")
    cal["real_start_pose"] = dict(
        note=("all 15 episodes start from the folded rest pose; LeRobot deg median [8.3, -88.1, 81.1, 71.7, 3.9, 0.7] "
              "(gripper 31-40 in green_cube_red_plate, 0.6-5.7 in green_cube_red_plate_1); mapped to sim deg median [8.3, -90.0, 72.6, 70.6, "
              "-71.9] (arm) - far from the sim task starts (median ~[1, -5, -15, 92, 11] deg)"))
    save_json(a.calib, cal)
    print("finalized", a.calib)


if __name__ == "__main__":
    main()
