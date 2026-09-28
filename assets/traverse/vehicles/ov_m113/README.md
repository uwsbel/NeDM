# ov_m113: private M113 data (offroad_vehicles_20260927, module M2)

- `M113_AutomaticTransmissionShafts_g4.json`: arm `m113_g4`. Chrono's `M113_AutomaticTransmissionShafts` C++ class
  (src/chrono_models/vehicle/m113/powertrain/M113_AutomaticTransmissionShafts.cpp) written as JSON, every value equal
  to the C++ class (inertias, torque-converter maps, shift points, shift latency 1.0 s) except the gear ratios, all
  divided by 4 (forward 0.06 / 0.10675 / 0.17125 / 0.2405, reverse -0.03775).
- `M113_AutomaticTransmissionShafts_x1_check.json`: the same file with the stock ratios; used only by
  `scripts/ov_m113_check.py rigid` to show that the JSON route reproduces the C++ transmission.
The vehicle itself is Chrono's M113 C++ model (no JSON); see scripts/ov_m113.py.
