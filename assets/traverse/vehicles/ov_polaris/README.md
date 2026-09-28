# Private Polaris vehicle data (offroad_vehicles_20260927, module M1)

Written by `scripts/ov_polaris_check.py make-data` from Chrono's stock `data/vehicle/Polaris/` files.
At run time `scripts/ov_vehicle.py` builds a temporary vehicle-data folder with two links, `Polaris_ov/` (this
folder's `Polaris_ov/`) and `Polaris/` (the Chrono build's stock folder), points Chrono's vehicle data path at it
while the vehicle is built, and restores the path afterwards. `MANIFEST.json` holds the sha256 of every file;
the vehicle switch refuses to build if any file differs.

- `Polaris_ov/Polaris_ovc_Chassis.json`: the stock chassis with the reference point moved from the front axle
  (0.397 m below the axle line, i.e. below the ground at rest) to mid-wheelbase, 0.023 m above the front axle
  line: every location + (1.35763, 0, -0.42) m. Same mass and inertia. The visual mesh line is dropped (it would
  be drawn 0.42 m too low; rendering only).
- `Polaris_ov/Polaris_ovc_stock.json`: the stock vehicle file with the same shift of every location; stock
  driveline (`Polaris/Polaris_DrivelineSimple.json`, which has Chrono's reduction defect). Primary arm `polaris`
  (and `polaris_w08`, which differs only in the soil wheels).
- `Polaris_ov/Polaris_ovc_pc.json` + `Polaris_ovc_DrivelineSimple_pc.json` +
  `Polaris_ovc_AutomaticTransmissionSimpleMap_pc.json`: power-corrected arm `polaris_pc` (conical ratios 1.0,
  every gear ratio x 0.25).
- `Polaris_ov/Polaris_ovc_4wd.json`: arm `polaris_4wd`, Chrono's shafts driveline `Polaris/Polaris_4WD.json`.
- `ov_polaris_belly.json`: points on the underside of the visual chassis mesh (lowest vertex per 0.10 m cell),
  in the re-framed chassis frame, for the belly-in-soil diagnostic.
