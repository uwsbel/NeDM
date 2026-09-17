# Independent smooth-hill mechanism analysis

The current broad hill produces bounded motion with wheel spin; the narrower hill produces repeated rollback. Neither current layout has a successful detour.

| Hill | Route | Goal | Bounded confirmation | Rollback time | Last5s XY diameter | Chassis contact |
|---|---|---:|---:|---:|---:|---:|
| h5p5_s3p5 | family_01 | False | none | 6.20s | 1.655m | 0.0N |
| h5p5_s3p5 | family_03 | False | 9.5 | 5.15s | 0.702m | 0.0N |
| h5p5_s3p5 | family_05 | False | none | 6.60s | 3.303m | 0.0N |
| h6_s4p5 | family_01 | False | 5.9 | 4.30s | 0.178m | 0.0N |
| h6_s4p5 | family_03 | False | none | 6.55s | 4.698m | 0.0N |
| h6_s4p5 | family_05 | False | none | 7.30s | 5.894m | 0.0N |

Both instrumented straight runs exactly match basic state, action, pose, terminal state/pose, power, contact and work arrays.

H5.5/sigma3.5 has a clean near-rest launch but no stationary bounded interval: rollback lasts6.2s and the last5s trajectory spans1.65m. The front wheels are frequently unloaded while spinning; the rear wheels continue supporting and pushing the vehicle.

H6/sigma4.5 confirms bounded motion at5.9s and its final5s span only.178m. Full throttle, zero brake, gear1 and loaded-wheel slip coexist with near-zero wheel-hub motion. This supports a slope-induced traction/driveline failure rather than a sampled chassis block. However, its initial scored frame is already moving backward at.663m/s and pitched35deg, with both front tires unloaded; it cannot be presented as a settled from-rest launch.

Move the broad hill farther ahead and repeat unchanged straight/detour probes before selecting the final scene. Moving the center to0 also lets the existing +8m detour reach its full offset near the hill. Longer observation of H5.5 is permissible as a persistence test, but its continuing rollback does not currently justify a stationary-stall label.

Contact-pair identity and exact underbody clearance remain unavailable. Zero20Hz chassis resultant supports but cannot prove absence of every brief contact. See the JSON for raw-source SHA256 values, time-window force/slip statistics and explicit measurement conventions.
