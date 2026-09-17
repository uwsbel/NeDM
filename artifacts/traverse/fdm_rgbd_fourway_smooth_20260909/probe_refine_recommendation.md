# Final smooth-hill recommendation

Use the centered Gaussian hill H=6.5 m, sigma=4.5 m, with the existing straight 4 m/s reference as the hill-climb failure. The same hill is traversable at 6 m/s; present that speed dependence explicitly.

The 4 m/s run starts near rest with all four tires loaded. Bounded motion is confirmed at 11.45 s and a strict two-second stop at 17.2 s; it times out at 25 s. The longest strict stop is 3.05 s. During the final five seconds, median pitch is 34.49 degrees, yaw 0.81 degrees, gear 1, throttle 1 and brake 0. Intermittent movement remains: the final five-second XY diameter is 0.572 m.

Raw world-vertical median tire forces are 3.88/3.73/8.49/6.94 kN (FL/FR/RL/RR). Horizontal wheel-heading speeds are 0.102/0.080/0.061/0.051 m/s; circumference speeds are 4.97/8.50/0.473/0.401 m/s. Direct tire slip, wheel support, continuing effort and little forward motion support slope-induced traction/driveline failure. No sampled chassis or asset contact occurs.

The quantized terrain's local 0.5 m secant grade is 33.93 degrees beneath the chassis, about 38.6 degrees beneath the front wheels and 25–26 degrees beneath the rears. These are diagnostic truth measurements, never planner inputs. Query-normal force projections are approximations; an exact friction or engine-power limit is not established.

H6.7 at 6 m/s has a more stationary ending, but first rolls backward and stops on a shallower flank with saturated steering, gear 3 and one extremely fast spinning wheel. H6.5 at 4 m/s is the clearer aligned hill-climb example.

Neighboring heights are not monotonic: H6.6 and H6.8 at 6 m/s pass; H6.7 and H6.9 fail differently. These runs were screened to construct a demonstration, not an independent benchmark.

See probe_refine_independent.json/.md, probe_strength_independent.json/.md and slope_normal_caveat.json for definitions, source hashes and statistics. Zero chassis resultant at 20 Hz cannot exclude every brief or cancelling contact; exact underbody clearance is unavailable.
