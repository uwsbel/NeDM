# Pool table in Chrono: setup and checks (night of 2026-10-02)

Two pool balls on cloth inside four cushions. Ball A starts at the head spot
(-0.635, 0) m with launch `(vx, vy)` and no spin, like a centre-ball strike.
Ball B rests on the centre spot (0, 0). The table is a 9-ft playing area,
2.54 x 1.27 m between cushion noses. The balls are standard: radius 28.575 mm,
mass 0.17 kg. There are no pockets. Code: `src/nedm/pool_ball/physics.py`,
config `configs/pool_ball/chrono_v1.json`.

## What had to change from the bouncing-ball setup

1. **Chrono's default rigid contact (NSC) makes sliding balls hop.** NSC
   solves a convex relaxation of Coulomb friction, and a sliding contact
   separates at mu x slip speed. A ball launched at 2.5 m/s with no spin left
   the cloth at 0.43 m/s and hopped about 1 cm. No solver, tolerance or
   timestep setting removes this. The bouncing ball never noticed: its
   contacts were brief impacts with friction 0.03. A pool ball slides on the
   cloth all the time, so this setup uses **Chrono's penalty contact (SMC,
   Hertz)**. With SMC the same launch stays within 9 um of the cloth.
2. **SMC applies no rolling or spinning friction.** Chrono computes those
   coefficients but never uses them. The cloth's rolling resistance
   (rho = 0.3 mm) and spinning friction (0.3 mm) are therefore applied as
   torques rho*m*g opposing the ball's horizontal and vertical spin, through a
   body accumulator every step.
3. **SMC's default tangential friction depends on the physics step.** Chrono's
   "OneStep" model adds kt*|v_t|*dt. That made B's position after a cushion
   hit change by up to 16 mm when the step was halved. The "None" model keeps
   min(gt*|v_t|, mu*F_n), which does not depend on the step.
4. **Cushion nose at 1.27 R.** With a vertical face at ball-centre height,
   cushion friction lifted rolling balls 2-3 mm. With the nose above the
   centre, as on a real table, the rebound presses the ball into the cloth.
   With cloth restitution 0.05 (slate and cloth absorb that push), hops are
   at most about 0.5 mm.
5. **Per-pair materials** (cloth 0.2 / -, ball-ball 0.06 / 0.95, ball-cushion
   0.2 / 0.8, as friction / restitution) are set in an add-contact callback.
   Chrono's default rule takes the minimum of the two bodies' values, which
   cannot give all three pairs at once.

## Calibration against textbook rolling physics

A single ball launched with no spin, with sliding friction mu = 0.2 and
rolling resistance rho = 0.3 mm. Analytic values include rho during the slide:
roll starts at `t = 2 v0 R / (g (7 mu R - 5 rho))` with speed `0.7032 v0`.

| v0 | roll starts (Chrono / analytic) | speed at roll (Chrono / analytic) | rolling deceleration (Chrono / analytic) |
|---|---|---|---|
| 1.5 m/s | 0.227 s / 0.227 s | 1.0549 / 1.0547 m/s | 0.07357 / 0.07357 m/s² |
| 3.0 m/s | 0.454 s / 0.454 s | 2.1098 / 2.1095 m/s | 0.07356 / 0.07357 m/s² |

These hold for contact stiffness 1e8-2e9 Pa and steps 12.5-50 us. Other
checks:
- Full hit: B leaves at 0.968 of A's speed (restitution 0.95 alone gives
  0.975) and A nearly stops.
- Cushion rebound returns 0.73 of the normal speed.
- A-B contact lasts about 0.7 ms; cushion contact 1-2 ms.

## Timestep convergence

B's position at t = 2.0 s compared with a 6.25 us run (three shots, after
cushion hits):

| physics step | max change of B at 2.0 s |
|---|---|
| 25 us | 1.2 mm |
| **12.5 us (used)** | **0.19 mm** |
| 25 us with the OneStep friction model | 15.9 mm |

The dataset uses 12.5 us steps with 1 ms records. One 2.5 s shot costs about
7.6 s on one AMD core.
