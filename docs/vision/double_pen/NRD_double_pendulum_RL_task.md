# NRD Double-Pendulum Goal-Reaching RL Task

**Status:** proposed first RL experiment after NRD dynamics training  
**Primary goal:** show that a policy using `[z1, z2]` can be trained efficiently inside the frozen, decoder-free NRD and transferred without fine-tuning to Chrono.

## 1. Experimental question

Train two goal-conditioned policies against the **same frozen NRD transition model**:

1. a policy whose dynamic-state observation is `z1`; and
2. a policy whose dynamic-state observation is `[z1, z2]`.

The NRD transition model consumes `[z1, z2, action]` and predicts both next `z1` and next `z2` in both experiments. Only the policy observation changes. This isolates the effect of exposing the visual latent to the policy without changing the learned dynamics.

The camera decoder is not part of policy training or deployment. It remains a diagnostic tool only.

## 2. System and action

The planar double pendulum has link lengths

\[
l_1=l_2=0.3\ \mathrm{m}, \qquad L=l_1+l_2=0.6\ \mathrm{m}.
\]

The normalized policy action controls elbow torque:

\[
a_t\in[-1,1], \qquad \tau_t=a_t\tau_{\max}, \qquad \tau_{\max}=1.5\ \mathrm{N\,m}.
\]

The NRD advances at 50 Hz (`dt = 0.02 s`). For the first policy experiment, hold each policy action for five NRD transitions, giving a 10 Hz policy rate. This is close to the action dwell times represented in the training data and reduces high-frequency model exploitation. Chrono evaluation must use the same policy rate and action hold.

## 3. Goal distribution

At reset, sample one end-effector goal in polar coordinates about the fixed shoulder pivot:

\[
\theta_g\sim\mathcal U(0,2\pi),
\]

\[
r_g\sim\mathcal U(0.5L,0.8L).
\]

Using the standard polar convention in the mechanism's world `X-Z` plane,

\[
x_g=r_g\cos\theta_g, \qquad z_g=r_g\sin\theta_g.
\]

For the current mechanism this gives

\[
r_g\in[0.30,0.48]\ \mathrm{m}.
\]

This annulus lies inside the reachable workspace. The goal remains fixed for the whole episode. Resample a goal if the initial end-effector distance is already within the 1 cm success tolerance.

Uniform sampling of `r` is intentional for the first experiment because it gives equal weight to radial bands. If uniform sampling by workspace area is wanted later, sample `r = sqrt(U(r_min^2, r_max^2))` as a separate ablation.

## 4. Reset and autonomous NRD rollout

Each vectorized NRD environment should initialize from a valid recorded 16-step context containing aligned `z1`, encoded `z2`, and actions. Do not initialize the transformer by repeating a single state and latent, because that history is outside the training distribution.

After reset:

1. freeze the encoder, transition model, and decoder parameters;
2. obtain the initial latent context from recorded frames or a cached encoder pass;
3. recursively predict both `z1` and `z2` at every 0.02 s NRD step;
4. call the policy once every five NRD steps and hold its action between calls; and
5. never call the decoder.

At every predicted transition, renormalize the `(cos q, sin q)` pair for each joint before computing kinematics, reward, or the next transition input.

## 5. Policy observations

Both policies receive identical goal information. Use a continuous goal representation without an angular wrap discontinuity:

\[
g=[x_g/L,\ z_g/L].
\]

Also provide the normalized Cartesian end-effector error

\[
e_t=[(x_g-x_{ee,t})/L,\ (z_g-z_{ee,t})/L].
\]

The two observation variants are

```text
z1 policy:
    [normalize_state(z1_t), g, e_t]

z1+z2 policy:
    [normalize_state(z1_t), normalize_z2(z2_t), g, e_t]
```

Here `normalize_state` and `normalize_z2` must use the statistics stored in the NRD checkpoint. Goal and error features are identical between policies, so the only experimental difference is the 64-dimensional `z2` input.

## 6. End-effector position

For

\[
z_1=[\cos q_1,\sin q_1,\cos q_2,\sin q_2,\omega_1,\omega_2],
\]

recover the angles with `atan2` and compute

\[
x_{ee}=l_1\sin q_1+l_2\sin(q_1+q_2),
\]

\[
z_{ee}=-l_1\cos q_1-l_2\cos(q_1+q_2).
\]

The goal distance is

\[
d_t=\sqrt{(x_{ee,t}-x_g)^2+(z_{ee,t}-z_g)^2}.
\]

Reward and success are calculated from predicted `z1`, never from a decoded image.

## 7. Reward

The reward must be identical for both observation variants. It contains:

- dense end-effector distance and progress shaping;
- a success bonus when the end effector is within 1 cm of the goal; and
- a penalty for fast changes in both generalized angular velocities.

A recommended first form is

\[
r_t=
-w_d\frac{d_t}{L}
+w_p\frac{d_{t-1}-d_t}{L}
+B_s\mathbf 1[d_t\le 0.01]
-w_{\Delta\omega}\sum_{i=1}^{2}
\left(\frac{\omega_{i,t}-\omega_{i,t-1}}{\sigma_{\omega_i}}\right)^2.
\]

`sigma_omega_i` is the corresponding angular-velocity scale from the processed-dataset or checkpoint normalization. Normalization prevents one joint from dominating the smoothness term because its raw velocity range is larger.

Suggested initial weights are:

```yaml
reward:
  distance_weight: 1.0
  progress_weight: 5.0
  success_bonus: 25.0
  angular_velocity_change_weight: 0.01
  success_tolerance_m: 0.01
```

These values are starting points and must be tuned using term magnitudes, success rate, and Chrono transfer rather than total reward alone.

In the equation, `t` indexes the 0.02 s NRD transitions. Compute the reward and success condition at every NRD transition, then sum the five substep rewards returned for one 0.1 s policy action. This avoids missing a brief success between policy observations and makes the angular-velocity-change term measure changes at the model's native time step.

The dense distance/progress terms are important. With only a 1 cm sparse success bonus and the angular-velocity-change penalty, remaining nearly stationary can become an easy local optimum.

Do not add an action-magnitude or action-rate penalty initially. Log both quantities, and add a small penalty only if the learned policy uses aggressive torque switching that harms Chrono transfer.

## 8. Success and termination

For the first experiment, success occurs when

\[
d_t\le 0.01\ \mathrm{m}.
\]

Award the success bonus once and terminate the episode. If later policies learn to pass rapidly through the tolerance region without settling, keep the 1 cm geometric tolerance but require it for several consecutive policy steps.

Also terminate on:

- the maximum episode duration (start with 5 s, then extend only if the transfer result shows that more time is needed);
- a non-finite state or latent;
- `|omega1|` or `|omega2|` exceeding the 35 rad/s data-collection guard;
- an action outside `[-1,1]` after scaling or clipping; or
- a normalized `z2` or state value outside a training-distribution guard chosen from the processed dataset.

Record out-of-distribution termination separately from ordinary timeout. A policy that achieves high NRD reward by frequently leaving the learned model's support is not successful.

## 9. Training comparison

Use the same PPO implementation, architecture apart from the required input width, hyperparameters, random seeds, reset contexts, sampled goals, reward, and termination rules for both policies.

Train:

```text
Policy A: frozen NRD transition; policy observes z1 + goal features
Policy B: frozen NRD transition; policy observes z1 + z2 + goal features
```

The state-only NeDM dynamics model is not part of the primary comparison. It is a fallback diagnostic only if neither NRD-trained policy transfers.

Track at least:

- success rate;
- environment steps and wall time to reach fixed success-rate thresholds;
- final and minimum end-effector distance;
- time to success;
- RMS and maximum angular-velocity change;
- action magnitude and action slew;
- timeout and out-of-distribution termination rates; and
- reward broken down by term.

## 10. Chrono transfer

Transfer each trained policy to the original Chrono system without policy fine-tuning.

For every Chrono policy step:

1. read the true Chrono `z1`;
2. for Policy B, encode the synchronized true camera frame to obtain `z2`;
3. construct exactly the same normalized observation and goal features used during NRD training;
4. apply the policy action for the same 0.1 s hold; and
5. calculate evaluation metrics from the true Chrono state.

The decoder remains off. Use a fixed held-out set of `(initial condition, goal)` pairs for both policies so their NRD and Chrono results are directly paired.

Report:

- NRD success rate and Chrono success rate;
- the per-policy transfer gap;
- median and percentile final error;
- median time to success;
- angular-velocity-change statistics; and
- any NRD-success/Chrono-failure cases.

## 11. Interpretation

This first double-pendulum scene is fully observed by `z1`, and the fixed-camera `z2` is largely redundant with mechanism pose. Therefore, the expected result is similar task performance for the two policies.

The main success criterion is not that `z2` improves reaching. It is that the `[z1,z2]` policy:

1. trains at useful batched throughput with the decoder disabled;
2. reaches held-out polar goals inside the frozen autonomous NRD; and
3. transfers to Chrono with an acceptable degradation relative to its NRD result.

If the `z1` policy transfers but the `[z1,z2]` policy does not, investigate the distribution shift between recursively predicted training latents and camera-encoded Chrono latents. If neither policy transfers, use the more accurate state-only dynamics model as a diagnostic control before changing the task or reward.
