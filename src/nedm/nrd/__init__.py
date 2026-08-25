"""Neural Reduced Dynamics (NRD): vision-in-the-loop extension of NeDM.

The NRD state is concat(z1, z2): the explicit physical state z1 (the NeDM
state) plus a compact camera latent z2 = E(x). One temporal model predicts
both parts; a decoder maps predicted latents back to frames for training and
evaluation, and is disabled during high-throughput rollouts.

See docs/vision/NRD_overall_project_plan.md and
docs/vision/double_pen/NRD_double_pendulum_study_plan.md.
"""
