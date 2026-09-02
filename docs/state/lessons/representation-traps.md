# Representation traps

Two failure modes that are invisible in the loss curve and will recur in every
vision study. Both measured in Study 1.

## Latents share a huge constant component — normalize before use

**Cost:** metrics that read 1.000 and mean nothing · **Found:** 2026-08-25

**Expected:** cosine similarity between latents of different frames is a
meaningful signal. **Happened:** raw pairwise cosine between *arbitrary* frames
is **0.9998**; per-dimension std across frames is only 0.013. **Cause:** the
encoder's `LayerNorm`'d output has a large common direction; the informative
part is a small residual on top of it. **Fix:** fit `z2_mean`/`z2_std` from the
frozen warm-started encoder at joint-trainer init
(`NRDDynamicsModel.set_z2_normalization`) and use the normalized space for
**tokens, losses and metrics alike**. Identity until fitted.

Consequence: any latent-space loss computed raw is dominated by a trivially
predictable constant, and any raw-cosine metric is meaningless.

## A plain reconstruction loss erases the object

**Cost:** a full autoencoder training run · **Found:** 2026-08-25

**Expected:** low reconstruction L1 means the encoder captured the scene.
**Happened:** val L1 ≈ 0.005 and PSNR ≈ 26 dB while the decoder reconstructed
**only the static background and erased the pendulum entirely** (verified
visually). **Cause:** the pendulum covers ~3% of pixels, so ignoring it is
cheaper than modeling it. **Fix:** per-pixel foreground weighting (1, plus 30 on
pixels differing from a static per-pixel-median background model — exact for a
fixed camera), applied to both the AE warm-up and the joint stage's decoded-frame
loss. The background model is stored in `ae_best.pt`.

**This gets worse, not better, at higher resolution.** In Study 3 the vehicle is
~15×7 px in a 256² frame — a smaller foreground fraction than the pendulum's 3%.
Foreground weighting is not optional there, and the plan additionally mandates
auxiliary heads (occupancy, vehicle heatmap + yaw, elevation) precisely so the
latent cannot get away with modeling background alone.

## Corollary: judge representations by downstream probes

The encoder fine-tuning criterion in the Study 3 plan is **occupancy-probe and
localization-probe performance, not latent-prediction plateau**. Both traps above
show why a loss can look healthy while the representation is useless.
