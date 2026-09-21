# Queue

Ordered. Top item is next. Move an item to STATE.md when done, with its result.

## Now

1. **How is the forward-speed effect distributed across surrogates?** Two v2 surrogates
   gave opposite answers (-36%/-20% and +57%). A third (seed 1) is training on north. Tune
   PPO in each, two seeds, evaluate paired; report the distribution, not the best.
2. **Surrogate ensemble with a disagreement penalty.** The standard answer to a policy
   exploiting one model's errors. The OOD kNN penalty did not matter (PPO works without
   it), so disagreement between models, not distance from data, is the next thing to price.

## Next

3. **Multi-command CRM evaluation.** Everything so far is vx 0.5 on one spawn set.
4. **Gate on the loop check.** It now separates faithful from broken loops on a real model.

## Open questions, not blocking

- **Does the 800-episode v1 model beat the 550?** Running on a3 as a capacity data point.
  If 800 is barely better than 550, the v2 model is data-limited in some other way than
  volume, and more episodes will not fix it.
- **Is 0.30 s long enough to learn anything?** It is what the model supports today. A
  better model buys longer branches, which PPO in particular should want.
- **Does truncation bias the corpus?** Episodes that drift most get cut shortest. v2
  truncates 19.4% of episodes (v1 12.6%) and keeps 94.4% of rows. The drift allowance is
  0.35 rad/s; if the fine-tune results depend on high-drift behaviour, raise it or size
  from the observed drift distribution.
- **Is the Chrono GPU fault reproducible?** One illegal memory access in
  `SphBceManager.cu` in ~1,950 episodes, the first under 4-per-node packing. Not enough
  to blame packing; worth a note to the Chrono FSI maintainers if it recurs.
- **Is there a machine effect on CRM at all?** Withdrawn as unproven, not established.
  Keep comparisons within one machine.
- **The four unexplained HMMWV divergences** in `docs/SOIL.md`, worth raising with Zhang.
- **Their bed is 0.25 m deep**, above the ~0.22 m at which we measured a CRM bed heaving.
