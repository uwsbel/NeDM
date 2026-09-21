# Queue

Ordered. Top item is next. Move an item to STATE.md when done, with its result.

## Now

1. **Collect `go2_crm_v2`.** 1200 episodes, 24 shards packed 4 per `mi2104x` node
   (`qrun/collect_v2_array.sbatch`), running as array job 430005. The packed 4-GPU smoke
   (429999) passed: obs_check reproduces the policy output on CRM exactly, and four
   collects per node run at single-node speed. Shards 0-15 reuse
   v1 seeds. Merge with `qrun/merge_v2.sbatch` on `devel`; `merge_corpus.py` refuses
   shards that disagree on `row_capture`, `policy_raw_order` or `policy`.
2. **Train the v2 NN-ROM on a3.** Same recipe as v1. Read the horizon profile before
   anything else: fine-tune branches are 0.30 s, so the model must beat the no-motion floor
   there.

## Next

3. **Fine-tune on v2, both methods**, at the default 15 control steps (0.30 s) with
   recorded commands. The start check must pass (it reports the agreement); if it fails,
   stop -- nothing downstream means anything.
4. **Paired Chrono evaluation** of base / analytic / PPO on north with `evaluate.py` and
   `paired_eval.py`: 16 spawns over +/-1 m, same cases for every arm, Gate 4 per arm.
5. **If a fine-tune helps, replicate it** with a second fine-tune seed and a second spawn
   set before calling it a result. If neither helps, check in-model gain against Chrono
   gain per arm -- the anti-correlation this project has seen before is the first suspect.

## Open questions, not blocking

- **Does the 800-episode v1 model beat the 550?** Running on a3 as a capacity data point.
  If 800 is barely better than 550, the v2 model is data-limited in some other way than
  volume, and more episodes will not fix it.
- **Is 0.30 s long enough to learn anything?** It is what the model supports today. A
  better model buys longer branches, which PPO in particular should want.
- **Does truncation bias the corpus?** Episodes that drift most get cut shortest. Report
  the residual rate per corpus rather than assuming zero.
- **Is there a machine effect on CRM at all?** Withdrawn as unproven, not established.
  Keep comparisons within one machine.
- **The four unexplained HMMWV divergences** in `docs/SOIL.md`, worth raising with Zhang.
- **Their bed is 0.25 m deep**, above the ~0.22 m at which we measured a CRM bed heaving.
