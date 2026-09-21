# Queue

Ordered. Top item is next. Move an item to STATE.md when done, with its result.

## Now

1. **Verify the v2 smoke corpus.** Six episodes at drift-aware bed sizing. Three in, all
   keeping every row, where v1 truncated two of six. Confirm the remaining three and run
   `corpus_check.py` for the four gates.
2. **Confirm the active domain.** Replication at seed 77 on sbel. The first ensemble gave
   no measurable bias for either 0.5 m or 1.0 m against the unapproximated solve, with
   0.5 m costing 1.7x less. If it replicates, 0.5 m is the setting and the corpus gets a
   1.6x speedup per episode.

## Next

3. **Stage hpcfund and validate on `devel`.** The tree, `src/nedm`, the policy and the
   URDF assets are not there. The branch is euler-local, so moving it is a deliberate
   step -- either rsync through a workstation or push to the remote, which is Kyle's call
   since it publishes to the shared lab repo. Then one episode on `devel` at 0.1x charge
   before anything larger, per the cluster rules.
4. **Full-scale collection.** Sized once the per-episode cost is settled by (2).
5. **`finetune.py`**, with `--method {analytic,ppo}`. Analytic is the method that works
   today; PPO is the one Kyle wants to get working to get away from an implementation
   that is hacky and only suits a simple task.
6. **Gate 3** (rollout horizon) in the training path.

## Open questions, not blocking

- **Does truncation bias the corpus?** Episodes that drift most get cut shortest, so
  surviving data over-represents low-drift behaviour. v2 sizing makes truncation rare,
  which shrinks the problem, but the residual rate should be reported per corpus rather
  than assumed to be zero.
- **Is there a machine effect on CRM at all?** The claim was withdrawn as unproven, not
  established. a3 cannot run the reference, so a like-for-like replication needs a third
  machine that can, or the question stays open. It costs nothing to keep comparisons
  within one machine meanwhile.
- **What is the real variance of the tracking metric?** 5.7 points is a floor, measured by
  perturbing spawn alone. A corpus varies command, soil realisation and initial state too,
  so the honest number should come from the episode distribution rather than one knob.
- **The four unexplained HMMWV divergences** in `docs/SOIL.md`: `free_surface_threshold`
  2.0 against the design doc's 0.8, `num_proximity_search_steps` 4 where the doc said not
  to raise it without comparing force and sinkage, our 4x coarser MBS/CFD coupling, and
  the uncalibrated active-domain sizes on both sides. Worth raising with Zhang rather than
  silently adopting either side.
- **Their bed is 0.25 m deep**, above the ~0.22 m at which we measured a CRM bed heaving
  and carrying bodies. Measured at our spacing and mass, so uncertain for theirs.
