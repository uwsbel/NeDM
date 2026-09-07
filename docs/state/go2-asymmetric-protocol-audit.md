# Audit: which reported results used the asymmetric protocol

The verdict harness runs the treated arm at `--action-mult k` and the baseline at
nominal. `go2-gain-confound.md` shows this moved the family split from +0.00854 to
+0.00003. This is the audit of prior results for the same defect, done **from a listing
of the artifacts on disk rather than from recall** -- the recalled count was one and the
actual count is eight.

```
   summary                       gain          matched?   recoverable from artifact?
   go2_verdict_armA_k0.60        0.60          no         filename convention only
   go2_verdict_armA_k0.65        0.65          no         filename convention only
   go2_verdict_armA_k075         0.75          no         filename convention only
   go2_verdict_armA_k0.85        0.85          no         filename convention only
   go2_verdict_armA_k0.90        0.90          no         filename convention only
   go2_verdict_anchor_sbel       UNKNOWN       unknown    NO -- no output dir either
   go2_verdict_v4_sbel           UNKNOWN       unknown    NO -- no output dir either
   go2_verdict_fresh3            0.75          no         control now running
   go2_verdict_fresh4            0.75          no         matched: +0.00003
   go2_verdict_fresh5            0.75          no         matched: +0.00154
```

**The five k-sweep runs are the discovery data for the family split.** The
pre-registration quotes their k=0.60/0.65/0.75 rows as the finding to be replicated.
Every one is armA-at-k against base-at-nominal, so the discovery itself was measured
under the protocol -- which the matched-gain result already implied and this makes
enumerable rather than inferred.

**`anchor_sbel` and `v4_sbel` cannot be classified at all.** Neither the summary nor a
surviving output directory records the gain, and the filename carries no k. Any claim
resting on them should say the comparison is unrecoverable.

## The provenance defect, and the fix

None of the eight summaries recorded which arms were compared. A file containing a
median paired difference, an interval and a McNemar p, with no record of the checkpoints
or the gains, is a number whose comparison cannot be reconstructed. The convention of
putting `k075` in the filename worked until two files did not use it.

Summaries now carry an `arms` block and the full `argv`:

```json
   "arms": {"treated_ckpt": "...go2_finetuned_exc25.pt", "treated_action_mult": 0.75,
            "baseline_ckpt": "...go2_cts_150k.pt", "baseline_action_mult": 1.0,
            "matched_gain": false}
```

`matched_gain` is the field to filter on. It would have made this audit a one-line query
instead of an archaeology exercise, and it is the same repair as recording collection
parameters in the episode sidecar: **the artifact should state the conditions that
produced it, because the filename, the directory and the operator's memory all fail
differently.**

## The gain had one record; the checkpoint had two

Following the coordinating session's check of its own artifacts, which have the identical
defect: the episode tree can recover the checkpoint and cannot recover the gain.

```
   parameter        records that exist                          auditable?
   checkpoint       filename AND controller.policy in every     YES, two records,
                    episode's collector_config.resolved.json    cross-checkable
   action mult      filename only                               NO, nothing to check
                                                                it against
```

**The parameter that turned out to be the confound was the one with nothing to
cross-check it against, and the parameter never in doubt had two records.**

The reason is worth stating because it will recur. `collector_config.resolved.json` is
named as though it captured the run's resolved parameters; it captures the parameters
that flow through the **config system**. `NEDM_ACTION_MULT` is applied at policy level
from the environment, outside that path, so it never entered the file. **A file that
looks like a complete record is complete for one subsystem, and the boundary is invisible
from inside the file.**

Both halves now fixed:

```
   summary        arms{treated_ckpt, treated_action_mult, baseline_ckpt,
                  baseline_action_mult, matched_gain} + argv
                  -- written on EVERY exit path including the aborts
   episode config effective_action_mult + action_mult_source
```

The episode config matters more than the summary: it is one file per run against many
per corpus, and the summary does not survive a cleaned output tree. The k-sweep episode
trees are gone, which is why those five runs are recoverable only by filename convention.

## Verification, constructed rather than assumed

The `matched_gain` field was verified on the case it was expected to fail, per the rule
that produced it:

```
   armA @0.75  vs base @1.0     matched_gain false   expected false   OK
   armA @nom   vs base @1.0     matched_gain true    expected true    OK
   base @nom   vs base @1.0     matched_gain true    expected true    OK
   filter on matched_gain -> excludes exactly the asymmetric run
```

Constructing the true case surfaced a separate defect: the summary was written only on
the success path, so the two matched runs produced **no file at all** -- armA diverges at
nominal gain, the harness correctly returned NOT MEASURABLE, and the provenance went with
it. **The runs most in need of a record were the ones writing none.** Aborts now write a
stub carrying the arms, the argv, and the reason.
