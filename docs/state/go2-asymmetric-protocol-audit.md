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
