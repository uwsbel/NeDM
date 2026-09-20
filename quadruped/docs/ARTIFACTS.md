# Artifacts: data, models, results

**Bytes never enter git. Manifests always do.**

`.git` is already 502 MB with artifacts tracked, one NeDM checkpoint is 911 MB, and a
corpus runs to tens of GB. Committing those makes the repository unusable and LFS only
relocates the problem. What we actually need from version control is TRACEABILITY, and a
manifest provides that completely: given one, the bytes can be located, verified, and the
command that produced them re-run.

## Layout

```
  data/<corpus_name>/manifest.json         committed
  data/<corpus_name>/bytes -> ...          gitignored, a path or symlink
  models/<model_name>/manifest.json        committed
  results/<run_id>/manifest.json           committed
  results/<run_id>/*.json                  committed if small, else referenced
```

`run_id` is `YYYYMMDD-HHMMSS-<short hash>` and is the primary key. Every artifact
references the `run_id`s of its inputs, so any result walks back to the exact data and
model that produced it.

## The manifest schema, identical everywhere

```json
{
  "run_id":       "20260920-174455-a1b2c3d",
  "kind":         "corpus | model | result",
  "created_utc":  "2026-09-20T17:44:55Z",
  "git": { "commit": "...", "branch": "...", "dirty": false },
  "command":      ["collect.py", "--corpus", "..."],
  "host":         { "name": "...", "gpu": "...", "cores": 16 },
  "chrono_build": { "path": "...", "md5": "..." },
  "versions":     { "python": "3.12.x", "torch": "...", "chrono": "..." },
  "inputs":       [ { "run_id": "...", "kind": "corpus", "sha256": "..." } ],
  "outputs":      [ { "path": "...", "sha256": "...", "bytes": 123456 } ],
  "metric_defs":  { "errdist": "v2", "val_loss": "v2-sampled" },
  "notes":        "free text"
}
```

### Why `dirty` is a field

A manifest whose `git.dirty` is true records a result that cannot be reproduced from any
commit. That is acceptable while iterating and disqualifying for anything quoted. Tools
that publish a number check this field.

### Why `metric_defs` is a field

A metric silently changed meaning mid-study -- `val_loss` moved from a prefix of the
validation split to a random sample on 2026-09-17 -- and every comparison spanning that
date was invalidated without anything looking wrong. Versioning the definition turns that
into a visible mismatch instead of a wrong number.

### Why `chrono_build` is a field

Replay is not build-invariant, which is why `crm_verdict.py` already refuses to pair
scores across builds. Recording the hash lets that check happen automatically rather than
by memory.

## Corpus manifests carry their excitation

A corpus manifest additionally records the excitation actually applied, not the flags
requested:

```json
  "excitation": {
    "action_injection": { "kind": "ou", "sigma_range": [..], "tau_s": 0.15,
                          "per_joint_independent": true },
    "probes":           { "fraction": 0.15, "kind": "single_joint_chirp" },
    "push":             { "enabled": true, "direction": "uniform_sphere",
                          "magnitude_range_n": [.., ..], "events_per_episode": 2 },
    "initial_state":    { "randomised": true, "bounds_ref": "params/excitation.yaml#init" }
  },
  "segmentation": { "split_at_pushes": true, "guard_steps": 2,
                    "segments": 1840, "rows_dropped": 12043 },
  "failures":     { "truncated_episodes": 31, "by_check": { "joint_step": 19, ... } }
```

The excitation block exists so a corpus can be audited for admissibility after the fact.
The reference corpus of the previous study was 83% pushed and nothing in its metadata made
that visible at a glance.
