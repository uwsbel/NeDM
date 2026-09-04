# Frozen collection provenance

Per-episode git provenance for collected datasets, captured at collection time.

## Why these live in `docs/` and not `artifacts/`

`artifacts/` is gitignored because it holds ~337 GB of episode CSVs and ~73 GB of
processed caches. These files are ~300 KB and their entire purpose is to survive,
so the rule that keeps bulk data out of git is the wrong rule for them.

## Why they exist at all

`reflog_head_at` recovers an episode's commit from its file **mtime**, and mtime
does not survive `cp -r`, `rsync` without `-a`, or archive-and-restore. Until the
mapping is written down, the provenance of a thousand episodes depends on a
filesystem attribute that ordinary handling silently discards.

Freezing it also makes the repair pass a **pure function of files**: re-running it
tomorrow would otherwise attribute commits differently from running it today,
because the input would be mtimes rather than data.

Keyed by episode **directory**, not `episode_id` — the ids were not unique, which
is the bug the repair exists to fix, so keying on them would have been circular.

## Note

The repair propagates `git_commit`, `git_tree` and `collection_code_digest` into
every episode's metadata, so these files are no longer the sole copy. They remain
the copy that is in version control.
