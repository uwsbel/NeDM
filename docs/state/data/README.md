# Data tracking

**What datasets exist, how big, on which machine, and how to get them.**

Datasets are the least portable thing in this project — they are large,
per-machine, and mostly not in git. This registry is what makes a new machine
usable.

| File | Covers |
|---|---|
| [`published.md`](published.md) | The paper's five datasets (Hugging Face) |
| [`vision-line.md`](vision-line.md) | Study 1 and Study 3 datasets |

## Rules

1. **Record the machine for every copy.** "The pilot dataset" is not a location.
2. **Almost none of this is on `kyle-sbel`.** Every dataset below lives on a
   machine under [`../machines/reference/`](../machines/reference/), which is
   not reachable. Assume a dataset is absent until `du -sh artifacts/*` says
   otherwise, and treat "fetch dataset X" as **blocked**, not as a step.
   The Hugging Face datasets in [`published.md`](published.md) are the one
   exception — they download from anywhere.
3. **Move only the trainable form.** Compressed episode-chunked stores and
   processed caches travel; raw frame dumps and uncompressed memmaps do not.
4. **Check both sides before rsync** (project pattern; not runnable from here):
   ```bash
   ssh newton "du -sh ~/NeDM/artifacts/<name>"
   df -h "$NEDM_ROOT"
   ```
5. **Processed caches reference the raw store, never duplicate it**
   (`processed_caches: "reference"` in the Study 3 manifest).

## Template

```markdown
| Dataset | Episodes / size | Location(s) | Collected | Notes |
|---|---|---|---|---|
```
