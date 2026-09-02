# Published datasets (the paper)

**Updated:** 2026-09-02. Authoritative detail:
[`docs/progress.md`](../../progress.md) and
[`docs/hf_dataset_card.md`](../../hf_dataset_card.md).

Five datasets and four processed caches are on Hugging Face:
<https://huggingface.co/datasets/harryzhang1018/NeDM> — 70 GB, float32 Parquet
plus `.npy`.

```bash
"$NEDM_PY" scripts/release/download_nedm_datasets.py            # raw
"$NEDM_PY" scripts/release/download_nedm_datasets.py --processed # caches → artifacts/training_datasets/
"$NEDM_PY" scripts/release/download_nedm_datasets.py --rehydrate # rebuild per-episode CSV tree
```

`--rehydrate` reconstructs `artifacts/datasets/` so every script in the repo runs
unchanged. They can also be regenerated from scratch with the collection and
preprocessing scripts.

## What is and is not in git

| | |
|---|---|
| **In git** (~2 GB via LFS) | Checkpoints, run metadata, Chrono evaluation output, reference sets |
| **Not in git** | Raw episode CSVs (`artifacts/datasets/`, ~337 GB); processed cache arrays (`artifacts/training_datasets/`, ~73 GB) |

The tracked artifact tree is an **allowlist** in `.gitignore`, so a paper
artifact missing a rule shows up in `git status` rather than staying silently
untracked. Keep it that way.
