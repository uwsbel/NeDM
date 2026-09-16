# The scoring environment, per machine

**A CRM verdict is only comparable to another verdict that used the same Chrono build.**
On kyle-sbel the two available builds differ by **53%** on the identical policy
(`w_h15r0`: mae_vx 0.0788 under the source build, 0.1685 under conda). Nothing about the
policy, the episodes, or the code changes between those two numbers.

## kyle-sbel: TWO pychrono installs, and PYTHONPATH picks one silently

| build | path | md5 | what it is |
|---|---|---|---|
| **source (use this)** | `/home/kyle/Documents/sbel/chrono-build/bin` | `3b0bd530d06f54a4` | Chrono SHA `698282895` + patches, the project pin |
| conda (do NOT use) | `miniconda3/envs/nedm/.../pychrono` | `8e9e386546fe0b33` | released `pychrono-10.0.0` |

Activating the `nedm` conda env and running the scorer picks the **conda** build, because
it is on `sys.path` by virtue of being installed in the active env. Nothing warns you.

```bash
cd /home/kyle/Documents/sbel/NeDM
export PYTHONPATH=/home/kyle/Documents/sbel/chrono-build/bin:$PWD/src
export NEDM_CHRONO_PYTHONPATH=/home/kyle/Documents/sbel/chrono-build/bin
export NEDM_REPO=$PWD
PY=/home/kyle/miniconda3/envs/nedm-src/bin/python     # nedm-src, NOT nedm

$PY scripts/evaluation/score_crm_tracking.py \
    --policy <policy_ts.pt> \
    --index  /home/kyle/sbel-artifacts/datasets/go2_crm_merged/score_subset_index.json \
    --split val --concurrency 4 --out crmtrack_<arm>_$(hostname -s).json
```

Note the interpreter differs too: the verdict harness uses the **`nedm-src`** env, not
`nedm`. `score_crm_tracking` passes `sys.executable` to the episode subprocess, so the
interpreter you launch with is the one that resolves pychrono.

## euler

One build, selected explicitly, so there is nothing to get wrong beyond forgetting it.

```bash
source /srv/home/kasha2/nedm/env-scoring.sh
export NEDM_CHRONO=/srv/home/kasha2/chrono-build-parsers   # Go2 needs ChParserURDF;
                                                           # plain chrono-build lacks it
export PYTHONPATH=$NEDM_CHRONO/bin:/srv/home/kasha2/nedm/NeDM/src:$PYTHONPATH
# torch is the cu124 wheel set, so its CUDA runtime libs live inside the venv and
# pychrono cannot dlopen libnvrtc.so.12 without this:
VENVNV=/srv/home/kasha2/venvs/nedm/lib/python3.12/site-packages/nvidia
for d in "$VENVNV"/*/lib; do export LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"; done
```

Euler builds from the same source pin (`6982828`) but produces a different binary
(md5 `4489e0f802125279`) because it is compiled independently. That is expected: the
**source SHA** is the thing that must match, not the md5.

Partition choice matters. A CRM verdict cannot checkpoint, so it belongs on the
non-preemptible `sbel` tier (priority 400, node `euler19`, 4x A100). The preemptible
`research` tier is fine for short work such as a fine-tune, which is minutes long.

## hpcfund

Staged but **not usable for verdicts yet**: pychrono imports, torch does not exist in the
venv, and it would need the ROCm build.

## The guard that now enforces this

`score_crm_tracking.py` probes the pychrono the episode subprocess will actually resolve
-- by asking the same interpreter with the same environment, which is the only
trustworthy way -- then:

1. prints `pychrono <path>  md5 <hash>` before scoring starts,
2. stamps `chrono_md5` and `chrono_path` into **every** episode record, so
   "which build produced these numbers?" is a grep rather than an unanswerable question,
3. **exits FATAL** if `NEDM_CHRONO_PYTHONPATH` is set but the import resolved elsewhere.

Point 3 catches the operator who believes they pinned the build and did not. Point 2
catches everything else, after the fact.

## Why this file exists

On 2026-09-16 a day was spent investigating a 21.8% "baseline drift", a GPU-driver
theory, and a "machine regression" on kyle-sbel. None of it was real. Every sbel verdict
that day had been scored against the conda Chrono while the September numbers and euler
used the source build.

`run_go2_finetune_verdict.py` already documented this exact trap in a docstring, and it
was read, understood, and fallen into anyway. **Documentation alone was not sufficient**,
which is why the guard above exists. Treat this file as the reference, not the defence.
