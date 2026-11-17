# Architecture ablation: one network or three, transformer or MLP

**Follow-on work, not part of the paper.** The contact NRD ([`contact_nrd/`](../README.md))
is three transformer networks: core, collision and contact. This study asks which part of that design
matters, with a 2 × 2:

| | Three networks | One network |
| --- | --- | --- |
| **Transformer trunk** | 3T (the contact NRD) | 1T |
| **MLP trunk** | 3M | 1M |

Only the trunk (the part that mixes the tokens) changes between T and M; only the number of networks
changes between 3 and 1.

**[`training_setup.md`](training_setup.md) compares how every model was trained**: data, sizes,
networks, schedule, selection, and what is not matched.

## Code

Everything runs from this repository alone, on `nedm.contact_nrd`; no simulator.

The source is in [`src/nedm/contact_nrd/arch_ablation/`](../../src/nedm/contact_nrd/arch_ablation), one module per architecture:

| Module | Contents |
| --- | --- |
| [`arch_3t.py`](../../src/nedm/contact_nrd/arch_ablation/arch_3t.py) | The 3T: the contact NRD and its trainer, unchanged |
| [`arch_3m.py`](../../src/nedm/contact_nrd/arch_ablation/arch_3m.py) | The 3M: the contact NRD with its three trunks replaced by residual MLPs of the same size; trained by `nedm.contact_nrd.train` itself with the trunks swapped |
| [`arch_1t.py`](../../src/nedm/contact_nrd/arch_ablation/arch_1t.py) | The 1T: the causal transformer backbone and its sizes |
| [`arch_1m.py`](../../src/nedm/contact_nrd/arch_ablation/arch_1m.py) | The 1M: the 1T with a residual MLP in place of the attention blocks, and its sizes |
| [`common.py`](../../src/nedm/contact_nrd/arch_ablation/common.py) | Shared by 3M and 1M: the residual MLP. Shared by all: the motion score |
| [`one_net.py`](../../src/nedm/contact_nrd/arch_ablation/one_net.py) | Shared by 1T and 1M: the network around the backbone, and the adapter that puts a one-network model behind the contact NRD's calls |
| [`one_train.py`](../../src/nedm/contact_nrd/arch_ablation/one_train.py) | Shared by 1T and 1M: the contact NRD's co-training with one network |
| [`checkpoints.py`](../../src/nedm/contact_nrd/arch_ablation/checkpoints.py) | The four architectures by name; checkpoint loading for all four |
| [`train.py`](../../src/nedm/contact_nrd/arch_ablation/train.py) | `python -m nedm.contact_nrd.arch_ablation.train --arch 3t\|3m\|1t\|1m` |
| [`test.py`](../../src/nedm/contact_nrd/arch_ablation/test.py) | `python -m nedm.contact_nrd.arch_ablation.test`: all four on a test directory, by the contact NRD's test |

The scripts are in [`scripts/contact_nrd/arch_ablation/`](../../scripts/contact_nrd/arch_ablation), one folder per study case:

| Path | Contents |
| --- | --- |
| [`reproduce.sh`](../../scripts/contact_nrd/arch_ablation/reproduce.sh) | One case end to end: download, 3M, 1T, 1M, test |
| `<case>/train.sbatch` | The case's three training runs as one Slurm array job (4 h limit; submit again to resume): [`bouncing_ball`](../../scripts/contact_nrd/arch_ablation/bouncing_ball/train.sbatch), [`pool`](../../scripts/contact_nrd/arch_ablation/pool/train.sbatch), [`so101_push_t`](../../scripts/contact_nrd/arch_ablation/so101_push_t/train.sbatch), [`g1_locomotion`](../../scripts/contact_nrd/arch_ablation/g1_locomotion/train.sbatch) |
| `<case>/*.py` | The case's figures, from the stored results of the original runs (also [`go2/`](../../scripts/contact_nrd/arch_ablation/go2)); [`layout.py`](../../scripts/contact_nrd/arch_ablation/layout.py) says where those are |

The tests are in [`tests/contact_nrd/arch_ablation/`](../../tests/contact_nrd/arch_ablation), one folder per study case:
`<case>/test_networks.py` gives the case's system and the sizes its four networks must have, and runs the checks of
[`checks.py`](../../tests/contact_nrd/arch_ablation/checks.py) on it (sizes, calls and checkpoints; CPU, no data).

The 3T is not retrained: it is the released contact NRD checkpoint, or `python -m nedm.contact_nrd.train`.

## Data and results

Both are kept in `artifacts/` of the checkout, as the contact NRD's own data and runs are. `artifacts/` is not in git.

| Folder | Contents |
| --- | --- |
| `artifacts/contact_nrd/<case>/{train,test,models}/` | The released contact NRD data and 3T checkpoints (`reproduce.sh` downloads them; `DATA_ROOT` points at an existing copy) |
| `artifacts/contact_nrd/arch_ablation/runs/<case>_<arch>_s<seed>/` | The runs of `reproduce.sh` |
| `artifacts/contact_nrd/arch_ablation/<case>/` | Scores and figures: `test_scores_nedm.json` from `reproduce.sh`, and the stored results of the original runs |
| `artifacts/contact_nrd/arch_ablation/videos/` | The three videos of the original runs |

The stored results of the original runs were copied from the NRDLab data folder (`$NRDLAB_DATA/arch_ablation/`,
default `~/Documents/nrdlab/data`). The original runs themselves (`$NRDLAB_DATA/runs/...`, named in the case READMEs)
are not copied.

## Running

```bash
scripts/contact_nrd/arch_ablation/reproduce.sh pool            # likewise bouncing_ball, so101_push_t, g1_locomotion
scripts/contact_nrd/arch_ablation/reproduce.sh pool 1t test    # single steps
TRAIN_ARGS=--smoke scripts/contact_nrd/arch_ablation/reproduce.sh bouncing_ball 3m 1t 1m   # a pipeline check, a few updates
PYTHONPATH=src python -m unittest discover -s tests/contact_nrd/arch_ablation -t .   # or one case: tests.contact_nrd.arch_ablation.pool.test_networks
sbatch scripts/contact_nrd/arch_ablation/pool/train.sbatch   # on a cluster: the case's 3M, 1T and 1M as one array job
```

A full run takes several hours on one GPU; an interrupted run continues from its last check when the same
command is run again. An existing run folder is always continued, so give `RUN_TAG` when a setting changes (the
smoke check uses `_smoke` by itself). Needs `torch` and `numpy` (`environment.nedm.yml` or `requirements.txt`), plus
`huggingface_hub` for the download.

## The systems

| System | Folder | NeDM implementation | Stored results of the original runs |
| --- | --- | --- | --- |
| Bouncing ball | [`bouncing_ball/`](bouncing_ball/README.md) | 3M, 1T, 1M | 3T, 3M, two 1Ts, 1M, in variants that are not ported |
| Pool | [`pool/`](pool/README.md) | 3M, 1T, 1M | 3T, 3M, 1T, 1M, in variants that are not ported |
| SO101 push-T | [`so101_push_t/`](so101_push_t/README.md) | 3M, 1T, 1M | 3T, 3M, 1T, 1M |
| G1 locomotion | [`g1_locomotion/`](g1_locomotion/README.md) | 3M, 1T, 1M | 3T, 3M, 1T, 1M |
| Go2, rigid ground | [`go2/`](go2/README.md) | not ported | 3T, 1T, 1M (the 3M run was not finished) |

Each folder's README is the record of that case's original runs.

## What the NeDM implementation is, and is not

The study was first run with another code base, NRDLab, whose contact NRD objective reproduces
`nedm.contact_nrd.train`. `nedm.contact_nrd.arch_ablation` is a port of that study onto NeDM's own contact NRD code.

**Checked**

- The networks have the sizes of the original runs to the parameter: SO101 3M 2 420 086, 1T 2 487 496,
  1M 2 421 128; G1 3M 2 531 477, 1T 2 506 632, 1M 2 530 376.
- All three architectures train, check, resume and finish on a subset of each of the four cases, and the test
  scores all four, including a released 3T checkpoint.
- The unit tests pass.

**Not checked**

- No full-length run has been made with the port, so the stored numbers have not been reproduced with it. A
  resumed or repeated run is not bit-identical to the original in any case: the random draws differ.

**Not ported**

- **Go2.** NeDM has no Go2 model, data loader or objective.
- **The original ball and pool variants.** The port trains one design on every case, the size-matched one of
  G1 and SO101. The ball's and the pool's stored results come from other variants (unmatched sizes, and for the
  ball the core network alone as the one-network model); see their READMEs.
- **The unified loss** of the ball case.
- **Two choices of the original G1 and SO101 runs.** The port's 3M is selected on the official score, as its
  trainer is the 3T's; the original G1 3M was selected on the motion score. The original SO101 models were
  trained on the larger push-T corpus and compared with that corpus's 3T; `reproduce.sh` does that only with the
  settings in its header.
- **Error-against-horizon evaluation and the pool rollouts.** The stored `horizon_errors.*`, `results.json` and
  `rollouts.npz` were written by the original scripts; the figure scripts only redraw from them.

The 1T-against-1M comparison on the paper's own models (HMMWV, M113, arm) is not part of this study.
