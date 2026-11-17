# Pool (billiards): the 3T against its architecture ablations

> **Record of the original runs.** The numbers and figures below come from the first implementation of this
> ablation, in the NRDLab code base. Scripts, run folders and `nrdlab` commands named in this text are that
> implementation's and are not in this repository. The NeDM implementation is `nedm.contact_nrd.arch_ablation`
> ([`contact_nrd/arch_ablation/README.md`](../README.md)). Result files named here are in `artifacts/contact_nrd/arch_ablation/pool/`.

The contact NRD study's pool case (`ContactNRD-Pool`): a cue ball A strikes ball B on a 2.54 m x 1.27 m
table with four cushions; the test asks where ball B is 2.0 s after the shot. The study's model, the
**3T**, is three transformer networks (core, collision, contact; 2.40 M parameters); its released
seed 61 is used here. The three ablations change only the network, with the study's data, seed (61) and
full schedule (the original `pool_arch.sbatch`,
one MI300X each; `ARCH_VARIANTS` in NRDLab's `src/nrdlab/tasks/contact_nrd/cfg.py`):

| Model | Networks | Parameters | Run |
| --- | --- | --- | --- |
| 3T (released) | three transformers | 2.40 M | `contact_nrd/pool/seed61`, `seed62` (`nrdlab hub pull contact-nrd --part pool-models`) |
| 3Mlp | three MLPs | 5.28 M | `ContactNRD-Pool-3Mlp/train-model/2026-11-11_12-53-51_arch_s61` |
| 1Mlp | one MLP over both balls | 1.65 M | `ContactNRD-Pool-1Mlp/train-model/2026-11-11_17-24-46_arch_s61` |
| 1Transformer | one transformer over both balls | 0.80 M | `ContactNRD-Pool-1Transformer/train-model/2026-11-11_19-50-22_arch_s61` |

The sizes are not matched: the variants keep the study's network settings, so the 3Mlp is 2.2 times
the 3T and the one-network models are smaller. The runs are under `$NRDLAB_DATA/runs/`.

## Running

With the NeDM implementation, from the repository root ([`reproduce.sh`](../../../scripts/contact_nrd/arch_ablation/reproduce.sh)):

```bash
scripts/contact_nrd/arch_ablation/reproduce.sh pool   # download, 3m, 1t, 1m, test
```

The NeDM implementation trains the size-matched design of the G1 and SO101 cases (3M, 1T and 1M at the 3T's
2.40 M parameters), not the unmatched variants of the table above, so its numbers will differ from those below.

Figures and the video scene from the stored `results.json` and `rollouts.npz` (`training.png` also reads the
original runs' `metrics.jsonl` under `$NRDLAB_DATA/runs/ContactNRD-Pool-*/train-model/`):

```bash
python scripts/contact_nrd/arch_ablation/pool/figures.py   # horizon.png, endpoints.png, training.png, the table
python scripts/contact_nrd/arch_ablation/pool/scene.py     # scene.npz for the video
<python with pychrono> scripts/contact_nrd/arch_ablation/pool/chrono_video.py \
    artifacts/contact_nrd/arch_ablation/pool/scene.npz artifacts/contact_nrd/arch_ablation/videos/pool_arch.mp4
```

The original evaluation scored every model with the study's test (`test_report`) on the 4,800 test shots and kept
every shot's free rollout. The released 3T's numbers equal the published ones (seed 61: 0.69 / 2.41 mm, path
0.41 / 1.31 mm, 4,787 events right).

The video (`videos/pool_arch.mp4`, rendered by Chrono at 0.25x) shows one shot, the one at the median of ball B's
error at 2 s, once per model: the recorded shot solid (A white, B red), the model's see-through in its colour, the
camera following ball B from half a metre (the errors are millimetres; the grid on the felt is 10 cm).

## Results

| model | parameters | ball B at 2 s, median / p95 (mm) | path RMS, median / p95 (mm) | events right | score |
| --- | --- | --- | --- | --- | --- |
| 3T s61 | 2.40 M | **0.69 / 2.41** | **0.41 / 1.31** | 4787 / 4800 | **0.0032** |
| 3T s62 | 2.40 M | 0.64 / 2.52 | 0.38 / 1.44 | 4783 / 4800 | 0.0034 |
| 3Mlp | 5.28 M | 0.82 / 2.86 | 0.50 / 1.61 | 4780 / 4800 | 0.0039 |
| 1Mlp | 1.65 M | 4.63 / 13.15 | 2.99 / 7.86 | 0 / 4800 | 0.0671 |
| 1Transformer | 0.80 M | 5.74 / 16.41 | 3.53 / 9.31 | 0 / 4800 | 0.0711 |

* **Three networks or one decides it, not the trunk.** The 3Mlp is within 20 % of the 3T (0.82 vs
  0.69 mm median at 2 s); both one-network models are 7-8 times worse (`horizon.png`, `endpoints.png`).
* **Where the one-network models fail** (`horizon.png`): before the cue ball reaches ball B (about
  0.3 s), the three-network models keep the resting ball still (errors about 0.001 mm); the
  one-network models already move it by 0.03 to 1 mm, and the 1Mlp misplaces the cue ball from the
  first step. The contact network's explicit switch is what keeps a ball at rest until it is hit.
* **Events right is 0 for the one-network models** because they have no contact switches to score,
  not because they mistime contacts; their position errors are the comparison.
* **Not converged** (`training.png`): every ablation's validation score was still falling at its last
  check (the 3Mlp from 0.0056 to 0.0039 over the last 10,000 updates), and each kept that last check.
  The study's schedule, tuned for the 3T, ends before they settle, so longer training may narrow the
  gaps (untested).
