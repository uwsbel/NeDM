# Diagnostics

Everything here answers a question about the pipeline or the physics. None of it is on
the collect, train, fine-tune, evaluate path, which lives one level up:

```
quadruped/collect.py     corpus from Chrono          quadruped/merge_corpus.py   shard merge
quadruped/train.py       NN-ROM                      quadruped/paired_eval.py    paired analysis
quadruped/finetune.py    analytic / PPO / ensemble   quadruped/doctor.py         preflight
quadruped/evaluate.py    paired Chrono scoring
```

Moved here from `quadruped/` on 2026-09-22 rather than deleted, because the docs cite
several of these as the source of recorded numbers, and a citation to a script that no
longer exists cannot be checked. Run them from the repository root, for example
`python quadruped/diagnostics/obs_truth.py ...`; each resolves `quadruped/` and `src/`
from its own location.

## Checks worth rerunning

| script | question | when to run |
|---|---|---|
| `obs_truth.py` | Does the observation rebuilt from a corpus row match what the policy actually received, block by block? | After any change to collection, `ObsBuilder` or `lib/policy.py`. Every block must agree to rounding. It found the post-step capture bug (`07f440a4`). |
| `obs_check.py` | Does the base policy, fed rebuilt observations, reproduce its recorded outputs across a whole corpus? | On a new corpus, before fine-tuning. `finetune.py` also runs this at start and refuses if it fails. |
| `corpus_check.py` | Is a corpus good enough to fit a model that will be differentiated (the four gates)? | After every collection. |
| `horizon_sweep.py` | errdist against horizon for a trained NN-ROM; `--restamp` writes the profile into the checkpoint. | For a checkpoint trained before per-checkpoint profiles existed. |
| `chaos_floor.py` | How fast Chrono diverges from ITSELF (twin episodes differing only by GPU rounding): the floor no surrogate can beat. | When a surrogate's long-horizon error needs a reference. Result: 0.003 at 2 s against ~1.0 for one-step surrogates. |
| `namecheck.py` | Names read in a function that nothing binds, before they cost a GPU-minute. | On any edited Python file: `python quadruped/diagnostics/namecheck.py <files>`. |
| `walk_check.py` | Does the policy walk, on rigid ground and on CRM? | After a policy, URDF or actuation change. |
| `establish_sign.py` | The joint sign convention, established by physics and written into `params/policy.yaml`. | Only if the sign is ever unset; `collect.py` refuses to run without it. |

## One-off studies (results recorded in the docs)

| script | question | where the answer lives |
|---|---|---|
| `patch_cost.py` | What does a bigger CRM patch cost? | `docs/COST.md` |
| `active_domain_study.py` | How small can the active domain get before it changes the physics? | `docs/COST.md` |
| `active_domain_ensemble.py` | Does the active domain bias behaviour, decided over an ensemble? | `docs/COST.md`, `results/active_domain_ensemble*.json` |
| `analyse_ensemble.py` | Pool the ensembles and decide the box size (0.5 m). | `docs/COST.md`, `docs/STATE.md` |
| `settling_test.py` | Is the soil settled when the robot arrives? (Yes; 0.1 s free flow is enough.) | `docs/COST.md` |
| `tracking_spread.py` | How much does the tracking number move between runs that should be identical? | `docs/EVALUATION.md`, `docs/SOIL.md` |
| `machine_probe.py` | Do two machines disagree about CRM from the first step? (No: bit-identical for two steps.) | `docs/EVALUATION.md` |
