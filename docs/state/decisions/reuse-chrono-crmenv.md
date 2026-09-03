# Reuse `chrono_crmenv.py` instead of reimplementing it

**Assessed 2026-09-03 on `kyle-sbel`. Recommendation: do it. Not yet done.**

## Why this came up

`scripts/quadruped_go2_crm.py` reimplements the checkpoint's input contract by
hand: joint reorder, sign flip, observation scaling, and the command slot. Its
docstring justifies this as *"a PORT, not a reuse. That work runs
`bochengzou::pychrono`; everything here runs the `nedm` environment."*

**That justification does not survive inspection, and it was never checked.**

## `bochengzou::pychrono` is an install instruction, not an artifact

There is **no bochengzou environment on either box** and no package from that
channel anywhere. It appears exactly once in the whole 2025 tree, in
`README.md:11`, as a `conda install` line. Local `pychrono` provenance is
9.0.0 (`chrono` env, local tarball) and 10.0.0 `py312h98ab86c_677` from
`projectchrono` (`nedm`, `nedm-src`).

So the "different simulator" that justified reimplementation **was never
instantiated here**. A channel name in someone else's README was carried as a
technical constraint for the length of the port. Same error class as the Genesis
attribution: **an inherited label read as a fact about the world.**

Whether that channel holds a fork or a vanilla build is now **moot** — the code
runs against ours, per the import test below.

## The observation path needs NOTHING. That is the whole result.

`_compute_observations` (`chrono_crmenv.py:475-510`) touches **no Chrono API at
all** — it is pure torch over buffers, and it contains the reorder, the
negation, the scaling and the hardcoded command. State reaches it only through
the robot wrapper (`get_base_pos`, `get_joint_pos`, `get_contact_force`, …), and
`Robots.py` reaches Chrono only via `GetChMotor`,
`CastToChLinkMotorRotation` and `GetChBody` — **none of which moved** between
10.0.0 and the pinned SHA; all three were called successfully today.

**So the contract defining the checkpoint's input stays byte-identical under
reuse.** Every blocker is in setup or display.

## Four blockers, all setup or visualisation

| # | blocker | fix |
|---|---|---|
| a | `import pychrono.vsg` (`chrono_crmenv.py:8`), `import pychrono.irrlicht` (`Robots.py:2`) | make optional — **display only**, the RL loop needs neither |
| b | `fsi.ChFsiVisualizationVSG` | same; among the 11 VSG symbols absent from our build |
| c | `SetElasticSPH`, `SetActiveDomainDelay`, `ElasticMaterialProperties` | **already shimmed** in `src/nedm/chrono_crm_compat.py` |
| d | `terrain.GetSystemFSI` | **new fourth rename** → `GetFsiSystemSPH` / `GetFluidSystemSPH`; add to the shim |

Our build has VSG off deliberately, because we render through Sensor.

## What reuse buys

Three of the four hand-maintained conventions stop being ours, and **one of them
we cannot explain**: the sign flip is a blanket negation of all twelve joint
positions and velocities at `chrono_crmenv.py:491-492`, with no comment and no
config field, and it is **not** the URDF axes (all twelve are clean —
hips `1 0 0`, thighs and calves `0 1 0`, no mirroring).

**Not maintaining a convention we cannot explain is strictly better than
maintaining it.** There is no test today that would catch it drifting.

The reorder is worth understanding too, because it is not what our docstring
claims. **Three orderings exist:**

| ordering | source |
|---|---|
| `[FL, FR, RL, RR]` | the URDF's actual declaration order |
| `[RR, RL, FR, FL]` | **hand-typed** `motor_name_list`, `Robots.py:35-38` and `153-156` |
| `[FR, FL, RR, RL]` | the RL library's canonical layout |

The URDF's order **never participates**, because the harness addresses joints by
name. So "Chrono orders joints `[RR, RL, FR, FL]`" attributes to the simulator
what is one developer's typing order. **The reorder is an accident of
authorship** — had that list been typed in the RL order, there would be nothing
to map.

## Carry it as a patch, not a copy

Making the imports optional means editing someone else's file, so reuse is a
fork of `chrono_crmenv.py` however small. **Carry it as a patch against a
recorded upstream commit**, the same discipline as `patches/0001` against
`chrono-src`, rather than copying the file and letting it drift.

## The verification gate for the switch

**The switch is only proven faithful if the Go2 reproduces its current
behaviour.** The reference numbers exist, measured on the source build:

```
travel 2.5623 m   max tilt 10.2 deg   base_z_end 0.5307
```

and that run is **bit-reproducible** on the source build, so the comparison has
a noise floor of exactly zero. Anything other than those digits means the switch
changed the observation path, which is the one thing it must not do.

Run on `--soil training`, since that is the preset the policy was finetuned on
and therefore in-distribution.
