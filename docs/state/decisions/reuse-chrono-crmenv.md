# Reuse `chrono_crmenv.py` instead of reimplementing it

**Assessed 2026-09-03 on `kyle-sbel`. Recommendation: do it. Not yet done.**

## Why this came up

`scripts/quadruped_go2_crm.py` (now `src/nedm/quadruped/policy.py`) reimplemented
the checkpoint's input contract by
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

## DONE 2026-09-03, commit `133427b`. Gate passed on the exact digits.

```
           travel   max tilt   base_z_end
measured   2.5623      10.2       0.5307
target     2.5623      10.2       0.5307
```

Not close — **the same digits**, on a configuration where the source build is
bit-reproducible, so the noise floor is exactly zero.

**And a tighter check ran first.** The end-to-end gate could in principle pass
while both paths were wrong in the same way, so both observation
implementations were run on the **same random state** — random joint angles,
velocities, angular velocity, projected gravity, last actions — and compared
element by element:

```
max abs diff 0.000e+00    IDENTICAL across all 45 elements
```

Bit-identical rather than close. So the adapter's plumbing is provably correct
*independently of the simulation*, and the gate then confirms it in situ. **Two
levels of evidence, not one** — the unit check proves equivalence, the gate
proves it under load.

### What is inherited, and what is not

**Inherited byte-identical:** the permutation, the sign negation, the rest-pose
subtraction, and the four scale factors — the last of these read from the
harness's own `Config` rather than retyped, without being asked.

**Ours:** the adapter plumbing, thirteen named attributes populated from our
robot. **The math is theirs, the plumbing is ours.** That is a large reduction in
what can silently drift, and it is not zero — which is what the two checks above
exist to cover.

`chrono_crmenv.py` is imported **byte-identical**; `git status` on that tree is
clean. The hand-written path survives as `_observe_local`, unused, for removal in
a separate commit once the inherited path has run more than once.

### The `default_dof_pos` lead resolved, and our reimplementation was right

`genesis_defaults` is a **local variable** inside `_compute_observations` (line
495), not an attribute — which is why it never appeared in an attribute list
gathered by grepping for `self.`. `dof_pos` is filled **absolute** at line 413
straight from `get_joint_pos()`, and the rest-pose subtraction happens inside the
observation at line 505. Our reimplementation did the same thing.

**No discrepancy — and the audit still paid**, because it established what the
code expects rather than what we assumed, and the attribute list was incomplete
in a way that changed how the adapter must be populated. The adapter feeds
absolute angles, documented where someone might otherwise "fix" it.

### The stub trap, recorded because it gives no clue about its own cause

A stub raising on **every** attribute including dunders breaks the import
machinery itself. `inspect` reads `__file__`, gets the raiser function back, and
dies inside `importlib` with:

```
'function' object has no attribute 'endswith'
```

Nowhere near the display API, and looking nothing like a stub problem. Dunders
must behave normally; only real attribute lookups raise. Stubs raise **on use**,
never silently.

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
