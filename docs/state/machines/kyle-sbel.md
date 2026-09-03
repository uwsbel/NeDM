# kyle-sbel

**Verified:** 2026-09-02 · **Owner:** Kyle · **Role:** Development, docs,
manuscript builds, light training and analysis.

| | |
|---|---|
| GPU | NVIDIA RTX 3090, 24 GB |
| CPU / RAM | 16 cores / 30 GB |
| Disk (free) | 1.8 T total, **1.3 T free** |
| Repo path | `/home/kyle/Documents/sbel/NeDM` |
| Interpreter | `/home/kyle/miniconda3/envs/chrono/bin/python` (**pychrono 9.0.0**, torch 2.6.0+cu124, CUDA available) |
| Reachable from | local only |

```bash
export NEDM_ROOT=/home/kyle/Documents/sbel/NeDM
export NEDM_PY=/home/kyle/miniconda3/envs/chrono/bin/python
export PYTHONPATH=$NEDM_ROOT/src
```

## What this machine is for

Reading and writing code, docs, and the manuscript; small training runs and
analysis; driving work on other boxes. Sibling repos live alongside it under
`/home/kyle/Documents/sbel/` — `Manuscripts`, `chrono_fork`, `chrono_hil`,
`ccta`, `sbel-reproducibility`.

## Constraints

This is the **only** machine available (2026-09-02). The project convention is
that raw dataset collection happens on `newton` or Euler, not on a training box
— but neither is reachable from here, so if collection is genuinely needed it
either happens locally or it is blocked. There is disk for it (1.3 T free);
what is missing is the 4090/32-core throughput the pilot tiers were sized
against. Budget from the measured rates in
[`reference/newton.md`](reference/newton.md) and expect worse.

Anything requiring `newton`, the 5090 box, or Euler is a **blocker to
escalate**, not a step to attempt. See [`reference/`](reference/).

## Gotchas

1. **`git-lfs` is not installed.** Checkpoint `.pt` files under `artifacts/` are
   LFS pointer stubs, not weights. Install `git-lfs`, then
   `git lfs install && git lfs pull`, before expecting any checkpoint to load.
2. **The `chrono` env is not equivalent to `nedm`, despite what the name
   suggests.** There is no `nedm` env on this box and never was. `chrono` carries
   **pychrono 9.0.0**, installed from a local tarball
   (`~/Downloads/pychrono-9.0.0-py310_4853.tar.bz2`, conda channel shows as
   `<unknown>`), which is older than the 10.0.0 in `environment.nedm.yml` *and*
   older than the 9.0.1 in `environment.yml` that is meant to be the legacy
   option. Use `$NEDM_PY`, never a hardcoded env name, and do not assume a
   version from either environment file. Verified 2026-09-02.
3. **No `pychrono.fsi`, so no CRM terrain on this box.** Submodules shipped are
   cascade, core, fea, irrlicht, pardisomkl, parsers, postprocess, robot, ros,
   sensor, vehicle. Any CRM work needs a source build of Chrono with FSI and
   Python bindings enabled, or C++. This blocks the CRM half of the proposed
   quadruped case study; see
   [`../progress/future-case-studies.md`](../progress/future-case-studies.md).
4. **`pychrono.parsers` is broken, not absent.** `_parsers.so` is present but
   `ldd` reports `libament_index_cpp.so` and `liburdfdom_model.so.3.0`
   unresolved: it was built against ROS2 and urdfdom 3.0, neither installed
   here. So `ChParserURDF` is unavailable, which blocks importing a Go2 URDF.
   `pychrono.robot` is unaffected and RoboSimian works.
5. **`newton` is not resolvable from here** (`Temporary failure in name
   resolution`, 2026-09-02). No SSH host entry and/or not on the network. Work
   destined for newton must be pushed to GitHub and pulled there by someone with
   access.
6. Local Chrono checkout is at `/home/kyle/Documents/sbel/chrono_fork/chrono`,
   which is the source tree, not the pychrono package the env uses.
