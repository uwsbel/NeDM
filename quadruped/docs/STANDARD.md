# Fleet standardisation

One env name, one Chrono source, one artifact root. Each rule fixes a confusion that has
already cost a result.

## 1. The env is called `nedm`, everywhere

| host | before | after |
|---|---|---|
| euler | venv `nedm` | unchanged |
| hpcfund | venv `nedm` | unchanged |
| sbel-ubuntu | conda **`nedm-src`** (plus an unused `nedm`) | conda `nedm` |
| north / a3 / d33 | conda `nedm` | unchanged |

Only sbel differs, and it carries BOTH names resolving to the same torch and pychrono,
which is the worst case: a script naming either one works, so nothing ever surfaces the
inconsistency. `nedm-src` is retired.

## 2. The env imports the PINNED SOURCE build, never conda-forge

Every desktop currently imports `pychrono-10.0.0-py312h98ab86c_677` from conda-forge,
md5 `8e9e386546fe0b33`, byte-identical across all four, while its own source build sits
on disk unused.

That binary is not the NeDM pin and does not carry patches 0001/0002. It has already
produced one wrong result: a capacity arm scored -9.5% on sbel against this build where
euler read -39.6% on the same policy. The identical hash everywhere made it LOOK like the
consistent, safe choice, which is exactly why it survived unnoticed.

Nothing needs recompiling. All four desktops already sit at pin `698282895` and already
have a build of it:

```
  host          pinned build                     md5
  sbel-ubuntu   ~/Documents/sbel/chrono-build    3b0bd530d06f54a4
  north-ubuntu  ~/chrono-build                   d1d0bd0ac8ee990a
  a3-ubuntu     ~/chrono-build                   cfbf8af6bcd2ec7d
  d33-ubuntu    ~/chrono-hip-build               53102025cf46c612
```

Same source, same commit, different toolchains, so hashes differ per host. That is fine.
What matters is that **one corpus is collected against one hash**, which `collect.py`
enforces by writing it into the manifest and refusing to append an episode carrying a
different one.

### Why a `.pth` and not PYTHONPATH

`PYTHONPATH` is an environment variable, so it is something to forget, and euler already
demonstrates the failure mode: its venv cannot import pychrono at all unless
`activate-nedm.sh` was sourced first. A `.pth` file makes the build part of the env.

But a `.pth` APPENDS to `sys.path` and conda's `pychrono` lives in `site-packages`, which
comes first. So the conda package must be removed, not merely shadowed:

```
  pip uninstall -y pychrono
  echo /path/to/chrono-build/bin > $SITE_PACKAGES/nedm_chrono.pth
```

`doctor.py` then asserts the resolved `_core.so` hash matches this host's entry in
`machines.yaml`, and fails the run outright if it sees the trap hash.

## 3. Artifacts live on the NAS

`//100.99.2.50/Main` is mounted at `/mnt/nas/Main` on **all four desktops**: 30 TB,
29 TB free, 3% used. A second share exists at `/mnt/nas/ProtectArchive`.

This solves two problems at once. Corpora and checkpoints stop competing with local disk
-- north is 90% full with 89 GB left and could not otherwise hold a collection run -- and
a corpus collected on one desktop is readable by the others without copying.

```
  /mnt/nas/Main/nedm/data/<corpus>/     bytes, referenced by the committed manifest
  /mnt/nas/Main/nedm/models/<model>/
  /mnt/nas/Main/nedm/results/<run_id>/
```

euler and hpcfund do not see the NAS and keep their own roots. Anything that must survive
goes to the NAS, because euler's backup status is unknown and hpcfund's `$WORK` has none.
