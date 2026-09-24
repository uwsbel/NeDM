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
  /mnt/nas/Main/nedm/archive/<name>/    retired trees, with SHA256SUMS and a README
```

`INDEX.md` at `/mnt/nas/Main/nedm/` describes what is in each directory.

**Archived: north's `~/sbel-artifacts` (2026-09-24).** The old pipeline's output
directory (branch `kyle/locomotion`, 2026-09-03 to 2026-09-20; 95,217 files, 214.5 GB:
old corpora, training datasets and runs, the imported `go2_cts_150k.pt`, the crmtrack and
finetune_crm results) is now at `/mnt/nas/Main/nedm/archive/sbel-artifacts-north/data/`.
It was verified file by file against SHA256SUMS taken on north before the copy, then
deleted from north; the empty `north:~/sbel-artifacts/` holds an `ARCHIVED.md` pointing
here. The 4,496 symlinks (all pointing back inside the tree) are listed in
`SYMLINKS.tsv` rather than copied; the archive's README has the restore command. The old
pipeline's scripts on `kyle/locomotion` hard-code `~/sbel-artifacts/...` in 167 files, so
on north they now find nothing until the needed parts are restored. The `sbel-artifacts`
directories on sbel, a3 and d33 are separate, different trees and were not touched.

Copying from north: north is on a home network, not the campus LAN, so its route to the
NAS is its home upload (~2-3 MB/s on the wire). Writing to the CIFS mount directly costs a
WAN round trip per file (~1.1 MB/s for many small files); rsync over ssh to the NAS host
`sliger-ubuntu` with `-z --compress-choice=zstd` reached ~5 MB/s effective. Verify on
sliger, which reads the NAS over its own LAN.

euler and hpcfund do not see the NAS and keep their own roots. Anything that must survive
goes to the NAS, because euler's backup status is unknown and hpcfund's `$WORK` has none.
