# SO101 pick and place: results on the real arm

A camera policy picks a 30 mm cube and places it on a plate with a real SO101 arm. The arm-and-cube dynamics were
distilled from CPU MuJoCo episodes into NeDM's [contact NRD](../contact_nrd/README.md). A state teacher was trained
inside that NRD, then a camera student learned from the teacher using ray-traced images of a calibrated scene.
The student ran on the real arm without further training. This is follow-on work, separate from the NeDM paper.

No real image was used as a training sample. Real recordings informed camera and scene calibration, table height,
servo ranges and robot limits; later training rounds responded to observed robot failures. Student selection also
used robot screening. The released student is `tb10` (`v12_tb10_s1210`, `it_01000`), trained with a table 10 mm lower
than the nominal simulation table.

NeDM keeps the results and pinned release links here. The study's implementation, configs, requirements,
training instructions, checkpoints and evidence are on
[Hugging Face](https://huggingface.co/datasets/harryzhang1018/NeDM/tree/main/so101_pick_place).

## Real-arm results

The operator labelled each episode; “placed” means the operator typed `p`. Intervals are Wilson 95 %.
The stress test was run on 2026-10-27 after student selection.

| Test | Placed | 95 % interval |
|---|---|---|
| Released student tb10, stress test | **36/50 = 72 %** | 58–83 % |
| Plate left of the cube (R→L) | 19/23 = 83 % | 63–93 % |
| Plate right of the cube (L→R) | 17/27 = 63 % | 44–78 % |
| Reference student D, sessions 2026-10-24 to 10-27 | 19/28 = 68 % | 49–82 % |

The six stress sessions scored 8/10, 2/2, 5/8, 7/10, 8/10 and 6/10. One run stopped at home before the policy ran
and is excluded. The 14 failures were six carries without release at the plate, five failed grasps and three
releases short of the plate. Cubes within 19 cm of the base were placed in 1/7 episodes.

D's 19/28 includes two `p` labels whose notes say off-mat or wrong location; excluding those gives 17/28.
The students were tested on different episodes, so these counts do not establish an improvement over D.
[The robot result tables](results/robot_results.md) retain the per-episode labels and the checkpoint records.
Directions and distances come from a released image-and-kinematics estimate whose generating script was not found.

## Selection and simulation limits

- tb10 failed its round's frozen simulation rule: nominal-table success was 3.5 % and the +3 mm table check had
  94 places, below its 144.9 threshold. A lower-table student was expected to fail the nominal-table check.
- A later lab rule exempted lower-table students from the nominal-table check and lowered the +3 mm placement
  threshold to 90. Earlier gate results were already known when that rule was written. tb10 was screened on the
  robot (4/4 placed) and then chosen by the user for the stress test.
- The 50 stress episodes were new, but the student choice used robot results and the stress protocol was not
  fixed in advance. There is no sealed student simulation test.
- One robot, one lab, one cube and one plate were tested. Videos contain the student's logged 112 px camera
  images; there is no full-resolution third-person recording.
- Session logs identify the bundle folder, rather than recording a checkpoint hash from the robot computer.
  The checkpoint mapping comes from the study's local bundle records.

## NRD evaluation

Both released NRDs load with main's unchanged contact-NRD model code. Two training-only geometry heads were
removed from the original checkpoints; prior release checks found bit-identical rollouts on 256 validation
episodes, on CPU and GPU. Every teacher and student used the fine-tuned policy NRD (`c1_s62`).

The 3,050-episode scripted MuJoCo test was evaluated on these two checkpoints on 2026-10-27. The selected-family
metrics below score 2,000 episodes in seven families; the all-family rows score all 3,050. The 1 s windows start
every 0.5 s from recorded four-state histories.

| Cube-position error | Base NRD c0 | Policy NRD c1 |
|---|---|---|
| Selected families, 1 s windows, median / p95 | 0.16 / 3.48 mm | 0.22 / 4.13 mm |
| Same families on validation, median / p95 | 0.16 / 3.26 mm | 0.21 / 4.04 mm |
| Selected families, full 4 s rollout, median / p95 | 1.20 / 9.41 mm | 1.37 / 11.03 mm |
| All 3,050 test episodes, 1 s windows, p95 | 3.84 mm | 4.61 mm |

The test contains only scripted families and is not fully blind: an earlier NRD design was evaluated on it before
the base NRD's configuration was made. The policy NRD's mixed scripted/expert/noisy/policy validation has no
matching test set. These errors do not certify the camera student's behavior or physical success.

## Release and reproduction

[`release_manifest.json`](release_manifest.json) pins the source and data independently, with SHA-256 checksums.
The original 503-file data manifest and stage provenance remain on Hugging Face. The source archive preserves
the original implementation bytes and executable permissions; unpack it over the exact base revision below.

```bash
pip install huggingface_hub
python - <<'PY'
import hashlib, json, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
m = json.loads(Path("so101_pick_place/release_manifest.json").read_text())
c = m["code"]
p = Path(hf_hub_download(m["repo_id"], c["archive"]["path"], repo_type=m["repo_type"],
                         revision=c["hf_revision"]))
assert p.stat().st_size == c["archive"]["bytes"]
assert hashlib.sha256(p.read_bytes()).hexdigest() == c["archive"]["sha256"]
shutil.copyfile(p, "source.tar.gz")
PY
GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout https://github.com/uwsbel/NeDM.git so101-study
cd so101-study
GIT_LFS_SKIP_SMUDGE=1 git checkout --detach dacc64968c006340393f60e7f1e15a2b65fc4aa7
tar -xzf ../source.tar.gz --strip-components=1
git fetch origin so101-pick-place-study
```

Run the block from the NeDM repository root. In the restored checkout,
`so101_pick_place/README.md` has the full original training and deployment guide. Install its GPU dependencies
and `requirements-so101-pick-place.txt`, then rebuild the robot tables from the archived logs:

```bash
bash scripts/so101_pick_place/reproduce.sh full --steps download,robot
```

This command reads archived logs. To repeat the NRD test, use `full --steps download,nrd_test`; full training is
documented in the archived guide. Three teacher stages and the base NRD used uncommitted code, so their reruns
are recipes rather than exact reconstructions. Their available code snapshots and run provenance are preserved.

Source files can also be browsed in the pinned
[Hugging Face code folder](https://huggingface.co/datasets/harryzhang1018/NeDM/tree/6039542eea5eb96bd925f64dcdaf94d647d5d67c/so101_pick_place/code); its
[reproduction guide](https://huggingface.co/datasets/harryzhang1018/NeDM/blob/6039542eea5eb96bd925f64dcdaf94d647d5d67c/so101_pick_place/code/REPRODUCE.md) explains the overlay. The original Git study history remains on
[`so101-pick-place-study`](https://github.com/uwsbel/NeDM/tree/so101-pick-place-study) for historical training steps.
The existing data, checkpoints, robot bundle, logs and videos retain their original pinned data revision.
