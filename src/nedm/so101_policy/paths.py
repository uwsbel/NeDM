"""Default locations of the data of the SO101 pick-and-place release (so101_pick_place/README.md).

One rule for every default input path of this package and its scripts:

- Files on the Hub (checkpoints, task sets, start banks, geometry, NRD data, robot recordings) are read from
  ``DATA = <repository>/artifacts/so101_pick_place``, the Hub folder ``so101_pick_place/`` after
  ``scripts/so101_pick_place/reproduce.sh --steps download``, with the Hub's sub-folders (``nrd/``,
  ``rl_inputs/``, ``teacher/``, ...).
- Small files in git (camera calibration, scene, servo fit, LeRobot motor calibration) are read from
  ``CONFIGS = <repository>/configs/so101_pick_place``.

Paths inside the JSON configs use the same two folders, written relative to the repository root
(``artifacts/so101_pick_place/...``, ``configs/so101_pick_place/...``): run the commands from the repository root.
To keep the data on another disk, link ``artifacts/`` there. The study code read the same files from home-folder
copies (``~/so101_policy/ckpt``, ``~/so101_r2s_claude``); the files are the same (sha256 in release_manifest.json).
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "artifacts" / "so101_pick_place"
CONFIGS = REPO / "configs" / "so101_pick_place"
