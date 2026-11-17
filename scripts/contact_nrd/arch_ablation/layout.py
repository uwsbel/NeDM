"""Where the ablation's results are kept: ``artifacts/contact_nrd/arch_ablation/<system>/`` of this checkout (scores
and figures; not in git).

``NRDLAB_RUNS`` is used by one figure only (the pool's training curves): the runs of the original implementation,
``$NRDLAB_DATA/runs`` (else ``$WORK/Documents/nrdlab/data/runs``, else ``~/Documents/nrdlab/data/runs``).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "artifacts" / "contact_nrd" / "arch_ablation"
NRDLAB_RUNS = Path(os.environ.get("NRDLAB_DATA") or Path(os.environ.get("WORK") or Path.home()) / "Documents" / "nrdlab" / "data") / "runs"
