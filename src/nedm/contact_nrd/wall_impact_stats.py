"""Ball state at the wall impact: training data against the moved-wall cohort (exploratory check)."""
import json
import sys
from pathlib import Path

import numpy as np


def stats(root):
    with np.load(Path(root) / "unified_data.npz") as d:
        states, contacts = d["states"], d["contacts"]
    rows = []
    for n in range(len(states)):
        k = np.flatnonzero(contacts[n, :, 1])
        if len(k):
            s = states[n, k[0], 0]
            rows.append((k[0] * 1e-3, s[2], s[3], s[5], s[7]))   # time, z, vx, vz, spin_y
    a = np.array(rows)
    return {name: [float(np.quantile(a[:, i], q)) for q in (0.0, 0.05, 0.5, 0.95, 1.0)]
            for i, name in enumerate(("time_s", "z_m", "vx_mps", "vz_mps", "spin_y_radps"))}


if __name__ == "__main__":
    print(json.dumps({root: stats(root) for root in sys.argv[1:]}, indent=1))
