#!/usr/bin/env python
"""Diagnose imagined energy: which z1 channels drift, and which power feature set transfers
from recorded data to imagined rollouts (judged against the Chrono candidate energies)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nedm.traverse.nrd_model import DT_S
from nedm.traverse.power_calib import KINDS, PowerModel, backward_accel

dump, scores, chrono_rows, calib, cache = [Path(a) for a in sys.argv[1:6]]
d = np.load(dump)
z1, act, active, head_p = d["z1"], d["act"], d["active"], d["power"]  # (T, N, ...)
keys, cands = d["keys"], d["candidates"]
T, N = active.shape
ch = {(r["key"], r["candidate"]): r for r in map(json.loads, chrono_rows.read_text().splitlines()) if "tracker" in r["controller"]}
NAMES = ["vx", "vy", "roll", "pitch", "roll_rate", "avy", "yaw_rate", "Fz1", "Fz2", "Fz3", "Fz4", "w1", "w2", "w3", "w4"]

# --- channel drift on the recorded-route rollouts vs the real recorded z1 -------------
print("z1 channel: imagined vs recorded on the same route (recorded-route rollouts, active steps)")
err, bias, scale = np.zeros(15), np.zeros(15), np.zeros(15)
n_ep = 0
for i in np.nonzero(cands == "recorded")[0]:
    with np.load(cache / f"{keys[i]}.npz") as c:
        real = c["z1"]
    t_act = active[:, i].sum()
    real_seg = real[16:16 + t_act]; img_seg = z1[:t_act, i]
    m = min(len(real_seg), len(img_seg))
    err += np.abs(img_seg[:m] - real_seg[:m]).mean(0); bias += (img_seg[:m] - real_seg[:m]).mean(0)
    scale += real_seg[:m].std(0) + 1e-6; n_ep += 1
for j, nme in enumerate(NAMES):
    print(f"  {nme:9s} MAE {err[j] / n_ep:8.3f}  bias {bias[j] / n_ep:+8.3f}  (recorded std {scale[j] / n_ep:.3f})")

# --- power feature sets: recorded-fit models applied to imagined rollouts --------------
ax = np.zeros_like(z1[..., 0]); ax[1:] = (z1[1:, :, 0] - z1[:-1, :, 0]) / DT_S
common = [i for i in range(N) if (keys[i], cands[i]) in ch]
e_ch = np.array([ch[(keys[i], cands[i])]["energy_kj"] for i in common])
t_ch = np.array([ch[(keys[i], cands[i])]["time_s"] for i in common])
print(f"\n{len(common)} candidates with Chrono energy; Chrono mean {e_ch.mean():.1f} kJ")
def report(name, e_img):
    e = e_img[common]
    # per-episode combined-cost pick agreement (time + energy/10), energy on Chrono scale via mean ratio
    ratio = e_ch.mean() / e.mean()
    agree = n_ep2 = 0
    for k in set(keys[common]):
        idx = [j for j, i in enumerate(common) if keys[i] == k and cands[i] != "recorded"]
        if len(idx) < 2: continue
        n_ep2 += 1
        ci = [t_ch[j] * 0 + d_time[common[j]] + ratio * e[j] / 10 for j in idx]
        cc = [t_ch[j] + e_ch[j] / 10 for j in idx]
        agree += int(np.argmin(ci) == np.argmin(cc))
    print(f"  {name:10s} imagined {e.mean():6.1f} kJ  ratio {ratio:.2f}  corr {np.corrcoef(e, e_ch)[0, 1]:+.3f}  pick agree {agree}/{n_ep2}")
d_time = active.sum(0) * DT_S
report("head", (head_p * active).sum(0) * DT_S)
for kind in KINDS:
    pm = PowerModel.load(calib, kind)
    p = pm.predict(z1, act, ax, np)  # (T, N)
    report(kind, (p * active).sum(0) * DT_S)
    report(kind + "|0", (np.maximum(p, 0) * active).sum(0) * DT_S)
