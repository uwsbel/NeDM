#!/usr/bin/env python
"""Compare imagined candidate scores (NRD rollout) with the same candidates driven in Chrono.

Per (episode, candidate): imagined time/energy vs Chrono time/energy under the PPO tracker.
Per episode: does the imagined pick (min time + energy/10 among safe candidates) match the
Chrono pick, and how far off is it (regret) when it doesn't? Also Spearman rank agreement.
"""
import json, sys
from pathlib import Path
import numpy as np

img = json.load(open(sys.argv[1]))["rows"]
rows = [json.loads(l) for l in Path(sys.argv[2]).read_text().splitlines() if l.strip()]
ctl = sys.argv[3] if len(sys.argv) > 3 else "wp3_tracker_v1"
ch = {(r["key"], r["candidate"]): r for r in rows if ctl in str(r["controller"])}
im = {(r["key"], r["candidate"]): r for r in img}
common = sorted(set(ch) & set(im))
print(f"chrono rows ({ctl}): {len(ch)}  imagined rows: {len(im)}  common: {len(common)}")
t_i = np.array([im[k]["time_s"] for k in common]); t_c = np.array([ch[k]["time_s"] for k in common])
e_i = np.array([im[k]["energy_kj"] for k in common]); e_c = np.array([ch[k]["energy_kj"] for k in common])
ok = np.array([ch[k]["completed"] for k in common])
print(f"chrono completed {ok.sum()}/{len(ok)}  contact {sum(ch[k]['contact'] for k in common)}")
print(f"time  : imagined {t_i[ok].mean():.2f}s chrono {t_c[ok].mean():.2f}s  corr {np.corrcoef(t_i[ok], t_c[ok])[0,1]:.3f}  MAE {np.abs(t_i-t_c)[ok].mean():.2f}s")
print(f"energy: imagined {e_i[ok].mean():.1f}kJ chrono {e_c[ok].mean():.1f}kJ  corr {np.corrcoef(e_i[ok], e_c[ok])[0,1]:.3f}  ratio {e_c[ok].mean()/e_i[ok].mean():.2f}")

def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1] if len(a) > 2 else np.nan

keys = sorted({k for k, _ in common})
cost = lambda r, scale: r["time_s"] + scale * r["energy_kj"] / 10.0
agree, regret, rho_t, rho_c, n_ep = 0, [], [], [], 0
agree_t, regret_t, detail = 0, [], []
for key in keys:
    cands = [c for k, c in common if k == key and c != "recorded"]
    if len(cands) < 2:
        continue
    n_ep += 1
    # imagined pick uses imagined energy; chrono pick uses chrono energy, rescaled to the imagined energy scale
    # so the two rank on the same objective (energy ratio handled by the scale factor)
    ratio = e_c[ok].mean() / e_i[ok].mean()
    ci = {c: cost(im[(key, c)], 1.0) for c in cands}
    cc = {c: cost(ch[(key, c)], 1.0 / ratio) if ch[(key, c)]["completed"] else 1e9 for c in cands}
    pi, pc = min(ci, key=ci.get), min(cc, key=cc.get)
    agree += pi == pc
    regret.append(cc[pi] - cc[pc])
    ti = {c: im[(key, c)]["time_s"] for c in cands}
    tc = {c: ch[(key, c)]["time_s"] if ch[(key, c)]["completed"] else 1e9 for c in cands}
    pti, ptc = min(ti, key=ti.get), min(tc, key=tc.get)
    agree_t += pti == ptc
    regret_t.append(tc[pti] - tc[ptc])
    detail.append((key.split("__", 1)[1], pi, pc, round(cc[pi] - cc[pc], 2), pti, ptc, round(tc[pti] - tc[ptc], 2)))
    rho_t.append(spearman([im[(key, c)]["time_s"] for c in cands], [ch[(key, c)]["time_s"] for c in cands]))
    rho_c.append(spearman([ci[c] for c in cands], [cc[c] for c in cands]))
print(f"episodes with >=2 candidates: {n_ep}; imagined pick == chrono pick: {agree}/{n_ep}")
print(f"regret of imagined pick under chrono cost: mean {np.mean(regret):.2f}s-equiv, max {np.max(regret):.2f}")
print(f"time-only objective: imagined pick == chrono pick {agree_t}/{n_ep}; regret mean {np.mean(regret_t):.2f}s max {np.max(regret_t):.2f}s")
for d in detail:
    print("  ", d)
print(f"mean per-episode spearman: time {np.nanmean(rho_t):.2f}, combined cost {np.nanmean(rho_c):.2f}")
