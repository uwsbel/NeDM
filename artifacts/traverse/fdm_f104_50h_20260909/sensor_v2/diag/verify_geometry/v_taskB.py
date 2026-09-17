"""Independent check of Task B: what the v1 raw-depth tensor (delta_d, sec) does and does not determine.

Corridor points are re-extracted here from the raw captures + the saved test/test2 routes; nothing is read
from the task_b_depth_information/points cache.  TerrainMap is an evaluation reference only.

  /home/harry/miniconda3/envs/nedm/bin/python <this file>
"""
import json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path("/home/harry/NeDM-traverse_mppi")
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from nedm.traverse.terrain import TerrainMap
import sensor_dataset as SD

BASE = ROOT / "artifacts/traverse/fdm_f104_50h_20260909/sensor_v1"
OUT = Path(__file__).parent
ARENAS = {"f104": "arena_f104_50h_v1", "g203": "arena_g203", "g216": "arena_g216",
          "g217": "arena_g217", "g228": "arena_g228", "g231": "arena_g231"}
S = 511.0 / 512.0


def bil(img, row, col):
    n0, n1 = img.shape[-2], img.shape[-1]
    r0 = np.clip(np.floor(row).astype(int), 0, n0 - 2); c0 = np.clip(np.floor(col).astype(int), 0, n1 - 2)
    fr = np.clip(row - r0, 0, 1); fc = np.clip(col - c0, 0, 1)
    return (img[..., r0, c0] * (1 - fr) * (1 - fc) + img[..., r0, c0 + 1] * (1 - fr) * fc
            + img[..., r0 + 1, c0] * fr * (1 - fc) + img[..., r0 + 1, c0 + 1] * fr * fc)


def extract(tag, arena):
    """Per corridor point: measured ray range d, sec, the route-start d0/sec0, and the back-projected height."""
    SD.init_map(str(BASE / f"maps/{arena}"))
    H = SD.G["cam_h"]; f = SD.G["f"]; ctrW = SD.G["ctrW"]; mppW = SD.G["mppW"]
    depth = SD.G["depth"].astype(np.float64)
    tm = TerrainMap.from_dir(ROOT / "assets/traverse" / arena)
    cols = defaultdict(list)
    files = sorted((BASE / f"test_{tag}/routes").glob("*.json")) + sorted((BASE / f"test2_{tag}/routes").glob("*.json"))
    for ri, fp in enumerate(files):
        r = json.load(open(fp))
        wp = np.asarray(r["waypoints"], float); st = np.asarray(r["stations"], float)
        gx, gy, _ = SD.corridor(wp, st)
        rw = ctrW - gy / mppW; cw = ctrW + gx / mppW
        d = bil(depth, rw, cw)
        rx = (cw - ctrW) / f; ry = -(rw - ctrW) / f
        sec = np.sqrt(1 + rx ** 2 + ry ** 2)
        ok = (d < SD.G["max_depth"] - 1e-3) & (np.abs(gx) < 39.5) & (np.abs(gy) < 39.5)
        d0 = d[0, 16]; s0 = sec[0, 16]
        if not ok[0, 16]:
            continue
        ax = d / sec
        z = H - ax
        xb = rx * ax; yb = ry * ax
        m = ok & (np.abs(xb) < 39.5) & (np.abs(yb) < 39.5)
        grp = fp.name.split("__")[0]
        cols["d"].append(d[m]); cols["sec"].append(sec[m]); cols["z"].append(z[m])
        cols["d0"].append(np.full(int(m.sum()), d0)); cols["s0"].append(np.full(int(m.sum()), s0))
        cols["z0"].append(np.full(int(m.sum()), H - d0 / s0))
        cols["group"].append(np.full(int(m.sum()), hash(grp) % (1 << 31)))
        cols["route"].append(np.full(int(m.sum()), ri))
        cols["station"].append(np.tile(np.arange(96)[:, None], (1, 32))[m])
        cols["lateral"].append(np.tile(np.arange(32)[None, :], (96, 1))[m])
        cols["xb"].append(xb[m]); cols["yb"].append(yb[m])
        cols["zsim"].append(tm.height(xb[m] * S, yb[m] * S))
    return {k: np.concatenate(v) for k, v in cols.items()}, len(files)


def within_bin_std(keys, target, groups, min_groups=2):
    """Std of `target` inside bins of `keys`, keeping only bins spanned by >= min_groups distinct groups."""
    order = np.lexsort(keys[::-1])
    k = np.stack([a[order] for a in keys], 1)
    t = target[order]; g = groups[order]
    newbin = np.any(np.diff(k, axis=0) != 0, axis=1)
    bid = np.r_[0, np.cumsum(newbin)]
    nb = bid[-1] + 1
    cnt = np.bincount(bid, minlength=nb)
    s1 = np.bincount(bid, weights=t, minlength=nb)
    s2 = np.bincount(bid, weights=t * t, minlength=nb)
    # distinct groups per bin
    gmin = np.full(nb, np.inf); gmax = np.full(nb, -np.inf)
    np.minimum.at(gmin, bid, g); np.maximum.at(gmax, bid, g)
    multi = (cnt >= 2) & (gmax > gmin)
    var = np.maximum(s2[multi] / cnt[multi] - (s1[multi] / cnt[multi]) ** 2, 0)
    w = cnt[multi]
    pooled = float(np.sqrt((var * w).sum() / w.sum()))
    return pooled, int(multi.sum()), int(w.sum())


def main():
    res = {}
    pool = {}
    for tag, arena in ARENAS.items():
        P, nf = extract(tag, arena)
        H = 110.0
        dd = P["d"] - P["d0"]
        zz = P["z"] - P["z0"]
        # identity checks
        idn = (H - P["z0"]) * (P["sec"] - P["s0"]) - zz * P["sec"]
        # (a) ambiguity given (delta_d, sec)
        kd = np.round(dd / 0.02).astype(np.int64)
        ks = np.round(P["sec"] / 2e-4).astype(np.int64)
        ks0 = np.round(P["s0"] / 2e-4).astype(np.int64)
        a_std, a_bins, a_n = within_bin_std([kd, ks], zz, P["group"])
        b_std, b_bins, b_n = within_bin_std([kd, ks, ks0], zz, P["group"])
        # (c) one-constant closed form, Qbar fitted on f104
        res[arena] = dict(
            routes=nf, points=int(dd.size),
            identity_max_abs=float(np.abs(idn - dd).max()),
            z_from_H_minus_d_over_sec_vs_sim_mae=float(np.abs(P["z"] - P["zsim"]).mean()),
            target_std=float(zz.std()),
            ambiguity_delta_d_sec=[a_std, a_bins, a_n],
            ambiguity_delta_d_sec_sec0=[b_std, b_bins, b_n],
            z0_std=float(np.unique(P["z0"]).std()), z0_min=float(P["z0"].min()), z0_max=float(P["z0"].max()),
            d0_mean=float(np.unique(P["d0"]).mean()), d0_std=float(np.unique(P["d0"]).std()),
            d0_min=float(P["d0"].min()), d0_max=float(P["d0"].max()),
            sens_dz_dd0_mean=float(np.abs(1 / P["s0"] - 1 / P["sec"]).mean()),
            sens_dz_dd0_max=float(np.abs(1 / P["s0"] - 1 / P["sec"]).max()),
            depth_rel_std=float(dd.std()),
            perspective_term_std=float(((H - P["z0"]) * (P["sec"] - P["s0"])).std()),
            terrain_term_std=float((zz * P["sec"]).std()),
            persp_over_terrain=float(((H - P["z0"]) * (P["sec"] - P["s0"])).std() / (zz * P["sec"]).std()),
        )
        pool[arena] = dict(dd=dd, sec=P["sec"], s0=P["s0"], zz=zz, z0=P["z0"],
                           station=P["station"], lateral=P["lateral"], route=P["route"], group=P["group"])
        print(f"{arena}: n={dd.size} identity {res[arena]['identity_max_abs']:.2e} | "
              f"std(z-z0) {res[arena]['target_std']:.3f} | amb(dd,sec) {a_std:.3f} ({a_bins} bins) | "
              f"amb(+sec0) {b_std:.4f} ({b_bins} bins) | persp/terr {res[arena]['persp_over_terrain']:.2f}", flush=True)

    # one-constant closed form, Q fitted on f104 only
    f = pool["arena_f104_50h_v1"]
    Q = float(np.mean(110.0 - f["z0"]))
    # least-squares Q on f104
    A = (1 - f["s0"] / f["sec"]); b = f["zz"] + f["dd"] / f["sec"]
    Qls = float((A * b).sum() / (A * A).sum())
    cf = {}
    for arena, P in pool.items():
        pred = Q * (1 - P["s0"] / P["sec"]) - P["dd"] / P["sec"]
        predls = Qls * (1 - P["s0"] / P["sec"]) - P["dd"] / P["sec"]
        cf[arena] = dict(rmse_Qmean=float(np.sqrt(((pred - P["zz"]) ** 2).mean())),
                         rmse_Qls=float(np.sqrt(((predls - P["zz"]) ** 2).mean())))
    # linear readout conditioning on f104 (z-z0 ~ a*depth_rel + b*sec + c*sec0 + d)
    X = np.stack([f["dd"], f["sec"], f["s0"], np.ones_like(f["dd"])], 1)
    coef, *_ = np.linalg.lstsq(X, f["zz"], rcond=None)
    rms = float(np.sqrt(((X @ coef - f["zz"]) ** 2).mean()))
    sd = dict(depth_rel=float(f["dd"].std()), sec1=float((f["sec"] - 1).std()), sec1_0=float((f["s0"] - 1).std()))
    out = dict(per_arena=res, closed_form=dict(Q_mean_f104=Q, Q_lstsq_f104=Qls, rmse=cf),
               linear_readout_f104=dict(coef=[float(c) for c in coef], rmse=rms, channel_std=sd,
                                        weight_ratio=float(abs(coef[1] / coef[0])),
                                        scale_ratio=float(sd["depth_rel"] / sd["sec1"])))
    json.dump(out, open(OUT / "v_taskB.json", "w"), indent=1)
    print(json.dumps(out["closed_form"], indent=1))
    print(json.dumps(out["linear_readout_f104"], indent=1))


if __name__ == "__main__":
    main()
