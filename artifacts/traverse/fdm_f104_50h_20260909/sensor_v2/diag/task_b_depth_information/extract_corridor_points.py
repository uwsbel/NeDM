"""Task B step 1: extract the depth-side quantities the risk network actually sees, per corridor point.

Read-only. Uses scripts/sensor_dataset.py itself (the deployed v1 sampler) so the pixel addresses, the bilinear
interpolation, the sec channel and the route-start range d0 are bit-for-bit what the D arm was trained on.

Per corridor point we store
    dep   d_i      raw Euclidean ray range read from the 1024x1024 depth image at the corridor pixel
    sec   s_i      sec(ray angle) at that pixel (channel 6 is s_i - 1)
    d0    d0       range at (station 0, lateral 16) of the route -- the value v1 subtracts and never passes
    s0    s0       sec at that same route-start pixel (the model does see this, at corridor cell [0,16])
    zpix  z_i      pinhole height of the observed point: H - d_i / s_i   (exact, camera model only)
    elev  v1 height channel (rgbd elevation * elevation_scale) at the same world address
    xbp,ybp        back-projected world position ray_dir * axial range
    zsim           simulator terrain height at (xbp, ybp)   <-- EVALUATION REFERENCE ONLY
    gx,gy          the flat-ground world address v1 uses to pick the pixel

  python extract_corridor_points.py --route-stride 2 --station-stride 2 --lateral-stride 2
"""
import argparse, glob, json, math, os, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
import sensor_dataset as SD                                    # the deployed v1 sampler
from nedm.traverse.terrain import TerrainMap                   # evaluation reference only

MAPS = ROOT / 'artifacts/traverse/fdm_f104_50h_20260909/sensor_v1/maps'
V1 = ROOT / 'artifacts/traverse/fdm_f104_50h_20260909/sensor_v1'
ARENAS = {'arena_f104_50h_v1': 'f104', 'arena_g203': 'g203', 'arena_g216': 'g216',
          'arena_g217': 'g217', 'arena_g228': 'g228', 'arena_g231': 'g231'}


def route_files(tag):
    out = []
    for wave in ('test', 'test2'):
        out += sorted(glob.glob(str(V1 / f'{wave}_{tag}' / 'routes' / '*.json')))
    return out


def corridor_fields(wp, st):
    """Everything tensor10 computes on the depth side, without the slope/speed channels."""
    gx, gy, grid = SD.corridor(np.asarray(wp, float), np.asarray(st, float))
    G = SD.G
    rowW = G['ctrW'] - gy / G['mppW']
    colW = G['ctrW'] + gx / G['mppW']
    depth = SD.bilinear(G['depth'], rowW, colW).astype(np.float64)
    sec = np.sqrt(1 + ((colW - G['ctrW']) / G['f']) ** 2 + ((rowW - G['ctrW']) / G['f']) ** 2)
    valid_px = (np.abs(gx) < 40.0) & (np.abs(gy) < 40.0)
    dvalid = valid_px & (depth < G['max_depth'] - 1e-3)
    # exactly sensor_dataset.tensor10's d0 rule
    if dvalid[0, SD.N_LATERAL // 2]:
        d0 = depth[0, SD.N_LATERAL // 2]
    elif dvalid[0].any():
        d0 = float(np.nanmean(np.where(dvalid[0], depth[0], np.nan)))
    else:
        d0 = G['cam_h']
    s0 = sec[0, SD.N_LATERAL // 2]
    # v1 height channel at the same world address (512-px encoded elevation)
    row = G['ctr'] - gy / G['mpp']
    col = G['ctr'] + gx / G['mpp']
    elev = SD.bilinear(G['rgbd'][3], row, col).astype(np.float64) * G['elev_scale']
    # back-projection (correct geometry)
    ray_x = (colW - G['ctrW']) / G['f']
    ray_y = (G['ctrW'] - rowW) / G['f']
    axial = depth / sec
    return dict(gx=gx, gy=gy, dep=depth, sec=sec, d0=float(d0), s0=float(s0), dvalid=dvalid,
                elev=elev, zpix=G['cam_h'] - axial, xbp=ray_x * axial, ybp=ray_y * axial,
                ray_x=ray_x, ray_y=ray_y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--route-stride', type=int, default=2)
    ap.add_argument('--station-stride', type=int, default=2)
    ap.add_argument('--lateral-stride', type=int, default=2)
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent / 'points'))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for arena, tag in ARENAS.items():
        SD.init_map(str(MAPS / arena))
        terrain = TerrainMap.from_dir(ROOT / 'assets/traverse' / arena)
        files = route_files(tag)
        # --- d0 over EVERY route/group (cheap: only the corridor's first station is needed)
        d0_all, s0_all, grp_all, arm_all = [], [], [], []
        for f in files:
            r = json.load(open(f))
            c = corridor_fields(r['waypoints'], r['stations'])
            d0_all.append(c['d0']); s0_all.append(c['s0'])
            grp_all.append(r['meta']['scene_id'])
            arm_all.append(os.path.basename(f).split('__')[1][:-5])
        # --- full corridor point cloud for a route subsample
        cols = {k: [] for k in ('dep', 'sec', 'd0', 's0', 'zpix', 'elev', 'xbp', 'ybp', 'zsim',
                                'gx', 'gy', 'ray_x', 'ray_y', 'station', 'lateral', 'route')}
        for ri, f in enumerate(files[::a.route_stride]):
            r = json.load(open(f))
            c = corridor_fields(r['waypoints'], r['stations'])
            ss = slice(None, None, a.station_stride); ls = slice(None, None, a.lateral_stride)
            m = c['dvalid'][ss, ls]
            st_i, lat_i = np.meshgrid(np.arange(SD.N_STATION)[ss], np.arange(SD.N_LATERAL)[ls], indexing='ij')
            take = lambda k: c[k][ss, ls][m]
            zs = terrain.height(take('xbp'), take('ybp'))
            n = int(m.sum())
            for k in ('dep', 'sec', 'zpix', 'elev', 'xbp', 'ybp', 'gx', 'gy', 'ray_x', 'ray_y'):
                cols[k].append(take(k))
            cols['zsim'].append(zs)
            cols['d0'].append(np.full(n, c['d0'])); cols['s0'].append(np.full(n, c['s0']))
            cols['station'].append(st_i[m]); cols['lateral'].append(lat_i[m])
            cols['route'].append(np.full(n, ri))
        data = {k: np.concatenate(v).astype(np.float32) for k, v in cols.items()}
        np.savez_compressed(out / f'{tag}.npz', **data,
                            d0_all=np.asarray(d0_all, np.float32), s0_all=np.asarray(s0_all, np.float32),
                            group_all=np.asarray(grp_all, object), arm_all=np.asarray(arm_all, object))
        summary[tag] = dict(arena=arena, n_routes_total=len(files),
                            n_routes_sampled=len(files[::a.route_stride]), n_points=int(len(data['dep'])),
                            n_groups=len(set(grp_all)))
        print(tag, summary[tag], flush=True)
    json.dump(summary, open(out / 'extract_summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
