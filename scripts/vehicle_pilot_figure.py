"""Visual check of the vehicle-included pilot: rendered vehicle, exclusion zone, and the masked corridor."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, 'scripts')
import sensor_map_v2 as M2, vehicle_corridor as VC

V = 'artifacts/traverse/fdm_f104_50h_20260909/sensor_v2'
SURF, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
MARGIN = 1.5


def main():
    R = json.load(open(f'{V}/vehicle_pilot_results.json'))
    cases = [R[0], R[9], R[18]]
    fig, axes = plt.subplots(len(cases), 4, figsize=(19, 4.4 * len(cases)), facecolor=SURF)
    for row, r in enumerate(cases):
        g, arena = r['group'], r['arena']
        frame = f'{V}/vehicle_frames/{g}'
        with np.load(frame + '/observation.npz') as o:
            rgb, depth = o['rgb'], o['depth_m']
        pose = np.asarray(r['measured_pose'], float)
        grid_v = M2.grid_from_capture(frame)
        gf = np.load(f'{V}/grids/arena_{arena}/grid.npz')
        n = grid_v['z'].shape[0]; half = 40.0; mpp = 80.0 / 512
        # 1: rendered scene around the vehicle (sensor image, cropped)
        f = (1024 / 2) / np.tan(np.radians(47) / 2)
        u = 1024 / 2 + pose[0] / 110.0 * f; v = 1024 / 2 - pose[1] / 110.0 * f
        s = 60
        u = min(max(u, s), 1024 - s); v = min(max(v, s), 1024 - s)
        ax = axes[row, 0]
        ax.imshow(rgb[int(v - s):int(v + s), int(u - s):int(u + s)])
        ax.set_title(f'{g}\nsensor image, vehicle present', fontsize=10, color=INK); ax.axis('off')
        # 2: depth difference (vehicle-included minus vehicle-free) with the exclusion rectangle
        ax = axes[row, 1]
        dz = np.nan_to_num(grid_v['z']) - np.nan_to_num(gf['z'])
        w = 12.0
        ext = [pose[0] - w, pose[0] + w, pose[1] - w, pose[1] + w]
        clip = lambda v: int(min(max(v, 0), n))
        i0 = clip((pose[1] - w + half) / mpp); i1 = clip((pose[1] + w + half) / mpp)
        j0 = clip((pose[0] - w + half) / mpp); j1 = clip((pose[0] + w + half) / mpp)
        sub = dz[i0:i1, j0:j1]
        ext = [-half + j0 * mpp, -half + j1 * mpp, -half + i0 * mpp, -half + i1 * mpp]
        im = ax.imshow(sub, extent=ext, origin='lower', cmap='RdBu_r', vmin=-2, vmax=2)
        mask = VC.exclusion_mask(pose, MARGIN, n=n, mpp=mpp, half=half)
        globals()['_mask'] = mask
        ax.contour(np.linspace(ext[0], ext[1], sub.shape[1]), np.linspace(ext[2], ext[3], sub.shape[0]),
                   mask[i0:i1, j0:j1].astype(float), levels=[0.5], colors='#0b0b0b', linewidths=1.6)
        aff = (np.abs(dz) > 0.15) | ((grid_v['cover'] == 0) & (gf['cover'] > 0))
        ys, xs = np.where(aff[i0:i1, j0:j1])
        ax.scatter(ext[0] + (xs + .5) * mpp, ext[2] + (ys + .5) * mpp, s=1.2, c='#eda100', alpha=.5,
                   label='vehicle-affected cells')
        ax.legend(fontsize=7.5, frameon=False, loc='upper right')
        ax.set_title('height change from the vehicle (m)\nblack = exclusion zone (footprint + 1.5 m)', fontsize=10, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        # 3/4: corridor height channel of the chosen route, A vs C
        # rebuild this case's own corridor for the route the height model chose without the mask
        import sensor_dataset_v2 as V2, vehicle_pilot as VP
        case, pose_case, cands = VP.pool_for(f"{V}/cases_{arena}/cases/{g}.json", g)
        route = cands[r['H']['pick']['A']]
        V2.init_grid(f'{V}/grids/arena_{arena}')
        XA, _, _ = VC.tensor12_excluded(np.asarray(route['waypoints']), np.asarray(route['speeds']), np.asarray(route['stations']), None)
        V2.G.update(z=grid_v['z'], range_m=grid_v['range_m'], sec=grid_v['sec'], rgb=grid_v['rgb'], cover=grid_v['cover'],
                    mpp=grid_v['meta']['mpp'], half=grid_v['meta']['half_extent_m'], n=grid_v['meta']['n'],
                    cam_h=grid_v['meta']['camera_height_m'])
        XC, _, infoC = VC.tensor12_excluded(np.asarray(route['waypoints']), np.asarray(route['speeds']), np.asarray(route['stations']), mask)
        ex = {'XA': XA, 'XC': XC}
        for k, (tag, title) in enumerate((('A', 'corridor: vehicle-free, no mask'), ('C', 'corridor: vehicle present + mask'))):
            ax = axes[row, 2 + k]
            X = ex['XA'] if tag == 'A' else ex['XC']
            z = np.where(X[4] > 0.5, X[0], np.nan)
            im2 = ax.imshow(z.T, aspect='auto', origin='lower', cmap='BrBG_r', vmin=-1.5, vmax=1.5,
                            extent=[0, 96, -6, 6])
            extra = '' if tag == 'A' else f"  |  {100*infoC['invalid_fraction']:.1f}% of samples invalid, reference station {infoC['reference_station']}"
            ax.set_title(title + '\n(white = invalid, no terrain information)' + extra, fontsize=9.5, color=INK)
            ax.set_xlabel('station', fontsize=9, color=INK2); ax.set_ylabel('lateral offset (m)', fontsize=9, color=INK2)
    for ax in axes.ravel():
        ax.set_facecolor(SURF)
    fig.suptitle('Vehicle-included depth input: the exclusion zone covers the vehicle, and the corridor keeps its coordinates',
                 fontsize=14, color=INK, x=0.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(f'{V}/vehicle_pilot.png', dpi=125, facecolor=SURF)
    print('wrote', f'{V}/vehicle_pilot.png')


if __name__ == '__main__':
    main()
