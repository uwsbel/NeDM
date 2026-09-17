"""Write range-limited copies of the v2 corridor datasets, so the matched trainer can be run on them unchanged.

A corridor sample at station s and lateral offset l sits hypot(s, l) from the vehicle, which is at the route start.
Marking everything beyond a radius invalid reproduces, in the stored data, what the runner's `--sense-radius-m`
does to a live frame. The vehicle footprint exclusion is applied too, so the training input matches deployment.
Channels are rewritten exactly as scripts/vehicle_corridor.tensor12_excluded does: the height and range references
move to the first station with enough valid samples, slopes are recomputed from the refilled height, and invalid
cells carry no terrain information.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vehicle_corridor import HALF_LENGTH_M, HALF_WIDTH_M_VEH, MIN_VALID_PER_STATION

HALF_W = 6.0


def remask_batch(X, route_len, margin, radius_m):
    """X: (n,12,S,L) float32, modified in place. Returns the mean valid fraction."""
    n, _, S, L = X.shape
    lateral = np.linspace(-HALF_W, HALF_W, L)
    ds = np.maximum(route_len / (S - 1), 1e-3)
    along = np.arange(S)[None, :] * ds[:, None]                      # (n, S)
    zone = (along <= HALF_LENGTH_M + margin)[:, :, None] & (np.abs(lateral) <= HALF_WIDTH_M_VEH + margin)[None, None, :]
    valid = (X[:, 4] > 0.5) & ~zone
    if np.isfinite(radius_m):
        valid &= np.hypot(along[:, :, None], lateral[None, None, :]) <= radius_m
    has = valid.sum(2) >= MIN_VALID_PER_STATION
    ref = np.where(has.any(1), has.argmax(1), 0)
    rows = np.arange(n)
    vr = valid[rows, ref]
    cnt = np.maximum(vr.sum(1), 1)
    z0 = np.where(has.any(1), (np.where(vr, X[rows, 0, ref], 0.)).sum(1) / cnt, 0.)
    r0 = np.where(has.any(1), (np.where(vr, X[rows, 5, ref], 0.)).sum(1) / cnt, X[:, 5].mean((1, 2)))
    zf = np.where(valid, X[:, 0] - z0[:, None, None], 0.).astype(np.float32)
    X[:, 0] = zf
    X[:, 1] = np.clip(np.gradient(zf, axis=1) / ds[:, None, None], -2, 2)
    X[:, 2] = np.clip(np.gradient(zf, 2 * HALF_W / (L - 1), axis=2), -2, 2)
    X[:, 6] = np.where(valid, X[:, 5] - r0[:, None, None], 0.)
    X[:, 5] = np.where(valid, X[:, 5], 110.0)
    X[:, 7] = np.where(valid, X[:, 7], 0.0)
    X[:, 11] = np.where(valid, X[:, 11], 0.0)
    X[:, 4] = valid.astype(np.float32)
    return float(valid.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', required=True); ap.add_argument('--out-dir', required=True)
    ap.add_argument('--radius-m', type=float, required=True); ap.add_argument('--margin', type=float, default=1.5)
    ap.add_argument('--chunk', type=int, default=4096)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for f in a.files:
        z = dict(np.load(f, allow_pickle=True))
        key = 'X12' if 'X12' in z else 'X'
        rl = (z['route_len12'] if 'route_len12' in z else z['route_len']).astype(float)
        X = z[key]
        out = np.empty(X.shape, np.float16)
        fv = []
        for i in range(0, len(X), a.chunk):
            chunk = X[i:i + a.chunk].astype(np.float32)
            fv.append(remask_batch(chunk, rl[i:i + a.chunk], a.margin, a.radius_m) * len(chunk))
            out[i:i + a.chunk] = chunk.astype(np.float16)
        z[key] = out
        dst = os.path.join(a.out_dir, os.path.basename(f))
        np.savez(dst, **z)
        print(f'{os.path.basename(f)}: {len(X)} routes, mean valid {sum(fv)/len(X):.3f} -> {dst}', flush=True)


if __name__ == '__main__':
    main()
