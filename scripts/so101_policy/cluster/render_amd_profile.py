"""Where the time of one SO101Renderer.render call goes (Newton backend): FK + transforms, the two copies into Warp, the
shape-BVH refit, the ray-tracing kernel, the uint8 copy. Same states and cameras as render_amd_bench.py bench.

    python render_amd_profile.py --states states.npz --assets so101_assets.npz --worlds 1024 --res 128 [--order tiled]
"""
import argparse
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_amd_bench as RB          # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--states", required=True)
    p.add_argument("--assets", default="")
    p.add_argument("--worlds", type=int, default=1024)
    p.add_argument("--res", type=int, default=128)
    p.add_argument("--no-shadows", action="store_true")
    p.add_argument("--order", default="pixel")
    p.add_argument("--tile", default="8x8")
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--out", default="")
    a = p.parse_args()
    import warp as wp
    r = RB._renderer(a.worlds, a.res, not a.no_shadows, a.assets, order=a.order, tile=a.tile)
    per = int(np.load(a.states)["per_batch"])
    batches = [RB._load(a.states, slice(b * per, b * per + a.worlds)) for b in range(4)]
    for b in batches:
        r.render(*b)
    be = r.renderer.backend
    newton = be._newton
    t = {k: [] for k in ("transforms", "put", "refit", "trace", "copy", "total")}

    def sync():
        torch.cuda.synchronize()
        wp.synchronize()

    for k in range(a.reps):
        q, cp, cq, g = batches[k % 4]
        sync()
        t0 = time.perf_counter()
        body_q, cam_q = r.scene.transforms(q, cp, cq, g)
        sync()
        t1 = time.perf_counter()
        be._put(be.state.body_q, body_q.reshape(be.num_worlds * be.num_bodies, 7))
        be._put(be._camera_xf, cam_q.transpose(0, 1))
        sync()
        t2 = time.perf_counter()
        newton.geometry.refit_bvh_shape(be.model, be.state)
        sync()
        t3 = time.perf_counter()
        be._camera.update(be.state, be._camera_xf, be._rays, color_image=be._color, depth_image=be._depth,
                          clear_data=be._clear)
        sync()
        t4 = time.perf_counter()
        rgb = wp.to_torch(be._color).view(torch.uint8).reshape(*be._color.shape, 4)[..., :3].clone()
        sync()
        t5 = time.perf_counter()
        for key, v in zip(t, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t5 - t0)):
            t[key].append(v * 1e3)
    row = dict(host=socket.gethostname(), device=torch.cuda.get_device_name(0), worlds=a.worlds, res=a.res,
               shadows=not a.no_shadows, order=a.order, tile=a.tile if a.order == "tiled" else "",
               **{f"{k}_ms": round(float(np.median(v)), 2) for k, v in t.items()}, rgb_shape=list(rgb.shape))
    print(json.dumps(row), flush=True)
    if a.out:
        with open(a.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
