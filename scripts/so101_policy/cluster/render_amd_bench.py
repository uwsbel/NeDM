"""SO101 scene with the Newton (Warp) batch renderer: correctness across machines and throughput (build8_amd_render.md).

The camera student's cameras: ``top`` (dataset.TOP_EYE, 0.48 m over (0.19, 0)) and ``wrist`` (the MJCF wrist camera),
one resolution, fovy 61 deg, goal plate drawn, shadows on unless --no-shadows.

    python render_amd_bench.py states --out states.npz                 # once, where the validation data are (luffy)
    python render_amd_bench.py parity --states states.npz --res 128 --out parity_<host>_128.npz
    python render_amd_bench.py compare parity_luffy_128.npz parity_mi350x_128.npz
    python render_amd_bench.py bench --states states.npz --worlds 1024 --res 128 [--no-shadows] --out bench.jsonl

States: recorded validation states (val_only split 1, random episodes and times; goal = the episode's final cube pose,
x, y and yaw), the same rule as luffy's bench_pair.py (build2_render.md), so speeds are comparable. ``bench`` times one
``SO101Renderer.render`` call (FK + transforms + ray tracing + uint8 copy) and cycles 4 different state batches. The
Newton backend is slower on diverse states than on repeated ones (build2_render.md), so never bench copies of one
state. GPU memory = device total - free (``torch.cuda.mem_get_info``: Warp's and torch's allocations together).
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

CAMS = ("top", "wrist")


def cmd_states(a):
    z = np.load(Path(a.val).expanduser())
    arm, cube, lengths = z["arm"], z["cube"], z["lengths"]
    rng = np.random.default_rng(a.seed)
    out = {}
    n = a.batches * a.n
    ep = rng.choice(len(lengths), n)
    t = (rng.random(n) * lengths[ep]).astype(int)
    c = cube[ep, lengths[ep] - 1]
    yaw = np.arctan2(2 * (c[:, 3] * c[:, 6] + c[:, 4] * c[:, 5]), 1 - 2 * (c[:, 5] ** 2 + c[:, 6] ** 2))
    out["q"] = arm[ep, t, :6].astype(np.float32)
    out["cube_pos"] = cube[ep, t, 0:3].astype(np.float32)
    out["cube_quat"] = cube[ep, t, 3:7].astype(np.float32)
    out["goal"] = np.stack([c[:, 0], c[:, 1], yaw], 1).astype(np.float32)
    out["episode"], out["t"] = ep.astype(np.int64), t.astype(np.int64)
    out["batches"], out["per_batch"] = np.int64(a.batches), np.int64(a.n)
    np.savez_compressed(a.out, **out)
    print(f"{n} states ({a.batches} x {a.n}) from {a.val} -> {a.out}")


def _load(path, sl=slice(None), device="cuda"):
    import torch
    z = np.load(path)
    return [torch.as_tensor(z[k][sl], device=device) for k in ("q", "cube_pos", "cube_quat", "goal")]


def _renderer(n, res, shadows, assets=None, goal="plate", order="pixel", tile="16x8"):
    """SO101Renderer (Newton backend, the student's cameras). ``order``: Newton's ray-to-thread order (RenderConfig
    render_order: pixel = the same pixel of every view first (Newton's default), view = one whole view after the
    other, tiled = tiles of ``tile`` (W x H) pixels of one view). Set on the backend's camera sensor before the first
    render (the render kernel is compiled per config)."""
    from nedm.render.so101 import SO101Assets, SO101Renderer
    from nedm.so101_policy.vision import dataset as V
    ass = SO101Assets.load(assets) if assets else None
    r = SO101Renderer(n, res, res, backend="newton", shadows=shadows, assets=ass, cameras=CAMS,
                      top_eye=V.TOP_EYE, top_target=V.TOP_TARGET, goal=goal)
    if order != "pixel":
        from newton._src.sensors.warp_raytrace.types import RenderOrder
        cfg = r.renderer.backend._camera.render_config
        cfg.render_order = {"view": RenderOrder.VIEW_PRIORITY, "tiled": RenderOrder.TILED}[order]
        cfg.tile_width, cfg.tile_height = (int(v) for v in tile.split("x"))
    return r


def _gpu_used_gb():
    import torch
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2 ** 30


def _device_name():
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def cmd_parity(a):
    import torch
    q, cp, cq, g = _load(a.states, slice(0, a.n))
    r = _renderer(a.n, a.res, not a.no_shadows, a.assets, order=a.order, tile=a.tile)
    rgb, depth = r.render(q, cp, cq, g, depth=True)
    torch.cuda.synchronize()
    np.savez_compressed(a.out, rgb=rgb.cpu().numpy(), depth=depth.cpu().numpy(), cameras=np.array(r.cameras),
                        host=np.array(socket.gethostname()), device=np.array(_device_name()), res=np.int64(a.res),
                        interop=np.array(str(r.renderer.backend.interop)))
    print(json.dumps(dict(out=a.out, host=socket.gethostname(), device=_device_name(), n=a.n, res=a.res,
                          interop=r.renderer.backend.interop, depth_hit_frac=float((depth > 0).float().mean()))))


def cmd_compare(a):
    x, y = np.load(a.ref), np.load(a.other)
    out = dict(ref=a.ref, other=a.other, ref_device=str(x["device"]), other_device=str(y["device"]), cameras={})
    for i, cam in enumerate(x["cameras"]):
        d0, d1 = x["depth"][:, i].astype(np.float64), y["depth"][:, i].astype(np.float64)
        hit0, hit1 = d0 > 0, d1 > 0
        both = hit0 & hit1
        dd = np.abs(d0 - d1)[both] * 1e3                      # mm
        r0, r1 = x["rgb"][:, i].astype(np.int16), y["rgb"][:, i].astype(np.int16)
        dr = np.abs(r0 - r1).max(-1)                           # per pixel, max over channels
        out["cameras"][str(cam)] = dict(
            pixels=int(d0.size), hit_miss_mismatch=int((hit0 != hit1).sum()),
            depth_abs_mm_p50=float(np.percentile(dd, 50)), depth_abs_mm_p99=float(np.percentile(dd, 99)),
            depth_abs_mm_p999=float(np.percentile(dd, 99.9)), depth_abs_mm_max=float(dd.max()),
            depth_frac_over_1mm=float((dd > 1.0).mean()),
            rgb_identical_frac=float((dr == 0).mean()), rgb_within_2_frac=float((dr <= 2).mean()),
            rgb_abs_mean=float(np.abs(r0 - r1).mean()), rgb_frac_over_16=float((dr > 16).mean()))
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


def cmd_bench(a):
    import torch
    if a.rebuild:     # rebuild the shape BVH at every call instead of refitting it (NewtonBackend refits after call 1)
        from nedm.render.backends import newton as NB
        render0 = NB.NewtonBackend.render

        def render(self, body_q, camera):
            self._bvh_built = False
            return render0(self, body_q, camera)
        NB.NewtonBackend.render = render
    torch.cuda.synchronize()
    used0 = _gpu_used_gb()
    t0 = time.perf_counter()
    r = _renderer(a.worlds, a.res, not a.no_shadows, a.assets, order=a.order, tile=a.tile)
    t_build = time.perf_counter() - t0
    nb = int(np.load(a.states)["batches"])
    per = int(np.load(a.states)["per_batch"])
    if a.worlds > per:
        raise SystemExit(f"{a.worlds} worlds > {per} states per batch")
    batches = [_load(a.states, slice(b * per, b * per + a.worlds)) for b in range(nb)]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    r.render(*batches[0])
    torch.cuda.synchronize()
    t_first = time.perf_counter() - t0
    for b in batches[1:]:
        r.render(*b)
    torch.cuda.synchronize()
    peak = _gpu_used_gb()
    ms, t_start, k = [], time.perf_counter(), 0
    while k < a.min_calls or (time.perf_counter() - t_start < a.min_seconds and k < a.max_calls):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        r.render(*batches[k % nb])
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t1) * 1e3)
        peak = max(peak, _gpu_used_gb())
        k += 1
    med = float(np.median(ms))
    row = dict(host=socket.gethostname(), device=_device_name(), worlds=a.worlds, res=a.res, cameras=list(CAMS),
               shadows=not a.no_shadows, order=a.order, tile=a.tile if a.order == "tiled" else "",
               bvh="rebuild" if a.rebuild else "refit", calls=len(ms), ms_median=round(med, 2),
               ms_min=round(min(ms), 2), ms_max=round(max(ms), 2), views_per_s=round(len(CAMS) * a.worlds / med * 1e3),
               build_s=round(t_build, 2), first_call_s=round(t_first, 2), gpu_used_gb_before=round(used0, 2),
               gpu_used_gb_peak=round(peak, 2), gpu_render_gb=round(peak - used0, 2),
               interop=r.renderer.backend.interop, triangles=r.scene.assets.triangles,
               job=__import__("os").environ.get("SLURM_JOB_ID", ""))
    print(json.dumps(row), flush=True)
    if a.out:
        with open(a.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("states")
    s.add_argument("--val", default="~/so101_nrd_fix/data/val_only/unified_data.npz")
    s.add_argument("--batches", type=int, default=4)
    s.add_argument("--n", type=int, default=4096)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    for name in ("parity", "bench"):
        s = sub.add_parser(name)
        s.add_argument("--states", required=True)
        s.add_argument("--assets", default="", help="SO101Assets .npz (no mujoco needed); default: export the MJCF")
        s.add_argument("--res", type=int, default=128)
        s.add_argument("--no-shadows", action="store_true")
        s.add_argument("--order", default="pixel", choices=["pixel", "view", "tiled"])
        s.add_argument("--tile", default="16x8", help="W x H pixels per tile (--order tiled)")
        s.add_argument("--out", default="")
    sub.choices["parity"].add_argument("--n", type=int, default=64)
    b = sub.choices["bench"]
    b.add_argument("--worlds", type=int, default=1024)
    b.add_argument("--rebuild", action="store_true", help="rebuild the shape BVH every call (default: refit)")
    b.add_argument("--min-calls", type=int, default=4)
    b.add_argument("--max-calls", type=int, default=40)
    b.add_argument("--min-seconds", type=float, default=15.0)
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("other")
    c.add_argument("--out", default="")
    a = p.parse_args(argv)
    {"states": cmd_states, "parity": cmd_parity, "compare": cmd_compare, "bench": cmd_bench}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
