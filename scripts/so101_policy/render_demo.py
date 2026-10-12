#!/usr/bin/env python
"""SO101 batch-renderer demos and measurements (nedm.render.so101). One renderer per process (Madrona's limit), so
each subcommand builds one.

    PY=~/miniconda3/envs/so101_render/bin/python      # luffy: torch 2.12 cu130, warp 1.13, newton 1.2, mujoco 3.15
    source ~/so101_policy/runs/render/madrona_luffy/env_luffy.sh      # only for --backend madrona (luffy build)
    PYTHONPATH=src $PY scripts/so101_policy/render_demo.py sheet --out OUT [--cameras front,top,wrist] [--res 224]
    PYTHONPATH=src $PY scripts/so101_policy/render_demo.py video --row 185 [--nrd CKPT] --out OUT
    PYTHONPATH=src $PY scripts/so101_policy/render_demo.py bench --worlds 1024 [--backend madrona] --out OUT
    PYTHONPATH=src $PY scripts/so101_policy/render_demo.py pairbench --worlds 1024 --out OUT  # Madrona env
    PYTHONPATH=src $PY scripts/so101_policy/render_demo.py parity --out OUT
    PYTHONPATH=src $PY scripts/so101_policy/render_demo.py appearance --out OUT

--data (default ``~/so101_nrd_fix/data/val_only``) = a data directory of the grasp NRD (system.json,
unified_data.npz); validation data only (split 1), never the sealed test. --cameras: a comma list from front, top,
wrist (default front,wrist); one resolution --res for all.

| Command | Output in OUT |
|---|---|
| sheet | ``grid_4x4_<tag>.png``: 16 recorded validation states (random episodes and times), each tile = the cameras
  side by side; ``_depth.png`` (z-depth, 0-1.2 m, near = bright); ``_x3.png`` (3x nearest upscale);
  ``framing_<tag>.png``: the cube at the 6 corners of the workspace annulus, fixed cameras only |
| video | ``episode_<row>_<tag>.mp4`` (+ ``_stills.png``): the recorded episode (MuJoCo) on the 20 ms model grid; with
  ``--nrd`` a second row with the NRD's open-loop rollout of the same commands from the same first state. One render
  call for the whole episode (one world per frame and source). Goal marker = the recorded final cube pose |
| bench | one JSON line appended to ``bench.jsonl``: ms per call of ``render`` (FK + transforms + render + copy), of
  ``render(copy=False)``, with depth, of the backend alone, views/s, GPU memory |
| pairbench | one JSON line appended to ``bench_pair.jsonl``: Newton and Madrona in ONE process, timed round robin
  (both see the same load on a shared GPU), median ms per ``render`` call and views/s; needs the Madrona env |
| parity | ``parity_<tag>.json``, ``.png`` (rows per camera: MuJoCo, ours), ``_arrays.npz``: z-depth against MuJoCo's
  own renderer (EGL) for 8 validation states, the analytic table plane, and (when the other backend's arrays exist) the
  other backend |
| appearance | ``appearance_<backend>.json``, ``.png`` (rows: own colours, random colours + light, cube recoloured):
  EXPERIMENTAL ``set_appearance`` checks |

<tag> = <backend>_<cameras joined by ->_<res>, e.g. newton_front-wrist_128.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

from nedm.render.so101 import FACE_COLORS, SO101Renderer, poses_from_nrd


def load_val(data: Path):
    z = np.load(data / "unified_data.npz")
    system = json.loads((data / "system.json").read_text())
    out = {k: z[k] for k in ("arm", "cube", "lengths", "family", "outcome", "splits", "action")}
    if (out["splits"] == 2).any():
        raise SystemExit("this data holds sealed-test episodes (split 2): use validation data only")
    return out, system


def states(v, ep, t) -> np.ndarray:
    """[k, 2, 13] states (arm 12 + pad, cube 13) of episodes ``ep`` at records ``t``."""
    x = np.zeros((len(ep), 2, 13), np.float32)
    x[:, 0, :12], x[:, 1] = v["arm"][ep, t], v["cube"][ep, t]
    return x


def final_goal(v, ep) -> np.ndarray:
    """Goal (x, y, yaw) [k, 3] = the recorded final cube pose of each episode."""
    c = v["cube"][ep, v["lengths"][ep] - 1]
    w, x, y, z = c[:, 3], c[:, 4], c[:, 5], c[:, 6]
    return np.stack([c[:, 0], c[:, 1], np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))], 1)


def tiles(rgb: np.ndarray) -> np.ndarray:
    """[k, C, H, W, 3] -> [k, H, C*W, 3] (cameras side by side)."""
    k, c, h, w, _ = rgb.shape
    return rgb.transpose(0, 2, 1, 3, 4).reshape(k, h, c * w, 3)


def depth_gray(depth: np.ndarray, far: float = 1.2) -> np.ndarray:
    g = np.where(depth > 0, 255 * (1 - np.clip(depth, 0, far) / far), 0).astype(np.uint8)
    return np.repeat(g[..., None], 3, -1)


def save_png(img: np.ndarray, path: Path, scale: int = 1) -> None:
    from PIL import Image
    im = Image.fromarray(img)
    if scale > 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    im.save(path)


def gpu_used_mb() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2 ** 20


def process_mb() -> float | None:
    """This process's GPU memory per nvidia-smi (MiB), None if unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in out.strip().splitlines():          # e.g. "Failed to initialize NVML" on luffy: no usable line
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == os.getpid():
            return float(parts[1])
    return None


def make_renderer(a, worlds: int) -> SO101Renderer:
    return SO101Renderer(worlds, a.res, a.res, backend=a.backend, shadows=not a.no_shadows,
                         cube_colors=FACE_COLORS if a.face_colors else None, goal=None if a.goal == "none" else a.goal,
                         cameras=a.cameras)


def tag(a) -> str:
    return f"{a.backend}_{'-'.join(a.cameras)}_{a.res}"


# ---- commands -------------------------------------------------------------------------------------------------------
def cmd_sheet(a):
    v, _ = load_val(a.data)
    rng = np.random.default_rng(a.seed)
    task = np.flatnonzero((v["family"] <= 6) & (v["splits"] == 1))          # the task families (not the probes)
    ep = rng.choice(task, 16, replace=False)
    t = (rng.random(16) * v["lengths"][ep]).astype(int)
    r = make_renderer(a, 16)
    rgb, depth = r.render(*poses_from_nrd(torch.tensor(states(v, ep, t))), torch.tensor(final_goal(v, ep)), depth=True)
    from nedm.render import grid
    name = f"grid_4x4_{tag(a)}"
    sheet = grid(tiles(rgb.cpu().numpy()), 4)
    save_png(sheet, a.out / f"{name}.png")
    save_png(sheet, a.out / f"{name}_x3.png", 3)
    save_png(grid(tiles(depth_gray(depth.cpu().numpy())), 4), a.out / f"{name}_depth.png")
    # framing: the cube at the 6 corners of the workspace annulus (r 0.15 / 0.27 m, -75 / 0 / +75 deg around the pan
    # axis), the goal at the opposite corner, arm of the first sheet state; fixed cameras only, one row each
    rr, ang = np.meshgrid([0.15, 0.27], np.radians([-75.0, 0.0, 75.0]))
    cx, cy = 0.0388 + rr.ravel() * np.cos(ang.ravel()), rr.ravel() * np.sin(ang.ravel())
    cube = np.zeros((6, 7))
    cube[:, 0], cube[:, 1], cube[:, 2], cube[:, 3] = cx, cy, 0.015, 1.0
    goal = np.stack([cx[::-1], cy[::-1], np.zeros(6)], 1)
    q = torch.tensor(states(v, ep[:1], t[:1])[0, 0, :6]).expand(6, 6)
    img = r.render(q, torch.tensor(cube[:, :3]), torch.tensor(cube[:, 3:]), torch.tensor(goal)).cpu().numpy()
    fixed = [c for c, cam in enumerate(r.cameras) if cam != "wrist"]
    if fixed:
        save_png(grid(np.concatenate([img[:, c] for c in fixed]), 6), a.out / f"framing_{tag(a)}.png", 2)
    json.dump({"rows": ep.tolist(), "records": t.tolist(), "res": a.res, "backend": a.backend,
               "tile": " | ".join(r.cameras), "goal": "recorded final cube pose of the episode"},
              open(a.out / f"{name}.json", "w"), indent=1)
    print("wrote", a.out / f"{name}.png")


def cmd_video(a):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    v, system = load_val(a.data)
    row, stride = a.row, 2                                    # 10 ms records -> 20 ms model grid
    n = (int(v["lengths"][row]) - 1) // stride + 1
    t = np.arange(n) * stride
    x = states(v, np.full(n, row), t)
    sources = [("recorded (MuJoCo)", x)]
    if a.nrd:
        from nedm.contact_nrd.model import load_model
        from nedm.contact_nrd.rollouts import load_rows, rollouts
        model, _ = load_model(a.nrd, "cuda:0")
        out = rollouts(model, load_rows(a.data, [row], "cuda:0", 0.02), "first", 0, None, 2)
        pred = out["predicted"][0, :n]
        sources.append((f"NRD {Path(a.nrd).stem}, open loop from the first state", pred[:, :, :13].astype(np.float32)))
    k = len(sources)
    r = make_renderer(a, k * n)
    q, cp, cq = poses_from_nrd(torch.tensor(np.concatenate([s for _, s in sources])))
    goal = torch.tensor(final_goal(v, np.array([row]))).expand(k * n, 3)
    t0 = time.perf_counter()
    rgb = r.render(q, cp, cq, goal).cpu().numpy()
    ms = 1e3 * (time.perf_counter() - t0)
    C = len(r.cameras)
    rgb = rgb.reshape(k, n, C, a.res, a.res, 3)
    scale, head = a.scale, 18
    name = f"episode_{row}_{tag(a)}"
    writer = imageio.get_writer(str(a.out / f"{name}.mp4"), fps=50 / a.slow, codec="libx264", quality=8,
                                macro_block_size=None)
    stills = []
    for i in range(n):
        rows = [np.concatenate(list(rgb[s, i]), 1) for s in range(k)]
        frame = Image.fromarray(np.concatenate(rows, 0)).resize((C * a.res * scale, k * a.res * scale), Image.NEAREST)
        canvas = Image.new("RGB", (frame.width, frame.height + k * head), (252, 252, 251))
        draw = ImageDraw.Draw(canvas)
        for s, (label, _) in enumerate(sources):
            y = s * (a.res * scale + head)
            draw.text((4, y + 3), f"{label}   t = {i * 0.02:.2f} s   {' | '.join(r.cameras)}", fill=(11, 11, 11))
            canvas.paste(frame.crop((0, s * a.res * scale, frame.width, (s + 1) * a.res * scale)), (0, y + head))
        writer.append_data(np.asarray(canvas))
        if i in (0, n // 3, n // 2, 2 * n // 3, n - 1):
            stills.append(np.asarray(canvas))
    writer.close()
    save_png(np.concatenate(stills, 0), a.out / f"{name}_stills.png")
    info = {"row": row, "frames": n, "sources": [s for s, _ in sources], "render_call_ms": round(ms, 1),
            "worlds": k * n, "cameras": list(r.cameras), "res": a.res, "backend": a.backend, "fps": 50 / a.slow,
            "family": system["families"][int(v["family"][row])]["name"]}
    if a.nrd:
        err = np.linalg.norm(pred[:, 1, :3] - x[:, 1, :3], axis=-1)
        info["nrd_cube_pos_err_mm"] = {"p50": float(np.median(err) * 1e3), "max": float(err.max() * 1e3),
                                       "final": float(err[-1] * 1e3)}
    json.dump(info, open(a.out / f"{name}.json", "w"), indent=1)
    print(json.dumps(info))


def cmd_bench(a):
    v, _ = load_val(a.data)
    rng = np.random.default_rng(a.seed)
    task = np.flatnonzero((v["family"] <= 6) & (v["splits"] == 1))
    used0 = gpu_used_mb()
    t0 = time.perf_counter()
    r = make_renderer(a, a.worlds)
    build_s = time.perf_counter() - t0
    batches = []
    for _ in range(4):
        ep = rng.choice(task, a.worlds)
        t = (rng.random(a.worlds) * v["lengths"][ep]).astype(int)
        q, cp, cq = (u.cuda() for u in poses_from_nrd(torch.tensor(states(v, ep, t))))
        batches.append((q, cp, cq, torch.tensor(final_goal(v, ep)).cuda()))
    t0 = time.perf_counter()
    r.render(*batches[0])
    first_s = time.perf_counter() - t0
    for i in range(5):
        r.render(*batches[i % 4])

    def timed(fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(a.frames):
            fn(i)
        torch.cuda.synchronize()
        return 1e3 * (time.perf_counter() - t0) / a.frames

    full = timed(lambda i: r.render(*batches[i % 4]))
    view = timed(lambda i: r.render(*batches[i % 4], copy=False))
    with_depth = timed(lambda i: r.render(*batches[i % 4], depth=True))
    pre = [r.scene.transforms(*b) for b in batches]
    backend = timed(lambda i: r.renderer.render(pre[i % 4][1], body_q=pre[i % 4][0]))
    fk = timed(lambda i: r.scene.transforms(*batches[i % 4]))
    appearance = None
    if a.backend == "madrona":           # re-init cost of set_appearance (colours + light per world), then reset
        colors = torch.rand(a.worlds, len(r.geom_names), 3)

        def recolor(i):
            r.set_appearance(colors, torch.tensor([0.3, -0.2, -1.0]).expand(a.worlds, 3))
            r.render(*batches[i % 4])
        appearance = timed(recolor)
        r.set_appearance(reset=True)
        r.render(*batches[0])
    views = len(r.cameras) * a.worlds
    triangles = sum(len(m.faces) for m in r.scene.scene.bodies) + sum(len(m.faces) for m, _ in r.scene.scene.statics)
    res = {"backend": a.backend, "gpu": torch.cuda.get_device_name(0), "worlds": a.worlds, "cameras": list(r.cameras),
           "res": a.res, "shadows": not a.no_shadows, "triangles_per_world": triangles,
           "build_s": round(build_s, 2), "first_render_s": round(first_s, 2),
           "ms_render": round(full, 2), "ms_render_view": round(view, 2), "ms_render_depth": round(with_depth, 2),
           "ms_backend_only": round(backend, 2), "ms_fk_transforms": round(fk, 2),
           "ms_set_appearance_render": None if appearance is None else round(appearance, 2),
           "views_per_s": round(views / full * 1e3), "views_per_s_backend_only": round(views / backend * 1e3),
           "gpu_used_mb_before": round(used0), "gpu_used_mb_after": round(gpu_used_mb()),
           "gpu_process_mb_nvidia_smi": process_mb(),
           "torch_max_allocated_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20), "frames": a.frames}
    print(json.dumps(res))
    with open(a.out / "bench.jsonl", "a") as fh:
        fh.write(json.dumps(res) + "\n")


def cmd_pairbench(a):
    """Both backends in one process, timed interleaved: a fair comparison when other jobs share the GPU."""
    v, _ = load_val(a.data)
    rng = np.random.default_rng(a.seed)
    task = np.flatnonzero((v["family"] <= 6) & (v["splits"] == 1))
    batches = []
    for _ in range(4):
        ep = rng.choice(task, a.worlds)
        t = (rng.random(a.worlds) * v["lengths"][ep]).astype(int)
        q, cp, cq = (u.cuda() for u in poses_from_nrd(torch.tensor(states(v, ep, t))))
        batches.append((q, cp, cq, torch.tensor(final_goal(v, ep)).cuda()))
    rs = {}
    for b in ("madrona", "newton"):      # Madrona first: its engine start needs the most free GPU memory
        a.backend = b
        rs[b] = make_renderer(a, a.worlds)
        for x in batches:
            rs[b].render(*x)
    torch.cuda.synchronize()
    ms = {b: [] for b in rs}
    for _ in range(a.rounds):
        for b, r in rs.items():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for x in batches:
                r.render(*x)
            torch.cuda.synchronize()
            ms[b].append(1e3 * (time.perf_counter() - t0) / len(batches))
    views = len(a.cameras) * a.worlds
    res = {"gpu": torch.cuda.get_device_name(0), "worlds": a.worlds, "cameras": list(a.cameras), "res": a.res,
           "shadows": not a.no_shadows, "rounds": a.rounds, "calls_per_round": len(batches)}
    for b, m in ms.items():
        med = float(np.median(m))
        res[b] = {"ms_median": round(med, 1), "ms_min": round(min(m), 1), "ms_max": round(max(m), 1),
                  "views_per_s_median": round(views / med * 1e3)}
    print(json.dumps(res))
    with open(a.out / "bench_pair.jsonl", "a") as fh:
        fh.write(json.dumps(res) + "\n")


def cmd_parity(a):
    """z-depth against MuJoCo's own renderer, same 8 states as the unit test protocol, numbers written out."""
    import mujoco
    from nedm.contact_nrd.collect import so101_grasp
    from nedm.render.so101 import GOAL_COLOR, GOAL_HALF, GOAL_THICKNESS
    v, _ = load_val(a.data)
    rng = np.random.default_rng(a.seed)
    K, H = 8, a.res
    task = np.flatnonzero((v["family"] <= 6) & (v["splits"] == 1))
    ep = rng.choice(task, K, replace=False)
    x = states(v, ep, (rng.random(K) * v["lengths"][ep]).astype(int))
    goal = final_goal(v, ep)
    r = make_renderer(a, K)
    rgb, depth = r.render(*poses_from_nrd(torch.tensor(x)), torch.tensor(goal), depth=True)
    rgb, depth = rgb.cpu().numpy(), depth.cpu().numpy()
    spec = so101_grasp.build_spec(so101_grasp.load_config())
    for fixed in ("front", "top"):
        c = getattr(r.scene, fixed).double().cpu().numpy()
        spec.worldbody.add_camera(name=fixed, pos=c[:3].tolist(), quat=[c[6], *c[3:6]], fovy=r.scene.fovy_deg)
    marker = spec.worldbody.add_body(name="goal_marker", mocap=True)     # world geoms do not move after compile
    marker.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, contype=0, conaffinity=0,
                    size=[GOAL_HALF, GOAL_HALF, GOAL_THICKNESS / 2], rgba=[*GOAL_COLOR, 1])
    m = spec.compile()
    d = mujoco.MjData(m)
    m.vis.map.znear, m.vis.map.zfar = 0.02 / m.stat.extent, 4.0 / m.stat.extent
    m.geom_rgba[m.geom("wrist_camera_lens").id, 3] = 0.0
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = opt.geomgroup[2] = 1
    mid = m.body("goal_marker").mocapid[0]
    cams = list(r.cameras)
    C = len(cams)
    ref = np.zeros((K, C, H, H), np.float32)
    mrgb = np.zeros((K, C, H, H, 3), np.uint8)
    rend = mujoco.Renderer(m, H, H)
    for k in range(K):
        d.qpos[:] = m.qpos0
        d.qpos[:6], d.qpos[6:13] = x[k, 0, :6], x[k, 1, :7]
        d.mocap_pos[mid] = [goal[k, 0], goal[k, 1], GOAL_THICKNESS / 2]
        d.mocap_quat[mid] = [math.cos(goal[k, 2] / 2), 0, 0, math.sin(goal[k, 2] / 2)]
        mujoco.mj_forward(m, d)
        for c, cam in enumerate("wrist_cam" if n == "wrist" else n for n in cams):
            rend.update_scene(d, camera=cam, scene_option=opt)
            mrgb[k, c] = rend.render()
            rend.enable_depth_rendering()
            rend.update_scene(d, camera=cam, scene_option=opt)
            ref[k, c] = rend.render()
            rend.disable_depth_rendering()
    rend.close()
    res = {"states": K, "res": H, "backend": a.backend, "cameras": cams, "rows": ep.tolist()}
    green = lambda im: (im[..., 1] > im[..., 0] + 30) & (im[..., 1] > im[..., 2] + 30)          # noqa: E731
    ours, theirs = green(rgb.astype(int)), green(mrgb.astype(int))
    res["goal_marker_iou"] = {name: float((ours[:, c] & theirs[:, c]).sum() / max(1, (ours[:, c] | theirs[:, c]).sum()))
                              for c, name in enumerate(cams)}
    res["goal_marker_pixels_ours_mujoco"] = {name: [int(ours[:, c].sum()), int(theirs[:, c].sum())]
                                             for c, name in enumerate(cams)}
    for c, name in enumerate(cams):
        hit = (depth[:, c] > 0) & (ref[:, c] < 3.9)
        e = np.abs(depth[:, c] - ref[:, c])[hit] * 1e3
        res[name] = {"pixels_both_hit": int(hit.sum()), "of": int(hit.size),
                     "abs_diff_mm_p50": float(np.percentile(e, 50)), "p90": float(np.percentile(e, 90)),
                     "p99": float(np.percentile(e, 99)), "frac_over_5mm": float((e > 5).mean()),
                     "hit_miss_mismatch_px": int(((depth[:, c] > 0) != (ref[:, c] < 3.9)).sum())}
    for c, name in enumerate(cams):
        if name == "wrist":
            continue
        plane = table_depth(r, name)[None]
        table = (depth[:, c] > 0) & (np.abs(ref[:, c] - plane) < 0.02) & (np.abs(depth[:, c] - plane) < 0.02)
        for who, dd in (("ours", depth[:, c]), ("mujoco", ref[:, c])):
            e = np.abs(dd - plane)[table] * 1e3
            res[f"table_plane_{name}_{who}_mm"] = {"p50": float(np.percentile(e, 50)),
                                                   "p99": float(np.percentile(e, 99)), "pixels": int(table.sum())}
    other_backend = "newton" if a.backend == "madrona" else "madrona"
    other = a.out / f"parity_{tag(a).replace(a.backend, other_backend, 1)}_arrays.npz"
    if other.exists():                      # the other backend's run of the same states: depth agreement
        o = np.load(other)
        both = (depth > 0) & (o["depth"] > 0)
        e = np.abs(depth - o["depth"])[both] * 1e3
        res["vs_other_backend_depth_mm"] = {"p50": float(np.percentile(e, 50)), "p99": float(np.percentile(e, 99)),
                                            "frac_over_5mm": float((e > 5).mean()),
                                            "hit_miss_mismatch_px": int(((depth > 0) != (o["depth"] > 0)).sum()),
                                            "rgb_equal_frac": float((rgb == o["rgb"]).all(-1).mean())}
    json.dump(res, open(a.out / f"parity_{tag(a)}.json", "w"), indent=1)
    np.savez_compressed(a.out / f"parity_{tag(a)}_arrays.npz", depth=depth, rgb=rgb, mujoco_depth=ref, mujoco_rgb=mrgb)
    from nedm.render import grid
    save_png(grid(np.concatenate([im for c in range(C) for im in (mrgb[:, c], rgb[:, c])]), K),
             a.out / f"parity_{tag(a)}.png")
    print(json.dumps(res, indent=1))


def table_depth(r: SO101Renderer, camera: str = "front") -> np.ndarray:
    """z-depth [H, W] of the table plane z = 0 in a fixed camera at pixel centres (inf where the ray misses it)."""
    return front_table_depth(r, camera).numpy()


def front_table_depth(r: SO101Renderer, camera: str = "front") -> torch.Tensor:
    """z-depth [H, W] of the table plane z = 0 in the front (or top) camera at pixel centres (inf: misses it)."""
    from nedm.render.so101 import qrot, wxyz
    cam = getattr(r.scene, camera).double().cpu()
    H, W = r.height, r.width
    f = (H / 2) / math.tan(math.radians(r.scene.fovy_deg) / 2)
    u = (torch.arange(W, dtype=torch.float64) + 0.5 - W / 2) / f
    v = (torch.arange(H, dtype=torch.float64) + 0.5 - H / 2) / f
    dirs = torch.stack(torch.broadcast_tensors(u[None, :], -v[:, None], torch.tensor(-1.0, dtype=torch.float64)), -1)
    world = qrot(wxyz(cam[3:]).expand(H, W, 4), dirs)
    t = -cam[2] / world[..., 2]
    return torch.where(t > 0, t, torch.full_like(t, float("inf"))).float()


def cmd_appearance(a):
    """Per-world colours (and on Madrona per-world light) after the first render: EXPERIMENTAL set_appearance."""
    v, _ = load_val(a.data)
    N = 16
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    task = np.flatnonzero((v["family"] <= 6) & (v["splits"] == 1))
    ep = rng.choice(task, N, replace=False)
    x = states(v, ep, (rng.random(N) * v["lengths"][ep]).astype(int))
    inputs = [u.cuda() for u in poses_from_nrd(torch.tensor(x))] + [torch.tensor(final_goal(v, ep)).cuda()]
    mad = a.backend == "madrona"
    r = make_renderer(a, N)
    base_rgb = r.render(*inputs)
    names = r.geom_names
    G = len(names)
    cube = [k for k, nm in enumerate(names) if nm.startswith("cube")]
    res = {"backend": a.backend, "worlds": N, "res": a.res, "geoms": G, "shadows": not a.no_shadows}
    # 1. random colours for every geom of every world (+ a random light from above per world on Madrona)
    colors = 0.15 + 0.8 * torch.rand(N, G, 3, generator=gen)
    light = torch.randn(N, 3, generator=gen)
    light[:, 2] = -light[:, 2].abs() - 0.5
    used0 = gpu_used_mb()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    r.set_appearance(colors, light if mad else None)
    rgb1 = r.render(*inputs)
    res["set_appearance_plus_render_ms"] = round(1e3 * (time.perf_counter() - t0), 1)
    t0 = time.perf_counter()
    r.render(*inputs)
    res["plain_render_ms"] = round(1e3 * (time.perf_counter() - t0), 1)
    res["worlds_changed"] = int((rgb1 != base_rgb).flatten(1).any(1).sum())
    # 2. the cube colour lands on the cube: change only the cube colour; the pixels that change are cube pixels, and
    # their colour (shading cancels in the ratio r : g : b) must follow the requested colour
    colors2 = colors.clone()
    colors2[:, cube] = (0.15 + 0.8 * torch.rand(N, 1, 3, generator=gen)).expand(N, len(cube), 3)
    r.set_appearance(colors2)
    rgb2 = r.render(*inputs)
    mask = (rgb1 != rgb2).any(-1)
    errs = []
    for k in range(N):
        for img, col in ((rgb1, colors), (rgb2, colors2)):
            px = img[k][mask[k]].float()
            px = px[px.sum(-1) > 60]                                    # skip near-black pixels (8-bit noise)
            if len(px) < 10:
                continue
            got = px.sum(0) / px.sum()
            want = col[k, cube[0]] / col[k, cube[0]].sum()
            errs.append(float((got.cpu() - want).abs().max()))
    res["cube_colour_ratio_err"] = {"max": max(errs), "median": float(np.median(errs)), "views": len(errs)} \
        if errs else None
    res["changed_pixels_only_on_cube_worlds"] = int(mask.flatten(1).any(1).sum())
    if mad:
        # 3. per-world light: table grey 0.5, shadows off; light straight down (n.l = 1) in worlds 0-7, horizontal
        # (n.l = 0 -> the 0.2 ambient floor) in worlds 8-15. Table pixels of the front camera found by depth.
        grey = colors.clone()
        grey[:, names.index("table")] = 0.5
        down = torch.tensor([0.0, 0.0, -1.0]).expand(N, 3).clone()
        down[N // 2:] = torch.tensor([1.0, 0.0, 0.0])
        r.set_appearance(grey, down, shadows=torch.zeros(N, dtype=torch.bool))
        rgb3, depth3 = r.render(*inputs, depth=True)
        front = r.cameras.index("front")
        plane = front_table_depth(r).to(depth3.device)
        table = (depth3[:, front] - plane).abs() < 3e-4
        level = [float(rgb3[k, front][table[k]].float().mean()) for k in range(N)]
        res["table_level_light_down"] = round(float(np.mean(level[: N // 2])), 1)
        res["table_level_light_horizontal"] = round(float(np.mean(level[N // 2:])), 1)
        res["table_level_expected"] = [round(0.5 * 255, 1), round(0.2 * round(0.5 * 255), 1)]
    # 4. back to the scene's own appearance: bit-identical to the first picture
    r.set_appearance(reset=True)
    res["reset_bit_identical"] = bool(torch.equal(r.render(*inputs), base_rgb))
    for _ in range(30):                                                       # memory growth check
        r.set_appearance(colors, light if mad else None)
        r.render(*inputs)
    torch.cuda.synchronize()
    res["gpu_used_mb_before"], res["gpu_used_mb_after_32_more_set_appearance"] = round(used0), round(gpu_used_mb())
    json.dump(res, open(a.out / f"appearance_{a.backend}.json", "w"), indent=1)
    from nedm.render import grid
    show = [tiles(t[:8].cpu().numpy()) for t in (base_rgb, rgb1, rgb2)]
    save_png(grid(np.concatenate(show), 8), a.out / f"appearance_{a.backend}.png")
    print(json.dumps(res, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["sheet", "video", "bench", "pairbench", "parity", "appearance"])
    p.add_argument("--data", type=Path, default=Path.home() / "so101_nrd_fix/data/val_only")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--backend", default="newton", choices=["madrona", "newton"])
    p.add_argument("--cameras", default="front,wrist", help="comma list from front, top, wrist")
    p.add_argument("--res", type=int, default=128)
    p.add_argument("--no-shadows", action="store_true")
    p.add_argument("--face-colors", action="store_true")
    p.add_argument("--goal", default="plate", choices=["plate", "frame", "none"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--row", type=int, default=185, help="video: validation row (185 = nominal, carried 38 cm)")
    p.add_argument("--nrd", default="", help="video: NRD checkpoint for a second row (open-loop rollout)")
    p.add_argument("--scale", type=int, default=3, help="video: nearest-neighbour upscale")
    p.add_argument("--slow", type=float, default=1.0, help="video: play at 50 / SLOW frames per second")
    p.add_argument("--worlds", type=int, default=1024)
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--rounds", type=int, default=6, help="pairbench: rounds of 4 calls per backend")
    a = p.parse_args()
    a.cameras = tuple(c.strip() for c in a.cameras.split(",") if c.strip())
    a.out.mkdir(parents=True, exist_ok=True)
    {"sheet": cmd_sheet, "video": cmd_video, "bench": cmd_bench, "pairbench": cmd_pairbench, "parity": cmd_parity,
     "appearance": cmd_appearance}[a.command](a)


if __name__ == "__main__":
    main()
