"""Reference renders of the real scene for the background-randomisation tests (``test_bgdr.py``): the student's views
(``viral.make_views``) of a scene file with the default options, nominal and with seeded per-world draws (random
per-call light), and ``RealDR.draw`` / ``SO101Renderer.render_dr`` directly (segment + hdr outputs). Only API that
predates the background change is used, so the same file runs against the base commit's code:

    PYTHONPATH=<src> python tests/so101_policy/bgdr_ref.py --scene real_scene_v2.json --calib calib_v1.json \
        --out ref.npz
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import torch


def states(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = (torch.rand(n, 6, generator=g) - 0.5) * torch.tensor([1.6, 1.0, 1.0, 1.0, 1.0, 0.6])
    q[:, 1] -= 0.6
    q[:, 2] += 0.8
    cp = torch.stack([0.18 + 0.12 * torch.rand(n, generator=g), 0.2 * (torch.rand(n, generator=g) - 0.5),
                      torch.full((n,), 0.015)], -1)
    cq = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n, 4).clone()
    goal = torch.stack([0.2 + 0.1 * torch.rand(n, generator=g), 0.2 * (torch.rand(n, generator=g) - 0.5),
                        torch.rand(n, generator=g)], -1)
    return q, cp, cq, goal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    from nedm.render.so101 import RealDR
    from nedm.so101_policy.vision import viral as VI
    n = a.n
    cfg = replace(VI.ViralConfig(), cameras=("external", "wrist"), res=224, scene=a.scene, calib=a.calib)
    views = VI.make_views(cfg, n, seed=3, device="cuda")
    q, cp, cq, goal = (t.cuda() for t in states(n))
    out = {"nominal": views.render(q, cp, cq, goal).cpu().numpy()}
    for k in range(2):
        d = views.draw(n)
        for key, v in d.items():
            out[f"draw{k}_{key}"] = v.cpu().numpy()
        out[f"dr{k}"] = views.render(q, cp, cq, goal, dr=d).cpu().numpy()
    nd = views.nominal_draw(n)
    out["nominal_draw_keys"] = np.array(sorted(nd))
    mask = torch.tensor([True, False] * (n // 2), device="cuda")
    md = views.merge_nominal(views.draw(n), mask)
    for key, v in md.items():
        out[f"merged_{key}"] = v.cpu().numpy()
    # RealDR + renderer directly (CPU generator, segment and hdr outputs)
    from nedm.render.so101 import SO101Renderer
    rs = VI.load_real_setup(cfg)
    r = SO101Renderer(n, 224, 224, backend="newton", cameras=("external", "wrist"), goal="block", real=rs.real,
                      segment=True)
    dr = RealDR.from_scene(rs.real)
    d = dr.draw(r.scene, n, torch.Generator().manual_seed(5))
    for key, v in d.items():
        out[f"rdraw_{key}"] = v.cpu().numpy()
    rgb, seg, hdr = r.render_dr(q, cp, cq, goal, draw=d, segment=True, hdr=True)
    out["rrgb"], out["rseg"], out["rhdr"] = rgb.cpu().numpy(), seg.cpu().numpy(), hdr.cpu().numpy()
    out["env_pose"] = r.scene.env_pose.cpu().numpy()
    np.savez_compressed(a.out, **out)


if __name__ == "__main__":
    main()
