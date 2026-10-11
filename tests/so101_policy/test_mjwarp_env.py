"""Tests of the MuJoCo Warp SO101 env (nedm.so101_policy.mjwarp_env).

CPU tests run the env's Warp kernels on Warp's CPU device (seconds; need ``warp``, and ``mujoco`` + the vendored
Menagerie model for the contact-class test). GPU tests need CUDA + ``mujoco_warp`` (conda env ``so101_mjwarp`` on
luffy) and skip otherwise; the first GPU run compiles MuJoCo Warp's kernels (minutes), later runs use Warp's cache.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch

wp = pytest.importorskip("warp")
E = pytest.importorskip("nedm.so101_policy.mjwarp_env")

VAL = os.path.expanduser("~/so101_nrd_fix/data/val_only/unified_data.npz")


def _rand_quat(rng, n):
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _data_rates(prev, cur, dt=0.02):
    """The NRD data rule (nedm.contact_nrd.data._step_average) on two records -> (qd [n, 6], v [n, 3], w [n, 3])."""
    from nedm.contact_nrd.data import _step_average
    arm_t = ["angle"] * 6 + ["angular_rate"] * 6
    cube_t = ["position"] * 3 + ["quaternion"] * 4 + ["velocity"] * 3 + ["angular_velocity"] * 3
    system = {"bodies": [dict(name="arm", kind="articulated", moving=True, channel_types=arm_t),
                         dict(name="cube", kind="rigid", moving=True, channel_types=cube_t)]}
    n = len(prev)
    st = torch.zeros(n, 2, 2, 13, dtype=torch.float64)
    for t, x in enumerate((prev, cur)):
        st[:, t, 0, :6] = torch.as_tensor(x[:, :6])
        st[:, t, 1, :7] = torch.as_tensor(x[:, 6:13])
    out = _step_average(st, dict(stride=1, dt=dt, system=system), {"articulated", "rigid"})
    return out[:, 1, 0, 6:12].numpy(), out[:, 1, 1, 7:10].numpy(), out[:, 1, 1, 10:13].numpy()


def _run_finish(prev13, qpos, qvel, fmax, offtab, instant=0, codes=False):
    """E._finish on Warp's CPU device. fmax [B, 7] (both records) or [B, 14]; offtab [B] (record 0) or [B, 2]."""
    dev = "cpu"
    B = len(qpos)
    fmax = np.asarray(fmax, np.float32)
    fmax = np.concatenate([fmax, fmax], 1) if fmax.shape[1] == 7 else fmax
    offtab = np.asarray(offtab, np.int32)
    offtab = np.stack([offtab, np.zeros_like(offtab)], 1) if offtab.ndim == 1 else offtab
    prev = wp.array(prev13, dtype=wp.float64, device=dev)
    out = wp.zeros((B, E.RAW), dtype=wp.float64, device=dev)
    flags = wp.zeros((B, E.NFLAG), dtype=wp.int32, device=dev)
    code = wp.zeros(B, dtype=wp.int32, device=dev)
    wp.launch(E._finish, dim=B, device=dev,
              inputs=[wp.array(qpos, dtype=float, device=dev), wp.array(qvel, dtype=float, device=dev), 6, 6,
                      wp.array(fmax, dtype=float, device=dev), wp.array(offtab, dtype=wp.int32, device=dev), 0.01,
                      wp.float64(50.0), instant, prev, out, flags, code])
    if codes:
        return out.numpy(), flags.numpy(), prev.numpy(), code.numpy()
    return out.numpy(), flags.numpy(), prev.numpy()


def test_finish_matches_data_step_average():
    """WorldState rates of a command = the NRD data's step-average rule on the float32 poses (<= 1e-9)."""
    rng = np.random.default_rng(0)
    n = 64
    q0 = np.concatenate([rng.uniform(-1, 1, (n, 6)), rng.uniform(-0.3, 0.3, (n, 3)), _rand_quat(rng, n)], 1).astype(np.float32)
    q0[:, 9:13] *= np.where(q0[:, 9:10] < 0, -1, 1)
    step = np.concatenate([rng.normal(0, 0.03, (n, 9)), rng.normal(0, 0.05, (n, 4))], 1)
    q1 = q0 + step
    q1[:, 9:13] /= np.linalg.norm(q1[:, 9:13], axis=1, keepdims=True)
    q1 = q1.astype(np.float32)
    q1[: n // 2, 9:13] *= -1                     # MuJoCo may hand back -q: the state must stay sign continuous
    qpos = np.concatenate([q1[:, :6], q1[:, 6:13]], 1)
    qvel = rng.normal(size=(n, 12)).astype(np.float32)
    out, flags, prev = _run_finish(q0.astype(np.float64), qpos, qvel, np.zeros((n, 7), np.float32), np.zeros(n, np.int32))
    want_q1 = q1.astype(np.float64)
    want_q1[: n // 2, 9:13] *= -1
    qd, v, w = _data_rates(q0.astype(np.float64), want_q1)
    assert np.abs(out[:, 0:6] - want_q1[:, :6]).max() == 0
    assert np.abs(out[:, 15:19] - want_q1[:, 9:13]).max() == 0
    assert np.abs(out[:, 6:12] - qd).max() < 1e-9
    assert np.abs(out[:, 19:22] - v).max() < 1e-9
    assert np.abs(out[:, 22:25] - w).max() < 1e-9
    assert np.abs(prev - want_q1).max() == 0      # the pose of this command is the start of the next average
    assert not flags.any()


def test_finish_instant_rates_and_flags():
    """'instant' cube rates = the collector record (world v, R(q) body spin); flags follow the 0.01 N rule."""
    rng = np.random.default_rng(1)
    n = 8
    quat = _rand_quat(rng, n)
    quat[:, 0] = np.abs(quat[:, 0])
    qpos = np.concatenate([np.zeros((n, 6)), np.zeros((n, 3)), quat], 1).astype(np.float32)
    qvel = rng.normal(size=(n, 12)).astype(np.float32)
    prev = np.concatenate([np.zeros((n, 9)), quat], 1)
    fmax = np.zeros((n, 7), np.float32)
    fmax[0, 0] = fmax[1, 1] = fmax[2, 2] = fmax[3, 3] = 0.02
    fmax[4, 4], fmax[5, 5], fmax[6, 6] = 0.5, 0.5, 0.5
    fmax[7, :] = 0.009                            # below the threshold: nothing
    off = np.zeros(n, np.int32)
    out, flags, _ = _run_finish(prev, qpos, qvel, fmax, off, instant=1)
    import mujoco
    for i in range(n):
        r = np.zeros(9)
        mujoco.mju_quat2Mat(r, quat[i].astype(np.float32).astype(np.float64))
        assert np.allclose(out[i, 22:25], r.reshape(3, 3) @ qvel[i, 9:12].astype(np.float64), atol=1e-6)
        assert np.allclose(out[i, 19:22], qvel[i, 6:9], atol=0)
    assert flags[:4, :4].tolist() == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    assert flags[:, 4].tolist() == [0, 0, 0, 0, 1, 1, 1, 0]
    off[0] = 1
    qpos[1, 2] = np.nan
    _, flags, _ = _run_finish(prev, qpos, qvel, np.zeros((n, 7), np.float32), off, instant=1)
    assert flags[:, 4].tolist() == [1, 1, 0, 0, 0, 0, 0, 0]


def test_end_code_precedence():
    """end code: non-finite wins; then the earlier 10 ms record; inside a record TERMINAL order, then off table.
    The invalid flag sees every non-arm-table end of the command, whatever its precedence."""
    n = 5
    quat = np.tile([1.0, 0, 0, 0], (n, 1))
    qpos = np.concatenate([np.zeros((n, 9)), quat], 1).astype(np.float32)
    qvel = np.zeros((n, 12), np.float32)
    prev = np.concatenate([np.zeros((n, 9)), quat], 1)
    fmax = np.zeros((n, 14), np.float32)
    off = np.zeros((n, 2), np.int32)
    fmax[0, 7 + 3] = 1.0; fmax[0, 0 + 5] = 1.0          # arm_self in record 0 beats arm_table in record 1
    fmax[1, 3] = 1.0; fmax[1, 4] = 1.0                  # same record: arm_table (1) before cube_other_link (2)
    off[2, 0] = 1; fmax[2, 7 + 6] = 1.0                 # off table in record 0 before cube_base in record 1
    fmax[3, 3] = 1.0; qpos[3, 0] = np.inf               # numerical wins
    off[4, 1] = 1
    _, flags, _, code = _run_finish(prev, qpos, qvel, fmax, off, codes=True)
    assert code.tolist() == [3, 1, 5, 7, 5]
    assert flags[:, 4].tolist() == [1, 1, 1, 1, 1] and flags[:, 3].tolist() == [1, 1, 0, 1, 0]
    assert E.end_names()[1] == "arm_table" and E.end_names()[E.NUMERICAL] == "numerical"


def test_place_world_spin_to_body_frame():
    """reset / set_cube take the cube spin in the world frame; MuJoCo's free joint wants it in the body frame."""
    import mujoco
    rng = np.random.default_rng(2)
    n = 6
    start = np.zeros((n, E.RAW))
    start[:, :6] = rng.uniform(-1, 1, (n, 6))
    start[:, 6:12] = rng.normal(size=(n, 6))
    start[:, 12:15] = rng.uniform(-0.2, 0.2, (n, 3))
    start[:, 15:19] = _rand_quat(rng, n)
    start[:, 19:25] = rng.normal(size=(n, 6))
    mask = np.array([1, 1, 1, 0, 1, 1], np.int32)
    dev = "cpu"
    qpos, qvel, ctrl = (wp.zeros((n, k), dtype=float, device=dev) for k in (13, 12, 6))
    prev, out = wp.zeros((n, 13), dtype=wp.float64, device=dev), wp.zeros((n, E.RAW), dtype=wp.float64, device=dev)
    lo, hi = np.full(6, -0.5), np.full(6, 0.5)
    hold = start[:, :6] + 0.1
    wp.launch(E._place, dim=n, device=dev,
              inputs=[wp.array(mask, dtype=wp.int32, device=dev), wp.array(start, dtype=wp.float64, device=dev),
                      wp.array(hold, dtype=wp.float64, device=dev),
                      wp.array(lo, dtype=wp.float64, device=dev), wp.array(hi, dtype=wp.float64, device=dev), 6, 6, 1,
                      qpos, qvel, ctrl, prev, out])
    qp, qv, c, o, p = qpos.numpy(), qvel.numpy(), ctrl.numpy(), out.numpy(), prev.numpy()
    for i in range(n):
        if not mask[i]:
            assert not qp[i].any() and not o[i].any()
            continue
        r = np.zeros(9)
        mujoco.mju_quat2Mat(r, start[i, 15:19])
        assert np.allclose(qv[i, 9:12], r.reshape(3, 3).T @ start[i, 22:25], atol=1e-5)
        assert np.allclose(c[i], np.clip(hold[i], lo, hi), atol=1e-7)
        sign = -1.0 if start[i, 15] < 0 else 1.0
        f32 = start[i].astype(np.float32).astype(np.float64)    # the state holds what the float32 simulation holds
        assert np.array_equal(o[i, 15:19], sign * f32[15:19]) and o[i, 15] >= 0
        assert np.array_equal(o[i, :15], f32[:15]) and np.array_equal(o[i, 19:25], f32[19:25])
        assert np.array_equal(p[i, :9], f32[[0, 1, 2, 3, 4, 5, 12, 13, 14]])


def test_contact_sum_matches_collector_rule():
    """Per-world summed normal force by contact kind = Scene.contact_forces' rule (first efc row of each contact,
    kind table of the collector), on random contacts of the real scene's geoms."""
    pytest.importorskip("mujoco")
    from nedm.contact_nrd.collect import so101_grasp as g
    try:
        s = g.Scene(g.load_config())
    except SystemExit as e:   # vendored Menagerie model missing
        pytest.skip(str(e))
    cls, kind = E.collector_classes(s)
    rng = np.random.default_rng(3)
    nw, nc, nj = 5, 400, 32
    geom = rng.integers(0, s.m.ngeom, (nc, 2)).astype(np.int32)
    world = rng.integers(0, nw, nc).astype(np.int32)
    adr = rng.integers(-1, nj, (nc, 6)).astype(np.int32)
    force = rng.uniform(0, 2, (nw, nj)).astype(np.float32)
    nacon = 350                                    # contacts past nacon are ignored
    dev = "cpu"
    fsum = wp.zeros((nw, 7), dtype=float, device=dev)
    wp.launch(E._sum_contacts, dim=nc, device=dev,
              inputs=[wp.array([nacon], dtype=int, device=dev), nc, wp.array(geom, dtype=wp.vec2i, device=dev),
                      wp.array(world, dtype=int, device=dev), wp.array(adr, dtype=int, device=dev),
                      wp.array(force, dtype=float, device=dev), nj, wp.array(cls, dtype=wp.int32, device=dev),
                      wp.array(kind, dtype=wp.int32, device=dev), fsum])
    want = np.zeros((nw, 7))
    for c in range(nacon):
        k = s.kind[s.cls[geom[c, 0]], s.cls[geom[c, 1]]]
        if adr[c, 0] >= 0 and k >= 0:
            want[world[c], k] += force[world[c], adr[c, 0]]
    assert np.allclose(fsum.numpy(), want, atol=1e-4)


# ---- GPU -------------------------------------------------------------------------------------------------------
def _gpu_ok():
    try:
        import mujoco_warp  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


gpu = pytest.mark.skipif(not _gpu_ok(), reason="needs CUDA and mujoco_warp (conda env so101_mjwarp)")


def _rest_start(n):
    q = torch.tensor([[0.0, -0.6, 0.9, 1.2, 0.0, 0.6]], dtype=torch.float64).repeat(n, 1)
    c = torch.tensor([[0.25, 0.0, 0.014993]], dtype=torch.float64).repeat(n, 1)
    c[:, 1] = torch.linspace(-0.1, 0.1, n, dtype=torch.float64)
    yaw = torch.linspace(0, 0.5, n, dtype=torch.float64)
    quat = torch.stack([torch.cos(yaw / 2), 0 * yaw, 0 * yaw, torch.sin(yaw / 2)], 1)
    return dict(q=q, qd=torch.zeros(n, 6, dtype=torch.float64), cube_pos=c, cube_quat=quat)


@gpu
def test_gpu_rest_reset_subset_and_set_cube():
    env = E.MjwarpEnv(4)
    st = env.reset(None, _rest_start(4), warmup=True)
    assert st.contact[:, 2].all() and not st.contact[:, :2].any() and not st.arm_table.any() and not st.invalid.any()
    p0 = st.cube_pos.clone()
    for _ in range(25):                           # hold: cube rests, arm stays (0.5 s)
        st = env.step(st.q)
    assert (st.cube_pos - p0).norm(dim=1).max() < 1e-4
    assert st.contact[:, 2].all() and st.qd.abs().max() < 0.05 and not st.invalid.any()
    keep = st.cube_pos[1:].clone()
    st = env.reset([0], {k: v[:1] for k, v in _rest_start(4).items()})      # others untouched
    assert torch.equal(st.cube_pos[1:], keep)
    pos = torch.tensor([[0.2, 0.05, 0.03]], dtype=torch.float64)
    quat = torch.tensor([[math.cos(0.3), 0.0, 0.0, math.sin(0.3)]], dtype=torch.float64)
    st = env.set_cube([2], pos, quat)
    assert torch.allclose(st.cube_pos[2].cpu(), pos[0], atol=1e-7) and torch.allclose(st.cube_quat[2].cpu(), quat[0], atol=1e-7)
    assert st.cube_v[2].abs().max() == 0 and not st.contact[2, 2]
    st = env.step(st.q)                           # 20 ms fall from 15 mm above rest: the jump is not a rate
    assert st.cube_v[2, 2] < 0 and st.cube_v[2, 2] > -0.3
    assert (env.overflow & 0b1111).eq(0).all()    # no constraint / contact buffer overflow


@gpu
@pytest.mark.skipif(not os.path.exists(VAL), reason="validation data not on this machine")
def test_gpu_matches_cpu_replay_1s():
    """4 nominal validation episodes, recorded first state + commands, 1 s: GPU vs CPU MuJoCo (collector Scene)."""
    import mujoco
    from nedm.contact_nrd.collect import so101_grasp as g
    z = np.load(VAL)
    rows = np.flatnonzero((z["family"] == 0) & (z["lengths"] == 601))[:4]
    arm, cube, act = z["arm"][rows, 0].astype(np.float64), z["cube"][rows, 0].astype(np.float64), z["action"][rows, 0:100:2]
    s = g.Scene(g.load_config())
    m, d = s.m, s.d
    cpu_pos, cpu_flag = [], []
    for i in range(len(rows)):
        mujoco.mj_resetData(m, d)
        d.qpos[:6], d.qvel[:6] = arm[i, :6], arm[i, 6:]
        d.qpos[6:13], d.qvel[6:9] = cube[i, :7], cube[i, 7:10]
        d.qvel[9:12] = g.quat_mat(cube[i, 3:7]).T @ cube[i, 10:13]
        mujoco.mj_forward(m, d)
        fl = []
        for k in range(50):
            d.ctrl[s.act] = np.clip(act[i, k], s.tlo, s.thi).astype(np.float32)
            fm = np.zeros(3)
            for _ in range(40):
                mujoco.mj_step(m, d)
                fm = np.maximum(fm, s.contact_forces()[0][:3])
            fl.append(fm > s.thr)
        cpu_pos.append(d.qpos[6:9].copy())
        cpu_flag.append(np.stack(fl))
    env = E.MjwarpEnv(len(rows))
    env.reset(None, dict(q=arm[:, :6], qd=arm[:, 6:], cube_pos=cube[:, :3], cube_quat=cube[:, 3:7],
                         cube_v=cube[:, 7:10], cube_w=cube[:, 10:13]))
    a = torch.tensor(act, device=env.device)
    flags = []
    for k in range(50):
        st = env.step(a[:, k])
        flags.append(st.contact.cpu().numpy())
    err = np.linalg.norm(st.cube_pos.cpu().numpy() - np.stack(cpu_pos), axis=1)
    assert err.max() < 2e-4
    agree = (np.stack(flags, 1) == np.stack(cpu_flag)).mean()
    assert agree > 0.98
