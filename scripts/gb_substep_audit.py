#!/usr/bin/env python
"""Substep timing audit of the UNMODIFIED f104 collectors (PLAN B2) + local CRM determinism / representativeness checks.

What the recorded ``action[k]`` is: the path follower's output at the last substep of interval k-1, applied from substep 0
of interval k and then refreshed at every physics substep (25 x 2 ms rigid, 50 x 1 ms CRM). This script logs the CLAMPED
triple actually handed to ``hmmwv.Synchronize`` at every substep and reports, per world and channel, the within-interval
max-min, the mean applied minus ``action[k]``, the shares of intervals off by > 0.05 / > 0.1, the throttle/brake flip rate,
and the same restricted to stalled and brake-onset frames.

Hooks (no repo file edited):
  rigid  the frozen runner (``traverse_fdm_rgbd_diverse_chrono.run_chrono`` through ``gen_collect.import_runner`` +
         ``adapted_function``, i.e. the production stop policy) calls ``frame_observer.on_substep(scene, frame, sub, dt,
         action3)`` with the clamped inputs; a small observer stores them. The source-manifest / runtime-fingerprint gates
         of ``gen_collect.main`` are bypassed exactly as ``rigid_moving_collect.py`` does (physics, driver, stop policy intact).
  crm    ``scripts/crm_collect.py`` is imported untouched; the frozen module it calls ``make_driver`` on is pre-imported
         and ``make_driver`` is swapped for one that wraps the real ``ChPathFollowerDriver`` in a proxy whose ``GetInputs``
         logs every call with the sim time. The loop mutates the returned DriverInputs copy (steering clamp) before
         ``hmmwv.Synchronize``, so the proxy reads the previous copy back at the next call = the applied triple.
  determinism   one CRM episode driven twice through ``scripts/crm_collect.py`` itself (subprocess), npz bytes compared.
  representativeness   3 recorded ``collect_v1`` episodes re-driven locally, full horizon, unmodified collector, compared
         with the cluster recording (status, elapsed, goal_reached, max |pose diff|).
Every CRM subprocess is wrapped in ``flock <lock>``; peak GPU memory is sampled from nvidia-smi by the parent.

Run (from the repo root, PYTHONPATH=src:scripts):
  python scripts/gb_substep_audit.py --stage all --out artifacts/traverse/generalist_20260921/B_tracker/audit \
      --crm-config artifacts/traverse/crm_f104_v1/configs/crm_main.json
Stages: select, rigid, crm, determinism, repr, analyze (or all). Outputs: selection.json, rigid/<id>/, crm/<id>/,
determinism/, repr/, substep_audit.json.
"""
from __future__ import annotations

import argparse, hashlib, json, os, re, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
CRM_RUNS = REPO / "artifacts/traverse/crm_f104_v1/collect_v1/runs"
RIGID_RUNS = REPO / "artifacts/traverse/fdm_f104_50h_20260909/production_v3/runs"
DEFAULT_OUT = REPO / "artifacts/traverse/generalist_20260921/B_tracker/audit"
DT, SETTLE_FRAMES = 0.05, 16
CHANNELS = ("steer", "throttle", "brake")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def read_json(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def write_json(p: Path, v) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(v, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o) if isinstance(o, np.generic) else str(o)))


def route_from_recording(run_dir: Path, dest: Path) -> Path:
    """The recorded reference route (command_reference.npz) as the route JSON ``read_route`` accepts."""
    with np.load(run_dir / "command_reference.npz", allow_pickle=False) as c:
        route = {"waypoints": c["reference_waypoints"].tolist(), "speeds": c["reference_speeds"].tolist(),
                 "stations": c["reference_stations"].tolist(), "headings": c["reference_headings"].tolist(),
                 "meta": {"family": "recorded_reference", "source": str(run_dir / "command_reference.npz")}}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(route))
    return dest


# ----------------------------------------------------------------------------------------------------------------- selection
def select_episodes(n_rigid: int, n_crm: int, rigid_horizon_s: float, crm_horizon_s: float, scan: int = 800) -> dict:
    """Deterministic pick (sorted ids, one episode per group): a mix of goal-reached, early-stall blockage and (rigid)
    timeout episodes so stalled and brake-onset intervals exist inside the short audit horizons."""
    def scan_runs(root: Path):
        rows = []
        for eid in sorted(os.listdir(root))[:scan]:
            o = read_json(root / eid / "outcome.json")
            w = o.get("bounded_blockage_v1_windows") or []
            rows.append({"id": eid, "group": re.match(r"^(.*_group_\d+)_", eid).group(1), "status": o["status"], "elapsed_s": o["elapsed_s"],
                         "stall_s": o.get("longest_consecutive_effortful_near_zero_speed_s", 0.), "first_window": (w[0] if w else None)})
        return rows

    def take(rows, pred, k, used):
        out = []
        for r in rows:
            if len(out) >= k:
                break
            if r["group"] in used or not pred(r):
                continue
            out.append(r["id"]); used.add(r["group"])
        return out

    rig = scan_runs(RIGID_RUNS); used = set()
    n_goal = n_rigid - n_rigid * 9 // 20 - 2  # 11 / 7 / 2 for n = 20
    rigid = (take(rig, lambda r: r["status"] == "goal_reached" and 10. <= r["elapsed_s"] <= rigid_horizon_s, n_goal, used)
             + take(rig, lambda r: r["status"] == "prolonged_blockage_terminated" and r["first_window"] is not None and r["first_window"] <= 400, n_rigid * 9 // 20, used)
             + take(rig, lambda r: r["status"] == "timeout", 2, used))
    rigid += take(rig, lambda r: r["status"] == "goal_reached", n_rigid - len(rigid), used)
    crm_rows = scan_runs(CRM_RUNS); used = set()
    n_g, n_b = n_crm * 2 // 5, n_crm * 3 // 10
    crm = (take(crm_rows, lambda r: r["status"] == "goal_reached" and r["elapsed_s"] <= crm_horizon_s, n_g, used)
           + take(crm_rows, lambda r: r["status"] == "soil_breakthrough_terminated" and r["elapsed_s"] <= crm_horizon_s and r["stall_s"] >= 1., n_b, used)
           + take(crm_rows, lambda r: r["status"] == "prolonged_blockage_terminated" and r["first_window"] is not None and r["first_window"] <= 300, n_crm - n_g - n_b, used))
    crm += take(crm_rows, lambda r: r["status"] == "goal_reached", n_crm - len(crm), used)
    # representativeness: the median-elapsed episode of each class among groups not used above
    repr_ids = []
    for st in ("goal_reached", "soil_breakthrough_terminated", "prolonged_blockage_terminated"):
        xs = sorted([r for r in crm_rows if r["status"] == st and r["group"] not in used], key=lambda r: (r["elapsed_s"], r["id"]))
        if xs:
            r = xs[len(xs) // 2]; repr_ids.append(r["id"]); used.add(r["group"])
    return {"rigid": rigid, "crm": crm, "determinism": crm[0], "repr": repr_ids,
            "meta": {"rigid": {r["id"]: r for r in rig if r["id"] in set(rigid)},
                     "crm": {r["id"]: r for r in crm_rows if r["id"] in set(crm) | set(repr_ids)}}}


# ------------------------------------------------------------------------------------------------------------- rigid worker
class SubstepRecorder:
    """frame_observer for the frozen rigid runner: keeps the clamped action at every substep (frame >= 0 only)."""

    def __init__(self, out: Path):
        self.out, self.rows, self.frames = out, [], []

    def on_frame(self, scene, frame, state17, pose3, action3, *, command_context=None):
        self.frames.append((frame, *[float(a) for a in action3], float(command_context["desired_speed_mps"]), bool(command_context["parked"])))

    def on_substep(self, scene, frame, substep, dt_s, action3=None):
        self.rows.append((frame, substep, float(scene.system.GetChTime()), *[float(a) for a in action3]))

    def on_post_substep(self, scene, frame, substep):
        pass

    def finish(self, scene, N, terminal_state17, terminal_pose3, last_action3):
        rows = np.asarray(self.rows, np.float64)
        np.savez_compressed(self.out / "substep_actions.npz", frame=rows[:, 0].astype(np.int64), sub=rows[:, 1].astype(np.int64),
                            t=rows[:, 2], applied=rows[:, 3:6], frame_action=np.asarray([f[1:4] for f in self.frames], np.float64),
                            frame_desired=np.asarray([f[4] for f in self.frames]), frame_parked=np.asarray([f[5] for f in self.frames], bool),
                            n_frames=np.int64(N), world=np.array("rigid"))


def rigid_worker(a) -> None:
    """One rigid episode through the frozen runner + gen_collect's stop-policy adapter, with the substep observer."""
    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    import gen_collect
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    module = gen_collect.import_runner(REPO)
    run, adapter = gen_collect.adapted_function(module)
    case = gen_collect.read(a.case)
    a.minimum_elapsed_s, a.confirm_s, a.recovery_tail_s, a.disable_early_stop = 24., 2., 8., False
    a.f104_policy = gen_collect.StopPolicy(a, case)
    a.command, a.backend, a.depth_ray_scale = "collect", "Vulkan_RT_lavapipe", 1.
    a.record_rgbd_stride, a.path_height_source, a.render_parity, a.rich_telemetry = 0, "truth", False, False
    a.frame_observer = SubstepRecorder(out)
    a.case, a.route, a.out = str(Path(a.case).resolve()), str(Path(a.route).resolve()), str(out)
    t0 = time.time()
    run(a)
    o = gen_collect.read(out / "outcome.json")
    write_json(out / "worker.json", {"world": "rigid", "status": o["status"], "elapsed_s": o["elapsed_s"], "frames": o["frames"],
                                     "wall_s": time.time() - t0, "adapter": adapter, "horizon_s": a.horizon_s,
                                     "gates_bypassed": "source_manifest / FDM_RUNTIME_FINGERPRINT (as rigid_moving_collect.py); physics, driver, stop policy unchanged"})


# --------------------------------------------------------------------------------------------------------------- crm worker
class DriverProxy:
    """Wraps the real ChPathFollowerDriver; logs every GetInputs() (raw PID output) and reads the previous DriverInputs
    copy back on the next call (after the loop's steering clamp) = the triple applied to hmmwv.Synchronize."""

    def __init__(self, real):
        self._real, self._t, self._last = real, None, None
        self.t_sync, self.raw, self.applied, self.desired = [], [], [], []

    def Synchronize(self, t):
        self._t = float(t)
        return self._real.Synchronize(t)

    def GetInputs(self):
        if self._last is not None:
            self.applied.append((float(self._last.m_steering), float(self._last.m_throttle), float(self._last.m_braking)))
        inputs = self._real.GetInputs()
        self.t_sync.append(self._t)
        self.raw.append((float(inputs.m_steering), float(inputs.m_throttle), float(inputs.m_braking)))
        self._last = inputs
        return inputs

    def Advance(self, dt):
        return self._real.Advance(dt)

    def SetDesiredSpeed(self, v):
        self.desired.append((len(self.raw), float(v)))
        return self._real.SetDesiredSpeed(v)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def flush(self):
        if self._last is not None and len(self.applied) < len(self.raw):
            self.applied.append((float(self._last.m_steering), float(self._last.m_throttle), float(self._last.m_braking)))


def crm_worker(a) -> None:
    """scripts/crm_collect.py, untouched, with the frozen module's make_driver swapped for a logging proxy."""
    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    import traverse_fdm_rgbd_diverse_chrono as frozen
    proxies = []
    real_make = frozen.make_driver

    def make_driver(chrono, veh, vehicle, route, tmap):
        p = DriverProxy(real_make(chrono, veh, vehicle, route, tmap)); proxies.append(p); return p

    frozen.make_driver = make_driver
    import crm_collect
    argv = ["--source-root", str(REPO), "--case", a.case, "--route", a.route, "--out", a.out, "--chrono-data", a.chrono_data,
            "--crm-config", a.crm_config, "--horizon-s", str(a.horizon_s)]
    t0 = time.time()
    crm_collect.main(argv)
    wall = time.time() - t0
    assert len(proxies) == 1, len(proxies)
    p = proxies[0]; p.flush()
    out = Path(a.out)
    o = read_json(out / "outcome.json")
    n_calls = len(p.raw)
    substeps = int(round(DT / float(o["crm"]["physics_dt_s"])))
    np.savez_compressed(out / "substep_actions.npz", t=np.asarray(p.t_sync), raw=np.asarray(p.raw), applied=np.asarray(p.applied),
                        desired_calls=np.asarray(p.desired, np.float64), substeps=np.int64(substeps), settle_frames=np.int64(SETTLE_FRAMES),
                        n_frames=np.int64(o["frames"]), world=np.array("crm"))
    write_json(out / "worker.json", {"world": "crm", "status": o["status"], "elapsed_s": o["elapsed_s"], "frames": o["frames"], "wall_s": wall,
                                     "collector_wall_s": o["wall_s"], "rtf": o["crm"]["rtf_sim_over_wall"], "physics_dt_s": o["crm"]["physics_dt_s"],
                                     "n_sph": o["crm"]["n_sph"], "get_inputs_calls": n_calls, "expected_calls": (SETTLE_FRAMES + o["frames"]) * substeps,
                                     "horizon_s": a.horizon_s, "hook": "frozen.make_driver -> DriverProxy (GetInputs logged); crm_collect.py unmodified"})


# ------------------------------------------------------------------------------------------------------ subprocess helpers
class GpuSampler(threading.Thread):
    """Peak nvidia-smi memory of every compute process whose cmdline contains ``tag`` (the run's --out path)."""

    def __init__(self, tag: str, period_s: float = 2.):
        super().__init__(daemon=True)
        self.tag, self.period, self.peak_mib, self.gpu_peak_mib, self.samples, self._halt = tag, period_s, 0, 0, 0, False  # not _stop: Thread.join calls Thread._stop()

    def run(self):
        while not self._halt:
            try:
                txt = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout
                for line in txt.strip().splitlines():
                    pid, mem = [s.strip() for s in line.split(",")[:2]]
                    try:
                        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
                    except OSError:
                        continue
                    if self.tag in cmd:
                        self.peak_mib = max(self.peak_mib, int(mem)); self.samples += 1
                tot = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
                self.gpu_peak_mib = max(self.gpu_peak_mib, int(tot.strip().splitlines()[0]))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(self.period)

    def stop(self):
        self._halt = True


def run_locked(cmd: list[str], lock: str, log: Path, tag: str, env: dict) -> dict:
    """flock <lock> cmd, stdout/err to log, GPU memory sampled; returns rc, wall, peak memory."""
    log.parent.mkdir(parents=True, exist_ok=True)
    sampler = GpuSampler(tag); sampler.start()
    t0 = time.time()
    with open(log, "w") as lg:
        rc = subprocess.run(["flock", lock] + cmd, stdout=lg, stderr=subprocess.STDOUT, env=env, cwd=str(REPO)).returncode
    wall = time.time() - t0
    sampler.stop(); sampler.join(timeout=15)
    return {"rc": rc, "wall_s": wall, "peak_process_mib": sampler.peak_mib, "peak_gpu_total_mib": sampler.gpu_peak_mib, "gpu_samples": sampler.samples}


def crm_env() -> dict:
    return dict(os.environ, OMP_NUM_THREADS=os.environ.get("CRM_OMP", "4"), PYTHONPATH=f"{REPO}/src:{REPO}/scripts")


def unmodified_crm_cmd(case: Path, route: Path, out: Path, a, horizon_s: float | None) -> list[str]:
    cmd = [PY, "-P", "-u", str(REPO / "scripts/crm_collect.py"), "--source-root", str(REPO), "--case", str(case), "--route", str(route),
           "--out", str(out), "--chrono-data", a.chrono_data, "--crm-config", a.crm_config]
    if horizon_s is not None:
        cmd += ["--horizon-s", str(horizon_s)]
    return cmd


# ------------------------------------------------------------------------------------------------------------------ stages
def stage_select(a, out: Path) -> dict:
    sel = select_episodes(a.n_rigid, a.n_crm, a.rigid_horizon_s, a.crm_horizon_s)
    for eid in sel["rigid"]:
        route_from_recording(RIGID_RUNS / eid, out / "routes" / f"{eid}@rigid.json")
    for eid in set(sel["crm"]) | set(sel["repr"]) | {sel["determinism"]}:
        route_from_recording(CRM_RUNS / eid, out / "routes" / f"{eid}@crm.json")
    sel["args"] = {"n_rigid": a.n_rigid, "n_crm": a.n_crm, "rigid_horizon_s": a.rigid_horizon_s, "crm_horizon_s": a.crm_horizon_s}
    write_json(out / "selection.json", sel)
    print(f"selected {len(sel['rigid'])} rigid, {len(sel['crm'])} crm, determinism {sel['determinism']}, repr {sel['repr']}", flush=True)
    return sel


def stage_rigid(a, out: Path, sel: dict) -> None:
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=f"{REPO}/src:{REPO}/scripts")

    def one(eid):
        d = out / "rigid" / eid
        if (d / "substep_actions.npz").exists():
            return eid, "cached", 0.
        cmd = [PY, "-P", "-u", __file__, "--worker", "rigid", "--case", str(RIGID_RUNS / eid / "case.json"), "--route", str(out / "routes" / f"{eid}@rigid.json"),
               "--out", str(d), "--chrono-data", a.chrono_data, "--horizon-s", str(a.rigid_horizon_s)]
        (out / "rigid/logs").mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(out / "rigid/logs" / f"{eid}.log", "w") as lg:
            rc = subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, env=env, cwd=str(REPO)).returncode
        return eid, f"rc={rc}", time.time() - t0

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, min(4, a.rigid_jobs))) as ex:
        for eid, st, w in ex.map(one, sel["rigid"]):
            print(f"  rigid {eid}: {st} ({w:.0f} s)", flush=True)
    write_json(out / "rigid/stage.json", {"wall_s": time.time() - t0, "jobs": a.rigid_jobs, "n": len(sel["rigid"])})
    print(f"rigid stage done in {time.time() - t0:.0f} s", flush=True)


def stage_crm(a, out: Path, sel: dict) -> None:
    t0 = time.time()
    for eid in sel["crm"]:
        d = out / "crm" / eid
        if (d / "substep_actions.npz").exists():
            print(f"  crm {eid}: cached", flush=True); continue
        cmd = [PY, "-P", "-u", __file__, "--worker", "crm", "--case", str(CRM_RUNS / eid / "case.json"), "--route", str(out / "routes" / f"{eid}@crm.json"),
               "--out", str(d), "--chrono-data", a.chrono_data, "--crm-config", a.crm_config, "--horizon-s", str(a.crm_horizon_s)]
        r = run_locked(cmd, a.lock, out / "crm/logs" / f"{eid}.log", str(d), crm_env())
        write_json(d / "run_meta.json", r)
        print(f"  crm {eid}: rc={r['rc']} ({r['wall_s']:.0f} s, peak {r['peak_process_mib']} MiB)", flush=True)
    write_json(out / "crm/stage.json", {"wall_s": time.time() - t0, "n": len(sel["crm"])})
    print(f"crm stage done in {time.time() - t0:.0f} s", flush=True)


def compare_npz(p1: Path, p2: Path) -> dict:
    same = sha256(p1) == sha256(p2)
    diff = {}
    with np.load(p1, allow_pickle=False) as x, np.load(p2, allow_pickle=False) as y:
        for k in x.files:
            if k in y.files and x[k].dtype.kind in "fiub" and x[k].shape == y[k].shape:
                diff[k] = float(np.max(np.abs(x[k].astype(np.float64) - y[k].astype(np.float64)))) if x[k].size else 0.
            else:
                diff[k] = None if k in y.files else "missing"
    return {"byte_identical": same, "sha256": [sha256(p1), sha256(p2)], "max_abs_diff": diff}


def stage_determinism(a, out: Path, sel: dict) -> dict:
    eid = sel["determinism"]
    case, route = CRM_RUNS / eid / "case.json", out / "routes" / f"{eid}@crm.json"
    res = {"episode": eid, "horizon_s": a.crm_horizon_s, "runs": {}}
    for name in ("run_a", "run_b"):
        d = out / "determinism" / name
        if not (d / "episode_complete.json").exists():
            r = run_locked(unmodified_crm_cmd(case, route, d, a, a.crm_horizon_s), a.lock, out / "determinism/logs" / f"{name}.log", str(d), crm_env())
            write_json(d / "run_meta.json", r)
        res["runs"][name] = read_json(d / "run_meta.json")
    da, db = out / "determinism/run_a", out / "determinism/run_b"
    res["files"] = {f: compare_npz(da / f, db / f) for f in ("trajectory.npz", "crm_extra.npz", "command_reference.npz")}
    res["outcome_equal"] = {k: read_json(da / "outcome.json")[k] == read_json(db / "outcome.json")[k] for k in ("status", "elapsed_s", "frames", "goal_reached")}
    audited = out / "crm" / eid / "trajectory.npz"
    if audited.exists():  # the proxied audit run of the same episode: the hook must not change the physics
        res["proxied_run_vs_run_a"] = compare_npz(audited, da / "trajectory.npz")
    res["deterministic"] = all(v["byte_identical"] for v in res["files"].values())
    write_json(out / "determinism/determinism.json", res)
    print(f"determinism: byte-identical {res['deterministic']} ({ {k: v['byte_identical'] for k, v in res['files'].items()} })", flush=True)
    return res


def compare_with_recording(local: Path, rec: Path) -> dict:
    lo, ro = read_json(local / "outcome.json"), read_json(rec / "outcome.json")
    with np.load(local / "trajectory.npz") as L, np.load(rec / "trajectory.npz") as R:
        n = min(len(L["pose"]), len(R["pose"]))
        d = np.linalg.norm(L["pose"][:n, :2] - R["pose"][:n, :2], axis=1)
        dyaw = np.abs(np.angle(np.exp(1j * (L["pose"][:n, 2] - R["pose"][:n, 2]))))
        ds = np.abs(L["state"][:n] - R["state"][:n]).max(0)
    first = lambda thr: (int(np.argmax(d > thr)) if (d > thr).any() else None)
    return {"cluster": {k: ro[k] for k in ("status", "elapsed_s", "frames", "goal_reached")}, "local": {k: lo[k] for k in ("status", "elapsed_s", "frames", "goal_reached")},
            "same_status": lo["status"] == ro["status"], "same_goal_reached": lo["goal_reached"] == ro["goal_reached"],
            "elapsed_diff_s": lo["elapsed_s"] - ro["elapsed_s"], "common_frames": int(n), "max_xy_diff_m": float(d.max()), "xy_diff_at_last_common_m": float(d[-1]),
            "xy_diff_at_frame0_m": float(d[0]),
            "max_yaw_diff_rad": float(dyaw.max()), "first_frame_xy_diff_gt_1cm": first(0.01), "first_frame_xy_diff_gt_10cm": first(0.1),
            "first_frame_xy_diff_gt_1m": first(1.0), "max_abs_state_diff_vx_mps": float(ds[0]),
            "cluster_host": ro.get("host"), "cluster_rtf": ro["crm"]["rtf_sim_over_wall"], "local_rtf": lo["crm"]["rtf_sim_over_wall"], "local_wall_s": lo["wall_s"]}


def stage_repr(a, out: Path, sel: dict) -> dict:
    res = {"full_horizon": {}, "audit_runs_first_20s": {}}
    for eid in sel["repr"]:
        d = out / "repr" / eid
        if not (d / "episode_complete.json").exists():
            r = run_locked(unmodified_crm_cmd(CRM_RUNS / eid / "case.json", out / "routes" / f"{eid}@crm.json", d, a, None), a.lock, out / "repr/logs" / f"{eid}.log", str(d), crm_env())
            write_json(d / "run_meta.json", r)
        res["full_horizon"][eid] = {**compare_with_recording(d, CRM_RUNS / eid), "run_meta": read_json(d / "run_meta.json")}
        print(f"  repr {eid}: {res['full_horizon'][eid]['local']['status']} vs cluster {res['full_horizon'][eid]['cluster']['status']}, "
              f"max xy diff {res['full_horizon'][eid]['max_xy_diff_m']:.3f} m", flush=True)
    for eid in sel["crm"]:  # the proxied 20 s audit drives against the same cluster recordings (prefix comparison)
        if (out / "crm" / eid / "trajectory.npz").exists():
            res["audit_runs_first_20s"][eid] = compare_with_recording(out / "crm" / eid, CRM_RUNS / eid)
    write_json(out / "repr/representativeness.json", res)
    return res


# ---------------------------------------------------------------------------------------------------------------- analysis
def load_intervals(d: Path, world: str):
    """(applied (n,S,3), action (n,3) recorded, z1 (n,17), parked (n,), sanity dict) for one audited run."""
    with np.load(d / "trajectory.npz", allow_pickle=False) as t:
        action, z1, parked = t["action"].astype(np.float64), t["state"].astype(np.float32), t["parked"]
    n = len(action)
    with np.load(d / "substep_actions.npz", allow_pickle=False) as s:
        if world == "rigid":
            frame, sub, ap = s["frame"], s["sub"], s["applied"]
            S = int(sub.max()) + 1
            assert len(ap) == n * S and (frame == np.repeat(np.arange(n), S)).all() and (sub == np.tile(np.arange(S), n)).all(), (len(ap), n, S)
            applied = ap.reshape(n, S, 3)
            fa = s["frame_action"]
            sanity = {"on_frame_vs_recorded_max_abs": float(np.abs(fa - action).max())}
        else:
            S, settle = int(s["substeps"]), int(s["settle_frames"])
            ap, raw = s["applied"], s["raw"]
            assert len(ap) == len(raw) == (settle + n) * S, (len(ap), len(raw), n, S)
            applied = ap.reshape(settle + n, S, 3)[settle:]
            raw_i = raw.reshape(settle + n, S, 3)[settle:]
            # the clamp only touches steering; throttle/brake applied == raw, settle steering forced to 0
            sanity = {"raw_vs_applied_throttle_brake_max_abs": float(np.abs(raw_i[:, :, 1:] - applied[:, :, 1:]).max()),
                      "settle_steering_max_abs": float(np.abs(ap.reshape(settle + n, S, 3)[:settle, :, 0]).max()),
                      "steer_clamp_active_share": float(np.mean(np.abs(raw_i[:, :, 0] - applied[:, :, 0]) > 1e-12))}
    sanity["sub0_vs_recorded_max_abs"] = float(np.abs(applied[:, 0, :].astype(np.float32) - action.astype(np.float32)).max())
    return applied, action, z1, parked, sanity


def channel_stats(applied: np.ndarray, action: np.ndarray, mask: np.ndarray) -> dict:
    """Per-channel within-interval range and mean-minus-recorded distributions over the intervals in ``mask``."""
    ap, ac = applied[mask], action[mask]
    n = int(mask.sum())
    if n == 0:
        return {"intervals": 0}
    out = {"intervals": n, "channels": {}}
    rng_all = ap.max(1) - ap.min(1)
    dev_all = ap.mean(1) - ac
    q = lambda x, p: float(np.percentile(x, p))
    for c, name in enumerate(CHANNELS):
        rng, dev = rng_all[:, c], dev_all[:, c]
        out["channels"][name] = {
            "range": {"mean": float(rng.mean()), "p50": q(rng, 50), "p90": q(rng, 90), "p99": q(rng, 99), "max": float(rng.max()),
                      "share_zero": float(np.mean(rng == 0)), "share_gt_0.05": float(np.mean(rng > 0.05)), "share_gt_0.1": float(np.mean(rng > 0.1))},
            "mean_minus_recorded": {"mean": float(dev.mean()), "abs_p50": q(np.abs(dev), 50), "abs_p90": q(np.abs(dev), 90), "abs_p99": q(np.abs(dev), 99),
                                    "abs_max": float(np.abs(dev).max()), "share_abs_gt_0.05": float(np.mean(np.abs(dev) > 0.05)), "share_abs_gt_0.1": float(np.mean(np.abs(dev) > 0.1))}}
    any_dev = np.abs(dev_all).max(1)
    flip_within = (ap[:, :, 1] > 0).any(1) & (ap[:, :, 2] > 0).any(1)
    mode0 = np.where(ac[:, 1] > 0, 1, np.where(ac[:, 2] > 0, 2, 0))
    other = np.where(mode0 == 1, (ap[:, :, 2] > 0).any(1), np.where(mode0 == 2, (ap[:, :, 1] > 0).any(1), ((ap[:, :, 1] > 0) | (ap[:, :, 2] > 0)).any(1)))
    out["any_channel"] = {"share_abs_mean_dev_gt_0.05": float(np.mean(any_dev > 0.05)), "share_abs_mean_dev_gt_0.1": float(np.mean(any_dev > 0.1)),
                          "share_range_gt_0.05": float(np.mean(rng_all.max(1) > 0.05)), "share_range_gt_0.1": float(np.mean(rng_all.max(1) > 0.1)),
                          "share_range_all_zero": float(np.mean(rng_all.max(1) == 0))}
    mid = (ac[:, 2] <= 0) & (ap[:, :, 2] > 0).any(1)  # recorded no brake at k, but the brake is on at some substep of k
    bshare = (ap[mid, :, 2] > 0).mean(1) if mid.any() else np.zeros(0)
    out["flips"] = {"within_interval_rate": float(flip_within.mean()), "within_interval_count": int(flip_within.sum()),
                    "mode_switch_after_sub0_rate": float(other.mean()),
                    "mid_interval_brake_onset_count": int(mid.sum()), "mid_interval_brake_onset_rate": float(mid.mean()),
                    "mid_interval_brake_substep_share": ({"median": float(np.median(bshare)), "mean": float(bshare.mean()), "min": float(bshare.min()),
                                                          "max": float(bshare.max())} if mid.any() else None)}
    return out


def analyze_world(out: Path, world: str, ids: list[str]) -> dict:
    from gb_build_cache import stalled_mask, hold_ok_mask, brake_onset_mask
    A, AC, masks, eps, sanity = [], [], {"stalled": [], "brake_onset": [], "pre_brake_onset": [], "hold_ok": [], "parked": [], "has_succ": [], "flip": []}, [], []
    for eid in ids:
        d = out / world / eid
        if not (d / "substep_actions.npz").exists():
            eps.append({"id": eid, "missing": True}); continue
        applied, action, z1, parked, san = load_intervals(d, world)
        st = stalled_mask(z1, action.astype(np.float32)); hold, flip = hold_ok_mask(action.astype(np.float32)); bo = brake_onset_mask(action.astype(np.float32))
        A.append(applied); AC.append(action)
        masks["stalled"].append(st); masks["brake_onset"].append(bo); masks["pre_brake_onset"].append(np.r_[bo[1:], False]); masks["hold_ok"].append(hold); masks["parked"].append(parked)
        has_succ = np.ones(len(action), bool); has_succ[-1] = False  # the last interval of an episode has no recorded successor: hold_ok undefined there
        masks["has_succ"].append(has_succ); masks["flip"].append(flip)
        w = read_json(d / "worker.json"); meta = read_json(d / "run_meta.json") if (d / "run_meta.json").exists() else {}
        eps.append({"id": eid, "status": w["status"], "elapsed_s": w["elapsed_s"], "frames": w["frames"], "wall_s": w["wall_s"], "stalled_frames": int(st.sum()),
                    "brake_onsets": int(bo.sum()), "recorded_flips": int(flip.sum()), "peak_process_mib": meta.get("peak_process_mib"), "rtf": w.get("rtf"), **san})
        sanity.append(san)
    if not A:
        return {"episodes": eps, "error": "no audited runs"}
    applied, action = np.concatenate(A), np.concatenate(AC)
    m = {k: np.concatenate(v) for k, v in masks.items()}
    n = len(action)
    allm = np.ones(n, bool)
    res = {"world": world, "substeps_per_interval": int(applied.shape[1]), "episodes": eps, "n_intervals": n,
           "sanity_max": {k: max(s[k] for s in sanity) for k in sanity[0]},
           "regimes": {"all": channel_stats(applied, action, allm), "stalled": channel_stats(applied, action, m["stalled"]),
                       "moving": channel_stats(applied, action, ~m["stalled"]), "brake_onset": channel_stats(applied, action, m["brake_onset"]),
                       "pre_brake_onset": channel_stats(applied, action, m["pre_brake_onset"]), "parked": channel_stats(applied, action, m["parked"]),
                       "hold_ok": channel_stats(applied, action, m["hold_ok"] & m["has_succ"]), "not_hold_ok": channel_stats(applied, action, ~m["hold_ok"] & m["has_succ"])},
           "n_transitions": int(m["has_succ"].sum()),
           "recorded_transition_flip_rate": float(m["flip"][m["has_succ"]].mean()),
           "hold_ok_share": float(m["hold_ok"][m["has_succ"]].mean()),
           "wall_s_total": sum(e.get("wall_s", 0.) for e in eps), "wall_s_per_sim_s": (sum(e.get("wall_s", 0.) for e in eps) / max(sum(e.get("elapsed_s", 0.) for e in eps), 1e-9)),
           "peak_process_mib_max": max([e.get("peak_process_mib") or 0 for e in eps] + [0])}
    return res


def stage_analyze(a, out: Path, sel: dict) -> dict:
    res = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "selection": {k: sel[k] for k in ("rigid", "crm", "determinism", "repr", "args")},
           "definitions": {"applied": "clamped [steer, throttle, brake] handed to hmmwv.Synchronize at each substep of interval k (frame >= 0)",
                           "range": "max - min over the substeps of one interval, per channel", "mean_minus_recorded": "mean over substeps minus recorded action[k]",
                           "flip_within_interval": "throttle > 0 at some substep and brake > 0 at another substep of the same interval",
                           "mode_switch_after_sub0": "the exclusive throttle/brake mode of action[k] is left at some later substep of interval k",
                           "stalled": "gb_build_cache.stalled_mask on the audited run (|vx|<0.3 & throttle>0.3, runs >= 20 frames)",
                           "brake_onset": "brake > 0 at k and <= 0 at k-1 (recorded); pre_brake_onset = the interval before",
                           "hold_ok": "gb_build_cache.hold_ok_mask on the audited run's recorded actions; the hold_ok / not_hold_ok regimes, hold_ok_share and "
                                      "recorded_transition_flip_rate cover only the n_transitions intervals with a recorded successor (the last interval of each episode is excluded, as in the cache's retention)",
                           "mid_interval_brake_substep_share": "among intervals with recorded brake <= 0 whose applied path brakes at some substep: the share of substeps with brake > 0",
                           "xy_diff_at_frame0_m": "distance between the local drive's and the cluster recording's pose at frame 0 (after the 0.8 s settle)"},
           "rigid": analyze_world(out, "rigid", sel["rigid"]), "crm": analyze_world(out, "crm", sel["crm"])}
    for name in ("determinism/determinism.json", "repr/representativeness.json", "rigid/stage.json", "crm/stage.json"):
        if (out / name).exists():
            res[name.split("/")[-1].replace(".json", "")] = read_json(out / name)
    res["rigid_local_vs_cluster_recording"] = {}
    for eid in sel["rigid"]:  # rigid is deterministic per machine only; this documents the local-vs-cluster divergence
        d = out / "rigid" / eid
        if (d / "trajectory.npz").exists():
            lo, ro = read_json(d / "outcome.json"), read_json(RIGID_RUNS / eid / "outcome.json")
            with np.load(d / "trajectory.npz") as L, np.load(RIGID_RUNS / eid / "trajectory.npz") as R:
                n = min(len(L["pose"]), len(R["pose"])); dxy = np.linalg.norm(L["pose"][:n, :2] - R["pose"][:n, :2], axis=1)
            res["rigid_local_vs_cluster_recording"][eid] = {"local_status": lo["status"], "cluster_status": ro["status"], "common_frames": int(n),
                                                             "max_xy_diff_m": float(dxy.max()), "xy_diff_at_frame0_m": float(dxy[0]),
                                                             "first_frame_xy_diff_gt_1cm": (int(np.argmax(dxy > .01)) if (dxy > .01).any() else None)}
    write_json(out / "substep_audit.json", res)
    for w in ("rigid", "crm"):
        r = res[w]
        if "error" in r:
            print(f"{w}: {r['error']}"); continue
        al = r["regimes"]["all"]
        print(f"{w}: {r['n_intervals']} intervals from {sum(1 for e in r['episodes'] if not e.get('missing'))} episodes; sanity {r['sanity_max']}")
        for c in CHANNELS:
            ch = al["channels"][c]
            print(f"   {c:8s} range p50/p90/p99/max {ch['range']['p50']:.4f}/{ch['range']['p90']:.4f}/{ch['range']['p99']:.4f}/{ch['range']['max']:.3f}  "
                  f"|mean-rec| p50/p90/p99 {ch['mean_minus_recorded']['abs_p50']:.4f}/{ch['mean_minus_recorded']['abs_p90']:.4f}/{ch['mean_minus_recorded']['abs_p99']:.4f}  "
                  f">0.05 {ch['mean_minus_recorded']['share_abs_gt_0.05']:.4f} >0.1 {ch['mean_minus_recorded']['share_abs_gt_0.1']:.4f}")
        print(f"   any channel |mean-rec| >0.05 {al['any_channel']['share_abs_mean_dev_gt_0.05']:.4f} >0.1 {al['any_channel']['share_abs_mean_dev_gt_0.1']:.4f}; "
              f"flip within interval {al['flips']['within_interval_rate']:.4f}; recorded flip rate {r['recorded_transition_flip_rate']:.4f}")
        for reg in ("stalled", "moving", "brake_onset"):
            s = r["regimes"][reg]
            if s["intervals"]:
                print(f"   {reg:12s} n={s['intervals']:6d} any |mean-rec| >0.05 {s['any_channel']['share_abs_mean_dev_gt_0.05']:.4f} >0.1 {s['any_channel']['share_abs_mean_dev_gt_0.1']:.4f} "
                      f"flip {s['flips']['within_interval_rate']:.4f}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--stage", default="all", choices=("all", "select", "rigid", "crm", "determinism", "repr", "analyze"))
    ap.add_argument("--n-rigid", type=int, default=20); ap.add_argument("--n-crm", type=int, default=10)
    ap.add_argument("--rigid-horizon-s", type=float, default=30.); ap.add_argument("--crm-horizon-s", type=float, default=20.)
    ap.add_argument("--rigid-jobs", type=int, default=4)
    ap.add_argument("--chrono-data", default="/home/harry/chrono/data")
    ap.add_argument("--crm-config", default=str(REPO / "artifacts/traverse/crm_f104_v1/configs/crm_main.json"))
    ap.add_argument("--lock", default="/tmp/luffy_crm.lock")
    ap.add_argument("--skip-repr", action="store_true"); ap.add_argument("--skip-determinism", action="store_true")
    # internal worker mode
    ap.add_argument("--worker", choices=("rigid", "crm")); ap.add_argument("--case"); ap.add_argument("--route"); ap.add_argument("--horizon-s", type=float)
    a = ap.parse_args()
    if a.worker == "rigid":
        rigid_worker(a); return
    if a.worker == "crm":
        crm_worker(a); return
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    assert Path(a.crm_config).exists(), a.crm_config
    stages = ["select", "rigid", "crm", "determinism", "repr", "analyze"] if a.stage == "all" else [a.stage]
    if a.skip_repr and "repr" in stages: stages.remove("repr")
    if a.skip_determinism and "determinism" in stages: stages.remove("determinism")
    sel = stage_select(a, out) if "select" in stages else read_json(out / "selection.json")
    t0 = time.time()
    if "rigid" in stages: stage_rigid(a, out, sel)
    if "crm" in stages: stage_crm(a, out, sel)
    if "determinism" in stages: stage_determinism(a, out, sel)
    if "repr" in stages: stage_repr(a, out, sel)
    if "analyze" in stages: stage_analyze(a, out, sel)
    print(f"stages {stages} done in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
