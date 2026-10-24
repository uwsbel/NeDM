"""Tests of the real2sim round v8 integration (branch claude/r2s-v8, 2026-10-24): the merged table DR
(``OnlineConfig.table_bias`` / ``table_render``), image latency (``OnlineConfig.img_lag``), scene v4 + base add-ons and
the memory recipe work together, and the variant configs are what ``~/so101_r2s_claude/v8/VARIANTS.md`` says.
- TableLagRender (CPU, kin world, jt_h110 + dev tasks): img_lag + table_render, one lag column and per camera: every
  render call draws the arm at Phi of ITS lagged state (``table_frame`` maps G blocks of image states), gets each
  world's plane and the held flag; the aux cube target = the first lag column's drawn cube (+ the plane height), the
  aux goal = the image's goal; the student's q input stays Phi(current q).
- Record (CPU): ``OnlineDAgger.ckpt_record`` is {} with every v8 option off (old checkpoint layout) and lists the
  options when on; a note-only img_lag is off; an img_lag dict without lo_ms / hi_ms is refused.
- Variants (CPU): each v8 online config = its family base config + exactly its switches' keys; the V+H student config =
  the scene4 student config + hist 4 + cmd_in; ``viral_r2s_v8_variants.json`` names existing configs; the DR ranges
  cover the measured real values (``~/so101_r2s_claude/v8/sysid/sysid_v8.json``, skipped without it).
- RobotLoad (CPU): a hist 4 + cmd_in student saved with the v8 record loads in ``hw_lib.StudentPolicy`` (no new robot
  input) and gives the chunk of the same student saved without the record.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -m unittest \
        tests.so101_policy.test_v8_integrate -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from nedm.so101_policy import table_bias as TB
from nedm.so101_policy.kin import SO101Kin
from nedm.so101_policy.vision import viral as VI

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import studentlab_ref as R  # noqa: E402

CFG = REPO / "configs" / "so101_policy" / "vision"
SYSID = Path("~/so101_r2s_claude/v8/sysid/sysid_v8.json").expanduser()


class PlaneViews(R.FakeViews):
    """FakeViews with the v8 render inputs (``plane``, ``held``); records every call."""
    table_render_ok = True

    def __init__(self, *a, plate_on=False, **k):
        super().__init__(*a, **k)
        self.plate_on = plate_on
        self.kw = []

    def render(self, q, cube_pos, cube_quat, goal_pose, dr=None, light=None, keep=False, plane=None, held=None):
        self.kw.append(dict(q=q.detach().clone(), cp=cube_pos.detach().clone(), gp=goal_pose.detach().clone(),
                            keep=bool(keep), plane=None if plane is None else plane.detach().clone(),
                            held=None if held is None else held.detach().clone()))
        return super().render(q, cube_pos, cube_quat, goal_pose, dr, light, keep)


def online_with(st, ocfg_over, views, seed=0, envs=16):
    from nedm.so101_policy import joint_env as JE
    o = dict(envs=envs, buffer=64, steps_per_iter=2, horizon=60, lr=1e-3, lr_backbone=1e-3)
    o.update(ocfg_over)
    with R.kin_world():
        return VI.OnlineDAgger(st, JE.JointTeacher(str(R.JT_PATH), "cpu"), R.load(), "unused-nrd.pt", views,
                               VI.OnlineConfig.from_dict(o), "cpu", seed, lambda s: None)


def start(od):
    A = torch.arange(od.env.B)
    od.env.reset()
    od._draw(A), od._lag_reset(A), od._bias_reset(A), od._qd_reset(A)


@unittest.skipUnless(R.HAVE, "jt_h110 / dev tasks missing")
class TableLagRender(unittest.TestCase):
    def run_case(self, per_camera):
        N = 16
        st = R.tiny_student(0)
        views = PlaneViews(st.cfg.res, st.cfg.cameras, plate_on=True)
        od = online_with(st, dict(horizon=40, buffer=4000, table_bias=dict(p=0.7, slope_y=[-20, 20]), table_render=True,
                                  img_lag=dict(lo_ms=10, hi_ms=80, per_camera=per_camera)), views, envs=N)
        self.assertTrue(od.tb_render)
        self.assertIsNotNone(od.il)
        start(od)
        kin = SO101Kin()
        G = len(st.cfg.cameras) if per_camera else 1
        rec = {}
        orig = od._lag_states

        def spy(gp):
            s = orig(gp)
            rec["s"] = [tuple(x.clone() for x in z) for z in s]
            return s

        od._lag_states = spy
        n_on, lags = 0, set()
        for t in range(16):
            od.it = t
            F = od.env.task.F
            pre = dict(plane=od.tb_plane.clone(), on=od.tb_on.clone(), ptr=od.buf.ptr, fresh=od.fresh.clone(),
                       held=None if F is None else F["held"].clone(), q=od.env.ws.q.clone())
            k0 = len(views.kw)
            od.act()
            calls = views.kw[k0:]
            self.assertEqual(len(calls), G, t)
            self.assertEqual(len(rec["s"]), G, t)
            on = pre["on"]
            for c in range(G):
                iq, icp, _, igp = rec["s"][c]
                want, _ = TB.observe(kin, iq.double(), pre["plane"])
                self.assertLess(float((calls[c]["q"].double() - want).abs().max()), 2e-6, (t, c))
                if bool((~on).any()):
                    self.assertTrue(torch.equal(calls[c]["q"][~on], iq[~on]), (t, c))   # no plane: the lagged arm
                self.assertTrue(torch.equal(calls[c]["plane"], pre["plane"]), (t, c))
                self.assertTrue(torch.equal(calls[c]["cp"], icp), (t, c))
                self.assertTrue(torch.equal(calls[c]["gp"], igp), (t, c))
                held = torch.zeros(N, dtype=torch.bool) if pre["held"] is None else pre["held"]
                self.assertTrue(torch.equal(calls[c]["held"], held & ~pre["fresh"]), (t, c))
                self.assertEqual(calls[c]["keep"], c > 0, (t, c))         # later cameras keep the per-call state
            # aux: the first column's drawn cube (+ plane height) and the image's goal
            _, icp0, _, igp0 = rec["s"][0]
            sl = (pre["ptr"] + torch.arange(N)) % od.buf.cap
            hz = torch.where(on, TB.height(pre["plane"], icp0[:, :2].double()), 0.0).float()
            want_cp = icp0 + torch.cat((torch.zeros(N, 2), hz[:, None]), -1)
            self.assertTrue(torch.equal(od.buf.aux[sl], VI.aux_targets(igp0, want_cp).float()), t)
            # the student's q input (buffer q) = Phi of the CURRENT state, not of the image state
            qnow, _ = TB.observe(kin, pre["q"].double(), pre["plane"])
            self.assertLess(float((od.buf.q[sl].double() - qnow).abs().max()), 2e-6, t)
            n_on += int(on.sum())
            lags.update(round(float(v), 3) for v in od.lag_x.flatten())
        self.assertGreater(n_on, 0.5 * N * 16)
        self.assertGreater(len(lags), 10)                                # continuous lags, redrawn

    def test_one_lag_column(self):
        self.run_case(False)

    def test_per_camera(self):
        self.run_case(True)


@unittest.skipUnless(R.HAVE, "jt_h110 / dev tasks missing")
class Record(unittest.TestCase):
    def test_ckpt_record_and_img_lag_validation(self):
        od, _ = R.make_online(R.tiny_student(0), {}, 0, 4)
        self.assertEqual(od.ckpt_record(), {})
        od, _ = R.make_online(R.tiny_student(0), dict(img_lag={"note": "off"}, table_bias={"note": "off"}), 0, 4)
        self.assertIsNone(od.il)
        self.assertIsNone(od.tb)
        self.assertEqual(od.ckpt_record(), {})
        with self.assertRaises(ValueError):
            VI.OnlineConfig.from_dict(dict(img_lag={"redraw_p": 1.0}))
        with self.assertRaises(ValueError):
            VI.OnlineConfig.from_dict(dict(img_lag={"lo_ms": 10.0}))
        VI.OnlineConfig.from_dict(dict(img_lag={"lo_ms": 0.0, "hi_ms": 0.0}))          # explicit zero: allowed
        st = R.tiny_student(0)
        od = online_with(st, dict(table_bias=dict(p=0.5), table_render=True, img_lag=dict(lo_ms=10, hi_ms=80)),
                         PlaneViews(st.cfg.res, st.cfg.cameras), envs=4)
        r = od.ckpt_record()["v8_online"]
        self.assertEqual(r["img_lag"]["hi_ms"], 80.0)
        self.assertEqual(r["table_bias"]["p"], 0.5)
        self.assertTrue(r["table_render"])
        self.assertNotIn("plate_mm", r)


def _load(name):
    d = json.loads((CFG / name).read_text())
    d.pop("note", None)
    return d


class Variants(unittest.TestCase):
    TABLE_KEYS = {"table_bias", "plate_mm", "table_render"}

    def test_online_configs_are_base_plus_switches(self):
        C, S, A3 = _load("viral_r2s_online_relabel_v4.json"), _load("viral_r2s_online_v1.json"), \
            _load("studentlab/a3_online.json")
        for name, base, sw in (("C_T", C, "T"), ("S_T", S, "T"), ("C_TL", C, "TL"), ("S_TL", S, "TL"),
                               ("S_TLH", S, "TLH")):
            got = _load(f"viral_r2s_online_v8_{name}.json")
            VI.OnlineConfig.from_dict(dict(got))
            want = dict(base)
            self.assertTrue(self.TABLE_KEYS <= set(got), name)
            want.update({k: got[k] for k in self.TABLE_KEYS})
            self.assertTrue(got["table_render"], name)
            if "L" in sw:
                self.assertNotIn("lag_range", got, name)
                want.pop("lag_range", None)
                want["img_lag"] = got["img_lag"]
                dyn = dict(base.get("dyn") or {"p": 0.8})
                dyn.update(d_lo=0, d_hi=2, delay_mode="queue")
                self.assertEqual(got["dyn"], dyn, name)
                want["dyn"] = dyn
                want["dyn_student_only"] = True
            else:
                self.assertNotIn("img_lag", got, name)
            if "H" in sw:
                for k in ("goal_jump", "hold_cap_s", "light_redraw_p"):
                    want[k] = A3[k]
            self.assertEqual(got, want, name)

    def test_student_configs(self):
        v, vh = _load("viral_r2s_v8_scene4_student.json"), _load("viral_r2s_v8_scene4_hist_student.json")
        a3 = _load("studentlab/a3_student.json")
        self.assertEqual({k: x for k, x in vh.items() if k not in ("hist", "cmd_in")}, v)
        self.assertEqual((vh["hist"], vh["cmd_in"]), (a3["hist"], a3["cmd_in"]))
        VI.ViralConfig.from_dict(vh)
        b1 = _load("studentlab/b1_student.json")          # V = b1 (background DR) + scene v4 + base add-ons
        self.assertEqual({k: x for k, x in v.items() if k not in ("scene", "scene_sha256", "real_dr")},
                         {k: x for k, x in b1.items() if k not in ("scene", "scene_sha256", "real_dr")})
        self.assertIn("base_addon", v["real_dr"])

    def test_variant_list(self):
        d = json.loads((CFG / "viral_r2s_v8_variants.json").read_text())
        names = [r["variant"] for r in d["variants"]]
        self.assertEqual(names, ["C_T", "S_T", "C_TV", "S_TV", "C_TL", "S_TL", "C_TVL", "S_TVLH"])
        for r in d["variants"]:
            self.assertTrue((REPO / r["student_cfg"]).exists(), r)
            self.assertTrue((REPO / r["online_cfg"]).exists(), r)
            sw = set(r["switches"])
            self.assertIn("T", sw)
            on = _load(Path(r["online_cfg"]).name)
            stc = _load(Path(r["student_cfg"]).name)
            self.assertEqual("img_lag" in on, "L" in sw, r["variant"])
            self.assertEqual("goal_jump" in on, "H" in sw, r["variant"])
            self.assertEqual(bool(stc.get("hist")), "H" in sw, r["variant"])
            self.assertEqual("base_addon" in (stc.get("real_dr") or {}), "V" in sw, r["variant"])
            self.assertEqual("v4" in r["scene"]["amd"], "V" in sw, r["variant"])
            want_init = "v4_relabel_s63" if r["family"] == "C" else "v2_online2_s28"
            self.assertIn(want_init, r["init"]["amd"])
            base = _load("viral_r2s_online_relabel_v4.json" if r["family"] == "C" else "viral_r2s_online_v1.json")
            self.assertEqual(on["envs"], base["envs"])
            self.assertEqual(on.get("relabel_cmd_speed"), base.get("relabel_cmd_speed"), r["variant"])

    @unittest.skipUnless(SYSID.exists(), "sysid_v8.json missing")
    def test_dr_covers_the_measured_ranges(self):
        s = json.loads(SYSID.read_text())
        c0, c1, c2 = s["table_plane"]["coef"]
        for name in ("C_T", "S_T", "C_TL", "S_TL", "S_TLH"):
            got = _load(f"viral_r2s_online_v8_{name}.json")
            tb = TB.TableBiasConfig.from_dict(got["table_bias"])
            drop = -(c0 + c1 * tb.x_ref_m) * 1e3                     # mm below the sim table at x_ref
            for dmm in (drop, drop + 5.0, -s["table_dz_range_m"]["lo"] * 1e3):   # real, the gate's real_m5, sysid lo
                self.assertTrue(tb.drop_mm[0] <= dmm <= tb.drop_mm[1], (name, dmm))
            self.assertEqual(tb.drop_mm[0], 0.0)                     # the sim table is in the range
            pm = s["table_dz_range_m"]["plane_option"]
            for v, (lo, hi), tol in ((c1, tb.slope_x, pm["c1_pm"]), (c2, tb.slope_y, pm["c2_pm"])):
                self.assertTrue(lo <= (v - tol) * 1e3 and (v + tol) * 1e3 <= hi, (name, v, lo, hi))
                self.assertTrue(lo <= 0.0 <= hi)
            lo, hi = got["plate_mm"]
            self.assertTrue(lo <= s["mat"]["plate_thickness_m"]["real"] * 1e3 <= hi)
            self.assertTrue(lo <= s["mat"]["plate_thickness_m"]["sim"] * 1e3 <= hi)
            if "img_lag" in got:
                il = VI.ImgLagConfig.from_dict(got["img_lag"])
                pr = s["image_lag_steps"]["plausible_range_p50_steps"]
                step = 20.0
                for cam in ("wrist", "external"):
                    self.assertTrue(il.lo_ms <= pr[cam][0] * step and pr[cam][1] * step <= il.hi_ms, (name, cam))
                bins = [int(k) for k, p in s["image_lag_steps"]["dr_recommendation"]["steps_probs"].items() if p > 0.05]
                self.assertTrue(il.lo_ms <= min(bins) * step and max(bins) * step <= il.hi_ms, (name, bins))
                for ms in (28.0, 62.0):                              # the latency fit's p1 / p99 (v8/latency)
                    self.assertTrue(il.lo_ms <= ms <= il.hi_ms, (name, ms))
                d = s["loop_period_ms"]["dr_recommendation"]["robot_dyn"]["d_lo"]
                self.assertTrue(got["dyn"]["d_lo"] <= d <= got["dyn"]["d_hi"], name)
                self.assertEqual(got["dyn"]["delay_mode"], "queue")


def _hw():
    hw = REPO / "scripts" / "so101_policy" / "real2sim" / "hw"
    if str(hw) not in sys.path:
        sys.path.insert(0, str(hw))
    import hw_lib
    return hw_lib


class RobotLoad(unittest.TestCase):
    def test_v8_record_loads_on_the_robot_side(self):
        H = _hw()
        rec = {"v8_online": {"img_lag": {"lo_ms": 10.0, "hi_ms": 80.0}, "table_bias": {"p": 0.7},
                             "table_render": True, "plate_mm": [1.0, 5.0]}}
        rng = np.random.default_rng(0)
        seq = []
        for _ in range(4):
            imgs = {c: rng.integers(0, 256, (16, 16, 3), dtype=np.uint8) for c in H.REAL_CAMERAS}
            q, qd = rng.uniform(-1, 1, 6), rng.uniform(-1, 1, 6)
            seq.append((imgs, q, qd, q + rng.uniform(-0.1, 0.1, (5, 6))))
        with tempfile.TemporaryDirectory() as tmp:
            st = VI.expand_student(R.tiny_student(0), hist=4, cmd_in=True, device="cpu")
            with torch.no_grad():
                w = st.head[0].weight
                w.add_(0.05 * torch.randn(w.shape, generator=torch.Generator().manual_seed(1)))
            a, b = Path(tmp) / "plain.pt", Path(tmp) / "v8.pt"
            VI.save_student(st, a, it=1)
            VI.save_student(st, b, it=1, **rec)
            out = []
            for ck in (a, b):
                pol = H.StudentPolicy(ck, "cpu")
                pol.reset()
                out.append([pol.query(i, q, qd, u_prev=u if k else None)["chunk"] for k, (i, q, qd, u) in
                            enumerate(seq)])
            self.assertEqual(H.StudentPolicy(b, "cpu").meta["v8_online"], rec["v8_online"])
            for x, y in zip(*out):
                np.testing.assert_array_equal(x, y)


if __name__ == "__main__":
    unittest.main()
