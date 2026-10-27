"""Tests of the round v13 student online config ``viral_r2s_online_v13_visRtb05.json`` (2026-10-27,
~/so101_r2s_claude/v13/quick/CONFIGS.md): the visual fix + the fixed 5 mm lower table.

- It loads (JSON, ``OnlineConfig.from_dict``) and differs from ``viral_r2s_online_v12_visR_tbpl.json`` ONLY in the table
  fields (``table_bias``, ``table_render``, ``student_floor``; in fact only ``table_bias.plane``) and the note.
- Its 3 table keys equal those of ``viral_r2s_online_v12_tb05.json``; everything else equals ``viral_r2s_online_v12_visR.json``.
- Its online floor raise (``VI.online_floor_raise``) equals tb05's (5 mm), and equals the floor raise of the fr05 teacher
  checkpoint (skipped without the local copy, ``NEDM_V13_FR05``).

    PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -m unittest tests.so101_policy.test_v13_configs -v
"""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VCFG = REPO / "configs/so101_policy/vision"
TABLE = ("table_bias", "table_render", "student_floor")
SHA_WB = "ab3e30edcf896cfcec8613ce897ecd89c68c1b0cc181a3815fd7c8bdaa8f41bf"
FR05 = Path(os.environ.get("NEDM_V13_FR05",
                           "~/so101_r2s_claude/ckpt/v12/teachers/tppo_v12_fr05_s121/model_00200.pt")).expanduser()
FR05_SHA = "7fc16cfa279cf57d0d55e4fe76477d1c88ba7e0c1ee33798902c1a3d0f9c360e"


def load(name):
    return json.loads((VCFG / name).read_text())


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        q = f"{p}.{k}" if p else k
        if isinstance(v, dict):
            out.update(flat(v, q))
        else:
            out[q] = v
    return out


class TestV13Config(unittest.TestCase):
    def setUp(self):
        self.new = load("viral_r2s_online_v13_visRtb05.json")
        self.base = load("viral_r2s_online_v12_visR_tbpl.json")
        self.tb05 = load("viral_r2s_online_v12_tb05.json")
        self.visr = load("viral_r2s_online_v12_visR.json")

    def test_only_table_fields_differ_from_visR_tbpl(self):
        self.assertEqual(list(self.new), list(self.base))                         # same keys, same order
        diff = {k for k in self.new if k != "note" and self.new[k] != self.base[k]}
        self.assertEqual(diff, {"table_bias"})
        fn, fb = flat(self.new), flat(self.base)
        self.assertEqual(set(fn), set(fb))
        self.assertEqual({k for k in fn if fn[k] != fb[k]}, {"note", "table_bias.plane"})
        self.assertTrue(set(diff) <= set(TABLE))

    def test_table_fields_are_tb05s(self):
        for k in TABLE:
            self.assertEqual(self.new[k], self.tb05[k], k)
        self.assertEqual(self.new["table_bias"], {"plane": [-0.005, 0.0, 0.0], "p": 1.0})
        self.assertEqual({k: v for k, v in self.new.items() if k not in ("note",) + TABLE},
                         {k: v for k, v in self.visr.items() if k != "note"})

    def test_loads_and_floor_raise(self):
        from nedm.so101_policy.vision import viral as VI
        oc = VI.OnlineConfig.from_dict(self.new)
        self.assertEqual(oc.student_floor, "observed")
        self.assertTrue(oc.table_render)
        self.assertEqual(oc.require_scene_sha256, SHA_WB)
        self.assertTrue(oc.gripper_look)
        fr = VI.online_floor_raise(oc)
        self.assertIsNotNone(fr)
        self.assertEqual(fr, VI.online_floor_raise(VI.OnlineConfig.from_dict(self.tb05)))
        self.assertNotEqual(fr, VI.online_floor_raise(VI.OnlineConfig.from_dict(self.base)))

    @unittest.skipUnless(FR05.exists(), f"no fr05 teacher at {FR05}")
    def test_fr05_teacher_floor_raise_matches(self):
        import hashlib
        import torch
        from nedm.so101_policy import joint_teacher as JT
        from nedm.so101_policy.vision import viral as VI
        self.assertEqual(hashlib.sha256(FR05.read_bytes()).hexdigest(), FR05_SHA)
        d = torch.load(str(FR05), map_location="cpu", weights_only=False)
        jf = JT.jt_floor_raise(SimpleNamespace(env_rec=d.get("env") or {}))
        fr = VI.online_floor_raise(VI.OnlineConfig.from_dict(self.new))
        self.assertIsNotNone(jf)
        self.assertLess(max(abs(a - b) for a, b in zip(fr, jf)), 1e-9)


if __name__ == "__main__":
    unittest.main()
