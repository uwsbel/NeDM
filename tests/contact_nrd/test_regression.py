"""The released cases are unchanged: the released code against the working tree (stdlib unittest; pytest also runs it).

Baseline: the code and train.json of the release commit, CONTACT_NRD_BASELINE_REV (default
41be3efccb7eb2f74e6741610853557b288a86bc, "contact NRD: one design for bouncing ball, pool and SO101 push-T"), written
from git into a temporary folder. The test prints the revision. Every subprocess runs from a new empty folder with
python -P (no folder in front of PYTHONPATH) and records the nedm modules it imported; the test checks their files
and SHA-256 against the baseline files or the working tree files (hashed when the test starts), and the
source_sha256 of each training run against the files of its side.

Data: a 540-episode subset of each released training set (its first 400 training, 120 validation and 20 test
episodes, original order) and the released seed-61 checkpoint, from either
- the download (python -m nedm.contact_nrd.download --part train models): CONTACT_NRD_DOWNLOAD, default
  artifacts/contact_nrd/ of this checkout, with <case>/train/ and <case>/models/seed61.pt; the test cuts the subset
  into a temporary folder; or
- a folder of ready subsets (CONTACT_NRD_RELEASE, default ~/contact_nrd_release: data/subsets/<case>/,
  ckpt/<case>/seed61.pt), used first if present.
It needs a GPU. Without the data, a GPU or the baseline revision the tests are skipped; with
CONTACT_NRD_REQUIRE_REGRESSION=1 they fail.

1. free_metrics of each released seed-61 checkpoint on its subset (all 540 episodes, step-average rates as in
   training): the baseline free_metrics and the new data.evaluate give the same report (JSON text), so the same selection
   score. Deterministic GPU algorithms on.
2. test.py of the baseline and of the working tree on each subset: same printed lines and the same output JSON.
3. train.py --smoke --deterministic of the baseline and of the working tree on each subset (CONTACT_NRD_SMOKE_CASES,
   default all three), each with its own default configuration (configs/contact_nrd/train.json of the baseline and
   of the working tree): the same masks, logs, JSON files and checkpoints (times and source or checkpoint hashes
   excluded).
4. The same without --smoke, with a reduced configuration (each side's own train.json with fewer and smaller updates,
   50 check episodes, a check every 10 updates; CONTACT_NRD_REDUCED_CASES, default all three): the non-smoke
   validation path (all validation episodes, the evenly spaced check subset, repeated checks).
Run comparison (differences): key order, types, list lengths, tensor dtype, shape and bits, float bits, at any depth.

    PYTHONPATH=src python -m unittest tests.contact_nrd.test_regression -v
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
RELEASE = Path(os.environ.get("CONTACT_NRD_RELEASE", Path.home() / "contact_nrd_release"))
DOWNLOAD = Path(os.environ.get("CONTACT_NRD_DOWNLOAD", REPO / "artifacts/contact_nrd"))
REQUIRE = os.environ.get("CONTACT_NRD_REQUIRE_REGRESSION") == "1"
BASELINE = os.environ.get("CONTACT_NRD_BASELINE_REV", "41be3efccb7eb2f74e6741610853557b288a86bc")
CASES = ("bouncing_ball", "pool", "so101_push_t")
SMOKE_CASES = os.environ.get("CONTACT_NRD_SMOKE_CASES", ",".join(CASES)).split(",")
REDUCED_CASES = os.environ.get("CONTACT_NRD_REDUCED_CASES", ",".join(CASES)).split(",")
REDUCED = {"check_episodes": 50, "evaluate_every": 10, "collision": {"updates": 30, "warmup": 5, "batch": 512},
           "cotrain": {"updates": 40, "warmup": 5, "core_batch": 512, "contact_batch": 512, "traj_batch": 32}}
SKIP = ("elapsed_s", "total_elapsed_s", "source_sha256", "checkpoint_sha256")   # times and code or checkpoint hashes
QUOTA = ((0, 400), (1, 120), (2, 20))   # (split, first episodes of it) of the subsets


def cut(src, dst):
    """Copy of a data directory with the QUOTA episodes only (original order; every array cut along the episodes)."""
    dst.mkdir(parents=True)
    with np.load(src / "unified_data.npz") as packet:
        splits = packet["splits"]
        keep = np.sort(np.concatenate([np.flatnonzero(splits == s)[:q] for s, q in QUOTA]))
        np.savez(dst / "unified_data.npz", **{k: packet[k][keep] for k in packet.files})
    shutil.copy2(src / "system.json", dst / "system.json")
    return dst


def released_cases(tmp):
    """{case: (subset directory, seed-61 checkpoint)} from the ready subsets or the download; None if missing."""
    ready = {c: (RELEASE / "data/subsets" / c, RELEASE / "ckpt" / c / "seed61.pt") for c in CASES}
    if all((d / "unified_data.npz").exists() and k.exists() for d, k in ready.values()):
        return ready
    found = {c: (DOWNLOAD / c / "train", DOWNLOAD / c / "models/seed61.pt") for c in CASES}
    if all((d / "unified_data.npz").exists() and k.exists() for d, k in found.values()):
        return {c: (cut(d, Path(tmp) / "subsets" / c), k) for c, (d, k) in found.items()}
    return None


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True).stdout


def baseline_package(root, rev):
    """Write src/nedm/__init__.py, src/nedm/contact_nrd/**.py and configs/contact_nrd/train.json of commit rev under
    root (bytes as in git)."""
    root = Path(root)
    files = [f for f in git("ls-tree", "-r", "--name-only", rev, "src/nedm/contact_nrd/").decode().split() if f.endswith(".py")]
    for rel in ["src/nedm/__init__.py", *files, "configs/contact_nrd/train.json"]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(git("show", f"{rev}:{rel}"))
    return root


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module_files(src):
    """{module name: (real path, sha256)} of every .py file of the nedm package under src."""
    out = {}
    for f in sorted((Path(src) / "nedm").rglob("*.py")):
        parts = f.relative_to(src).with_suffix("").parts
        out[".".join(parts[:-1] if parts[-1] == "__init__" else parts)] = (os.path.realpath(f), sha256(f))
    return out


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Runs a module as python -m does and records the nedm modules it imported: argv = record file, module, arguments.
RUNNER = """
import hashlib, importlib.util, json, os, runpy, sys
record, name = sys.argv[1], sys.argv[2]
sys.argv = [name] + sys.argv[3:]
main = importlib.util.find_spec(name).origin
try:
    runpy.run_module(name, run_name="__main__", alter_sys=True)
finally:
    files = {k: m.__file__ for k, m in list(sys.modules.items())
             if (k == "nedm" or k.startswith("nedm.")) and getattr(m, "__file__", None)}
    files[name] = main
    with open(record, "w") as fh:
        json.dump({k: [os.path.realpath(f), hashlib.sha256(open(f, "rb").read()).hexdigest()] for k, f in sorted(files.items())}, fh)
"""


def run(src, expect, work, module, *args):
    """python -m module args with PYTHONPATH=src, from a new empty folder under work -> (stdout, {imported module:
    sha256}). Fails unless every imported nedm module is the file of expect (module_files of src)."""
    env = {k: v for k, v in os.environ.items() if k not in ("SLURM_JOB_ID", "CUBLAS_WORKSPACE_CONFIG", "PYTHONSTARTUP")}
    env["PYTHONPATH"] = str(src)
    cwd = Path(tempfile.mkdtemp(dir=work))
    record = cwd.with_name(cwd.name + "_imports.json")
    out = subprocess.run([sys.executable, "-P", "-c", RUNNER, str(record), module, *args], cwd=cwd, env=env,
                         capture_output=True, text=True, check=True).stdout
    imported = {k: tuple(v) for k, v in json.loads(record.read_text()).items()}
    wrong = {k: v for k, v in imported.items() if expect.get(k) != v}
    if wrong or module not in imported:
        raise AssertionError(f"{module} under {src} imported other code: {wrong or 'no record of ' + module}")
    return out, {k: v[1] for k, v in imported.items()}


def bits(t):
    return t.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def differences(x, y, path="", skip=SKIP):
    """Paths where two nested values differ: types, key order, list lengths, tensor dtype, shape and bits, floats by
    their bits (-0.0 is not 0.0; NaN equals NaN of the same bits), other values by ==; dict keys in skip are ignored
    at any depth."""
    if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
        same = (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and x.dtype == y.dtype
                and x.shape == y.shape and torch.equal(bits(x), bits(y)))
        return [] if same else [f"{path}: tensors differ"]
    if type(x) is not type(y):
        return [f"{path}: {type(x).__name__} vs {type(y).__name__}"]
    if isinstance(x, dict):
        kx, ky = [k for k in x if k not in skip], [k for k in y if k not in skip]
        if kx != ky:
            return [f"{path}: keys {kx} vs {ky}"]
        return [d for k in kx for d in differences(x[k], y[k], f"{path}/{k}", skip)]
    if isinstance(x, (list, tuple)):
        if len(x) != len(y):
            return [f"{path}: length {len(x)} vs {len(y)}"]
        return [d for i, (a, b) in enumerate(zip(x, y)) for d in differences(a, b, f"{path}[{i}]", skip)]
    if isinstance(x, float):
        return [] if struct.pack("<d", x) == struct.pack("<d", y) else [f"{path}: {x!r} vs {y!r}"]
    return [] if x == y else [f"{path}: {x!r} vs {y!r}"]


def compare_runs(old, new):
    """Differences between two run directories (SKIP keys excluded; other files byte for byte)."""
    problems = []
    names = sorted(p.name for p in old.iterdir())
    if names != sorted(p.name for p in new.iterdir()):
        problems.append(f"files {names} vs {sorted(p.name for p in new.iterdir())}")
    for name in names:
        a, b = old / name, new / name
        if name.endswith(".pt"):
            problems += differences(torch.load(a, map_location="cpu", weights_only=False),
                                    torch.load(b, map_location="cpu", weights_only=False), name)
        elif name.endswith(".jsonl"):
            problems += differences([json.loads(x) for x in a.read_text().splitlines()],
                                    [json.loads(x) for x in b.read_text().splitlines()], name)
        elif name in ("run_config.json", "complete.json"):
            problems += differences(json.loads(a.read_text()), json.loads(b.read_text()), name)
        elif a.read_bytes() != b.read_bytes():
            problems.append(f"{name} differs")
    return problems


def reduced_config(src, dst):
    """src train.json with the REDUCED entries -> dst."""
    cfg = json.loads(Path(src).read_text())
    for k, v in REDUCED.items():
        cfg[k] = {**cfg[k], **v} if isinstance(v, dict) else v
    Path(dst).write_text(json.dumps(cfg, indent=1))
    return dst


class Comparator(unittest.TestCase):
    def test_differences(self):
        t = torch.arange(3.0)
        base = {"a": t, "b": [1, {"c": t}], "elapsed_s": 1.0}
        self.assertEqual(differences(base, {"a": t.clone(), "b": [1, {"c": t.clone()}], "elapsed_s": 2.0}), [])
        for other in ({"b": [1, {"c": t}], "a": t},                       # key order
                      {"a": t.double(), "b": [1, {"c": t}]},              # dtype
                      {"a": t, "b": [1, {"c": t + 1e-7}]},                # tensor inside a list
                      {"a": t, "b": [1]},                                 # list length
                      {"a": t, "b": [1.0, {"c": t}]}):                    # type
            self.assertTrue(differences(base, other), other)
        self.assertTrue(differences(torch.tensor([0.0]), torch.tensor([-0.0])))       # equal values, other bits
        self.assertTrue(differences(0.0, -0.0))
        self.assertEqual(differences(torch.tensor([float("nan")]), torch.tensor([float("nan")])), [])
        self.assertEqual(differences(float("nan"), float("nan")), [])
        self.assertEqual(differences(torch.zeros(()), torch.zeros(())), [])            # 0-dim tensors

    def test_run_records_imports(self):
        """A package nedm in the working folder of a run would shadow PYTHONPATH with plain python -m; run() starts in
        an empty folder with -P and names the files that the run imported."""
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "fake/src"
            (fake / "nedm").mkdir(parents=True)
            (fake / "nedm/__init__.py").write_text("")
            (fake / "nedm/probe.py").write_text("print('probe ran')\n")
            out, imported = run(fake, module_files(fake), tmp, "nedm.probe")
            self.assertEqual(out, "probe ran\n")
            self.assertEqual(set(imported), {"nedm", "nedm.probe"})
            other = dict(module_files(fake), **{"nedm.probe": ("/elsewhere/probe.py", "0" * 64)})
            with self.assertRaises(AssertionError):
                run(fake, other, tmp, "nedm.probe")


class ReleasedCasesUnchanged(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        try:
            cls.rev = git("rev-parse", "--verify", f"{BASELINE}^{{commit}}").decode().strip()
        except subprocess.CalledProcessError:
            cls.rev = None
        cls.cases = released_cases(cls.tmp.name) if torch.cuda.is_available() and cls.rev else None
        if cls.cases is None:
            cls.tmp.cleanup()
            why = (f"no GPU, no baseline revision {BASELINE} in this clone, or no released data: {RELEASE}/data/subsets "
                   f"and ckpt, or {DOWNLOAD}/<case>/train and models/seed61.pt")
            if REQUIRE:
                raise RuntimeError(f"CONTACT_NRD_REQUIRE_REGRESSION=1: {why}")
            raise unittest.SkipTest(why)
        cls.work = Path(cls.tmp.name) / "work"
        cls.work.mkdir()
        cls.base = baseline_package(Path(cls.tmp.name) / "baseline", cls.rev)
        cls.sides = {"old": (cls.base, module_files(cls.base / "src")), "new": (REPO, module_files(REPO / "src"))}
        print(f"\nbaseline revision {cls.rev} (CONTACT_NRD_BASELINE_REV={BASELINE}); working tree {REPO}")
        for tag, (root, files) in cls.sides.items():
            print(f"  {tag}: " + ", ".join(f"{k.split('.')[-1]} {files[k][1][:12]}" for k in sorted(files)
                                           if k.startswith("nedm.contact_nrd.") and k.count(".") == 2))
        cls.old_data = module("baseline_contact_nrd_data", cls.base / "src/nedm/contact_nrd/data.py")
        cls.old_model = module("baseline_contact_nrd_model", cls.base / "src/nedm/contact_nrd/model.py")

    def run_side(self, tag, module_name, *args):
        """run() of one side ("old": baseline, "new": working tree) -> stdout; prints the imported modules' hashes."""
        root, files = self.sides[tag]
        out, imported = run(root / "src", files, self.work, module_name, *args)
        print(f"  {tag} {module_name}: imported the expected files: "
              + ", ".join(f"{k.split('.')[-1]} {h[:12]}" for k, h in sorted(imported.items()) if k.startswith("nedm.contact_nrd.")))
        return out

    def check_sources(self, tag, run_dir):
        """The run's own record of its code (run_config.json source_sha256) is the code of its side."""
        root = self.sides[tag][0] / "src/nedm/contact_nrd"
        recorded = json.loads((run_dir / "run_config.json").read_text())["source_sha256"]
        self.assertEqual(recorded, {f.name: sha256(f) for f in root.glob("*.py")})

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_free_metrics_and_score(self):
        from nedm.contact_nrd import data as new_data
        from nedm.contact_nrd import model as new_model
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            for case in CASES:
                with self.subTest(case=case):
                    root, ckpt = self.cases[case]
                    for mod, tag in ((self.old_data, "old"), (self.old_model, "old"), (new_data, "new"), (new_model, "new")):
                        expect = self.sides[tag][1]["nedm.contact_nrd." + Path(mod.__file__).stem]
                        self.assertEqual((os.path.realpath(mod.__file__), sha256(mod.__file__)), expect)
                    reports = []
                    for dmod, mmod in ((self.old_data, self.old_model), (new_data, new_model)):
                        model, packet = mmod.load_model(ckpt, "cuda")
                        kinds = set(packet["model_config"]["integrate"]["kinds"])
                        data = dmod.step_average_rates(dmod.load_data(root, "cuda", model_step_s=model.dt), kinds)
                        ids = torch.arange(len(data["splits"]), device="cuda")
                        fn = dmod.free_metrics if dmod is self.old_data else new_data.evaluate
                        reports.append(json.dumps(fn(model, data, ids)))
                    self.assertEqual(reports[0], reports[1])
                    print(f"{case}: free_metrics identical, selection score {json.loads(reports[1])['selection_score']!r}")
        finally:
            torch.use_deterministic_algorithms(False)

    def test_test_script(self):
        for case in CASES:
            with self.subTest(case=case):
                outs = []
                for tag in ("old", "new"):
                    path = Path(self.tmp.name) / f"test_{case}_{tag}.json"
                    root, ckpt = self.cases[case]
                    text = self.run_side(tag, "nedm.contact_nrd.test", "--data", str(root), "--model", f"seed61={ckpt}",
                                         "--output", str(path))
                    outs.append((text, path.read_text()))
                self.assertEqual(outs[0][0], outs[1][0])
                self.assertEqual(outs[0][1], outs[1][1])
                print(f"{case}: test.py output identical: {outs[1][0].strip()}")

    def test_train_smoke(self):
        for case in SMOKE_CASES:
            with self.subTest(case=case):
                dirs = []
                for tag in ("old", "new"):
                    out = Path(self.tmp.name) / f"smoke_{case}_{tag}"
                    self.run_side(tag, "nedm.contact_nrd.train", "--data", str(self.cases[case][0]), "--output", str(out),
                                  "--smoke", "--deterministic")
                    self.check_sources(tag, out)
                    dirs.append(out)
                problems = compare_runs(*dirs)
                self.assertEqual(problems, [])
                masks = json.loads((dirs[1] / "run_config.json").read_text())["masks"]
                print(f"{case}: smoke runs identical; masks {masks}")

    def test_train_reduced(self):
        for case in REDUCED_CASES:
            with self.subTest(case=case):
                dirs = []
                for tag in ("old", "new"):
                    out = Path(self.tmp.name) / f"reduced_{case}_{tag}"
                    root = self.sides[tag][0]
                    cfg = reduced_config(root / "configs/contact_nrd/train.json", Path(self.tmp.name) / f"reduced_{case}_{tag}.json")
                    self.run_side(tag, "nedm.contact_nrd.train", "--data", str(self.cases[case][0]), "--output", str(out),
                                  "--config", str(cfg), "--deterministic")
                    self.check_sources(tag, out)
                    dirs.append(out)
                problems = compare_runs(*dirs)
                self.assertEqual(problems, [])
                done = json.loads((dirs[1] / "complete.json").read_text())
                print(f"{case}: reduced runs identical; {done['checkpoints_compared']} checks, "
                      f"validation score {done['validation_score']!r}")


if __name__ == "__main__":
    unittest.main()
