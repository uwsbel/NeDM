"""Freeze every checkpoint before the fresh cohorts exist, then derive the cohort seeds.

Copies each unified run's best.pt (+ run_config.json, validation.json) into
<root>/frozen/<name>/, lists every unified and reference checkpoint with its
SHA256 in <root>/frozen/MANIFEST.txt, and writes the fresh-cohort campaign
files with seed = int(sha256(manifest_bytes + system)[:8], 16).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--name", default="", help="per-system freeze: <root>/frozen_<name>")
    parser.add_argument("--unified", type=Path, nargs="+", required=True, help="run directories")
    parser.add_argument("--reference", nargs="+", required=True, help="name=checkpoint")
    parser.add_argument("--cohort", nargs="+", required=True, help="system=template_campaign.json:episodes_per_cell")
    args = parser.parse_args()
    frozen = args.root / (f"frozen_{args.name}" if args.name else "frozen")
    if (frozen / "MANIFEST.txt").exists():
        raise FileExistsError("already frozen")
    lines = []
    for run in sorted(args.unified):
        target = frozen / run.name
        target.mkdir(parents=True)
        for name in ("best.pt", "run_config.json", "validation.json", "core_validation.json"):
            if (run / name).exists():
                shutil.copy2(run / name, target / name)
        if sha(run / "best.pt") != sha(target / "best.pt"):
            raise RuntimeError(f"copy mismatch {run}")
        lines.append(f"{sha(target / 'best.pt')}  unified  {run.name}  {target / 'best.pt'}")
    for item in sorted(args.reference):
        name, path = item.split("=", 1)
        lines.append(f"{sha(path)}  reference  {name}  {path}")
    manifest = ("\n".join(lines) + "\n").encode()
    (frozen / "MANIFEST.txt").write_bytes(manifest)
    cohorts = {}
    for item in args.cohort:
        system, rest = item.split("=", 1)
        template, per_cell = rest.rsplit(":", 1)
        seed = int(hashlib.sha256(manifest + system.encode()).hexdigest()[:8], 16)
        campaign = json.loads(Path(template).read_text())
        campaign.update(name=f"{system}_unified_fresh_{seed}", seed=seed, episodes_per_cell={"test": int(per_cell)})
        path = frozen / f"cohort_{system}.json"
        path.write_text(json.dumps(campaign, indent=1))
        cohorts[system] = {"seed": seed, "campaign": str(path)}
    (frozen / "COHORTS.json").write_text(json.dumps({"manifest_sha256": hashlib.sha256(manifest).hexdigest(), "cohorts": cohorts}, indent=1))
    print(manifest.decode())
    print(json.dumps(cohorts, indent=1))


if __name__ == "__main__":
    main()
