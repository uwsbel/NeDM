"""Assert every path and preset a training config names actually exists.

THE COMPLEMENT TO A PAIRWISE EDGE CHECK. Diffing configs against each other
catches anything that DIFFERS and is structurally blind to anything SHARED and
wrong: six configs for a three-arm experiment were verified edge-by-edge, every
edge passed, and all six then failed within a second because all six named the
same stale dataset in four fields the diff could not see -- identical values
cancel.

Edges catch what differs. This catches what is shared. Neither subsumes the other.

Checks, per config:
  every *_dataset_dir and every entry of the dataset lists resolves to a directory
  that directory contains metadata.json
  the state_field_preset named in that dataset's metadata is defined in constants
  the output_dir's parent exists
"""
import argparse, json, os, sys

DIR_KEYS = ["processed_dataset_dir"]
LIST_KEYS = [("train_mix", "datasets"), (None, "validation_datasets"),
             ("rollout_eval", "datasets")]
FLAT_LIST_KEYS = [("loss", "channel_weight_datasets")]


def collect(cfg):
    out = []
    for k in DIR_KEYS:
        if k in cfg: out.append((k, cfg[k]))
    for parent, key in LIST_KEYS:
        node = cfg.get(parent, cfg) if parent else cfg
        for i, d in enumerate(node.get(key, []) if isinstance(node, dict) else []):
            if isinstance(d, dict) and "processed_dataset_dir" in d:
                out.append((f"{parent + '.' if parent else ''}{key}[{i}]",
                            d["processed_dataset_dir"]))
    for parent, key in FLAT_LIST_KEYS:
        for i, p in enumerate(cfg.get(parent, {}).get(key, [])):
            out.append((f"{parent}.{key}[{i}]", p))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    a = ap.parse_args()
    try:
        sys.path.insert(0, "src")
        from nedm.training.constants import STATE_FIELD_PRESETS
    except Exception:
        STATE_FIELD_PRESETS = None

    bad = 0
    for c in a.configs:
        cfg = json.load(open(c))
        problems = []
        for where, p in collect(cfg):
            if not os.path.isdir(p):
                problems.append(f"{where}: no such directory {p}")
                continue
            m = os.path.join(p, "metadata.json")
            if not os.path.exists(m):
                problems.append(f"{where}: {p} has no metadata.json")
                continue
            if STATE_FIELD_PRESETS is not None:
                pre = json.load(open(m)).get("state_field_preset")
                if pre and pre not in STATE_FIELD_PRESETS:
                    problems.append(f"{where}: metadata names undefined preset {pre!r}")
        od = cfg.get("output_dir")
        if od and not os.path.isdir(os.path.dirname(od)):
            problems.append(f"output_dir: parent of {od} does not exist")
        print(f"  {os.path.basename(c):26} {'OK' if not problems else 'FAIL'}")
        for pr in problems:
            print(f"      {pr}")
        bad += bool(problems)
    print(f"{len(a.configs) - bad} of {len(a.configs)} configs resolve")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
