"""Compare equiv.py outputs of the original file in the training code and the released file in main's code.

    python -I equiv_compare.py OUT.json A_PREFIX B_PREFIX [A_PREFIX B_PREFIX ...]

Per pair: every rollout tensor bit for bit (raw float64 bytes), the max abs difference, and the evaluate reports
as canonical JSON (sorted keys) for exact equality.

Release check of the SO101 pick-and-place study (so101_pick_place/README.md, "The NRD"; reproduce.sh steps
nrd_val, nrd_strip).
"""
import json
import sys

import torch


def main():
    out, pairs = sys.argv[1], sys.argv[2:]
    rows = []
    for a, b in zip(pairs[::2], pairs[1::2]):
        ta, tb = torch.load(a + ".pt", weights_only=True), torch.load(b + ".pt", weights_only=True)
        ja, jb = json.load(open(a + ".json")), json.load(open(b + ".json"))
        for name in ta:
            row = dict(a=a, b=b, data=name, episodes=ja["info"][name]["episodes"])
            for k in ta[name]:
                x, y = ta[name][k], tb[name][k]
                row[k] = dict(shape=list(x.shape), bitwise_equal=bool(
                    x.dtype == y.dtype and x.shape == y.shape
                    and torch.equal(x.contiguous().view(-1).view(torch.uint8), y.contiguous().view(-1).view(torch.uint8))),
                    max_abs_diff=float((x.double() - y.double()).abs().nan_to_num(0).max()))
            ca = json.dumps(ja["reports"][name], sort_keys=True)
            cb = json.dumps(jb["reports"][name], sort_keys=True)
            row["evaluate_json_equal"] = ca == cb
            row["selection_score"] = [ja["reports"][name]["selection_score"], jb["reports"][name]["selection_score"]]
            rows.append(row)
            print(name, a.split("/")[-1], "vs", b.split("/")[-1], {k: row[k]["bitwise_equal"] for k in ta[name]},
                  "evaluate equal", row["evaluate_json_equal"], flush=True)
    ok = all(r["evaluate_json_equal"] and all(v["bitwise_equal"] for k, v in r.items() if isinstance(v, dict))
             for r in rows)
    with open(out, "w") as f:
        json.dump(dict(all_equal=ok, rows=rows), f, indent=1)
    print("ALL EQUAL" if ok else "DIFFERENCES FOUND")


if __name__ == "__main__":
    main()
