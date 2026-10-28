"""Check a contact NRD data directory against the collection contract and count what training needs.

    python -m nedm.contact_nrd.collect.validate DIR [--output report.json]

Hard checks (exit code 1 if one fails): array shapes and dtypes, finite valid records, unit and sign-continuous
quaternions, 0/1 contacts, the padding rule, commands constant inside each control period. Frame check: the
step-average rates computed from the poses (as contact NRD training does) are compared with the mean of the
recorded instantaneous rates at both ends of the step; a large relative error points to a wrong frame (body instead
of world angular velocity), a wrong channel order or a time shift. It uses up to 400 evenly spaced episodes of the
families that are not core-only (rate_consistency; the same episodes as without core-only families) and up to 400
core-only episodes apart (rate_consistency_core_only). Coverage (the windows contact NRD training
draws, with its labels: contact in the step or in the step before): contact-free windows of each moving body
(all, and on the core grid of train.py: every free_phase_step-th record),
windows with exactly one active pair (per pair), contact onsets, by family and split. Families with
"use": "core_only" (train.py: core only) are counted apart: the split blocks count the other families as before and
add core_only_* counts (contact-free and core-grid windows only; core_fit_<body> = what the core fit draws from), and
a core-only family block has only its contact-free counts. A core-only family must have select: false; its episodes
need a branch_start [N] array (the branch record) with no contact label from it on. numpy only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

POSE_TYPES = {"position": 0, "angle": 0, "quaternion": 1, "velocity": 2, "angular_rate": 2, "angular_velocity": 3}


def window_labels(contacts, stride):
    """[N, T-1, P] interval labels -> [N, T-stride, P]: pair in contact during the step starting at each record."""
    c = contacts.astype(np.int32)
    cum = np.concatenate((np.zeros_like(c[:, :1]), c.cumsum(1)), 1)
    return (cum[:, stride:] - cum[:, :-stride]) > 0


def training_windows(system, contacts, lengths, splits):
    """The labels and window masks of contact NRD training (train.py): labels [N, W, P] (step or step before),
    involved [N, W, D] (body touched now or in the record interval before), valid [N, W] (command grid)."""
    stride = round(system["model_step_s"] / system["record_step_s"])
    lab = window_labels(contacts, stride)
    spread = np.zeros_like(lab)
    spread[:, stride:] = lab[:, :-stride]
    lab = lab | spread
    n, w, p = lab.shape
    prev = np.concatenate((np.zeros_like(lab[:, :1]), window_labels(contacts, 1)[:, : w - 1]), 1)
    after = prev & ~lab
    touched = lab | after
    moving = [k for k, b in enumerate(system["bodies"]) if b["moving"]]
    slot = {k: s for s, k in enumerate(moving)}
    involved = np.zeros((n, w, len(moving)), bool)
    for q, (i, j) in enumerate(system["pairs"]):
        involved[..., slot[i]] |= touched[..., q]
        if j in slot:
            involved[..., slot[j]] |= touched[..., q]
    acting = any(int(b.get("action_dim", 0)) for b in system["bodies"] if b["moving"])
    grid = (np.arange(w) % stride == 0) if acting else np.ones(w, bool)
    valid = (np.arange(w)[None, :] < (lengths - stride)[:, None]) & grid[None, :]
    return lab, involved, valid, stride


def step_average(states, types, stride, dt):
    """Rate channels from pose differences over one model step (as data.step_average_rates) -> [N, T-stride, k]."""
    pc = [c for c, t in enumerate(types) if POSE_TYPES[t] == 0]
    qc = [c for c, t in enumerate(types) if POSE_TYPES[t] == 1]
    vc = [c for c, t in enumerate(types) if POSE_TYPES[t] == 2]
    wc = [c for c, t in enumerate(types) if POSE_TYPES[t] == 3]
    out = {}
    if pc and len(pc) == len(vc):
        out["linear"] = (vc, (states[:, stride:, pc] - states[:, :-stride, pc]) / dt)
    if len(qc) == 4 and len(wc) == 3:
        q1, q0 = states[:, stride:][..., qc].astype(np.float64), states[:, :-stride][..., qc].astype(np.float64)
        w0, v0 = q0[..., :1], -q0[..., 1:]
        w1, v1 = q1[..., :1], q1[..., 1:]
        w = w1 * w0 - (v1 * v0).sum(-1, keepdims=True)
        v = w1 * v0 + w0 * v1 + np.cross(v1, v0)
        sign = np.where(w < 0, -1.0, 1.0)
        w, v = w * sign, v * sign
        nrm = np.linalg.norm(v, axis=-1, keepdims=True)
        ang = 2 * np.arctan2(nrm, w)
        rot = np.where(nrm > 1e-12, v / np.maximum(nrm, 1e-12) * ang, 2 * v)
        out["angular"] = (wc, rot / dt)
    return out


def validate(root, max_frame_episodes=400):
    root = Path(root)
    system = json.loads((root / "system.json").read_text())
    with np.load(root / "unified_data.npz") as f:
        arrays = {k: f[k] for k in f.files}
    fails, warns, report = [], [], {"dir": str(root), "system": system.get("name")}
    moving = [b for b in system["bodies"] if b["moving"]]
    lengths, splits, contacts = arrays["lengths"], arrays["splits"], arrays["contacts"]
    n = len(lengths)
    t_rec = int(system["records_per_episode"])
    report["episodes"] = int(n)
    report["valid_records"] = int(lengths.sum())
    rec_dt = float(system["record_step_s"])
    report["hours"] = float(((lengths - 1) * rec_dt).sum() / 3600)
    try:
        from nedm.contact_nrd.collect.export import check_arrays
        check_arrays(system, arrays)
    except ValueError as exc:
        fails.append(f"shape: {exc}")
        return dict(report, fails=fails, warns=warns, ok=False)
    valid_rec = np.arange(t_rec)[None, :] < lengths[:, None]                                          # [N, T]
    if (lengths < 2).any() or (lengths > t_rec).any():
        fails.append("lengths outside [2, records_per_episode]")
    if not np.isin(contacts, (0, 1)).all():
        fails.append("contacts are not 0/1")
    pad_int = np.arange(t_rec - 1)[None, :] >= (lengths - 1)[:, None]
    if contacts[pad_int].any():
        fails.append("contacts after the last valid interval are not zero")
    for b in moving:
        x = arrays[b["name"]]
        if x.dtype != np.float32:
            warns.append(f"{b['name']}: dtype {x.dtype} (contract: float32)")
        if not np.isfinite(x[valid_rec]).all():
            fails.append(f"{b['name']}: non-finite values in valid records")
        last = x[np.arange(n), lengths - 1]
        if not np.array_equal(np.where(valid_rec[..., None], x, last[:, None]), x):
            fails.append(f"{b['name']}: padding does not repeat the last valid record")
        qc = [c for c, t in enumerate(b["channel_types"]) if t == "quaternion"]
        if qc:
            q = x[..., qc].astype(np.float64)
            norm_err = np.abs(np.linalg.norm(q, axis=-1) - 1)[valid_rec]
            report[f"{b['name']}_quat_norm_err_max"] = float(norm_err.max())
            if norm_err.max() > 1e-4:
                fails.append(f"{b['name']}: quaternion norm error {norm_err.max():.2e}")
            dots = (q[:, 1:] * q[:, :-1]).sum(-1)[valid_rec[:, 1:]]
            if (dots < 0).any():
                fails.append(f"{b['name']}: quaternion sign flips ({int((dots < 0).sum())} records)")
            if (q[:, 0, 0] < 0).any():
                fails.append(f"{b['name']}: quaternion w < 0 at the first record")
    if "action" in arrays:
        act = arrays["action"]
        if not np.isfinite(act[valid_rec]).all():
            fails.append("action: non-finite values in valid records")
        ctrl = system.get("control_step_s", system["model_step_s"])
        per = round(ctrl / rec_dt)
        if per > 1:
            k = np.arange(1, t_rec)
            inside = (k % per != 0)[None, :] & valid_rec[:, 1:]
            changed = np.abs(act[:, 1:] - act[:, :-1]).max(-1) > 0
            if (changed & inside).any():
                fails.append(f"action changes inside a control period ({int((changed & inside).sum())} records)")
    # families and the core-only rules
    pair_names = [m.get("name", str(q)) for q, m in enumerate(system.get("pair_metadata", [{}] * len(system["pairs"])))]
    fam = arrays.get("family")
    fam_names = [f["name"] for f in system.get("families", [])]
    if (fam is not None) != bool(fam_names):
        fails.append("the family array and system.json families must come together")
    elif fam is not None:
        if fam.min() < 0 or fam.max() >= len(fam_names):
            fails.append("family values outside the families list")
        elif not all(isinstance(f.get("select"), bool) for f in system["families"]):
            fails.append("every system.json family needs select: true or false")
        else:
            sel = np.array([f["select"] for f in system["families"]])
            # test campaigns (split 2 only) and supplementary training folders (split 0 only, merged with a campaign
            # that has the validation episodes) are exempt
            if (splits == 1).any() and (splits == 0).any() and not (sel[fam] & (splits == 1)).any():
                fails.append("no validation episode in a family with select: true (checkpoint selection needs them)")
    uses = [f.get("use") for f in system.get("families", [])]
    core_ep = np.zeros(n, bool)
    if fam is not None and fam_names and fam.min() >= 0 and fam.max() < len(fam_names):
        if set(uses) - {None, "core_only"}:
            fails.append("family use must be absent or core_only")
        if any(u == "core_only" and f.get("select") for u, f in zip(uses, system["families"])):
            fails.append("a core-only family must have select: false")
        core_ep = np.isin(fam, [k for k, u in enumerate(uses) if u == "core_only"])
    if core_ep.any():
        bs = arrays.get("branch_start")
        if bs is None or bs.shape != (n,):
            fails.append("core-only episodes need a branch_start [N] array")
        else:
            b = bs[core_ep].astype(np.int64)
            if (b < 1).any() or (b >= lengths[core_ep]).any():
                fails.append("branch_start of core-only episodes outside [1, length - 1]")
            else:
                after = np.arange(t_rec - 1)[None, :] >= b[:, None]
                if contacts[core_ep][after].any():
                    fails.append("core-only episodes have contact labels from their branch_start on")
    # frame check (relative RMS error of the step-average rates) on up to max_frame_episodes evenly spaced episodes
    # of the families other than core-only ones (all episodes without them) and, apart, of the core-only families
    stride = round(system["model_step_s"] / rec_dt)
    dt = stride * rec_dt

    def frame_check(ids, what=""):
        pick = ids[np.linspace(0, len(ids) - 1, min(len(ids), max_frame_episodes)).round().astype(int)]
        frame = {}
        for b in moving:
            x = arrays[b["name"]][pick].astype(np.float64)
            ok = (np.arange(t_rec - stride)[None, :] < (lengths[pick] - stride)[:, None])
            for kind, (cols, avg) in step_average(x, b["channel_types"], stride, dt).items():
                if stride == 2:   # Simpson: the midpoint record catches speed peaks inside a step
                    inst = (x[:, :-2, cols] + 4 * x[:, 1:-1, cols] + x[:, 2:, cols]) / 6
                else:
                    inst = 0.5 * (x[:, stride:, cols] + x[:, :-stride, cols])
                err = (avg - inst)[ok]
                # relative to the largest channel RMS of the group: near-constant channels do not inflate the ratio
                scale = np.sqrt((x[:, :, cols][valid_rec[pick]] ** 2).mean(0)).max() + 1e-9
                rel = np.sqrt((err ** 2).mean(0)) / scale
                frame[f"{b['name']}_{kind}"] = dict(channels=[b["channels"][c] for c in cols],
                                                    relative_rms=[round(float(r), 4) for r in rel])
                if (rel > 0.5).any():
                    warns.append(f"{what}{b['name']} {kind}: step-average vs recorded rates differ (relative RMS up "
                                 f"to {rel.max():.2f}); check frames, channel order and timing")
        return frame, len(pick)
    others = np.flatnonzero(~core_ep)
    if len(others):
        report["rate_consistency"], n_other = frame_check(others)
    if core_ep.any():
        report["rate_consistency_core_only"], n_core = frame_check(np.flatnonzero(core_ep), "core-only episodes: ")
        report["rate_consistency_episodes"] = dict(other=n_other if len(others) else 0, core_only=n_core)
    # coverage
    lab, involved, valid, _ = training_windows(system, contacts, lengths, splits)
    train_cfg = Path(__file__).resolve().parents[4] / "configs/contact_nrd/train.json"
    free_phase_step = json.loads(train_cfg.read_text())["free_phase_step"] if train_cfg.exists() else 5

    def counts(sel, core_only=False):
        if not core_only:
            sel = sel & ~core_ep   # the other families, counted as without core-only episodes
        v = valid & sel[:, None]
        out = {"windows": int(v.sum())}
        phase = (np.arange(v.shape[1]) % free_phase_step == 0)[None, :]
        if core_only:   # contact-free windows only: core-only windows are never contact or single-pair windows
            for s, b in enumerate(moving):
                out[f"free_{b['name']}"] = int((v & ~involved[..., s]).sum())
                out[f"core_{b['name']}"] = int((v & phase & ~involved[..., s]).sum())
            return out
        for s, b in enumerate(moving):
            out[f"free_{b['name']}"] = int((v & ~involved[..., s]).sum())
            out[f"core_{b['name']}"] = int((v & phase & ~involved[..., s]).sum())
        single = lab.sum(-1) == 1
        for q, name in enumerate(pair_names):
            out[f"single_{name}"] = int((v & single & lab[..., q]).sum())
            out[f"on_{name}"] = int((v & lab[..., q]).sum())
        out["multi_contact"] = int((v & (lab.sum(-1) >= 2)).sum())
        c = contacts[sel].astype(bool)
        onsets = (c[:, 1:] & ~c[:, :-1]).sum((0, 1)) + c[:, 0].sum(0)
        out.update({f"onsets_{name}": int(onsets[q]) for q, name in enumerate(pair_names)})
        return out

    def block(sel):
        out = counts(sel)
        if (sel & core_ep).any():
            extra = counts(sel & core_ep, core_only=True)
            out["core_only_episodes"] = int((sel & core_ep).sum())
            out.update({f"core_only_{k}": v for k, v in extra.items()})
            out.update({f"core_fit_{b['name']}": out[f"core_{b['name']}"] + extra[f"core_{b['name']}"] for b in moving})
        return out
    report["coverage"] = {"all": block(np.ones(n, bool))}
    for s in (0, 1, 2):
        if (splits == s).any():
            report["coverage"][f"split{s}"] = block(splits == s)
    if fam is not None and fam_names:
        for f, name in enumerate(fam_names):
            sel = fam == f
            if sel.any():
                core_fam = uses[f] == "core_only" if f < len(uses) else False
                report["coverage"][f"family_{name}"] = dict(episodes=int(sel.sum()),
                                                            hours=float(((lengths[sel] - 1) * rec_dt).sum() / 3600),
                                                            **({"use": "core_only"} if core_fam else {}),
                                                            **counts(sel, core_only=core_fam))
    if "terminal_reason" in arrays and "terminal_reasons" in system:
        tr = arrays["terminal_reason"]
        report["terminal_reasons"] = {name: int((tr == k).sum()) for k, name in enumerate(system["terminal_reasons"])}
    train = report["coverage"].get("split0", {})
    for s, b in enumerate(moving):
        core_fit = train.get(f"core_fit_{b['name']}", train.get(f"core_{b['name']}", 0)) if train else 0
        if train and core_fit < 1000:
            warns.append(f"{b['name']}: only {core_fit} contact-free training windows on the core grid (every "
                         "free_phase_step-th record, as train.py draws them; core-only families included)")
    for name in pair_names:
        if train and train.get(f"single_{name}", 0) < 20:
            warns.append(f"pair {name}: only {train.get('single_' + name, 0)} single-pair training windows "
                         "(the contact fit skips pairs with fewer than 20)")
    report.update(fails=fails, warns=warns, ok=not fails)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.data)
    text = json.dumps(report, indent=1)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
