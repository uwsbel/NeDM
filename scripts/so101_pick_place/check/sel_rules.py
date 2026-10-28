"""Addendum B of SELECTION_v12.md (results/sim_checks/ on the Hub) as pure functions (no file I/O), used by
gates_summary.py.

- names: v12_<arm>_s<seed> (a run) and v12_<arm>_s<seed>_it<1000|1500> (a checkpoint); "visR" is a prefix of
  "visRtbpl", so the arm is matched as a whole token between "v12_" and "_s<digits>".
- B1 tolerances: G2 = max(2.5 SD sqrt(1 + 1/3), 4) (A1: one run vs the mean of D's 3), G3 = max(2.5 SD sqrt(2), 4)
  (one run vs ONE D run).
- B2 own look: visR and visRtbpl = the real-look parts, every other arm (and D) = the grey parts.
- B3 veto: grasp releases at >= 2 distinct starts in block 613,000 AND at >= 2 distinct starts in block 613,300; it
  applies to every arm except ctrl.
- B4 (1) checkpoint: it 1500 if it passes G1, G2, G3 and is not vetoed, else it 1000 with the same test, else out;
  (2) seed: more G2 places (own look, sim lenient rule); a difference below 20 places is a tie -> the lower seed.
- B5 slots: 1 visual (visR, visRtbpl, visH), 2 table (tbpl, tb10, tb15, tb05), 3 table + visual (visRtbpl, visH, then
  the next unused table arm), 4 control (ctrl, visH, then the next unused table arm); empty if no candidate.
"""
from __future__ import annotations

import math
import re

ARMS = ("ctrl", "tb05", "tb10", "tb15", "tbpl", "visR", "visH", "visRtbpl")
_ARM_RX = "|".join(sorted(ARMS, key=len, reverse=True))          # longest first (visRtbpl before visR)
RUN_RX = re.compile(rf"^v12_({_ARM_RX})_s(\d+)$")
CKPT_RX = re.compile(rf"^(v12_(?:{_ARM_RX})_s\d+)_it(1000|1500)$")
ITS = ("it1500", "it1000")                                          # B4 (1) order
# SELECTION_v12.md section 1 (12 runs, 24 checkpoints; the "26" in section 1 is a miscount)
EXPECTED_RUNS = ("v12_ctrl_s1201", "v12_ctrl_s1202", "v12_tb05_s1205", "v12_tb10_s1210", "v12_tb15_s1215",
                 "v12_tbpl_s1221", "v12_tbpl_s1222", "v12_visR_s1231", "v12_visR_s1232", "v12_visH_s1235",
                 "v12_visRtbpl_s1241", "v12_visRtbpl_s1242")
REAL_LOOK_ARMS = ("visR", "visRtbpl")
TIE_PLACES = 20
VETO_MIN_STARTS = 2
SLOT1 = ("visR", "visRtbpl", "visH")
TABLE_ORDER = ("tbpl", "tb10", "tb15", "tb05")
SLOT3 = ("visRtbpl", "visH") + TABLE_ORDER
SLOT4 = ("ctrl", "visH") + TABLE_ORDER
SLOTS = ((1, "visual", SLOT1), (2, "table", TABLE_ORDER), (3, "table + visual", SLOT3), (4, "control", SLOT4))


def parse_run(run):
    """'v12_visRtbpl_s1241' -> ('visRtbpl', 1241); anything else -> (None, None)."""
    m = RUN_RX.match(run or "")
    return (m.group(1), int(m.group(2))) if m else (None, None)


def parse_ckpt(name):
    """'v12_visR_s1231_it1500' -> ('v12_visR_s1231', 'it1500'); anything else -> (None, None)."""
    m = CKPT_RX.match(name or "")
    return (m.group(1), f"it{m.group(2)}") if m else (None, None)


def arm_of(name):
    """The arm of a run or checkpoint name (None if the name is not a v12 candidate)."""
    run, _ = parse_ckpt(name)
    return parse_run(run or name)[0]


def own_look(arm):
    """B2: the look of a student's pinned scene: 'real' for visR / visRtbpl, else 'grey' (D and the other arms)."""
    return "real" if arm in REAL_LOOK_ARMS else "grey"


def tol_g2(sd):
    """A1 (kept by B1): one candidate run vs the mean of D's three runs."""
    return max(2.5 * sd * math.sqrt(1.0 + 1.0 / 3.0), 4.0)


def tol_g3(sd):
    """B1: one candidate run vs ONE D run at each plane."""
    return max(2.5 * sd * math.sqrt(2.0), 4.0)


def vetoed(starts_a, starts_b):
    """B3: grasp releases at >= 2 distinct starts in block 613,000 AND in block 613,300 (None if a block is missing)."""
    if starts_a is None or starts_b is None:
        return None
    return starts_a >= VETO_MIN_STARTS and starts_b >= VETO_MIN_STARTS


def veto_applies(arm):
    """B3: every arm except ctrl (the training-length control)."""
    return arm != "ctrl"


def ckpt_test(summary, arm):
    """B4 (1) test of one checkpoint: True (passes G1, G2, G3 and, unless ctrl, is not vetoed), False, or None
    (undecided: a summary or a required run is missing, or an error)."""
    if not summary or summary.get("errors"):
        return None
    g = summary.get("pass_G123")
    if g is None:
        return None
    if not g:
        return False
    if not veto_applies(arm):
        return True
    v = summary.get("veto", {}).get("vetoed")
    if v is None:
        return None
    return not v


def pick_checkpoint(its, arm):
    """B4 (1): its = {'it1500': summary, 'it1000': summary}. Returns dict(pick, status, why); status 'pass', 'out' or
    'undecided' (an undecided it 1500 is not skipped: the run is undecided until it is complete)."""
    why = []
    for it in ITS:
        ok = ckpt_test(its.get(it), arm)
        if ok is None:
            why.append(f"{it} undecided")
            return dict(pick=None, status="undecided", why="; ".join(why))
        if ok:
            why.append(f"{it} passes" + ("" if veto_applies(arm) else " G1-G3 (ctrl: veto exempt)"))
            return dict(pick=it, status="pass", why="; ".join(why))
        why.append(f"{it} fails ({fail_reason(its[it], arm)})")
    return dict(pick=None, status="out", why="; ".join(why) + " -> the run is out")


def fail_reason(summary, arm):
    out = [g for g in ("G1", "G2", "G3") if not summary.get("gates", {}).get(g, {}).get("pass", False)]
    if veto_applies(arm) and summary.get("veto", {}).get("vetoed"):
        out.append("vetoed")
    return ", ".join(out) or "?"


def pick_seed(cands):
    """B4 (2): cands = [dict(run, seed, places)] (passing runs of one arm). The run with more G2 places; a difference
    below TIE_PLACES is a tie and a tie goes to the lower seed. Returns (chosen dict, why)."""
    if not cands:
        return None, "no passing run"
    if len(cands) == 1:
        return cands[0], "the only passing run"
    best = max(c["places"] for c in cands)
    tied = [c for c in cands if best - c["places"] < TIE_PLACES]
    ch = min(tied, key=lambda c: c["seed"])
    pl = ", ".join(f"s{c['seed']} {c['places']}" for c in sorted(cands, key=lambda c: c["seed"]))
    if len(tied) > 1:
        return ch, f"G2 places {pl}: difference below {TIE_PLACES} = tie -> the lower seed s{ch['seed']}"
    return ch, f"G2 places {pl}: s{ch['seed']} has {TIE_PLACES} or more places more"


def fill_slots(passing):
    """B5: passing = the set of arms with a chosen run. Returns [dict(slot, family, arm or None, rule)]."""
    used, out = {}, []
    for k, fam, order in SLOTS:
        notes, arm = [], None
        for a in order:
            if a not in passing:
                notes.append(f"{a}: no passing run")
            elif a in used:
                notes.append(f"{a}: used in slot {used[a]}")
            else:
                arm = a
                break
        if arm is not None:
            used[arm] = k
            rule = f"B5 slot {k} order {' > '.join(order)}: " + "; ".join(notes + [f"-> {arm}"])
        else:
            rule = f"B5 slot {k} order {' > '.join(order)}: " + "; ".join(notes) + " -> empty"
        out.append(dict(slot=k, family=fam, arm=arm, rule=rule))
    return out
