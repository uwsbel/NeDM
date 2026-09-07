"""The LEGACY derivation is frozen, and must NOT be realigned with the driver.

The verdict harness re-derives prewalk and ground tilt from a seeded RNG for corpora
that do not record them. The driver draws the same quantities from its own RNG. Two
copies of one contract, and when the driver's pitch range was capped to +-1.5 while
the harness kept +-3.0, every corpus collected afterwards failed the bit-identical
replay check -- looking like simulator non-determinism rather than a code divergence.

THE OBVIOUS FIX IS WRONG. Realigning the harness to the driver's current +-1.5 would
break `go2_joint_off3000000`, which was collected BEFORE the cap with +-3.0 and is the
baseline behind every scored result in this project. The legacy derivation must keep
reproducing the corpus it was written for, so the two files SHOULD differ from here on.

So this pins the legacy constants against being "fixed", rather than asserting the two
agree. New corpora record their parameters in the sidecar and do not use this path at
all; it exists only to replay what already exists.
"""
import random, re, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

# What go2_joint_off3000000 was collected with, pre-cap. Changing these silently
# invalidates the only replay-compatible baseline corpus in the project.
FROZEN = {"prewalk": ("0.0", "3.0"), "roll": ("-3.0", "3.0"), "pitch": ("-3.0", "3.0")}


def test_legacy_derivation_is_frozen():
    src = (ROOT / "scripts/evaluation/run_go2_finetune_verdict.py").read_text()
    for name, want in FROZEN.items():
        m = re.search(rf'{name} = tr\.uniform\(([-\d.]+), ([-\d.]+)\)', src)
        assert m is not None, f"legacy {name} draw not found -- did the fallback move?"
        assert m.groups() == want, (
            f"legacy {name} is uniform{m.groups()}, frozen value is uniform{want}. "
            f"This path exists to replay go2_joint_off3000000, collected before the "
            f"ground-pitch cap. Realigning it with the current driver breaks the only "
            f"replay-compatible baseline in the project. If a corpus needs different "
            f"values it should RECORD them, not change this."
        )


def test_new_corpora_record_rather_than_derive():
    """The collector must write what the harness would otherwise have to guess."""
    src = (ROOT / "scripts/collection/collect_go2_smoke.py").read_text()
    for field in ("prewalk_s", "ground_tilt_roll_deg", "ground_tilt_pitch_deg",
                  "perturb_peak_n"):
        assert f'"{field}"' in src, f"{field} is not recorded in the episode sidecar"
    h = (ROOT / "scripts/evaluation/run_go2_finetune_verdict.py").read_text()
    assert "LEGACY_DERIVATION" in h, "the fallback is not gated behind an explicit flag"
    assert "--legacy-derivation" in h, "the flag is missing, so the fallback is silent"


def test_absence_is_fatal_by_default():
    """A sidecar without the fields must abort, not quietly guess."""
    h = (ROOT / "scripts/evaluation/run_go2_finetune_verdict.py").read_text()
    assert "sidecar records no collection parameters" in h, (
        "absence must produce an explicit refusal: it is ambiguous between a pre-cap "
        "corpus, where deriving is right, and a post-cap one, where it is wrong.")
