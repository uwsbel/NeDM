# Overview — where everything stands

**Updated:** 2026-09-02.

| Workstream | Status | Next action |
|---|---|---|
| State-only NRD (the paper) | Submitted to Elsevier; arXiv:2608.19375v1 | Nothing blocking. Manuscript needs `BibFiles/` to build citations |
| Study 1 — double pendulum + RGB | **Complete.** Joint rollouts, policy transfer, camera-only distillation all pass | Written up; nothing pending |
| Study 3 — HMMWV + overhead RGB-D | **WP0 complete, WP1 not started** | Build the 4-channel RGB-D encoder — see [`vision-study3-traverse.md`](vision-study3-traverse.md) |
| Study 2 — tabletop manipulation | Deferred; Study 3 was pulled ahead of it | — |
| Future case studies | Quadruped-on-CRM proposed by the author; excavation as second | Scope a WP0-style plan doc |

## Branch map

| Branch | Contents |
|---|---|
| `main` | The published state-only paper: HMMWV, tracked vehicle + arm. No vision code |
| `nrd_vision` | All vision work. 25 commits ahead of `main`, +43.7k lines, 322 files. Not merged |
| `blender-render` | Rendering |
| `kyle/agent-context` | Kyle's branch. This doc tree. Based on `nrd_vision` |

## The one fact that reframes everything

**No RGB-D has reached a model yet.** Depth exists only in the collected Study 3
dataset. Everything trained so far is RGB-only: `src/nedm/nrd/vision.py:41`
hardcodes `_conv_block(3, channels[0])` at 128² with `z2_dim=64`. The plan's
4-channel / 256² / `z2_dim=128` encoder does not exist, and `configs/nrd/`
contains only the four `dpend_*` configs.

Study 3 is therefore a **fully de-risked data pipeline one work package short of
its first training run**, not a partially trained model.

## Architecture, in one paragraph

`z2` is appended to `z1`, never substituted. The token is `[z1, z2, a]` into the
same `ContinuousTransformer` backbone the paper uses, with two heads — `Δz1` in
normalized-target space (the existing NeDM convention) and next-`z2` in
normalized latent space. Encoder and decoder live inside the module so one
checkpoint carries the whole surrogate; the decoder is off during policy
rollouts. See `src/nedm/nrd/model.py`.
