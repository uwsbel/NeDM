"""Plot existing diagnostic artifacts only; no model or simulation execution."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
draws = json.loads((ROOT / "replayed_draws_v1/replayed_draws.json").read_text())
exposure = json.loads((ROOT / "exposure_v1/exposure.json").read_text())
patch = json.loads((ROOT / "patch_geometry_v1/patch_geometry.json").read_text())
colors = ["#3267a8", "#d97932"]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 3, figsize=(15, 4.9), constrained_layout=True)
names = ["Narrow\n4-second forecast", "Diverse\n12-second forecast"]
prior = [exposure["packs"][k]["splits"]["train"]["groups"]["prior_any_failure"]["fraction"] * 100 for k in ["narrow", "diverse"]]
axs[0].bar(names, 100 - np.array(prior), label="Before first failure", color="#4c956c")
axs[0].bar(names, prior, bottom=100 - np.array(prior), label="After some failure", color="#bebebe")
for i, p in enumerate(prior):
    axs[0].text(i, 100 - p / 2, f"{p:.1f}%", ha="center", va="center")
axs[0].set(ylabel="Share of training windows (%)", title="The training mixture changed", ylim=(0, 103))
axs[0].legend(loc="lower left", fontsize=9)
for panel, group, title in [(1, "anchor0", "Actual launch-positive presentations"), (2, "pre_first_any_failure", "All fresh-positive label elements")]:
    for i, key in enumerate(["narrow", "diverse"]):
        metric = "at_last_horizon_row_draws" if panel == 1 else "positive_horizon_element_draws"
        vals = [draws["packs"][key]["groups"][group]["positive_exposure"][event][metric] for event in ["contact", "bounded"]]
        bars = axs[panel].bar(np.arange(2) + (i - .5) * .35, vals, .35, label=key.title(), color=colors[i])
        axs[panel].bar_label(bars, labels=[f"{v:,}" for v in vals], padding=3, fontsize=9)
    axs[panel].set_xticks([0, 1], ["Contact", "Bounded motion"])
    axs[panel].set(title=title, ylabel="Repeated training presentations", yscale="log")
    axs[panel].set_ylim(10 if panel == 1 else 1e4, 4e4 if panel == 1 else 1e6)
    axs[panel].legend(fontsize=9)
fig.suptitle("Exact seed-11 sampler replay: 5,000 updates; narrow batch 64, diverse batch 32", fontsize=14)
fig.text(.5, -.035, "Counts are correlated training exposure, not independent trials. Labels differ: narrow asset contact / 4 s; diverse asset-or-chassis contact / 12 s.", ha="center", fontsize=10)
fig.savefig(ROOT / "training_exposure.png", dpi=170, bbox_inches="tight")
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
for i, (family, asset_id) in enumerate([("rough_mosaic", 1), ("rolling_hills", 0)]):
    asset = patch["scenes"][f"diverse_v1_test_{family}_00"]["assets"][asset_id]
    vals = [asset[k] for k in ["raw_projected_edge_pixels", "candidate_input_edge_pixels", "global_context_edge_pixels"]]
    bars = axs[0].bar(np.arange(3) + (i - .5) * .35, vals, .35, label=family.replace("_", " "), color=colors[i])
    axs[0].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
axs[0].axhline(1, color="gray", linestyle=":", linewidth=1)
axs[0].set_xticks([0, 1, 2], ["Raw RGB-D\n1024²", "Candidate source\n512²", "Global context\n128²"])
axs[0].set(ylabel="Projected rock edge (pixels)", title="The rocks exist, but global context is coarse", ylim=(0, 9))
axs[0].legend(fontsize=9)
for i, key in enumerate(["narrow", "diverse"]):
    vals = [.3397, .6794, .6794 if key == "narrow" else 2.7176, .5333]
    bars = axs[1].bar(np.arange(4) + (i - .5) * .35, vals, .35, label=key.title(), color=colors[i])
    axs[1].bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
axs[1].set_xticks(range(4), ["Raw", "Candidate\nsource", "Global\ncontext", "Local patch\nsampling"])
axs[1].set(ylabel="Ground spacing (m)", title="Local patch resolution was preserved", ylim=(0, 3.3))
axs[1].legend(fontsize=9)
fig.suptitle("Measured representation limits; these alone do not establish the cause of prediction errors", fontsize=13)
fig.savefig(ROOT / "sensor_resolution.png", dpi=170)
plt.close(fig)
