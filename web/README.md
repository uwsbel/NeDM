# Project page

The public page for *Learning the Right Abstraction: Neural Reduced Dynamics for
Complex Robot Control*, served by GitHub Pages at
**<https://uwsbel.github.io/NeDM/>**.

Everything here is static and self-contained — no build step, no CDN, no
JavaScript dependencies. `index.html` plus `assets/style.css` is the whole site.

```
web/
  index.html                  # the page
  assets/style.css            # all styling, light/dark aware
  assets/figures/*.png        # figures exported from the manuscript
  assets/videos/*.mp4         # 9 Blender renders of Chrono rollouts (~11 MB)
  assets/videos/posters/*.jpg # first-frame stills, shown before a clip loads
```

## Preview locally

```bash
python3 -m http.server 8000 --directory web
# then open http://localhost:8000
```

Open it through a server rather than `file://` — the clips are fetched by
relative path.

All nine videos sit in the page at once as a 3×3 matrix (terrain × policy).
They carry `preload="none"`, and an `IntersectionObserver` plays a row only
while it is on screen, so the ~11 MB arrives a row at a time rather than up
front.

## Publish

`.github/workflows/pages.yml` uploads `web/` to Pages on every push that
touches it, and `workflow_dispatch` re-runs it by hand from the Actions tab.

**The site has to be created once, by hand:**

**Settings → Pages → Build and deployment → Source: *GitHub Actions***

The workflow deploys to that site but cannot create it. `configure-pages` has
an `enablement` input that looks like it would, and it is a dead end — its own
`action.yml` says it *"requires a token other than `GITHUB_TOKEN`"*, needing
`administration:write`, which no workflow token carries. A PAT would work and
is not worth a repository secret for a step done once.

Two prerequisites had to clear before that setting even appears, both now done:

| Blocker | Symptom | Resolution |
|---|---|---|
| Free org + private repo | No "Build and deployment" section at all; `Get Pages site failed … Not Found` | Repository made public 2026-08-13 |
| `enablement: true` with the default token | `Create Pages site failed … Resource not accessible by integration` | Input removed; flip the setting by hand instead |

**Where the work lives.** This page and its workflow sit on the `project-page`
branch, kept off `main` while the workflow could not succeed. To prove the
deployment from the branch, allow it under Settings → Environments →
`github-pages` → Deployment branches, which defaults to the default branch
only. Then squash-merge to `main` and drop `project-page` from the workflow's
branch list.

Note that Git LFS files are **not** served by Pages (visitors get the pointer
text, not the file), so everything under `assets/` is committed as a regular
file. Keep it that way.

## Updating the media

Figures come from the manuscript's image archive; the PDF-only ones are
rasterised:

```bash
IMG=../Manuscripts/ImageArchive/journals/2026/neural-dynamics-model
pdftocairo -png -r 150 -singlefile $IMG/fpp.pdf          web/assets/figures/overview
pdftocairo -png -r 200 -singlefile $IMG/study-case-2.pdf web/assets/figures/study-case-2
cp $IMG/hmmwv_policy_transfer_bars.png     web/assets/figures/policy-transfer-bars.png
cp $IMG/hmmwv_policy_trajectories_grid.png web/assets/figures/policy-trajectories.png
```

The trajectory grid still prints "zero-shot" in panel (c); the page says
out-of-distribution everywhere it writes its own prose. Regenerating the figure
with `scripts/figures/` would settle the mismatch, at the cost of diverging from
the manuscript's own artifact.

Videos are the fixed-camera Blender renders from
`artifacts/blender_exports/videos/`, remuxed so the `moov` atom is at the front
(the browser can then start playing before the whole file arrives), and named
`<terrain>_<policy>.mp4` — the naming `index.html` relies on:

```bash
ffmpeg -i <src>.mp4 -c copy -movflags +faststart web/assets/videos/<terrain>_<policy>.mp4
ffmpeg -ss 8.5 -i <src>.mp4 -frames:v 1 -vf scale=800:-2 -q:v 5 \
       web/assets/videos/posters/<terrain>_<policy>.jpg
```

`terrain ∈ {rigid, crm, bumpy}`, `policy ∈ {generalist, rigid_only, crm_only}`.

## Adding the paper PDF

The Paper button in the header is disabled until there is a preprint to point
at. To turn it on, replace the `btn-disabled` anchor in `index.html` with a real
link — either an arXiv URL or a PDF committed to `web/assets/paper.pdf`.
