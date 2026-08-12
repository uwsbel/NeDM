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

Open it through a server rather than `file://` — the video swapper fetches
clips by relative path.

## Publish

`.github/workflows/pages.yml` uploads `web/` to Pages on every push to `main`
that touches it, and `workflow_dispatch` re-runs it by hand from the Actions
tab. It passes `enablement: true`, so the first successful run creates the
Pages site itself — no one needs to visit Settings → Pages first.

**Blocked as of 2026-08-12:** GitHub Pages requires the repository to be public
unless the owning account is on Pro/Team/Enterprise ("If the account that owns
the repository uses GitHub Free or GitHub Free for organizations, the
repository must be public"). `uwsbel/NeDM` is private under a Free org, so
every run fails at `configure-pages` with *Get Pages site failed … Not Found*,
and Settings → Pages shows no "Build and deployment" section at all — that is
the feature being unavailable, not a permissions problem. Making the repo
public, or upgrading the org, unblocks it with no change to this workflow.

Until then the page can be served from any public repo you own: copy `web/`
into it and point the same workflow at it.

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
cp $IMG/tracked_stress_trajectories.png    web/assets/figures/tracked-stress.png
```

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
