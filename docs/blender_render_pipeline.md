# Blender render pipeline (HMMWV)

Turns a Chrono policy rollout into an mp4. Three stages, two camera modes.

```
eval_hmmwv_rl_chrono_tracking.py --blender-output-dir   ->  exported.assets.py
                                                            output/stateNNNNN.py
                                                            crm_surface/*.ply   (CRM only)
        |
        v
blender-render/render_hmmwv_scene.py                    ->  frames/frameNNNN.png
        |
        v
scripts/rendering/render_rollout_video.sh   (fixed camera)
scripts/rendering/render_follow_video.sh    (chase camera)  ->  videos/*.mp4
```

## Stage 1 — export from Chrono

`scripts/evaluation/eval_hmmwv_rl_chrono_tracking.py --blender-output-dir <dir>` writes the
Chrono postprocess export plus, on CRM, one reconstructed soil surface per exported frame
(see `src/nedm/crm_surface.py`). The soil is the expensive part: ~4.5 GB per 180-frame
rollout, so render one policy at a time and delete `crm_surface/` once the mp4 exists.

`--crm-surface-margin-m` sets how far the reconstruction window extends past the reference
bounding box. 3 m is enough for the fixed top-down camera; **a chase camera needs 6 m**,
because it looks along the path toward the window edge instead of down at it.

### Soil surface quality (close-up renders)

Run with `--crm-surface-keep-particles`. The particle dumps are the expensive product — a
CRM rollout is non-deterministic, so re-running gives a *different* rollout rather than the
same one at higher quality — and keeping them means any later change to resolution,
smoothing or threshold is a `scripts/rendering/rebuild_crm_surface.py` run of a couple of
seconds per frame instead of a re-simulation.

Settings that matter at close range, in order of impact:

| flag | top-down | close-up | why |
|---|---|---|---|
| `--crm-surface-threshold` | 0.6 | **0.8** | The tyres throw individual particles up to 0.5 m into the air. At 0.6 each reconstructs as its own blob — the "field of shards" look. 0.8 drops the isolated flyers and keeps the churned band. 1.0 also erases the band. |
| `--crm-surface-cube-size` | 2.0 | 1.5 | 2.0 = 8 cm cells = the particle spacing, so every surface particle becomes its own bump. Below ~1.0 you are just resolving individual particles more precisely; the information is not there. |
| `--crm-surface-smoothing-iters` | 0 | 25 | Feature-weighted, so rut walls survive. |
| `--crm-surface-smoothing-length` | 1.5 | 2.0 | Wider kernel blends neighbouring particles into a continuous surface. |
| `--soil-shading` (Blender) | `flat` | `smooth` | Flat keeps relief legible at 30 m; up close it turns every 6 cm facet into a hard plane. |

Note that `--mesh-smoothing-weights` (on whenever smoothing is enabled) explicitly
*preserves isolated particles*, so no amount of smoothing removes the flyers — only the
threshold does.

### What the soil can actually show

Measured from the particle field of a ref-18 rollout (`initial_spacing_m` 0.08,
`depth_m` 0.25 — i.e. **three particle layers**): the wheels leave a **raised** band of
loosened soil, +3 to +10 cm above the original surface and ~2.6 m wide, and the deepest
point anywhere in the track is −0.9 cm. There is no sunken rut, because a tyre cannot sink
into a three-layer bed; it shoves material sideways and up. 0.085% of particles are
airborne, up to 0.5 m above the surface.

So a close-up cannot be made to show a carved wheel track by rendering alone. That needs
the physics changed — a deeper bed and a finer `initial_spacing_m`, or softer soil
(`cohesion` 5000 Pa / `young_modulus_pa` 1e6 in `configs/hmmwv_crm_eval.json` is fairly
stiff) — which also changes the domain the policy was trained and evaluated on.

## Stage 2 — Blender

`render_hmmwv_scene.py` dresses the scene per terrain kind (`rigid` / `bumpy` / `crm`),
overlays the reference and driven trajectories from the eval `.npz`, and renders stills or
a frame range. Terrain handling and the CRM rut-lighting recipe are documented in the
module docstring and in `--help`.

## Bumpy terrain — making the relief visible

`configs/hmmwv_bumpy_eval.json` is a 256 px heightmap over a 500 m patch (~2 m/cell,
±0.5 m), i.e. a **long-wavelength undulation of ~1–3% grade**, not a rock field. Relief
across the 5 m vehicle footprint is ~0.10 m. A vehicle-scale shot of it is legitimately
near-flat and no lighting fixes that.

What works is `--contour-interval-m`: topographic contour lines keyed to world height,
drawn into the ground material. At 0.05 m over a reference that traverses ~0.5 m of relief
you get ~9 lines across the frame and the hills and hollows are unmistakable, with the
vehicle visibly driving across them. The geometry is untouched — vertical exaggeration
would be the obvious alternative but it decouples the vehicle from the ground it is
actually driving on. State the interval in the caption so the lines are not read as
texture.

Two framing details:

* **Camera azimuth sets which screen axis the path runs along.** `camera_basis` gives
  screen-right = `(-sin az, cos az)`, so `az=180` puts world **y** across the frame and a
  long east–west path ends up squeezed into the short axis — `az=270` fixes that. It only
  bites on a path that is strongly axis-aligned: a diagonal one costs little either way
  (ref 12 auto-frames to 68.9 m at az=180 against 63.2 m at az=270), so keep `az=180` and
  the shared framing unless the numbers say otherwise. Print them with `auto_frame_ortho_scale`
  before choosing.
* Pick the reference for **curvature and relief**, not just for the error ordering. Every
  `bumpy_field` heightmap is deliberately flat within 32–35 m of the origin so the vehicle
  settles level at spawn, so no map has relief near (0, 0) and the only way onto real
  ground is a reference that drives out of that disc. Ref 12
  (`sustained_turn/b10_s001_sustained_turn_00048`, map 33) is the one worth rendering: a
  107° continuous sweep bending 20.9 m off the start–end chord, 97 m long, 37% of it on
  non-flat ground, with the climb (+0.28 m) falling in the tightest part of the turn. It
  also has the widest error spread of any bumpy reference (mixture 0.138 < crm_only 0.225
  < rigid_only 0.412 m). Ref 14 was the earlier pick and turns only 7°.

### Two bugs the bumpy path exposed (fixed 2026-08-11)

The chrono_import add-on's `frame_change_post` handler **deletes and rebuilds every object
in `chrono_frame_objects` on each frame change**. `build_terrain()` and the "hide
everything that is not vehicle or terrain" pass both ran only once, at setup, so from the
first rendered frame onwards:

1. the bumpy patch reverted to Chrono's default flat grey, losing the ground material and
   the contours (invisible on rigid/CRM, where the patch is hidden and substituted); and
2. Chrono's helper glyphs came back visible — a green Y-axis glyph near the origin can
   fill an entire orthographic frame.

Both are now re-applied per frame in `render_frame` via `TerrainStyler` and
`hide_scene_clutter`. Anything else added at setup that the handler can recreate needs the
same treatment; note `hide_scene_clutter` exempts `render_*` names because our trajectory
tubes are curves and would otherwise be caught by the rule that hides Chrono's glyphs.

## Camera modes

**Fixed (default).** One auto-framed orthographic camera covering the whole trajectory,
identical for every policy on the same reference — which is what makes a
mixture-vs-experts comparison legible. Use `render_rollout_video.sh`.

**Chase (`--follow`).** Perspective camera riding with the vehicle, for demo clips where
the subject is the vehicle and its interaction with the ground rather than the tracking
error. Use `render_follow_video.sh`.

### The three-terrain comparison set

Nine videos — {rigid, bumpy, CRM} × {mixture, rigid-only, CRM-only} — read as one family
only if the shot is identical, so these are fixed:

| | |
|---|---|
| camera | fixed, auto-framed ortho, `AZIMUTH=180 ELEVATION=24` |
| format | `1600×900`, `SAMPLES=128`, 20 fps |
| caption | `"<Terrain>  <policy>"` — `Rigid` / `Bumpy` / `CRM` × `mixture (flat+CRM)` / `rigid-only policy` / `CRM-only policy` |

No metric is burned into the frame. RMSE in a caption dates the video to one eval run and
invites reading three separately-framed clips as a like-for-like measurement; the number
belongs in the paper text, next to the table it came from.

Lighting stays per-terrain, since each terrain has a different thing to make legible: CRM
a grazing sun for the ruts, bumpy the elevation contours, rigid the defaults. Elevation 24
is a compromise that suits all three — it keeps the CRM ruts in silhouette while still
showing the vehicle in three-quarter view rather than roof-down.

The chase camera is driven from the eval's own pose log (`pose` = x, y, yaw), not from the
per-frame vehicle mesh: the bounding box of the wheels and suspension breathes as the body
rolls, and a camera locked to it shakes. The track is smoothed with a centred moving
average (`--follow-smooth-frames`, yaw averaged as a unit vector), so the vehicle floats
slightly within the frame the way a real chase shot does.

| knob | default | effect |
|---|---|---|
| `--follow-mode chase\|world` | `chase` | `chase` holds a constant view relative to the vehicle's heading; `world` keeps a fixed bearing and only translates |
| `--follow-azimuth-deg` | 38 | bearing from straight behind; 0 hides the wheels behind the body, 90 loses the sense of travel |
| `--follow-elevation-deg` | 15 | low keeps ruts in silhouette; above ~25 they flatten out |
| `--follow-distance-m` | 9 | with a 32 mm lens this frames the body plus all four contact patches |
| `--follow-target-z-offset-m` | −0.30 | aims at the wheels rather than the roof |
| `--follow-lead-m` | 0 | shifts the look-at along the heading; keep under ~1 m or the vehicle walks out of frame |
| `--trajectory-radius-scale` | 1.0 | **use ~0.35 when following** — tubes sized for a 30 m top-down frame become foreground pipes lying along the ruts |
| `--no-trajectory` | off | camera still uses the npz, overlay is not drawn |

Two things stay world-fixed while following, deliberately: the sun (a key light swinging
with the camera would flip the rut shadows on every turn — `--camera-azimuth-deg` remains
the bearing the sun is offset from) and the fill softbox's *direction*, though the softbox
itself is translated with the vehicle because a 7 m area light does not carry.

The substituted ground plane is a disc centred on the trajectory; at 15° of elevation the
camera looks toward its rim, so `--terrain-radius-m 400` replaces the 140 m the top-down
camera gets away with.

## Stage 3 — encode

Both drivers render in chunks with a fresh Blender process per chunk plus
`--skip-existing`, retrying until every frame exists, because Blender segfaults in native
code under sustained load on these boxes; a crash costs one chunk, not the sequence. The
optional 4th argument burns a caption into the mp4.

```bash
# demo clip, CRM soil, chase camera
EXTRA_ARGS="--crm-surface-dir $OUT/crm_surface --terrain-z-m 0.25 \
  --soil-shading flat --crm-filler-plane --crm-filler-drop-m 0.02 \
  --sun-elevation-deg 20 --sun-azimuth-offset-deg 90 --sun-energy 4.0 \
  --fill-energy 60 --sky-strength 0.22 --trajectory-radius-scale 0.35" \
TERRAIN=crm WIDTH=1600 HEIGHT=900 SAMPLES=128 \
  scripts/rendering/render_follow_video.sh \
    "$OUT" "${OUT}_eval/chrono_tracking_18.npz" videos/crm_follow.mp4 "caption"
```
