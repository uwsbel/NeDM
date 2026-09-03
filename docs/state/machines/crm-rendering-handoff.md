# CRM rendering: what to know before touching it again

**Status 2026-09-03: working.** A Go2 walks on a lit, opaque, granular CRM bed
rendered through Chrono::Sensor, from Python, on two GPU architectures.

This page exists because the knowledge was spread across ~20 commit messages and
would cost someone a day to reassemble. Everything here is measured.

## Where the code lives

```
src/nedm/quadruped/constants.py    conventions, scales, soil presets
                   robot.py        Go2Robot
                   policy.py       the inherited observation contract
                   terrain.py      CRM and rigid construction
                   camera.py       sprite render + the four camera modes
                   soilprobe.py    z95 sinkage with the ejecta filter
scripts/quadruped_go2_crm.py       argparse + the sim loop
```

Everyday use is six flags: `--seconds --terrain --soil --camera --out --no-policy`.
Everything else sits in an `advanced` argparse group.

```
python scripts/quadruped_go2_crm.py --camera overhead --out runs/demo
```

## To reproduce a render, you need all four of these

Miss any one and you get a plausible-looking failure rather than an error.

1. **The source build**, pinned at `6982828952a920bb4e857625e74cedcf46d3573a`.
   The feature postdates tag 10.0.0 by 272 commits, so conda `pychrono` 10.0.0
   cannot do this at all. See [`chrono-build.md`](chrono-build.md).
2. **`patches/0001-expose-fsi-sph-render-options.patch`** applied to
   `chrono-src`. Without it `ChFsiSphRenderOptions` is unbound, and the only
   configuration reachable from Python is the one that renders nothing.
3. **The sprite settings** below. These are not cosmetic tuning — three of the
   five are the difference between an image and a blank frame.
4. **`load_normals=True`** when loading the sprite meshes. This one is the
   difference between a lit surface and flat black.

### The build must apply the patch, and the code now refuses without it

`chrono-src` is a **clean checkout**; the patch is **not** baked into the tree.
So the build procedure is **checkout SHA → apply `patches/0001` → build**, and a
rebuild that skips the patch produces a sensor module whose only reachable
configuration renders nothing.

**That is not left to anyone remembering.** `attach_sph_rendering` in
`src/nedm/quadruped/camera.py` **raises** if `ChFsiSphRenderOptions` is absent,
naming the patch and this page. It previously attached anyway and reported
*"attached (defaults)"* — the silent no-op pattern, in our own code.

Note the compiled `_sensor.so` retains the binding regardless of the source
tree's state, so an existing build keeps working after the tree is cleaned. The
hazard is only on the *next* rebuild.

## The settings, and the two that read backwards

Constants in `src/nedm/quadruped/camera.py`, each carrying the measurement that
fixed it. **They are no longer CLI flags** — they are settled, so they are not
decisions to make at the command line.

| setting | value | consequence if wrong |
|---|---|---|
| `load_normals` | `True` | `False` → flat black `[0,0,0]`, looks like static |
| `render_particle_spacing` | `0.01` | **raising it removes sprites** |
| mesh scale (baked) | `2.0` | too small → background shows through |
| rotated variants | `24` | too few → triangular lattice + banding |
| `sprite_position_jitter` | `0.005` | `0` → position moiré in the near field |

**`render_particle_spacing` is a resampling target, not a sprite size, and its
effect is cubic.** From `ChOptixEngine.cpp`:

```
render_count = num_markers * (source_spacing / render_particle_spacing)^3
```

0.01 against an 0.02 source draws **8×** the markers. Raising this value to
"close the gaps" does the opposite. Exposed background measured 30.3% at 0.01,
45.8% at 0.02, 68.2% at 0.026.

**Sprite size is the mesh's own scale** (`fsi_sph_render.cu` uses
`template_scale[template_id]`), and `ChFsiSphRenderOptions` has no field for it —
nor for orientation. Both are still controllable, **because the caller owns the
mesh list**: N pre-rotated, pre-scaled copies baked into the vertex data via
`ChTriangleMeshConnected.Transform(...)`. The renderer cycles the list, so the
list *is* the variety.

## What cannot be done from Python

**The bed cannot be coloured.** It renders near-white. Two independent routes
fail: `AddMaterial` on the sprite shapes has *zero* effect (output identical to
one decimal on all three channels), and the vertex-colour arrays cannot be
populated because `std::vector<ChColor>` has no SWIG template. Cosmetic; recorded
so nobody spends an afternoon on it.

## The physics answer, which supersedes the rendering question

**The soil response under a walking Go2 is +1 to +5 mm, and the sign is UP.**
Measured from SPH particle z, renderer-independent, at 179–195 N peak stance.
The soil **piles** rather than bowls. On a bed discretised at 20 mm that is a
tenth of one particle diameter.

**So visible foot-soil interaction is not a rendering problem.** No sprite
setting will show it. The levers are **soil stiffness, particle size, or robot
mass**, and that is a physics-design decision.

Two traps in measuring this again:

- **A 95th-percentile z is a surface estimator only where there is no ejecta.**
  A footfall throws particles airborne; the unfiltered walking measurement
  returned +0.28 m on a bed topping out at 0.20 m. Filter to particles within
  3 cm of the control surface. The estimator was valid for a static foot and
  invalid for a walking one with no code change at all.
- **The "4 cm differential sinkage" in older notes was never sinkage.** It was
  differential *foot height* from foot body positions — the robot pitching
  nose-down.

## Not filed: the upstream report

Six items against Chrono, **none submitted**. Three share one signature —
*accepted, no diagnostic, no effect* — and that theme is the report's argument,
so file them together rather than as unrelated bugs.

| # | item |
|---|---|
| 1 | `ChFsiSphRenderOptions` unbound — makes the bound `AttachFsiSphSystem` unconfigurable. Patch exists. Necessary, **not** sufficient. |
| 2 | Primitive sprite shapes (`ChVisualShapeSphere`) silently draw nothing, though `sprite_shapes` declares `ChVisualShape` |
| 3 | `demo_SEN_CRM_Rendering` passes `load_normals=false` while shipping meshes **containing** normals, so the reference program's own output is unlit |
| 4 | Sprite material ignored — `AddMaterial` has no effect |
| 5 | Vertex colour arrays unpopulatable from Python (no `vector<ChColor>` template) |
| 6 | `Background` not constructible; SWIG reports no destructor |

Plus, separately: WP0c's `ChDepthCamera` `ray_scale`.

**Also unshipped:** `patches/0001` exists and round-trips, but only these two
machines have it. Anyone building past the feature commit needs it.
