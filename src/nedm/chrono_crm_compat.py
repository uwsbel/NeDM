"""CRM API shim spanning pychrono 10.0.0 and the pinned source build.

WHY THIS EXISTS, AND WHEN TO DELETE IT
--------------------------------------
Chrono renamed several CRM entry points between release 10.0.0 (the conda
pychrono everything in this repo was measured against) and the main-branch SHA
6982828952a920bb4e857625e74cedcf46d3573a that we build from source:

    CRMTerrain.SetElasticSPH        -> CRMTerrain.SetCrmSPH
    CRMTerrain.SetActiveDomainDelay -> CRMTerrain.SetFreeFlowDuration
    fsi.ElasticMaterialProperties   -> fsi.SoilProperties

Every physics number this repo currently reports was produced through the OLD
names. Migrating outright would leave nothing to compare the migration against,
so this module lets ONE script run under BOTH interpreters -- which is the only
way to test whether the API change moved the physics.

That is its whole job. Once the Go2 has been re-run on the source build and the
two APIs are shown to agree, DELETE THIS MODULE and the conda path with it. A
compatibility layer that outlives its migration acquires users and then acquires
permanence; this one has an end condition, and that is it.

DESIGN CONSTRAINTS, and each is a reaction to a real failure
------------------------------------------------------------
Resolution happens ONCE at import, not per call: a decision needed once should
not be re-made hundreds of times, and per-call dispatch admits a behaviour change
mid-run.

Missing on BOTH sides raises. Never no-op, never silently default -- the failure
family this whole codebase spent a day cataloguing is calls that succeed while
doing nothing.

Present on BOTH sides ALSO raises. Two names for one operation means a build
neither branch was written for, and quietly picking one would reproduce that same
failure inside the tool written to prevent it.
"""

from __future__ import annotations

_OLD = "chrono-10.0.0"
_NEW = "chrono-main-6982828"


def _resolve(new_name, old_name, holder, what):
    """Pick exactly one of two names, or refuse."""
    has_new, has_old = hasattr(holder, new_name), hasattr(holder, old_name)
    if has_new and has_old:
        raise RuntimeError(
            f"{what}: BOTH {new_name!r} and {old_name!r} are present on "
            f"{getattr(holder, '__name__', holder)!r}. This shim knows two API "
            f"generations and this build matches neither cleanly; choosing one "
            f"silently is exactly the bug this module exists to avoid."
        )
    if has_new:
        return new_name, _NEW
    if has_old:
        return old_name, _OLD
    raise RuntimeError(
        f"{what}: NEITHER {new_name!r} nor {old_name!r} found on "
        f"{getattr(holder, '__name__', holder)!r}. Available: "
        f"{sorted(n for n in dir(holder) if not n.startswith('_'))}"
    )


def _detect():
    import pychrono.fsi as fsi
    import pychrono.vehicle as veh

    soil_name, gen_a = _resolve("SoilProperties", "ElasticMaterialProperties", fsi, "soil properties type")
    crm_name, gen_b = _resolve("SetCrmSPH", "SetElasticSPH", veh.CRMTerrain, "CRM soil setter")
    delay_name, gen_c = _resolve("SetFreeFlowDuration", "SetActiveDomainDelay", veh.CRMTerrain, "active-domain delay")

    # GetFsiSystemSPH / GetSystemFSI IS NOT A GENERATION MARKER AND IS NOT
    # RESOLVED HERE. It was, briefly, and it broke every CRM run on this
    # machine: conda pychrono 10.0.0 carries GetFsiSystemSPH together with the
    # OLD names for all three pairings above, so the four-way agreement check
    # reported "mixed API generations" on the exact build every physics number
    # in this repo was measured against. The pairing was asserted from the
    # rename list, not verified against the installed module.
    #
    # The one consumer is this module's own fsi_system(), which now resolves the
    # accessor by fallback at call time -- correct, because the name is not a
    # generation marker. Every caller outside this file already bypassed the
    # shim for it: camera.py does its own getattr chain and
    # collect_hmmwv_crm_smoke.py calls GetFsiSystemSPH directly.
    #
    # The three pairings that remain are the three documented renames, each
    # verified present-on-exactly-one-side in this build. If a future build is
    # found where the accessor genuinely tracks the generation, add it back WITH
    # the observation that established it.
    gens = {gen_a, gen_b, gen_c}
    if len(gens) != 1:
        raise RuntimeError(
            f"Mixed Chrono API generations in one build: soil type is {gen_a}, "
            f"CRM setter is {gen_b}, delay setter is {gen_c}. Refusing to guess."
        )
    return gens.pop(), fsi, soil_name, crm_name, delay_name


API_GENERATION, _fsi, _SOIL_NAME, _CRM_NAME, _DELAY_NAME = _detect()


def soil_properties():
    """Empty soil-property struct. Field names are IDENTICAL across both
    generations -- all 13 members verified equal -- so only the type name moved."""
    return getattr(_fsi, _SOIL_NAME)()


def set_crm_soil(terrain, props):
    """Apply soil properties to a CRMTerrain."""
    return getattr(terrain, _CRM_NAME)(props)


def set_free_flow_duration(terrain, seconds):
    """Delay before the active domain begins tracking. Same signature both sides."""
    return getattr(terrain, _DELAY_NAME)(seconds)


def fsi_system(terrain):
    """The terrain's FSI system.

    RESOLVED BY FALLBACK, NOT BY GENERATION. This accessor does NOT track the
    API generation -- conda pychrono 10.0.0 exposes GetFsiSystemSPH while
    carrying the OLD names for the three pairings in _detect -- so binding it to
    the generation is what made this shim reject the build everything was
    measured on. Take whichever name is present, and raise if neither is.
    """
    for name in ("GetFsiSystemSPH", "GetFluidSystemSPH", "GetSystemFSI"):
        getter = getattr(terrain, name, None)
        if getter is not None:
            return getter()
    raise AttributeError(
        "CRMTerrain exposes none of GetFsiSystemSPH, GetFluidSystemSPH or "
        f"GetSystemFSI. Available: {sorted(n for n in dir(terrain) if not n.startswith('_'))}"
    )


def stamp():
    """Tag for output artifacts. A shim that makes results comparable while
    leaving no record of which side produced them defeats its own purpose."""
    return API_GENERATION


# ---------------------------------------------------------------------------
# EXTERNAL-BUG workaround. Not one of ours; do NOT delete as obsolete.
#
# pychrono.sensor.Background is referenced by the public API on the source build
# -- ChOptixScene.SetBackground(Background) and GetBackground() -> Background --
# but the type is no longer constructible from Python, and the object returned by
# GetBackground comes back as a raw SwigPyObject with SWIG reporting:
#
#     swig/python detected a memory leak of type 'Background *', no destructor found.
#
# That is a wrapping defect upstream, not a rename: there is no new name to move
# to. Reported alongside the WP0c ChDepthCamera ray_scale issue.
#
# Consequence: the solid-colour background cannot be set on the source build.
# Frames render against the renderer default instead. This changes the BACKDROP
# only, never geometry or particles -- but it does mean a frame-to-frame pixel
# comparison across the two builds will differ in background pixels for reasons
# that have nothing to do with physics.
# ---------------------------------------------------------------------------

BACKGROUND_CONSTRUCTIBLE = None


def set_solid_background(scene, color_zenith):
    """Set a solid-colour background if this build allows it.

    Returns True if applied, False if skipped because of the upstream defect.
    Never raises: an unsettable backdrop must not stop a render.
    """
    global BACKGROUND_CONSTRUCTIBLE
    import pychrono.sensor as sens

    if BACKGROUND_CONSTRUCTIBLE is None:
        BACKGROUND_CONSTRUCTIBLE = hasattr(sens, "Background")

    if not BACKGROUND_CONSTRUCTIBLE:
        return False
    bg = sens.Background()
    bg.mode = sens.BackgroundMode_SOLID_COLOR
    bg.color_zenith = color_zenith
    scene.SetBackground(bg)
    return True


# ---------------------------------------------------------------------------
# Reusing the 2025 RL harness UNMODIFIED.
#
# chrono_crmenv.py defines the input contract that model_2999.pt was trained
# against: the leg reorder, the blanket sign negation, the observation scaling
# and the hardcoded command slot. Reimplementing those by hand means maintaining
# four conventions, one of which -- the negation -- has no recorded explanation
# anywhere in the source. Reusing the original file inherits all four correctly
# BY CONSTRUCTION, and we never need to know why the negation is there.
#
# The file is imported byte-identical. It fails to import here only because it
# pulls pychrono.vsg and pychrono.irrlicht at module scope for run-time display,
# and our build has neither. Patching those imports out would fork the one file
# whose value is being provably the original, so instead the missing modules are
# satisfied from outside via sys.modules.
#
# The stubs RAISE on any attribute access. They are not silent no-ops: an API
# that accepts a call and does nothing is the exact failure family this project
# has spent days cataloguing, and introducing our own would be indefensible.
# ---------------------------------------------------------------------------

import sys as _sys
import types as _types


def _stub_module(name):
    mod = _types.ModuleType(name)

    def _raise(*_a, **_k):
        raise RuntimeError(
            f"{name} is not built in this environment. It is imported only for "
            f"run-time display; headless simulation does not need it. Something "
            f"asked for it, which means a display path is being exercised."
        )

    # Dunders must behave normally: import machinery and inspect read __file__,
    # __spec__ and friends, and a stub that raises for those breaks the importer
    # long before anything touches the display API. Only real attribute lookups
    # raise.
    def _module_getattr(attr):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        return _raise

    mod.__getattr__ = _module_getattr
    mod.__file__ = f"<stub {name}>"
    mod.__all__ = []
    _sys.modules[name] = mod
    return mod


def install_display_stubs():
    """Satisfy display-only imports so the RL harness loads unmodified.

    Returns the list of names stubbed, so a caller can report what is absent
    rather than discovering it at first use.
    """
    stubbed = []
    for name in ("pychrono.vsg", "pychrono.irrlicht"):
        try:
            __import__(name)
        except ImportError:
            _stub_module(name)
            stubbed.append(name)
    import pychrono.fsi as fsi
    if not hasattr(fsi, "ChFsiVisualizationVSG"):
        def _raise_vsg(*_a, **_k):
            raise RuntimeError(
                "fsi.ChFsiVisualizationVSG is absent from this Chrono build "
                "(VSG not enabled). Display only; headless does not need it."
            )
        fsi.ChFsiVisualizationVSG = _raise_vsg
        stubbed.append("pychrono.fsi.ChFsiVisualizationVSG")
    return stubbed
