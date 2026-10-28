"""Renderer backends. Each draws a :class:`~nedm.render.scene.Scene` for many worlds at once.

A backend is constructed with the scene and the image settings, and has one method:
``render(body_q, camera)`` with body transforms (N, bodies, 7) and camera transforms
(N, cameras, 7), both ``xyzw`` and both torch tensors, returning a :class:`Frames`.
"""

from __future__ import annotations

BACKENDS = ("newton",)    # the study's Madrona backend is not part of this release (so101-pick-place-study)


def make(name: str):
    if name == "newton":
        from nedm.render.backends.newton import NewtonBackend
        return NewtonBackend
    raise ValueError(f"unknown renderer backend {name!r}, choose from {BACKENDS}")
