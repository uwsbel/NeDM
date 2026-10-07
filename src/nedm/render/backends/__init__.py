"""Renderer backends. Each draws a :class:`~nedm.render.scene.Scene` for many worlds at once.

A backend is constructed with the scene and the image settings, and has one method:
``render(body_q, camera)`` with body transforms (N, bodies, 7) and camera transforms
(N, cameras, 7), both ``xyzw`` and both torch tensors, returning a :class:`Frames`.
"""

from __future__ import annotations

BACKENDS = ("newton", "madrona")


def make(name: str):
    if name == "newton":
        from nedm.render.backends.newton import NewtonBackend
        return NewtonBackend
    if name == "madrona":
        from nedm.render.backends.madrona import MadronaBackend
        return MadronaBackend
    raise ValueError(f"unknown renderer backend {name!r}, choose from {BACKENDS}")
