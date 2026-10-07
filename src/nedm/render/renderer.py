"""Batch renderer: meshes and per-world poses in, one image per world out.

A renderer only. The NRD model stays the dynamics, and nothing is simulated to make a
picture. The drawing is done by a backend (``nedm.render.backends``) chosen by name, and the
scene, poses and cameras are the same whichever one draws them.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from nedm.render import backends
from nedm.render.scene import Scene


class BatchRenderer:
    """``num_worlds`` copies of a :class:`Scene`, each with its own poses and cameras.

    Poses and camera transforms may be torch tensors or numpy arrays. Torch tensors on the
    render device are read in place. Anything else is copied there first.
    """

    def __init__(
        self,
        scene: Scene,
        num_worlds: int,
        width: int = 128,
        height: int = 128,
        cameras: int = 1,
        fov_deg: float = 45.0,
        device: Any = None,
        shadows: bool = True,
        sky: tuple[int, int, int] = (200, 215, 230),
        backend: str = "newton",
    ) -> None:
        self.scene, self.num_worlds = scene, int(num_worlds)
        self.width, self.height, self.cameras = int(width), int(height), int(cameras)
        self.body_names = scene.body_names
        self.body_index = {name: i for i, name in enumerate(self.body_names)}
        self.joint_index, self.coord_count = scene.joint_index, scene.coord_count
        self.backend_name = backend
        self.backend = backends.make(backend)(scene, self.num_worlds, self.width, self.height, self.cameras,
                                              fov_deg, device, shadows, sky)
        self.device = self.backend.device
        self._torch_device = torch.device("cuda:0" if "cuda" in str(self.device) else "cpu")
        self._body_q: torch.Tensor | None = None

    @property
    def interop(self) -> str | None:
        """``"zero-copy"`` or ``"host-copy"``, once something has been rendered."""
        return self.backend.interop

    # ---- poses -----------------------------------------------------------------------

    def set_joint_q(self, joint_q) -> None:
        """Joint coordinates (N, coord_count) of a URDF scene. Runs forward kinematics."""
        if not self.scene.articulated:
            raise RuntimeError("this scene has no joints, give body transforms with set_body_q")
        self._body_q = self.scene.body_transforms(self._tensor(joint_q, (self.num_worlds, self.coord_count)))

    def set_body_q(self, body_q) -> None:
        """World transforms (N, bodies, 7) of every body, in :attr:`body_names` order."""
        self._body_q = self._tensor(body_q, (self.num_worlds, len(self.body_names), 7))

    def body_q(self) -> torch.Tensor:
        """Current body transforms (N, bodies, 7), on the render device."""
        if self._body_q is None:
            raise RuntimeError("no poses yet: call set_joint_q or set_body_q first")
        return self._body_q

    def joint_map(self, state_fields: Sequence[str], mapping: Mapping[str, Any]):
        """A :class:`~nedm.render.poses.JointMap` from state fields to this scene's joints."""
        from nedm.render.poses import JointMap
        return JointMap(self.joint_index, self.coord_count, state_fields, mapping,
                        base_coords=7 if self.scene.floating else 0)

    # ---- render ----------------------------------------------------------------------

    def render(self, camera, *, joint_q=None, body_q=None):
        """Render every world. ``camera`` is (N, 7), or (N, C, 7) with several cameras per world."""
        if joint_q is not None:
            self.set_joint_q(joint_q)
        if body_q is not None:
            self.set_body_q(body_q)
        return self.backend.render(self.body_q(), self._tensor(camera, (self.num_worlds, self.cameras, 7)))

    def _tensor(self, value, shape: tuple[int, ...]) -> torch.Tensor:
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))
        return value.detach().to(self._torch_device, torch.float32).reshape(shape)
