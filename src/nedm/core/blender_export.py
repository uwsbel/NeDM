"""Small PyChrono Blender postprocess wrapper.

Chrono's :class:`pychrono.postprocess.ChBlender` writes an ``*.assets.py``
script plus per-frame ``output/stateNNNNN.py/.dat`` files for Blender's
``chrono_import.py`` add-on.  This module keeps the setup/cadence logic in one
place so rollout scripts can call the Python API without depending on Irrlicht.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pychrono as chrono
import pychrono.postprocess as postprocess


def _vec(values: tuple[float, float, float]) -> Any:
    return chrono.ChVector3d(float(values[0]), float(values[1]), float(values[2]))


class BlenderFrameExporter:
    """Time-throttled wrapper around ``postprocess.ChBlender``.

    Call the instance at simulation/control frame boundaries.  It exports at
    most one Blender state frame per ``1 / fps`` seconds of Chrono time.
    """

    def __init__(
        self,
        system: Any,
        output_dir: str | Path,
        *,
        fps: float = 20.0,
        max_frames: int | None = None,
        script_name: str = "exported",
        picture_size: tuple[int, int] = (1280, 720),
        camera_location: tuple[float, float, float] = (-10.0, -12.0, 5.0),
        camera_aim: tuple[float, float, float] = (0.0, 0.0, 1.0),
        camera_angle_deg: float = 45.0,
        orthographic: bool = False,
        light_location: tuple[float, float, float] = (-10.0, -8.0, 12.0),
        include_contacts: bool = False,
        include_other_physics: bool = False,
        include_links: bool = False,
        show_item_frames: bool = False,
        clean: bool = False,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if max_frames is not None and max_frames <= 0:
            raise ValueError(f"max_frames must be positive when provided, got {max_frames}")

        self.system = system
        self.output_dir = Path(output_dir).expanduser().resolve()
        if clean:
            self._clean_known_outputs(self.output_dir, script_name)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.period_s = 1.0 / self.fps
        self.max_frames = int(max_frames) if max_frames is not None else None
        self.script_name = script_name
        self.include_other_physics = bool(include_other_physics)
        self.include_links = bool(include_links)
        self.frame_count = 0
        self.next_export_time_s = float(system.GetChTime())

        self.exporter = postprocess.ChBlender(system)
        self.exporter.SetBasePath(str(self.output_dir))
        self.exporter.SetOutputScriptFile(script_name)
        self.exporter.SetPictureFilebase("picture")
        self.exporter.SetPictureSize(int(picture_size[0]), int(picture_size[1]))
        self.exporter.SetBlenderUp_is_ChronoZ()
        self.exporter.SetCamera(
            _vec(camera_location),
            _vec(camera_aim),
            float(camera_angle_deg),
            bool(orthographic),
        )
        self.exporter.SetLight(_vec(light_location), chrono.ChColor(1.0, 1.0, 1.0), True)
        self.exporter.SetBackground(chrono.ChColor(0.78, 0.84, 0.90))
        self.exporter.SetShowItemsFrames(bool(show_item_frames), 0.25)
        self.exporter.SetShowLinksFrames(False, 0.0)
        if not include_contacts:
            self.exporter.SetShowContactsOff()
        self.exporter.SetUseSingleAssetFile(True)
        self._add_render_items()
        self.exporter.ExportScript()
        self._remove_link_frame_globals()

    def __call__(self, system: Any | None = None) -> None:
        # arm_data.set_frame_hook passes the current ChSystem.  The exporter
        # already owns the same system; this argument is accepted for that hook.
        self.maybe_export()

    def maybe_export(self, *, force: bool = False) -> bool:
        if self.max_frames is not None and self.frame_count >= self.max_frames:
            return False
        time_s = float(self.system.GetChTime())
        if not force and time_s + 1.0e-9 < self.next_export_time_s:
            return False
        self._add_render_items()
        self.exporter.ExportData()
        self.frame_count += 1
        if force:
            self.next_export_time_s = time_s + self.period_s
        else:
            while self.next_export_time_s <= time_s + 1.0e-9:
                self.next_export_time_s += self.period_s
        return True

    def summary(self) -> dict[str, Any]:
        output_path = self.output_dir / "output"
        state_files = sorted(output_path.glob("state*.py")) if output_path.is_dir() else []
        return {
            "output_dir": str(self.output_dir),
            "assets_script": str(self.output_dir / f"{self.script_name}.assets.py"),
            "state_file_count": len(state_files),
            "captured_frames": int(self.frame_count),
            "fps": float(self.fps),
            "max_frames": self.max_frames,
            "include_other_physics": self.include_other_physics,
            "include_links": self.include_links,
            "hide_frame_parent_empties": True,
        }

    def write_summary(self, path: str | Path | None = None) -> Path:
        self._hide_state_frame_parent_empties()
        summary_path = Path(path) if path is not None else self.output_dir / "blender_export_summary.json"
        summary_path.write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
        return summary_path

    @staticmethod
    def _clean_known_outputs(output_dir: Path, script_name: str) -> None:
        if not output_dir.exists():
            return
        for child in (output_dir / "output", output_dir / "anim"):
            if child.exists():
                shutil.rmtree(child)
        for file_path in output_dir.glob("*.assets.py"):
            file_path.unlink()
        summary = output_dir / "blender_export_summary.json"
        if summary.exists():
            summary.unlink()
        script_assets = output_dir / f"{script_name}.assets.py"
        if script_assets.exists():
            script_assets.unlink()

    def _add_render_items(self) -> None:
        for body in self.system.GetBodies():
            self.exporter.Add(body)
        get_meshes = getattr(self.system, "GetMeshes", None)
        if get_meshes is not None:
            for mesh in get_meshes():
                self.exporter.Add(mesh)
        if self.include_other_physics:
            get_other = getattr(self.system, "GetOtherPhysicsItems", None)
            if get_other is not None:
                for item in get_other():
                    self.exporter.Add(item)
        if self.include_links:
            for link in self.system.GetLinks():
                self.exporter.Add(link)

    def _remove_link_frame_globals(self) -> None:
        assets_path = self.output_dir / f"{self.script_name}.assets.py"
        if not assets_path.is_file():
            return
        lines = assets_path.read_text(encoding="utf-8").splitlines()
        filtered = [
            line
            for line in lines
            if not line.startswith("chrono_view_link_csys")
            and not line.startswith("chrono_view_link_csys_size")
        ]
        assets_path.write_text("\n".join(filtered) + "\n", encoding="utf-8")

    def _hide_state_frame_parent_empties(self) -> None:
        output_path = self.output_dir / "output"
        if not output_path.is_dir():
            return
        marker = "# NeDM: hide Chrono frame parent empties"
        patch = f"""{marker}

def make_chrono_object_assetlist(
    mname, mpos, mrot, masset_list,
    _nedm_original_make_chrono_object_assetlist=make_chrono_object_assetlist,
):
    result = _nedm_original_make_chrono_object_assetlist(mname, mpos, mrot, masset_list)
    chobject = chrono_frame_objects.objects.get(mname)
    if chobject and chobject.type == 'EMPTY':
        chobject.empty_display_size = 0.0
        chobject.hide_select = True
    return result

"""
        for state_path in sorted(output_path.glob("state*.py")):
            text = state_path.read_text(encoding="utf-8")
            if marker in text:
                continue
            state_path.write_text(patch + text, encoding="utf-8")
