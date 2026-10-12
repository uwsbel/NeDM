"""Tests of nedm.render.terrain: the ground a rollout is drawn on.

The terrain builders are checked as arrays. One renderer check then looks straight down at a
stepped height map from known heights, so a pass means the relief reached the renderer at the
ground height, the right way round, with the plane just under it.

    PYTHONPATH=src python -m unittest discover -s tests/render -p "test_terrain.py" -v

Madrona allows one renderer per process: its check runs with NEDM_MADRONA_TEST=terrain.
"""
import importlib.util
import os
import unittest

import numpy as np
import torch

from nedm.render import terrain
from nedm.render.scene import Ground, MeshBody, Scene

HAVE_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None
HAVE_MADRONA = bool(os.environ.get("NEDM_MADRONA_BUILD")) and torch.cuda.is_available()

RES, FOV_DEG, CAMERA_Z, GROUND_Z = 16, 45.0, 5.0, 0.25
OFF_AXIS = float(np.sqrt(1.0 + 2.0 * (np.tan(np.radians(FOV_DEG) / 2) / RES) ** 2))     # see test_renderer.py
BOX_V = np.array([[x, y, z] for x in (-0.1, 0.1) for y in (-0.1, 0.1) for z in (-0.1, 0.1)], dtype=np.float32)
BOX_F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.int32)


class TerrainBuilderTest(unittest.TestCase):
    def test_heightmap_puts_each_height_where_it_was_asked(self):
        z = np.arange(12, dtype=np.float32).reshape(4, 3) * 0.01
        ground = terrain.heightmap(z, spacing=0.5, origin=(1.0, 2.0))
        vertices, faces = ground.relief.arrays()
        self.assertEqual(vertices.shape, (12, 3))
        self.assertEqual(faces.shape, (2 * 3 * 2, 3))
        for i in range(4):
            for j in range(3):
                np.testing.assert_allclose(vertices[i * 3 + j], [1.0 + 0.5 * i, 2.0 + 0.5 * j, z[i, j]], atol=1e-6)
        a, b, c = (vertices[faces[:, k]] for k in range(3))
        self.assertTrue((np.cross(b - a, c - a)[:, 2] > 0).all())         # every triangle faces up

    def test_heightmap_is_centered_without_an_origin(self):
        vertices, _ = terrain.heightmap(np.zeros((5, 3)), spacing=0.5).relief.arrays()
        np.testing.assert_allclose(vertices[:, :2].min(axis=0), [-1.0, -0.5], atol=1e-6)
        np.testing.assert_allclose(vertices[:, :2].max(axis=0), [1.0, 0.5], atol=1e-6)

    def test_soil_is_reproducible_zero_mean_and_bounded(self):
        options = dict(extent=(-2.0, 2.0, -1.0, 1.0), spacing=0.02, roughness=0.01)
        z = terrain.soil(seed=1, **options).relief.arrays()[0][:, 2]
        np.testing.assert_array_equal(z, terrain.soil(seed=1, **options).relief.arrays()[0][:, 2])
        self.assertFalse(np.array_equal(z, terrain.soil(seed=2, **options).relief.arrays()[0][:, 2]))
        self.assertLess(abs(float(z.mean())), 1e-3)
        self.assertLessEqual(float(np.abs(z).max()), terrain.CLIP * 0.01 + 1e-6)
        # RMS is the roughness asked for in the middle, where the fade toward the edges has not started
        ground = terrain.soil(seed=1, **options)
        vertices = ground.relief.arrays()[0]
        middle = (np.abs(vertices[:, 0]) < 1.0) & (np.abs(vertices[:, 1]) < 0.5)
        self.assertAlmostEqual(float(vertices[middle, 2].std()), 0.01, delta=0.0015)

    def test_soil_fades_to_flat_at_its_edges(self):
        vertices = terrain.soil(extent=(-2.0, 2.0, -1.0, 1.0), spacing=0.02, roughness=0.01).relief.arrays()[0]
        on_edge = (np.abs(np.abs(vertices[:, 0]) - 2.0) < 1e-6) | (np.abs(np.abs(vertices[:, 1]) - 1.0) < 1e-6)
        self.assertGreater(int(on_edge.sum()), 100)
        np.testing.assert_allclose(vertices[on_edge, 2], 0.0, atol=1e-7)

    def test_soil_covers_its_extent_and_carries_on_flat_beyond_it(self):
        extent = (-2.0, 2.0, -1.0, 1.0)
        vertices, _ = terrain.soil(extent=extent, spacing=0.1, skirt=0.0).relief.arrays()
        np.testing.assert_allclose(vertices[:, :2].min(axis=0), [-2.0, -1.0], atol=1e-5)
        np.testing.assert_allclose(vertices[:, :2].max(axis=0), [2.0, 1.0], atol=1e-5)
        wide, faces = terrain.soil(extent=extent, spacing=0.1, skirt=50.0).relief.arrays()
        self.assertEqual(len(wide), len(vertices) + 8)
        np.testing.assert_allclose(wide[:, :2].min(axis=0), [-52.0, -51.0], atol=1e-4)
        np.testing.assert_allclose(wide[:, :2].max(axis=0), [52.0, 51.0], atol=1e-4)
        np.testing.assert_allclose(wide[-8:, 2], 0.0)
        a, b, c = (wide[faces[:, k]] for k in range(3))
        self.assertTrue((np.cross(b - a, c - a)[:, 2] > 0).all())         # the frame faces up too
        # the frame's area is the outer rectangle minus the extent
        frame = 0.5 * np.cross(b - a, c - a)[-8:, 2].sum()
        self.assertAlmostEqual(float(frame), 104.0 * 102.0 - 4.0 * 2.0, delta=0.5)

    def test_the_plane_sits_just_under_the_relief(self):
        ground = terrain.heightmap(np.array([[0.3, 0.1], [0.2, 0.4]]), spacing=1.0)
        ground.height = 2.0
        self.assertAlmostEqual(ground.plane_height, 2.0 + 0.1 - 0.002, places=6)
        self.assertEqual(terrain.checker().plane_height, 0.0)

    def test_terrains_by_name(self):
        self.assertIsNotNone(terrain.make("checker").tile_mesh())
        bare = terrain.make("plain")
        self.assertIsNone(bare.tile_mesh())
        self.assertIsNone(bare.relief)
        loose = terrain.make("soil", extent=(-1.0, 1.0, -1.0, 1.0), spacing=0.1)
        self.assertIsNone(loose.tile_mesh())
        self.assertEqual(loose.relief.arrays()[0].shape, (21 * 21 + 8, 3))      # the grid and the flat frame around it
        with self.assertRaises(ValueError):
            terrain.make("lava")
        ready = Ground(tile_size=None)
        self.assertIs(terrain.make(ready), ready)
        with self.assertRaises(TypeError):
            terrain.make(ready, color=(1.0, 0.0, 0.0))

    def test_a_scene_takes_a_terrain_by_name_or_ready_made(self):
        box = MeshBody("box", vertices=BOX_V, faces=BOX_F)
        by_name = Scene.from_meshes([box]).add_ground(height=0.3, terrain="plain").ground
        self.assertEqual((by_name.name, by_name.height), ("plain", 0.3))
        ready = Scene.from_meshes([box]).add_ground(height=0.3, terrain=terrain.soil(extent=(-1, 1, -1, 1))).ground
        self.assertEqual((ready.name, ready.height), ("soil", 0.3))
        self.assertEqual(Scene.from_meshes([box]).add_ground().ground.name, "checker")     # unchanged default


class TerrainSceneChecks:
    """A height map with a step: 0.5 m where x >= 0.5, 0.1 m where x <= 0, on ground at GROUND_Z."""

    backend = ""

    def test_relief_and_plane_are_drawn_where_they_should_be(self):
        from nedm.render import BatchRenderer, cameras
        z = np.full((5, 5), 0.1, dtype=np.float32)
        z[3:, :] = 0.5                                                  # rows are x: x = -1, -0.5, 0, 0.5, 1
        scene = Scene.from_meshes([MeshBody("box", vertices=BOX_V, faces=BOX_F)])
        scene.add_ground(height=GROUND_Z, terrain=terrain.heightmap(z, spacing=0.5))
        renderer = BatchRenderer(scene, num_worlds=3, width=RES, height=RES, fov_deg=FOV_DEG, shadows=False,
                                 backend=self.backend)
        # one camera per world, straight down: over the high side, over the low side, and off the map.
        # x and y differ in sign, so a map drawn with its axes swapped would fail.
        over = torch.tensor([[0.75, -0.75], [-0.75, 0.75], [5.0, 5.0]])
        eye = torch.cat([over, torch.full((3, 1), CAMERA_Z)], dim=1)
        down = cameras.look_at(eye, eye * torch.tensor([1.0, 1.0, 0.0]), up=(1.0, 0.0, 0.0))
        body_q = np.zeros((3, 1, 7), dtype=np.float32)
        body_q[:, 0, :3], body_q[:, 0, 6] = (0.0, 0.0, -50.0), 1.0     # the box is out of the way, under the ground
        depth = renderer.render(down, body_q=body_q).depth[:, 0, RES // 2, RES // 2]
        surface = GROUND_Z + np.array([0.5, 0.1, 0.1 - 0.002])          # the last is the plane, just under the map
        np.testing.assert_allclose(depth, OFF_AXIS * (CAMERA_Z - surface), atol=1e-4)
        del renderer


@unittest.skipUnless(HAVE_NEWTON, "newton and warp are not installed (see requirements-render.txt)")
class NewtonTerrainSceneTest(TerrainSceneChecks, unittest.TestCase):
    backend = "newton"


@unittest.skipUnless(HAVE_MADRONA and os.environ.get("NEDM_MADRONA_TEST") == "terrain",
                     "needs NEDM_MADRONA_BUILD, a GPU and NEDM_MADRONA_TEST=terrain")
class MadronaTerrainSceneTest(TerrainSceneChecks, unittest.TestCase):
    backend = "madrona"


if __name__ == "__main__":
    unittest.main()
