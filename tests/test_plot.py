import os
import numpy as np
import pytest
import pyvista as pv


def _make_test_voxel_grid(material_ids_3d):
    """Create a pv.ImageData with cell_data['material_id'] from a 3D array."""
    nz, ny, nx = material_ids_3d.shape
    grid = pv.ImageData(
        dimensions=(nx + 1, ny + 1, nz + 1),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )
    grid.cell_data["material_id"] = material_ids_3d.ravel(order="F")
    return grid


@pytest.fixture
def sample_particle(soot_material):
    from aerosol3d import AerosolParticle, create_sphere

    p = AerosolParticle(name="test", unit="nm")
    p.add_mesh("core", create_sphere((0, 0, 0), 50.0), soot_material)
    return p


class TestSaveScreenshot:
    def test_saves_png(self, sample_particle, tmp_path):
        from aerosol3d.utils.plot import save_screenshot

        path = str(tmp_path / "out.png")
        save_screenshot(sample_particle, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_custom_colors(self, sample_particle, tmp_path):
        from aerosol3d.utils.plot import save_screenshot

        path = str(tmp_path / "out.png")
        save_screenshot(sample_particle, path, colors={"core": "red"})
        assert os.path.exists(path)


class TestSaveRotationVideo:
    def test_saves_mp4(self, sample_particle, tmp_path):
        from aerosol3d.utils.plot import save_rotation_video

        path = str(tmp_path / "out.mp4")
        save_rotation_video(sample_particle, path, n_frames=6, fps=6)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_rejects_non_mp4(self, sample_particle, tmp_path):
        from aerosol3d.utils.plot import save_rotation_video

        path = str(tmp_path / "out.gif")
        with pytest.raises(ValueError, match="mp4"):
            save_rotation_video(sample_particle, path)


class TestBuildVoxelGlyphMesh:
    def test_builds_glyph_mesh(self):
        from aerosol3d.utils.plot import _build_voxel_glyph_mesh

        ids = np.zeros((3, 3, 3), dtype=np.int32)
        ids[1, 1, 1] = 1
        grid = _make_test_voxel_grid(ids)

        glyphs = _build_voxel_glyph_mesh(grid)
        assert glyphs.n_cells > 0

    def test_preserves_material_id(self):
        from aerosol3d.utils.plot import _build_voxel_glyph_mesh

        ids = np.zeros((4, 4, 4), dtype=np.int32)
        ids[1:3, 1:3, 1:3] = 2
        grid = _make_test_voxel_grid(ids)

        glyphs = _build_voxel_glyph_mesh(grid)
        assert "material_id" in glyphs.cell_data
        unique = np.unique(glyphs.cell_data["material_id"])
        assert 2 in unique

    def test_empty_grid_raises(self):
        from aerosol3d.utils.plot import _build_voxel_glyph_mesh

        ids = np.zeros((3, 3, 3), dtype=np.int32)
        grid = _make_test_voxel_grid(ids)

        with pytest.raises(ValueError, match="No occupied voxels"):
            _build_voxel_glyph_mesh(grid)

    def test_missing_material_id_raises(self):
        from aerosol3d.utils.plot import _build_voxel_glyph_mesh

        grid = pv.ImageData(dimensions=(3, 3, 3), spacing=(1.0, 1.0, 1.0))
        # no material_id
        with pytest.raises(ValueError, match="material_id"):
            _build_voxel_glyph_mesh(grid)


class TestResolveVoxelColors:
    def test_auto_assigns_colors(self):
        from aerosol3d.utils.plot import _resolve_voxel_colors

        ids = np.zeros((3, 3, 3), dtype=np.int32)
        ids[1, 1, 1] = 1
        ids[1, 1, 2] = 2
        grid = _make_test_voxel_grid(ids)

        colors = _resolve_voxel_colors(grid, None)
        assert 1 in colors
        assert 2 in colors
        assert colors[1] != colors[2]

    def test_respects_custom_colors(self):
        from aerosol3d.utils.plot import _resolve_voxel_colors

        ids = np.zeros((3, 3, 3), dtype=np.int32)
        ids[1, 1, 1] = 1
        grid = _make_test_voxel_grid(ids)

        colors = _resolve_voxel_colors(grid, {1: "black"})
        assert colors[1] == "black"

    def test_fills_missing_with_auto(self):
        from aerosol3d.utils.plot import _resolve_voxel_colors

        ids = np.zeros((3, 3, 3), dtype=np.int32)
        ids[1, 1, 1] = 1
        ids[1, 1, 2] = 2
        grid = _make_test_voxel_grid(ids)

        colors = _resolve_voxel_colors(grid, {1: "black"})
        assert colors[1] == "black"
        assert 2 in colors
        assert colors[2] != "black"
