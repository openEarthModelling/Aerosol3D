import os
import numpy as np
import pytest


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
