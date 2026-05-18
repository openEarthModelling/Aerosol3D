# tests/test_ema.py
import pytest


class TestVolumeWeighted:
    def test_two_component(self):
        from Aerosol3D.core.ema import volume_weighted

        volumes = [100.0, 400.0]
        ri = [complex(1.95, 0.79), complex(1.53, 0.0)]
        result = volume_weighted(volumes, ri)
        expected = 0.2 * complex(1.95, 0.79) + 0.8 * complex(1.53, 0.0)
        assert abs(result - expected) < 1e-10

    def test_single_component(self):
        from Aerosol3D.core.ema import volume_weighted

        result = volume_weighted([100.0], [complex(1.95, 0.79)])
        assert abs(result - complex(1.95, 0.79)) < 1e-10

    def test_empty_raises(self):
        from Aerosol3D.core.ema import volume_weighted

        with pytest.raises(ValueError, match="empty"):
            volume_weighted([], [])

    def test_equal_materials(self):
        from Aerosol3D.core.ema import volume_weighted

        m = complex(1.5, 0.0)
        result = volume_weighted([100.0, 200.0], [m, m])
        assert abs(result - m) < 1e-10
