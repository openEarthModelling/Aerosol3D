# tests/test_ema.py
import numpy as np
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


class TestMaxwellGarnett:
    def test_two_component_bc_sulfate(self):
        from Aerosol3D.core.ema import maxwell_garnett

        # BC (20%) + sulfate (80%)
        volumes = [100.0, 400.0]
        ri = [complex(1.95, 0.79), complex(1.53, 0.0)]
        result = maxwell_garnett(volumes, ri)

        # Verify against formula: host = sulfate (largest volume)
        m_h = complex(1.53, 0.0)
        m_i = complex(1.95, 0.79)
        f_i = 100.0 / 500.0
        beta = f_i * (m_i**2 - m_h**2) / (m_i**2 + 2 * m_h**2)
        eps_eff = m_h**2 * (1 + 2 * beta) / (1 - beta)
        expected = np.sqrt(eps_eff)
        assert abs(result - expected) < 1e-10

    def test_single_component(self):
        from Aerosol3D.core.ema import maxwell_garnett

        m = complex(1.95, 0.79)
        result = maxwell_garnett([100.0], [m])
        assert abs(result - m) < 1e-10

    def test_equal_materials(self):
        from Aerosol3D.core.ema import maxwell_garnett

        m = complex(1.5, 0.0)
        result = maxwell_garnett([100.0, 200.0], [m, m])
        assert abs(result - m) < 1e-10

    def test_real_valued(self):
        from Aerosol3D.core.ema import maxwell_garnett

        # Purely real RI: host=1.5, inclusion=2.0, f_i=0.25
        volumes = [1.0, 3.0]
        ri = [complex(2.0, 0.0), complex(1.5, 0.0)]
        result = maxwell_garnett(volumes, ri)
        m_h = complex(1.5, 0.0)
        m_i = complex(2.0, 0.0)
        f_i = 0.25
        beta = f_i * (m_i**2 - m_h**2) / (m_i**2 + 2 * m_h**2)
        expected = np.sqrt(m_h**2 * (1 + 2 * beta) / (1 - beta))
        assert abs(result - expected) < 1e-10
        assert result.imag == pytest.approx(0.0, abs=1e-12)


class TestBruggeman:
    def test_two_component_residual(self):
        from Aerosol3D.core.ema import bruggeman

        # BC (20%) + sulfate (80%)
        volumes = [100.0, 400.0]
        ri = [complex(1.95, 0.79), complex(1.53, 0.0)]
        result = bruggeman(volumes, ri)

        # Verify the Bruggeman equation is satisfied: residual ≈ 0
        eps_eff = result**2
        total = sum(volumes)
        residual = sum(
            (v / total) * (m**2 - eps_eff) / (m**2 + 2 * eps_eff) for v, m in zip(volumes, ri)
        )
        assert abs(residual) < 1e-8

    def test_single_component(self):
        from Aerosol3D.core.ema import bruggeman

        m = complex(1.95, 0.79)
        result = bruggeman([100.0], [m])
        assert abs(result - m) < 1e-10

    def test_equal_materials(self):
        from Aerosol3D.core.ema import bruggeman

        m = complex(1.5, 0.0)
        result = bruggeman([100.0, 200.0], [m, m])
        assert abs(result - m) < 1e-10

    def test_real_valued_residual(self):
        from Aerosol3D.core.ema import bruggeman

        volumes = [1.0, 3.0]
        ri = [complex(2.0, 0.0), complex(1.5, 0.0)]
        result = bruggeman(volumes, ri)
        eps_eff = result**2
        total = sum(volumes)
        residual = sum(
            (v / total) * (m**2 - eps_eff) / (m**2 + 2 * eps_eff) for v, m in zip(volumes, ri)
        )
        assert abs(residual) < 1e-8
        assert result.imag == pytest.approx(0.0, abs=1e-12)
