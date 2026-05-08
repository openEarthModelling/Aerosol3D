import os
import numpy as np
import pytest

JULIA_AVAILABLE = os.environ.get("SKIP_JULIA_TESTS") != "1"


@pytest.fixture(scope="module")
def julia_available():
    if not JULIA_AVAILABLE:
        pytest.skip("Julia runtime not available")


class TestMultiWavelength:
    def test_returns_list_for_wavelength_list(self, julia_available, soot_material):
        from aerosol3d import AerosolParticle, create_sphere
        from aerosol3d.optics.datastructs import SimulationConfig
        from aerosol3d.optics.dda_solver import solve_optics

        p = AerosolParticle(name="soot_sphere", unit="nm")
        p.add_mesh("core", create_sphere((0, 0, 0), 50.0), soot_material)
        config = SimulationConfig(wavelength=[400.0, 550.0, 700.0], dipole_spacing=10.0)

        results = solve_optics(p, config, voxel_size=10.0, verbose=False)

        assert isinstance(results, list)
        assert len(results) == 3
        for r in results:
            assert r.n_dipoles > 0
            assert r.cross_sections.C_ext > 0

    def test_backward_compat_single_float(self, julia_available, soot_material):
        from aerosol3d import AerosolParticle, create_sphere
        from aerosol3d.optics.datastructs import SimulationConfig
        from aerosol3d.optics.dda_solver import solve_optics

        p = AerosolParticle(name="soot_sphere", unit="nm")
        p.add_mesh("core", create_sphere((0, 0, 0), 50.0), soot_material)
        config = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)

        result = solve_optics(p, config, voxel_size=10.0, verbose=False)

        assert not isinstance(result, list)
        assert result.cross_sections.C_ext > 0
