import os

import pytest

JULIA_AVAILABLE = os.environ.get("SKIP_JULIA_TESTS") != "1"


@pytest.fixture(scope="module")
def julia_available():
    if not JULIA_AVAILABLE:
        pytest.skip("Julia runtime not available")


class TestOrientationalAverage:
    def test_sphere_isotropic(self, julia_available, soot_material):
        """Sphere should be isotropic: orientational average equals single direction."""
        from Aerosol3D import AerosolParticle, create_sphere
        from Aerosol3D.optics.datastructs import SimulationConfig
        from Aerosol3D.optics.dda_solver import solve_optics

        p = AerosolParticle(name="soot_sphere", unit="nm")
        p.add_mesh("core", create_sphere((0, 0, 0), 50.0), soot_material)
        config = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)

        single = solve_optics(p, config, voxel_size=10.0, verbose=False)

        propagations = [
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.707, 0.707, 0.0),
            (0.577, 0.577, 0.577),
        ]
        averaged = solve_optics(
            p,
            config,
            voxel_size=10.0,
            propagations=propagations,
            show_progress=False,
            verbose=False,
        )

        assert averaged.cross_sections.C_ext == pytest.approx(single.cross_sections.C_ext, rel=0.05)
        assert averaged.cross_sections.C_sca == pytest.approx(single.cross_sections.C_sca, rel=0.05)
        assert averaged.cross_sections.g == pytest.approx(single.cross_sections.g, abs=0.05)
