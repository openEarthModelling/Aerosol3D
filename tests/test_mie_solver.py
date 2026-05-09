import numpy as np
import pytest


def test_optical_result_has_solver_field():
    from Aerosol3D.optics.datastructs import CrossSections, OpticalResult, SimulationConfig

    cs = CrossSections(
        wavelength=550.0,
        C_ext=100.0,
        C_sca=80.0,
        C_abs=20.0,
        Q_ext=2.0,
        Q_sca=1.6,
        Q_abs=0.4,
        SSA=0.8,
        g=0.7,
        r_eff=50.0,
    )
    cfg = SimulationConfig(wavelength=550.0)
    result = OpticalResult(config=cfg, cross_sections=cs)
    assert hasattr(result, "solver")
    assert result.solver == "DDA"


def test_optical_result_solver_can_be_set():
    from Aerosol3D.optics.datastructs import CrossSections, OpticalResult, SimulationConfig

    cs = CrossSections(
        wavelength=550.0,
        C_ext=100.0,
        C_sca=80.0,
        C_abs=20.0,
        Q_ext=2.0,
        Q_sca=1.6,
        Q_abs=0.4,
        SSA=0.8,
        g=0.7,
        r_eff=50.0,
    )
    cfg = SimulationConfig(wavelength=550.0)
    result = OpticalResult(config=cfg, cross_sections=cs, solver="MIE")
    assert result.solver == "MIE"


class TestMieSolver:
    def test_solve_mie_importable(self):
        from Aerosol3D.optics.mie_solver import solve_mie

        assert callable(solve_mie)

    def test_solve_mie_returns_optical_result(self, soot_material):
        pytest.importorskip("PyMieScatt")
        from Aerosol3D import AerosolParticle, create_sphere
        from Aerosol3D.optics.datastructs import SimulationConfig
        from Aerosol3D.optics.mie_solver import solve_mie

        p = AerosolParticle(name="test")
        p.add_mesh("core", create_sphere((0, 0, 0), 50.0), soot_material)
        config = SimulationConfig(wavelength=550.0)

        result = solve_mie(p, config, verbose=False)

        assert result.solver == "MIE"
        assert result.cross_sections.Q_ext > 0
        assert result.cross_sections.Q_sca > 0
        assert result.cross_sections.g >= -1.0
        assert result.cross_sections.g <= 1.0
        assert result.voxel_grid is None
        assert result.n_dipoles == 0

    def test_solve_mie_with_phase_function(self, soot_material):
        pytest.importorskip("PyMieScatt")
        from Aerosol3D import AerosolParticle, create_sphere
        from Aerosol3D.optics.datastructs import SimulationConfig
        from Aerosol3D.optics.mie_solver import solve_mie

        p = AerosolParticle(name="test")
        p.add_mesh("core", create_sphere((0, 0, 0), 50.0), soot_material)
        config = SimulationConfig(wavelength=550.0)

        result = solve_mie(p, config, compute_phase_func=True, verbose=False)

        assert result.phase_function is not None
        assert result.phase_function.P11.shape[0] == 181
        assert result.phase_function.P11.shape[1] == 1
        theta = result.phase_function.theta
        P11 = result.phase_function.P11[:, 0]
        integral = 2 * np.pi * np.trapz(P11 * np.sin(theta), theta)
        assert integral == pytest.approx(1.0, abs=0.01)

    def test_solve_mie_ssa_consistency(self, soot_material):
        pytest.importorskip("PyMieScatt")
        from Aerosol3D import AerosolParticle, create_sphere
        from Aerosol3D.optics.datastructs import SimulationConfig
        from Aerosol3D.optics.mie_solver import solve_mie

        p = AerosolParticle(name="test")
        p.add_mesh("core", create_sphere((0, 0, 0), 50.0), soot_material)
        config = SimulationConfig(wavelength=550.0)

        result = solve_mie(p, config, verbose=False)
        cs = result.cross_sections

        # Q_ext = Q_sca + Q_abs
        assert cs.Q_ext == pytest.approx(cs.Q_sca + cs.Q_abs, abs=0.01)
        # SSA = Q_sca / Q_ext
        assert cs.SSA == pytest.approx(cs.Q_sca / cs.Q_ext, abs=0.01)

    def test_solve_mie_absorbing_material(self):
        pytest.importorskip("PyMieScatt")
        from Aerosol3D import AerosolParticle, Material, create_sphere
        from Aerosol3D.optics.datastructs import SimulationConfig
        from Aerosol3D.optics.mie_solver import solve_mie

        # Strongly absorbing material
        mat = Material(name="soot", refractive_index=complex(1.8, 0.7), density=1.8)
        p = AerosolParticle(name="test")
        p.add_mesh("core", create_sphere((0, 0, 0), 50.0), mat)
        config = SimulationConfig(wavelength=550.0)

        result = solve_mie(p, config, verbose=False)
        assert result.cross_sections.Q_ext > 0
        assert result.cross_sections.Q_sca > 0
        assert result.cross_sections.Q_abs > 0
        assert result.cross_sections.g >= -1.0
        assert result.cross_sections.g <= 1.0
