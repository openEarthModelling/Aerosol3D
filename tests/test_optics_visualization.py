# tests/test_optics_visualization.py
import os

import pytest

JULIA_AVAILABLE = os.environ.get("SKIP_JULIA_TESTS") != "1"


@pytest.fixture(scope="module")
def julia_available():
    if not JULIA_AVAILABLE:
        pytest.skip("Julia runtime not available")


@pytest.fixture(scope="module")
def optical_result(julia_available, soot_material):
    """Pre-computed optical result for visualization tests."""
    from Aerosol3D import AerosolParticle, create_sphere
    from Aerosol3D.optics.datastructs import SimulationConfig
    from Aerosol3D.optics.dda_solver import solve_optics

    p = AerosolParticle(name="soot_sphere", unit="nm")
    p.add_mesh("core", create_sphere((0, 0, 0), 50.0), soot_material)
    config = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)
    return solve_optics(p, config, voxel_size=10.0, compute_phase_func=True)


class TestPlotPhaseFunction2D:
    def test_creates_figure(self, optical_result, tmp_path):
        from Aerosol3D.optics.visualization import plot_phase_function_2d

        fig_path = str(tmp_path / "phase_func.png")
        plot_phase_function_2d(optical_result, save_path=fig_path)
        assert os.path.exists(fig_path)

    def test_log_scale(self, optical_result, tmp_path):
        from Aerosol3D.optics.visualization import plot_phase_function_2d

        fig_path = str(tmp_path / "phase_func_log.png")
        plot_phase_function_2d(optical_result, log_scale=True, save_path=fig_path)
        assert os.path.exists(fig_path)


class TestPlotNearField:
    def test_creates_plotter(self, optical_result):
        from Aerosol3D.optics.visualization import plot_near_field

        # Should not raise -- just verify it returns a plotter
        plotter = plot_near_field(optical_result, show=False)
        assert plotter is not None


class TestPlotMacroscopic:
    def test_prints_properties(self, optical_result, capsys):
        from Aerosol3D.optics.visualization import print_macroscopic

        print_macroscopic(optical_result)
        captured = capsys.readouterr()
        assert "C_ext" in captured.out
        assert "SSA" in captured.out
        assert "g" in captured.out
