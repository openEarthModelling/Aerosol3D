# tests/test_optics_integration.py
"""Full pipeline test: geometry -> coating -> DDA solve -> visualize."""

import os

import pytest

JULIA_AVAILABLE = os.environ.get("SKIP_JULIA_TESTS") != "1"


@pytest.fixture(scope="module")
def julia_available():
    if not JULIA_AVAILABLE:
        pytest.skip("Julia runtime not available")


class TestFullOpticsPipeline:
    def test_coated_sphere_full_pipeline(self, julia_available, tmp_path):
        """BC sphere + sulfate coating -> DDA -> optical properties -> visualization."""
        from Aerosol3D import (
            AerosolParticle,
            Material,
            apply_distance_coating,
            create_sphere,
        )
        from Aerosol3D.optics import SimulationConfig, solve_optics
        from Aerosol3D.optics.visualization import plot_phase_function_2d, print_macroscopic

        soot = Material("soot", complex(1.8, 0.7), 1.8)
        sulfate = Material("sulfate", complex(1.4, 0.0), 1.8)

        # Create coated particle
        p = AerosolParticle("coated", unit="nm")
        p.add_mesh("core", create_sphere((0, 0, 0), 50.0), soot)
        apply_distance_coating(p, thickness=10.0, material=sulfate)

        # Solve optics
        config = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)
        result = solve_optics(
            p,
            config,
            voxel_size=10.0,
            compute_near_field=True,
            compute_phase_func=True,
        )

        # Verify results
        assert result.n_dipoles > 0
        assert result.cross_sections.C_ext > 0
        assert 0 <= result.cross_sections.SSA <= 1
        assert -1 <= result.cross_sections.g <= 1
        assert result.phase_function is not None
        assert result.voxel_grid is not None

        # Print macroscopic properties (stdout verification)
        print_macroscopic(result)

        # Visualize
        plot_phase_function_2d(result, save_path=str(tmp_path / "phase.png"))
        assert os.path.exists(tmp_path / "phase.png")

    def test_phase_function_normalization(self, julia_available):
        """P11 must integrate to approximately 1 over the full sphere."""
        import numpy as np

        _trapz = getattr(np, "trapezoid", np.trapz)

        from Aerosol3D import (
            AerosolParticle,
            Material,
            apply_distance_coating,
            create_sphere,
        )
        from Aerosol3D.optics import SimulationConfig, solve_optics

        soot = Material("soot", complex(1.8, 0.7), 1.8)
        sulfate = Material("sulfate", complex(1.4, 0.0), 1.8)

        p = AerosolParticle("coated", unit="nm")
        p.add_mesh("core", create_sphere((0, 0, 0), 50.0), soot)
        apply_distance_coating(p, thickness=10.0, material=sulfate)

        config = SimulationConfig(wavelength=550.0, dipole_spacing=10.0)
        result = solve_optics(
            p,
            config,
            voxel_size=10.0,
            compute_phase_func=True,
            orientational_average=True,
            n_dirs=1,
            show_progress=False,
        )

        assert result.phase_function is not None
        theta = result.phase_function.theta
        P11 = result.phase_function.P11

        # Integrate P11 over the sphere using trapezoidal rule
        # ∫ P11 dΩ = ∫₀^{2π} ∫₀^π P11(θ,φ) sin(θ) dθ dφ
        sin_theta = np.sin(theta)
        # Average over phi first
        P11_theta = np.mean(P11, axis=1)
        integrand = P11_theta * sin_theta
        integral = _trapz(integrand, theta) * 2 * np.pi

        assert integral == pytest.approx(1.0, abs=0.05)

    def test_import_from_top_level(self, julia_available):
        """Verify optics exports are accessible from Aerosol3D top-level."""
        import Aerosol3D

        assert hasattr(Aerosol3D, "solve_optics")
        assert hasattr(Aerosol3D, "SimulationConfig")
