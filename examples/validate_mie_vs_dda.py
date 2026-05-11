"""Validate DDA against Mie theory for a spherical particle.

This example compares MIE (exact) and DDA (approximate) optical properties
for a simple sphere, demonstrating solver consistency.
"""

import numpy as np

from Aerosol3D import AerosolParticle, Material, SimulationConfig, create_sphere, solve_optics


def main():
    material = Material(name="test", refractive_index=complex(1.5, 0.01), density=1.0)

    particle = AerosolParticle(name="Sphere100")
    particle.add_mesh("core", create_sphere((0, 0, 0), 100.0), material)

    print(f"Equivalent diameter: {particle.equivalent_diameter:.2f} nm")
    print(f"Effective refractive index: {particle.effective_refractive_index}")

    config = SimulationConfig(wavelength=550.0, n_host=1.0, solver="CPU")

    print("\n--- MIE solve ---")
    mie_result = solve_optics(
        particle, config, solver="MIE", compute_phase_func=True, verbose=False
    )
    print(f"Q_ext = {mie_result.cross_sections.Q_ext:.4f}")
    print(f"Q_sca = {mie_result.cross_sections.Q_sca:.4f}")
    print(f"g     = {mie_result.cross_sections.g:.4f}")

    print("\n--- DDA solve (single orientation) ---")
    dda_result = solve_optics(
        particle, config, solver="DDA", compute_phase_func=True, verbose=False
    )
    print(f"Q_ext = {dda_result.cross_sections.Q_ext:.4f}")
    print(f"Q_sca = {dda_result.cross_sections.Q_sca:.4f}")
    print(f"g     = {dda_result.cross_sections.g:.4f}")

    print("\n--- DDA solve (orientational average, n_dirs=100) ---")
    dda_avg_result = solve_optics(
        particle,
        config,
        solver="DDA",
        orientational_average=True,
        n_dirs=100,
        compute_phase_func=True,
        verbose=False,
    )
    print(f"Q_ext = {dda_avg_result.cross_sections.Q_ext:.4f}")
    print(f"Q_sca = {dda_avg_result.cross_sections.Q_sca:.4f}")
    print(f"g     = {dda_avg_result.cross_sections.g:.4f}")

    print("\n--- Comparison ---")
    print(
        f"Q_ext: MIE={mie_result.cross_sections.Q_ext:.4f}, "
        f"DDA={dda_result.cross_sections.Q_ext:.4f}, "
        f"DDA_avg={dda_avg_result.cross_sections.Q_ext:.4f}"
    )
    print(
        f"Q_sca: MIE={mie_result.cross_sections.Q_sca:.4f}, "
        f"DDA={dda_result.cross_sections.Q_sca:.4f}, "
        f"DDA_avg={dda_avg_result.cross_sections.Q_sca:.4f}"
    )
    print(
        f"g:     MIE={mie_result.cross_sections.g:.4f}, "
        f"DDA={dda_result.cross_sections.g:.4f}, "
        f"DDA_avg={dda_avg_result.cross_sections.g:.4f}"
    )

    # --- Validation assertions ---
    mie_g = mie_result.cross_sections.g
    dda_g = dda_result.cross_sections.g
    dda_avg_g = dda_avg_result.cross_sections.g

    # g agreement within 10% relative error
    assert abs(dda_g - mie_g) / abs(mie_g) < 0.10, (
        f"g mismatch too large: DDA={dda_g:.4f}, MIE={mie_g:.4f}"
    )
    print("PASS: DDA g agrees with MIE within 10%")

    # Orientational-averaged g should be even closer
    assert abs(dda_avg_g - mie_g) / abs(mie_g) < 0.10, (
        f"Averaged g mismatch too large: DDA_avg={dda_avg_g:.4f}, MIE={mie_g:.4f}"
    )
    print("PASS: DDA averaged g agrees with MIE within 10%")

    # Phase function comparison
    try:
        import matplotlib.pyplot as plt

        theta_mie = np.degrees(mie_result.phase_function.theta)
        P11_mie = mie_result.phase_function.P11[:, 0]

        theta_dda = np.degrees(dda_result.phase_function.theta)
        P11_dda = np.mean(dda_result.phase_function.P11, axis=1)

        theta_dda_avg = np.degrees(dda_avg_result.phase_function.theta)
        P11_dda_avg = np.mean(dda_avg_result.phase_function.P11, axis=1)

        # P11 correlation check (compare on common theta grid)
        from numpy import corrcoef

        # Interpolate DDA to MIE theta grid for fair comparison
        P11_dda_interp = np.interp(theta_mie, theta_dda, P11_dda)
        r = corrcoef(np.log10(P11_mie), np.log10(P11_dda_interp))[0, 1]
        assert r > 0.90, f"P11 correlation too low: r={r:.3f}"
        print(f"PASS: P11 correlation r={r:.3f}")

        plt.figure(figsize=(8, 5))
        plt.semilogy(theta_mie, P11_mie, label="MIE", linewidth=2)
        plt.semilogy(theta_dda, P11_dda, label="DDA", linewidth=1.5)
        plt.semilogy(theta_dda_avg, P11_dda_avg, label="DDA avg", linewidth=1.5)
        plt.xlabel("Scattering angle (degrees)")
        plt.ylabel("P11")
        plt.legend()
        plt.title("Phase Function Comparison: MIE vs DDA")
        plt.tight_layout()
        plt.savefig("mie_vs_dda_phase_function.png", dpi=150)
        print("\nPhase function plot saved: mie_vs_dda_phase_function.png")
    except ImportError:
        print("\nmatplotlib not available, skipping phase function plot")


if __name__ == "__main__":
    main()
