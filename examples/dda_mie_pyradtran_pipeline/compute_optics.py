"""Stage 1: Compute optical properties using DDA and Mie, save to NetCDF.

Usage:
    python compute_optics.py
    python compute_optics.py --solver MIE   # Mie only (fast, for testing)
    python compute_optics.py --solver DDA   # DDA only
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from Aerosol3D import (
    AerosolParticle,
    create_sphere,
    MixingState,
    preset_material,
    SimulationConfig,
    solve_optics,
)
from config import (
    PARTICLE_CONFIG,
    DDA_CONFIG,
    MIE_CONFIG,
    OPTICS_DDA_NC,
    OPTICS_MIE_NC,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def create_particle() -> AerosolParticle:
    """Create a bare black carbon sphere."""
    material = preset_material(PARTICLE_CONFIG["material"])
    radius = PARTICLE_CONFIG["radius_nm"]

    particle = AerosolParticle(
        name="bc_sphere",
        mixing_state=MixingState.INTERNAL,
        unit="nm",
    )
    particle.add_mesh("core", create_sphere((0, 0, 0), radius), material)
    return particle


def compute_optics(particle: AerosolParticle, solver: str) -> list:
    """Run optical computation for all wavelengths.

    Args:
        particle: The aerosol particle to compute.
        solver: "DDA" or "MIE".

    Returns:
        List of OpticalResult, one per wavelength.
    """
    wavelengths = PARTICLE_CONFIG["wavelengths_nm"]
    config = SimulationConfig(
        wavelength=wavelengths,
        source="solar",
        precision=DDA_CONFIG["precision"] if solver == "DDA" else "medium",
    )

    logger.info(f"Running {solver} solve for {len(wavelengths)} wavelengths: "
                f"{wavelengths[0]:.0f}-{wavelengths[-1]:.0f} nm")

    results = solve_optics(
        particle,
        config,
        solver=solver,
        compute_phase_func=True,
    )

    # solve_optics may return a single result or a list depending on wavelength type
    if not isinstance(results, list):
        results = [results]

    logger.info(f"{solver} solve complete: {len(results)} results")
    for r in results:
        cs = r.cross_sections
        logger.info(f"  λ={cs.wavelength:.0f}nm: Qext={cs.Q_ext:.4f}, "
                    f"Qsca={cs.Q_sca:.4f}, SSA={cs.SSA:.4f}, g={cs.g:.4f}")

    return results


def get_refractive_index(material, wavelength_nm: float) -> tuple[float, float]:
    """Get refractive index (n, k) at a given wavelength.

    Args:
        material: Aerosol3D Material object.
        wavelength_nm: Wavelength in nm.

    Returns:
        (n_real, n_imag) tuple.
    """
    # Material stores wavelength in um internally
    wavelength_um = wavelength_nm / 1000.0

    # Try interpolation from material's refractive index data
    if hasattr(material, 'refractive_index'):
        ri = material.refractive_index
        if hasattr(ri, 'wavelength_um') and hasattr(ri, 'n_real'):
            n_real = np.interp(
                wavelength_um, ri.wavelength_um, ri.n_real
            )
            n_imag = np.interp(
                wavelength_um, ri.wavelength_um, ri.k_imag
            )
            return float(n_real), float(n_imag)

    # Fallback: use preset values for black carbon at visible wavelengths
    logger.warning(f"Could not get refractive index from material, using fallback")
    return 1.75, 0.44


def results_to_dataset(results: list, solver: str) -> xr.Dataset:
    """Convert list of OpticalResult to xarray Dataset.

    Args:
        results: List of OpticalResult.
        solver: "DDA" or "MIE" for metadata.

    Returns:
        xarray Dataset with all optical properties.
    """
    n_wl = len(results)
    wavelengths = np.array([r.cross_sections.wavelength for r in results])

    # Phase function angles (use first result's angles)
    if results[0].phase_function is not None:
        theta = results[0].phase_function.theta  # radians
        phi = results[0].phase_function.phi      # radians
        n_theta = len(theta)
        n_phi = len(phi)
    else:
        theta = np.array([0.0])
        phi = np.array([0.0])
        n_theta = 1
        n_phi = 1

    # Extract scalar quantities
    C_ext = np.array([r.cross_sections.C_ext for r in results])
    C_sca = np.array([r.cross_sections.C_sca for r in results])
    C_abs = np.array([r.cross_sections.C_abs for r in results])
    Q_ext = np.array([r.cross_sections.Q_ext for r in results])
    Q_sca = np.array([r.cross_sections.Q_sca for r in results])
    Q_abs = np.array([r.cross_sections.Q_abs for r in results])
    SSA = np.array([r.cross_sections.SSA for r in results])
    g = np.array([r.cross_sections.g for r in results])
    r_eff = results[0].cross_sections.r_eff

    # Extract P11
    P11 = np.zeros((n_wl, n_theta, n_phi))
    for i, r in enumerate(results):
        if r.phase_function is not None:
            P11[i, :, :] = r.phase_function.P11

    # Get refractive indices
    material = preset_material(PARTICLE_CONFIG["material"])
    n_real_arr = np.zeros(n_wl)
    n_imag_arr = np.zeros(n_wl)
    for i, r in enumerate(results):
        n_real_arr[i], n_imag_arr[i] = get_refractive_index(
            material, r.cross_sections.wavelength
        )

    # Compute particle volume
    radius_nm = PARTICLE_CONFIG["radius_nm"]
    volume_nm3 = (4.0 / 3.0) * np.pi * radius_nm ** 3

    # Build dataset
    ds = xr.Dataset(
        {
            "C_ext_nm2": (["wavelength"], C_ext),
            "C_sca_nm2": (["wavelength"], C_sca),
            "C_abs_nm2": (["wavelength"], C_abs),
            "Q_ext": (["wavelength"], Q_ext),
            "Q_sca": (["wavelength"], Q_sca),
            "Q_abs": (["wavelength"], Q_abs),
            "SSA": (["wavelength"], SSA),
            "g": (["wavelength"], g),
            "P11": (["wavelength", "theta", "phi"], P11),
            "refractive_index_real": (["wavelength"], n_real_arr),
            "refractive_index_imag": (["wavelength"], n_imag_arr),
        },
        coords={
            "wavelength_nm": (["wavelength"], wavelengths),
            "theta_deg": (["theta"], np.degrees(theta)),
            "phi_deg": (["phi"], np.degrees(phi)),
        },
        attrs={
            "solver": solver,
            "material": PARTICLE_CONFIG["material"],
            "radius_nm": float(radius_nm),
            "r_eff_nm": float(r_eff),
            "particle_volume_nm3": float(volume_nm3),
            "n_wavelengths": n_wl,
            "n_theta": n_theta,
            "n_phi": n_phi,
        },
    )

    return ds


def plot_p11_comparison(dda_results: list, mie_results: list, output_dir: Path):
    """Plot P11 phase function comparison for each wavelength.

    Args:
        dda_results: List of DDA OpticalResult.
        mie_results: List of Mie OpticalResult.
        output_dir: Directory to save plots.
    """
    n_wl = len(dda_results)

    for i in range(n_wl):
        wl = dda_results[i].cross_sections.wavelength
        dda_p11 = dda_results[i].phase_function.P11[:, 0]  # Use first phi
        mie_p11 = mie_results[i].phase_function.P11[:, 0]
        dda_theta_deg = np.degrees(dda_results[i].phase_function.theta)
        mie_theta_deg = np.degrees(mie_results[i].phase_function.theta)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Linear plot
        ax = axes[0]
        ax.semilogy(dda_theta_deg, dda_p11, "b-", label="DDA", linewidth=1.5)
        ax.semilogy(mie_theta_deg, mie_p11, "r--", label="Mie", linewidth=1.5)
        ax.set_xlabel("Scattering angle θ (°)")
        ax.set_ylabel("P11")
        ax.set_title(f"P11(θ) @ {wl:.0f} nm (log scale)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Polar plot
        ax = axes[1]
        dda_theta_rad = np.radians(dda_theta_deg)
        mie_theta_rad = np.radians(mie_theta_deg)
        ax.semilogy(dda_theta_rad, dda_p11, "b-", label="DDA", linewidth=1.5)
        ax.semilogy(mie_theta_rad, mie_p11, "r--", label="Mie", linewidth=1.5)
        ax.set_xlabel("θ (rad)")
        ax.set_ylabel("P11")
        ax.set_title("P11(θ) (polar view)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Relative difference - interpolate Mie to DDA grid for comparison
        ax = axes[2]
        mie_p11_interp = np.interp(dda_theta_deg, mie_theta_deg, mie_p11)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_diff = (dda_p11 - mie_p11_interp) / mie_p11_interp * 100.0
        rel_diff = np.where(np.isfinite(rel_diff), rel_diff, 0.0)
        ax.plot(dda_theta_deg, rel_diff, "g-", linewidth=1.0)
        ax.axhline(0, color="k", linestyle="-", linewidth=0.5)
        ax.set_xlabel("Scattering angle θ (°)")
        ax.set_ylabel("Relative difference (%)")
        ax.set_title("(DDA - Mie) / Mie × 100%")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / f"p11_comparison_{wl:.0f}nm.png", dpi=150)
        plt.close(fig)
        logger.info(f"Saved P11 comparison plot for {wl:.0f} nm")


def print_optics_summary(dda_results: list, mie_results: list):
    """Print summary of optical property differences."""
    logger.info("\n" + "=" * 70)
    logger.info("Optical Property Comparison: DDA vs Mie")
    logger.info("=" * 70)
    logger.info(f"{'λ (nm)':>8} {'ΔQ_ext (%)':>12} {'ΔQ_sca (%)':>12} "
                f"{'ΔSSA (%)':>10} {'Δg (%)':>10}")
    logger.info("-" * 70)

    for dda, mie in zip(dda_results, mie_results):
        wl = dda.cross_sections.wavelength
        dq_ext = (dda.cross_sections.Q_ext - mie.cross_sections.Q_ext) / mie.cross_sections.Q_ext * 100
        dq_sca = (dda.cross_sections.Q_sca - mie.cross_sections.Q_sca) / mie.cross_sections.Q_sca * 100
        dssa = (dda.cross_sections.SSA - mie.cross_sections.SSA) / mie.cross_sections.SSA * 100
        dg = (dda.cross_sections.g - mie.cross_sections.g) / abs(mie.cross_sections.g) * 100 if mie.cross_sections.g != 0 else 0

        logger.info(f"{wl:>8.0f} {dq_ext:>12.2f} {dq_sca:>12.2f} "
                    f"{dssa:>10.2f} {dg:>10.2f}")

    logger.info("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solver",
        choices=["DDA", "MIE", "both"],
        default="both",
        help="Which solver to run (default: both)",
    )
    args = parser.parse_args()

    # Create particle (shared between solvers)
    logger.info("Creating black carbon sphere particle...")
    particle = create_particle()
    logger.info(f"Particle: radius={PARTICLE_CONFIG['radius_nm']} nm, "
                f"material={PARTICLE_CONFIG['material']}")

    dda_results = None
    mie_results = None

    # Run DDA
    if args.solver in ("DDA", "both"):
        dda_results = compute_optics(particle, "DDA")
        dda_ds = results_to_dataset(dda_results, "DDA")
        dda_ds.to_netcdf(OPTICS_DDA_NC)
        logger.info(f"Saved DDA optics to {OPTICS_DDA_NC}")

    # Run Mie
    if args.solver in ("MIE", "both"):
        mie_results = compute_optics(particle, "MIE")
        mie_ds = results_to_dataset(mie_results, "MIE")
        mie_ds.to_netcdf(OPTICS_MIE_NC)
        logger.info(f"Saved Mie optics to {OPTICS_MIE_NC}")

    # Comparison plots and summary
    if dda_results and mie_results:
        print_optics_summary(dda_results, mie_results)
        plot_p11_comparison(dda_results, mie_results, OPTICS_DDA_NC.parent)
        logger.info("P11 comparison plots saved")

    logger.info("Stage 1 complete.")


if __name__ == "__main__":
    main()
