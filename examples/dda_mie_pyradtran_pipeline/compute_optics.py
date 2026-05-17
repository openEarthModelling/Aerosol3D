"""Stage 1: Compute optical properties using DDA and Mie, save to NetCDF.

Usage:
    python compute_optics.py
    python compute_optics.py --solver MIE   # Mie only (fast, for testing)
    python compute_optics.py --solver DDA   # DDA only
"""

import argparse
import logging

from Aerosol3D import (
    AerosolParticle,
    create_sphere,
    MixingState,
    preset_material,
    SimulationConfig,
    solve_optics,
)
from Aerosol3D.optics.optics_export import from_optical_results
from Aerosol3D.optics.visualization import (
    plot_optical_comparison,
    plot_phase_function_comparison,
    generate_comparison_summary,
)

from config import (
    PARTICLE_CONFIG,
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
    """Run optical computation for all wavelengths."""
    wavelengths = PARTICLE_CONFIG["wavelengths_nm"]

    logger.info(f"Running {solver} solve for {len(wavelengths)} wavelengths: "
                f"{wavelengths[0]:.0f}-{wavelengths[-1]:.0f} nm")

    results = []
    for wl in wavelengths:
        config = SimulationConfig(
            wavelength=wl,
            source="solar",
            precision="medium",
        )
        result = solve_optics(
            particle,
            config,
            solver=solver,
            voxel_size=None,
            compute_phase_func=True,
        )
        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)

    logger.info(f"{solver} solve complete: {len(results)} results")
    for r in results:
        cs = r.cross_sections
        logger.info(f"  λ={cs.wavelength:.0f}nm: Qext={cs.Q_ext:.4f}, "
                    f"Qsca={cs.Q_sca:.4f}, SSA={cs.SSA:.4f}, g={cs.g:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solver",
        choices=["DDA", "MIE", "both"],
        default="both",
        help="Which solver to run (default: both)",
    )
    args = parser.parse_args()

    logger.info("Creating black carbon sphere particle...")
    particle = create_particle()
    logger.info(f"Particle: radius={PARTICLE_CONFIG['radius_nm']} nm, "
                f"material={PARTICLE_CONFIG['material']}")

    dda_data = None
    mie_data = None

    if args.solver in ("DDA", "both"):
        dda_results = compute_optics(particle, "DDA")
        dda_data = from_optical_results(
            dda_results, n_legendre=32, material_name=PARTICLE_CONFIG["material"]
        )
        dda_data.to_netcdf(str(OPTICS_DDA_NC))
        logger.info(f"Saved DDA optics to {OPTICS_DDA_NC}")

    if args.solver in ("MIE", "both"):
        mie_results = compute_optics(particle, "MIE")
        mie_data = from_optical_results(
            mie_results, n_legendre=32, material_name=PARTICLE_CONFIG["material"]
        )
        mie_data.to_netcdf(str(OPTICS_MIE_NC))
        logger.info(f"Saved Mie optics to {OPTICS_MIE_NC}")

    # Comparison
    if dda_data and mie_data:
        output_dir = OPTICS_DDA_NC.parent

        summary = generate_comparison_summary([dda_data, mie_data], ["DDA", "Mie"])
        logger.info("\n" + summary)

        plot_optical_comparison([dda_data, mie_data], ["DDA", "Mie"], str(output_dir))
        logger.info("Saved optical comparison plot")

        plot_phase_function_comparison([dda_data, mie_data], ["DDA", "Mie"], str(output_dir))
        logger.info("Saved P11 comparison plots")

    logger.info("Stage 1 complete.")


if __name__ == "__main__":
    main()
