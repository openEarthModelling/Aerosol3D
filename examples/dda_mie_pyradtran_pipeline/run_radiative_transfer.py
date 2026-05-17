"""Stage 2: Run radiative transfer with pyRadtran using precomputed optics.

Usage:
    python run_radiative_transfer.py
    python run_radiative_transfer.py --source MIE   # Use Mie optics only
"""

import argparse
import logging
import os

import numpy as np

from Aerosol3D.optics.optics_export import AerosolOpticsData

from pyradtran import Scene, Runner, IntegrationConfig
from pyradtran.models.aerosol_composite import (
    CompositeAerosol,
    LoadedSpecies,
    ParticleOptics,
    PrecomputedSpecies,
    SizeDistribution,
)

from config import (
    OPTICS_DDA_NC,
    OPTICS_MIE_NC,
    RT_DDA_NC,
    RT_MIE_NC,
    SCENE_CONFIG,
    AEROSOL_PROFILE,
    SIZE_DISTRIBUTION,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_mass_profile(optical_depth_550: float, altitude_grid_km: np.ndarray,
                         scale_height_km: float, Q_ext_at_550: float,
                         r_eff_um: float, density_kg_m3: float) -> np.ndarray:
    """Compute mass concentration profile from target optical depth."""
    geom_cross_section_um2 = np.pi * r_eff_um ** 2
    geom_cross_section_m2 = geom_cross_section_um2 * 1e-12

    C_ext_550_m2 = Q_ext_at_550 * geom_cross_section_m2

    r_eff_m = r_eff_um * 1e-6
    volume_m3 = (4.0 / 3.0) * np.pi * r_eff_m ** 3
    mass_per_particle_kg = density_kg_m3 * volume_m3

    H_m = scale_height_km * 1000.0
    n0_m3 = optical_depth_550 / (C_ext_550_m2 * H_m)

    alt_mid_km = 0.5 * (altitude_grid_km[:-1] + altitude_grid_km[1:])
    n_z_m3 = n0_m3 * np.exp(-alt_mid_km / scale_height_km)
    mass_profile = n_z_m3 * mass_per_particle_kg

    return mass_profile


def build_composite_aerosol(optics: AerosolOpticsData) -> CompositeAerosol:
    """Build pyRadtran CompositeAerosol from AerosolOpticsData."""
    wavelengths_um = optics.wavelength_nm / 1000.0
    r_eff_um = optics.r_eff_nm / 1000.0

    C_ext_um2 = optics.C_ext * 1e-6  # nm² -> um²
    C_sca_um2 = optics.C_sca * 1e-6
    g_arr = optics.g.reshape((-1, 1))

    pf_kwargs = dict(
        wavelength_um=wavelengths_um.tolist(),
        radius_um=[r_eff_um],
        Cext_um2=C_ext_um2.reshape((-1, 1)),
        Csca_um2=C_sca_um2.reshape((-1, 1)),
        g=g_arr,
    )
    if optics.legendre_moments is not None:
        pf_kwargs["legendre_moments"] = optics.legendre_moments.reshape(
            (-1, 1, optics.n_legendre)
        )
        logger.info(f"Passing Legendre moments ({optics.n_legendre} terms) to pyRadtran")
    else:
        logger.warning("No Legendre moments available — pyRadtran will use HG fallback")

    particle_optics = ParticleOptics.from_cross_sections(**pf_kwargs)

    size_dist = SizeDistribution(
        kind=SIZE_DISTRIBUTION["kind"],
        params={
            "r_g_um": r_eff_um,
            "sigma_g": SIZE_DISTRIBUTION["sigma_g"],
        },
    )

    precomputed = PrecomputedSpecies(
        particle_optics=particle_optics,
        size_distribution=size_dist,
        particle_density_kg_m3=AEROSOL_PROFILE["particle_density_kg_m3"],
        integration_config=IntegrationConfig(),
    )

    idx_550 = np.argmin(np.abs(optics.wavelength_nm - 550.0))
    Q_ext_550 = optics.C_ext[idx_550] / (np.pi * optics.r_eff_nm ** 2)

    altitude_grid_km = np.array(AEROSOL_PROFILE["altitude_grid_km"])
    mass_profile = compute_mass_profile(
        optical_depth_550=AEROSOL_PROFILE["total_optical_depth_550"],
        altitude_grid_km=altitude_grid_km,
        scale_height_km=AEROSOL_PROFILE["scale_height_km"],
        Q_ext_at_550=Q_ext_550,
        r_eff_um=r_eff_um,
        density_kg_m3=AEROSOL_PROFILE["particle_density_kg_m3"],
    )

    n_layers = len(mass_profile)
    altitude_km = altitude_grid_km[:n_layers + 1]
    altitude_km_desc = altitude_km[::-1]
    mass_profile_desc = mass_profile[::-1]

    loaded = LoadedSpecies(
        species=precomputed,
        mass_profile_kg_m3=mass_profile_desc.tolist(),
        altitude_km=altitude_km_desc.tolist(),
    )

    aerosol = CompositeAerosol(
        sources=[loaded],
        wavelength_grid_um=wavelengths_um.tolist(),
        altitude_grid_km=altitude_grid_km[::-1].tolist(),
        n_legendre=optics.n_legendre,
    )

    return aerosol


def build_scene(aerosol: CompositeAerosol) -> Scene:
    """Build pyRadtran Scene with aerosol."""
    cfg = SCENE_CONFIG

    scene = (
        Scene()
        .set_atmosphere(
            profile=cfg["atmosphere"]["profile"],
            altitude=cfg["atmosphere"]["altitude"],
        )
        .set_source_solar(sza=cfg["source"]["sza"])
        .set_wavelength(
            cfg["wavelength"]["min_nm"],
            cfg["wavelength"]["max_nm"],
        )
        .set_solver(
            method=cfg["solver"]["method"],
            streams=cfg["solver"]["streams"],
            disort_intcor=cfg["solver"].get("disort_intcor"),
        )
        .set_surface(albedo=cfg["surface"]["albedo"])
        .set_output(
            quantities=cfg["output"]["quantities"],
            quantity=cfg["output"]["quantity"],
            format=cfg["output"]["format"],
        )
        .set_aerosol(aerosol)
    )

    return scene


def run_radiative_transfer(optics_path: str, output_path: str):
    """Run radiative transfer for given optics and save result."""
    logger.info(f"\nRunning radiative transfer for {optics_path}")

    optics = AerosolOpticsData.from_netcdf(optics_path)
    logger.info(f"Loaded {len(optics.wavelength_nm)} wavelengths, "
                f"legendre_moments={'yes' if optics.legendre_moments is not None else 'no'}")

    aerosol = build_composite_aerosol(optics)
    scene = build_scene(aerosol)

    data_path = os.environ.get("PYRADTRAN_DATA_PATH", "/usr/local/share/libRadtran/data")
    if not os.path.exists(data_path):
        logger.warning(f"libRadtran data path not found: {data_path}")
        logger.warning("Set PYRADTRAN_DATA_PATH environment variable")

    logger.info("Executing pyRadtran...")
    result = Runner.execute(scene, data_path=data_path)

    result.to_netcdf(output_path)
    logger.info(f"Saved RT result to {output_path}")
    logger.info(f"Result dimensions: {dict(result.dims)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["DDA", "MIE", "both"],
        default="both",
        help="Which optics source to use (default: both)",
    )
    args = parser.parse_args()

    if args.source in ("DDA", "both"):
        if OPTICS_DDA_NC.exists():
            run_radiative_transfer(str(OPTICS_DDA_NC), str(RT_DDA_NC))
        else:
            logger.error(f"DDA optics not found: {OPTICS_DDA_NC}")
            logger.error("Run compute_optics.py --solver DDA first")

    if args.source in ("MIE", "both"):
        if OPTICS_MIE_NC.exists():
            run_radiative_transfer(str(OPTICS_MIE_NC), str(RT_MIE_NC))
        else:
            logger.error(f"Mie optics not found: {OPTICS_MIE_NC}")
            logger.error("Run compute_optics.py --solver MIE first")

    logger.info("\nStage 2 complete.")


if __name__ == "__main__":
    main()
