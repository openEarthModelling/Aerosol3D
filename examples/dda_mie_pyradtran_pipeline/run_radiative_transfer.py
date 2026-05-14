"""Stage 2: Run radiative transfer with pyRadtran using precomputed optics.

Usage:
    python run_radiative_transfer.py
    python run_radiative_transfer.py --source MIE   # Use Mie optics only
"""

import argparse
import logging
import os

import numpy as np
import xarray as xr

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


def load_optics_dataset(path: str) -> xr.Dataset:
    """Load optical properties from NetCDF."""
    logger.info(f"Loading optics from {path}")
    ds = xr.open_dataset(path)
    return ds


def compute_mass_profile(optical_depth_550: float, altitude_grid_km: np.ndarray,
                         scale_height_km: float, Q_ext_at_550: float,
                         r_eff_um: float, density_kg_m3: float) -> np.ndarray:
    """Compute mass concentration profile from target optical depth.

    Uses exponential decay profile and integrates to match target tau.

    Args:
        optical_depth_550: Target optical depth at 550 nm.
        altitude_grid_km: Altitude layer boundaries in km.
        scale_height_km: Exponential scale height in km.
        Q_ext_at_550: Extinction efficiency at 550 nm.
        r_eff_um: Effective radius in micrometers.
        density_kg_m3: Particle density in kg/m3.

    Returns:
        Mass concentration per layer (kg/m3).
    """
    # Compute geometric cross-section
    geom_cross_section_um2 = np.pi * r_eff_um ** 2  # um2
    geom_cross_section_m2 = geom_cross_section_um2 * 1e-12  # m2

    # Extinction cross-section at 550 nm
    C_ext_550_m2 = Q_ext_at_550 * geom_cross_section_m2

    # Particle volume
    r_eff_m = r_eff_um * 1e-6
    volume_m3 = (4.0 / 3.0) * np.pi * r_eff_m ** 3

    # Mass per particle
    mass_per_particle_kg = density_kg_m3 * volume_m3

    # Number concentration needed for tau = 0.3 over scale height
    # tau = integral n(z) * C_ext dz
    # n(z) = n0 * exp(-z/H)
    # tau = n0 * C_ext * H (assuming H in meters)
    H_m = scale_height_km * 1000.0
    n0_m3 = optical_depth_550 / (C_ext_550_m2 * H_m)

    # Mass concentration at each altitude
    alt_mid_km = 0.5 * (altitude_grid_km[:-1] + altitude_grid_km[1:])
    n_z_m3 = n0_m3 * np.exp(-alt_mid_km / scale_height_km)
    mass_profile = n_z_m3 * mass_per_particle_kg  # kg/m3

    return mass_profile


def build_composite_aerosol(optics_ds: xr.Dataset) -> CompositeAerosol:
    """Build pyRadtran CompositeAerosol from optical dataset.

    Args:
        optics_ds: xarray Dataset with optical properties.

    Returns:
        pyRadtran CompositeAerosol object.
    """
    # Extract data
    wavelengths_nm = optics_ds["wavelength_nm"].values
    wavelengths_um = wavelengths_nm / 1000.0

    C_ext_nm2 = optics_ds["C_ext_nm2"].values  # (n_wl,)
    C_sca_nm2 = optics_ds["C_sca_nm2"].values
    g = optics_ds["g"].values
    r_eff_nm = float(optics_ds.attrs["r_eff_nm"])
    r_eff_um = r_eff_nm / 1000.0

    # Convert nm2 to um2 for pyRadtran
    C_ext_um2 = C_ext_nm2 * 1e-6
    C_sca_um2 = C_sca_nm2 * 1e-6

    # Build arrays with shape (n_wl, 1) for single radius
    n_wl = len(wavelengths_um)
    radius_um = [r_eff_um]

    Cext_arr = C_ext_um2.reshape((n_wl, 1))
    Csca_arr = C_sca_um2.reshape((n_wl, 1))
    g_arr = g.reshape((n_wl, 1))

    # Create ParticleOptics from cross-sections
    particle_optics = ParticleOptics.from_cross_sections(
        wavelength_um=wavelengths_um.tolist(),
        radius_um=radius_um,
        Cext_um2=Cext_arr,
        Csca_um2=Csca_arr,
        g=g_arr,
    )

    # Create size distribution (narrow lognormal)
    size_dist = SizeDistribution(
        kind=SIZE_DISTRIBUTION["kind"],
        params={
            "r_g_um": r_eff_um,
            "sigma_g": SIZE_DISTRIBUTION["sigma_g"],
        },
    )

    # Create precomputed species
    precomputed = PrecomputedSpecies(
        particle_optics=particle_optics,
        size_distribution=size_dist,
        particle_density_kg_m3=AEROSOL_PROFILE["particle_density_kg_m3"],
        integration_config=IntegrationConfig(),
    )

    # Compute mass profile
    # Find Q_ext closest to 550 nm
    idx_550 = np.argmin(np.abs(wavelengths_nm - 550.0))
    Q_ext_550 = float(optics_ds["Q_ext"].values[idx_550])

    altitude_grid_km = np.array(AEROSOL_PROFILE["altitude_grid_km"])
    mass_profile = compute_mass_profile(
        optical_depth_550=AEROSOL_PROFILE["total_optical_depth_550"],
        altitude_grid_km=altitude_grid_km,
        scale_height_km=AEROSOL_PROFILE["scale_height_km"],
        Q_ext_at_550=Q_ext_550,
        r_eff_um=r_eff_um,
        density_kg_m3=AEROSOL_PROFILE["particle_density_kg_m3"],
    )

    # Layer altitudes (mass_profile has n_layers = n_boundaries - 1)
    # Need altitude_km to have len(mass_profile) + 1 elements
    # Use the bottom n_layers+1 boundaries
    n_layers = len(mass_profile)
    altitude_km = altitude_grid_km[:n_layers + 1]

    loaded = LoadedSpecies(
        species=precomputed,
        mass_profile_kg_m3=mass_profile.tolist(),
        altitude_km=altitude_km.tolist(),
    )

    # Build CompositeAerosol
    aerosol = CompositeAerosol(
        sources=[loaded],
        wavelength_grid_um=wavelengths_um.tolist(),
        altitude_grid_km=altitude_grid_km.tolist(),
        n_legendre=32,
    )

    return aerosol


def build_scene(aerosol: CompositeAerosol) -> Scene:
    """Build pyRadtran Scene with aerosol.

    Args:
        aerosol: CompositeAerosol object.

    Returns:
        Configured Scene.
    """
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
    """Run radiative transfer for given optics and save result.

    Args:
        optics_path: Path to optics NetCDF file.
        output_path: Path to save RT result NetCDF.
    """
    logger.info(f"\nRunning radiative transfer for {optics_path}")

    # Load optics
    optics_ds = load_optics_dataset(optics_path)

    # Build aerosol and scene
    aerosol = build_composite_aerosol(optics_ds)
    scene = build_scene(aerosol)

    # Check for libRadtran data path
    data_path = os.environ.get("PYRADTRAN_DATA_PATH", "/usr/local/share/libRadtran/data")
    if not os.path.exists(data_path):
        logger.warning(f"libRadtran data path not found: {data_path}")
        logger.warning("Set PYRADTRAN_DATA_PATH environment variable")

    # Run
    logger.info("Executing pyRadtran...")
    result = Runner.execute(scene, data_path=data_path)

    # Save
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
