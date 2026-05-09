"""MIE optical solver using PyMieScatt."""

import numpy as np

from .datastructs import CrossSections, OpticalResult, PhaseFunction, SimulationConfig


def _mie_phase_function(m, d, wavelength, n_theta=181):
    """Compute phase function P11(theta) using PyMieScatt.ScatteringFunction."""
    import scipy.integrate

    if not hasattr(scipy.integrate, "trapz"):
        scipy.integrate.trapz = scipy.integrate.trapezoid
    import PyMieScatt as pms

    angular_resolution = 180.0 / (n_theta - 1) if n_theta > 1 else 1.0
    theta_rad, _, _, SU = pms.ScatteringFunction(
        m,
        wavelength,
        d,
        nMedium=1.0,
        minAngle=0,
        maxAngle=180,
        angularResolution=angular_resolution,
    )
    sin_theta = np.sin(theta_rad)
    norm = 2 * np.pi * np.trapz(SU * sin_theta, theta_rad)
    P11 = SU / norm if norm > 0 else SU
    return theta_rad, P11


def solve_mie(
    particle,
    config: SimulationConfig,
    compute_phase_func: bool = False,
    n_theta: int = 181,
    verbose: bool = True,
) -> OpticalResult:
    """Solve optics using Mie theory (PyMieScatt)."""
    import scipy.integrate

    if not hasattr(scipy.integrate, "trapz"):
        scipy.integrate.trapz = scipy.integrate.trapezoid
    import PyMieScatt as pms

    m = particle.effective_refractive_index / config.n_host
    d = particle.equivalent_diameter
    wavelength = config.wavelength

    if verbose:
        print(f"{'=' * 52}")
        print("  MIE Simulation Configuration")
        print(f"{'=' * 52}")
        print(f"  wavelength     = {wavelength:.1f} nm")
        print(f"  n_host         = {config.n_host}")
        print(f"  d_ve           = {d:.2f} nm")
        print(f"  m              = {m}")
        print(f"  x              = {np.pi * d / wavelength:.4f} (size parameter)")
        print(f"{'=' * 52}")

    Qext, Qsca, Qabs, g, _, _, _ = pms.MieQ(
        m, wavelength, d, nMedium=config.n_host
    )

    r_eff = d / 2.0
    geo_cs = np.pi * r_eff**2

    C_ext = Qext * geo_cs
    C_sca = Qsca * geo_cs
    C_abs = Qabs * geo_cs
    SSA = C_sca / C_ext if C_ext > 0 else 0.0

    cross_sections = CrossSections(
        wavelength=wavelength,
        C_ext=C_ext,
        C_sca=C_sca,
        C_abs=C_abs,
        Q_ext=Qext,
        Q_sca=Qsca,
        Q_abs=Qabs,
        SSA=SSA,
        g=g,
        r_eff=r_eff,
    )

    phase_function = None
    if compute_phase_func:
        theta_rad, P11 = _mie_phase_function(m, d, wavelength, n_theta=n_theta)
        phi = np.array([0.0])
        P11_2d = P11[:, np.newaxis]
        phase_function = PhaseFunction(theta=theta_rad, phi=phi, P11=P11_2d)

    return OpticalResult(
        config=config,
        cross_sections=cross_sections,
        phase_function=phase_function,
        voxel_grid=None,
        n_dipoles=0,
        validity=None,
        solve_time=0.0,
        solver="MIE",
    )
