"""Export DDA optical results to pyRadtran-compatible data structures.

No dependency on pyRadtran — produces plain dicts with numpy arrays.
Users construct pyRadtran.ParticleOptics on their side.
"""

import numpy as np

from .legendre import compute_legendre_moments


def optical_results_to_pyradtran_data(
    results: list,
    n_legendre: int = 32,
) -> dict:
    """Convert DDA multi-wavelength results to pyRadtran ParticleOptics-compatible dict.

    Args:
        results: List of OpticalResult, one per wavelength.
        n_legendre: Number of Legendre moments to compute.

    Returns:
        dict with keys:
            - wavelength_um: list[float]  (nm -> um)
            - radius_um: list[float]  (single-element [r_eff])
            - Qext: np.ndarray, shape (n_wl, 1)
            - Qsca: np.ndarray, shape (n_wl, 1)
            - g: np.ndarray, shape (n_wl, 1)
            - legendre_moments: np.ndarray, shape (n_wl, 1, n_legendre)

    Raises:
        ValueError: If results list is empty, or if any result lacks phase_function.
    """
    if not results:
        raise ValueError("results list cannot be empty")

    n_wl = len(results)
    wavelength_um = [r.cross_sections.wavelength / 1000.0 for r in results]
    radius_um = [results[0].cross_sections.r_eff]

    Qext = np.zeros((n_wl, 1))
    Qsca = np.zeros((n_wl, 1))
    g = np.zeros((n_wl, 1))
    legendre_moments = np.zeros((n_wl, 1, n_legendre))

    for i, result in enumerate(results):
        cs = result.cross_sections
        Qext[i, 0] = cs.Q_ext
        Qsca[i, 0] = cs.Q_sca
        g[i, 0] = cs.g

        if result.phase_function is None:
            raise ValueError(
                f"Result at index {i} has no phase_function. "
                "Re-run solve_optics with compute_phase_func=True."
            )
        moments = compute_legendre_moments(result.phase_function, n_legendre=n_legendre)
        legendre_moments[i, 0, :] = moments

    return {
        "wavelength_um": wavelength_um,
        "radius_um": radius_um,
        "Qext": Qext,
        "Qsca": Qsca,
        "g": g,
        "legendre_moments": legendre_moments,
    }
