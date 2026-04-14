# aerosol3d/optics/bridge.py
"""PyJulia bridge to CoupledElectricMagneticDipoles.jl.

Lazy-loads Julia runtime on first use. All data crossing the Python/Julia
boundary is strictly typed as float64 or complex128 with C-contiguous layout.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

_julia_initialized = False


def is_julia_ready() -> bool:
    """Check whether the Julia runtime has been initialized."""
    return _julia_initialized


def _ensure_julia():
    """Start Julia and load CEMD package. No-op if already initialized."""
    global _julia_initialized
    if _julia_initialized:
        return
    from julia.api import Julia
    Julia(compiled_modules=False)
    from julia import Main  # noqa: F401
    Main.eval("using CoupledElectricMagneticDipoles")
    # JIT warmup with a trivial problem (avoids compilation lag on first real call)
    Main.eval("DDACore.solve_DDA_e([0.0 0.0 0.0], [1.0im])")
    _julia_initialized = True
    logger.info("Julia runtime initialized, CEMD package loaded.")


def _enforce_types(
    positions: np.ndarray, alpha_e: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Enforce strict dtype and memory layout for Julia interop."""
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    alpha_e = np.ascontiguousarray(alpha_e, dtype=np.complex128)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3), got {positions.shape}")
    if alpha_e.ndim != 1:
        raise ValueError(f"alpha_e must be (N,), got {alpha_e.shape}")
    if positions.shape[0] != alpha_e.shape[0]:
        raise ValueError(
            f"positions and alpha_e must have same N: "
            f"{positions.shape[0]} vs {alpha_e.shape[0]}"
        )
    return positions, alpha_e


def _prepare_alpha_e(alpha_e: np.ndarray):
    """Convert alpha_e for Julia Alphas.dispatch_e compatibility.

    When N=1 and alpha_e is a 1-element array, Julia's dispatch_e
    interprets it as "same scalar for all dipoles" and wraps it in
    Vector{Vector}. To avoid this, pass a Python scalar instead.
    """
    if alpha_e.ndim == 1 and alpha_e.shape[0] == 1:
        return complex(alpha_e[0])
    return alpha_e


def solve_dda(
    positions: np.ndarray,
    alpha_e: np.ndarray,
    config,
) -> dict:
    """Call Julia DDA solver for electric-only dipoles.

    Args:
        positions: (N, 3) dipole positions (in same unit as config.wavelength).
        alpha_e: (N,) dimensionless polarizabilities (complex128).
        config: SimulationConfig.

    Returns:
        dict with "phi_inc": (N, 3) complex128 incident field at each dipole.
    """
    _ensure_julia()
    from julia import CoupledElectricMagneticDipoles as CEMD

    positions, alpha_e = _enforce_types(positions, alpha_e)
    alpha_e_jl = _prepare_alpha_e(alpha_e)

    k = 2.0 * np.pi / config.wavelength
    kr = k * positions

    # Generate plane wave input field
    khat = list(config.propagation)
    e0 = list(config.polarization)
    input_field = CEMD.InputFields.plane_wave_e(kr, khat=khat, e0=e0)

    phi_inc = CEMD.DDACore.solve_DDA_e(
        kr, alpha_e_jl, input_field=input_field, solver=config.solver
    )

    return {"phi_inc": np.asarray(phi_inc)}


def compute_cross_sections(
    positions: np.ndarray,
    alpha_e: np.ndarray,
    dda_result: dict,
    config,
) -> np.ndarray:
    """Compute C_ext, C_abs, C_sca via Julia PostProcessing.

    Returns:
        (3,) float64 array: [C_ext, C_abs, C_sca].
        Values are in units of 1/k^2 (dimensionless).
    """
    _ensure_julia()
    from julia import CoupledElectricMagneticDipoles as CEMD

    positions, alpha_e = _enforce_types(positions, alpha_e)
    alpha_e_jl = _prepare_alpha_e(alpha_e)

    k = 2.0 * np.pi / config.wavelength
    kr = k * positions

    khat = list(config.propagation)
    e0 = list(config.polarization)
    input_field = CEMD.InputFields.plane_wave_e(kr, khat=khat, e0=e0)

    cs = CEMD.PostProcessing.compute_cross_sections_e(
        k, kr, dda_result["phi_inc"], alpha_e_jl, input_field
    )
    return np.atleast_1d(np.asarray(cs, dtype=np.float64)).ravel()


def compute_diff_scattering(
    positions: np.ndarray,
    alpha_e: np.ndarray,
    dda_result: dict,
    config,
    directions: np.ndarray,
) -> np.ndarray:
    """Compute differential scattering cross section dC_sca/dOmega.

    Args:
        directions: (Nu, 3) unit direction vectors (auto-normalized by Julia).

    Returns:
        (Nu,) float64 array of dC_sca/dOmega in units of 1/k^2.
    """
    _ensure_julia()
    from julia import CoupledElectricMagneticDipoles as CEMD

    positions, alpha_e = _enforce_types(positions, alpha_e)
    alpha_e_jl = _prepare_alpha_e(alpha_e)

    k = 2.0 * np.pi / config.wavelength
    kr = k * positions

    khat = list(config.propagation)
    e0 = list(config.polarization)
    input_field = CEMD.InputFields.plane_wave_e(kr, khat=khat, e0=e0)

    # Ensure directions is 2D (Nu, 3) -- PyJulia may flatten 1-row arrays
    directions = np.atleast_2d(
        np.ascontiguousarray(directions, dtype=np.float64)
    )
    dcs = CEMD.PostProcessing.diff_scattering_cross_section_e(
        k, kr, dda_result["phi_inc"], alpha_e_jl, input_field, directions
    )
    return np.asarray(dcs, dtype=np.float64)
