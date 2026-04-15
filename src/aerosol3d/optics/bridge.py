# aerosol3d/optics/bridge.py
"""PyJulia bridge to CoupledElectricMagneticDipoles.jl.

Lazy-loads Julia runtime on first use. All data crossing the Python/Julia
boundary is strictly typed as float64 or complex128 with C-contiguous layout.
"""

import logging
import copy

import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

_julia_initialized = False
_gpu_checked: Optional[bool] = None


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


_gpu_checked: Optional[bool] = None


def check_gpu_available() -> bool:
    """Check whether Julia CUDA.jl is functional.

    Returns:
        True if CUDA is available on the Julia side, False otherwise.
    """
    global _gpu_checked
    if _gpu_checked is not None:
        return _gpu_checked
    try:
        _ensure_julia()
        from julia import Main
        Main.eval("CUDA.functional()")
        _gpu_checked = True
        logger.info("GPU (CUDA) is available via Julia.")
        return True
    except Exception:
        _gpu_checked = False
        logger.info("GPU (CUDA) is not available.")
        return False


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

    if config.solver == "GPU" and not check_gpu_available():
        logger.warning(
            "GPU solver requested but CUDA is not available. Falling back to CPU."
        )
        config = copy.copy(config)
        config.solver = "CPU"

    from julia import CoupledElectricMagneticDipoles as CEMD

    positions, alpha_e = _enforce_types(positions, alpha_e)
    alpha_e_jl = _prepare_alpha_e(alpha_e)

    k = 2.0 * np.pi / config.wavelength
    kr = k * positions

    # Generate plane wave input field
    khat = list(config.propagation)
    e0 = list(config.polarization) if config.polarization is not None else [1.0, 0.0, 0.0] if config.polarization is not None else [1.0, 0.0, 0.0]
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
        Values are in units of knorm^{-2} (same length unit as input positions).
    """
    _ensure_julia()
    from julia import CoupledElectricMagneticDipoles as CEMD

    positions, alpha_e = _enforce_types(positions, alpha_e)
    alpha_e_jl = _prepare_alpha_e(alpha_e)

    k = 2.0 * np.pi / config.wavelength
    kr = k * positions

    khat = list(config.propagation)
    e0 = list(config.polarization) if config.polarization is not None else [1.0, 0.0, 0.0]
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
    e0 = list(config.polarization) if config.polarization is not None else [1.0, 0.0, 0.0]
    input_field = CEMD.InputFields.plane_wave_e(kr, khat=khat, e0=e0)

    # Ensure directions is 2D (Nu, 3) -- PyJulia may flatten 1-row arrays
    directions = np.atleast_2d(
        np.ascontiguousarray(directions, dtype=np.float64)
    )
    dcs = CEMD.PostProcessing.diff_scattering_cross_section_e(
        k, kr, dda_result["phi_inc"], alpha_e_jl, input_field, directions
    )
    return np.asarray(dcs, dtype=np.float64)


def compute_asymmetry_parameter(
    positions: np.ndarray,
    alpha_e: np.ndarray,
    dda_result: dict,
    config,
    C_sca: float,
    n_points: int = 5804,
) -> float:
    """Compute asymmetry parameter g = <cos(theta)> via spherical quadrature.

    Uses uniform spherical grid for numerical integration of the
    differential scattering cross section weighted by cos(theta).

    Args:
        positions, alpha_e, dda_result, config: DDA inputs.
        C_sca: Scattering cross section (denominator for normalization).
        n_points: Approximate number of quadrature points.

    Returns:
        float: asymmetry parameter g in [-1, 1].
    """
    theta, phi, weights = _spherical_grid(n_points)

    directions = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])

    dcs = compute_diff_scattering(positions, alpha_e, dda_result, config, directions)

    cos_theta = np.cos(theta)
    if C_sca > 0:
        g = (4.0 * np.pi / C_sca) * np.sum(cos_theta * dcs * weights)
    else:
        g = 0.0

    return float(g)


def _spherical_grid(n_points: int = 5804):
    """Generate uniform spherical quadrature points and weights.

    Returns:
        theta: (N,) polar angles [0, pi]
        phi: (N,) azimuthal angles [0, 2*pi]
        weights: (N,) quadrature weights (sum to 1/(4*pi))
    """
    n_side = int(np.ceil(n_points ** (1.0 / 3.0)))
    u = np.linspace(0, 1, n_side + 1)[1:]
    theta = np.arccos(2 * u - 1)
    phi = np.linspace(0, 2 * np.pi, 2 * n_side, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    theta_flat = theta_grid.ravel()
    phi_flat = phi_grid.ravel()

    sin_theta = np.sin(theta_flat)
    weights = sin_theta / (4.0 * np.pi * sin_theta.sum())

    return theta_flat, phi_flat, weights
