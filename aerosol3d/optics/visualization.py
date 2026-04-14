"""PyVista and matplotlib visualization for DDA optical results."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .datastructs import OpticalResult

logger = logging.getLogger(__name__)


def plot_phase_function_2d(
    result: OpticalResult,
    plane: str = "xz",
    log_scale: bool = True,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Plot 2D polar phase function P11(theta) in a specified scattering plane.

    This is the standard format used in aerosol research publications.

    Args:
        result: OpticalResult with phase_function containing P11.
        plane: Scattering plane ("xz", "xy", "yz").
        log_scale: If True, use log scale for radial axis.
        save_path: If provided, save figure to this path.
        show: If True, call plt.show().
    """
    import matplotlib
    if "matplotlib.backends" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if result.phase_function is None:
        raise ValueError("OpticalResult has no phase_function. "
                         "Re-run solve_optics with compute_phase_func=True.")

    theta = result.phase_function.theta
    P11 = result.phase_function.P11

    # Select phi index for the specified plane
    phi_idx = {"xz": 0, "xy": 1, "yz": 2}.get(plane, 0)
    P11_slice = P11[:, phi_idx]

    # Mirror for full polar plot (theta: 0 -> pi -> 2*pi)
    theta_full = np.concatenate([theta, 2 * np.pi - theta[::-1]])
    P11_full = np.concatenate([P11_slice, P11_slice[::-1]])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    if log_scale:
        P11_plot = np.maximum(P11_full, 1e-30)
        ax.semilogy(theta_full, P11_plot)
    else:
        ax.plot(theta_full, P11_full)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_title(f"P11({plane}-plane) @ {result.cross_sections.wavelength} nm", va="bottom")
    ax.set_xlabel("Scattering angle (rad)")
    ax.set_ylabel("P11" + (" (log)" if log_scale else ""))

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Phase function saved to %s", save_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_near_field(
    result: OpticalResult,
    component: str = "E",
    log_scale: bool = True,
    show: bool = False,
):
    """Plot orthogonal slices of |E|^2 through the voxel grid.

    Useful for diagnosing localized absorption/heating in mixed aerosol.

    Args:
        result: OpticalResult with voxel_grid containing E_intensity cell_data.
        component: Field component ("E" only for DDA mode).
        log_scale: If True, use log color scale.
        show: If True, call plotter.show().

    Returns:
        pv.Plotter instance.
    """
    import pyvista as pv

    if result.voxel_grid is None:
        raise ValueError("OpticalResult has no voxel_grid. "
                         "Re-run solve_optics with compute_near_field=True.")
    if "E_intensity" not in result.voxel_grid.cell_data:
        raise ValueError("voxel_grid has no E_intensity. "
                         "Re-run solve_optics with compute_near_field=True.")

    grid = result.voxel_grid
    plotter = pv.Plotter(title=f"|E|^2 @ {result.cross_sections.wavelength} nm")

    slices = grid.slice_orthogonal(scalars="E_intensity")
    if log_scale:
        slices["E_intensity_log"] = np.log10(
            slices["E_intensity"] + 1e-30
        )
        plotter.add_mesh(slices, scalars="E_intensity_log", cmap="hot",
                         show_scalar_bar=True)
    else:
        plotter.add_mesh(slices, scalars="E_intensity", cmap="hot",
                         show_scalar_bar=True)

    if show:
        plotter.show()
    return plotter


def print_macroscopic(result: OpticalResult):
    """Print core aerosol macroscopic optical properties to stdout.

    Outputs: wavelength, C_ext, C_sca, C_abs, SSA, g, Q_ext, Q_sca, Q_abs, r_eff.
    """
    cs = result.cross_sections
    print(f"{'='*50}")
    print(f"  Aerosol Optical Properties @ {cs.wavelength} nm")
    print(f"{'='*50}")
    print(f"  C_ext  = {cs.C_ext:12.4f}  (nm^2)")
    print(f"  C_sca  = {cs.C_sca:12.4f}  (nm^2)")
    print(f"  C_abs  = {cs.C_abs:12.4f}  (nm^2)")
    print(f"  SSA    = {cs.SSA:12.6f}")
    print(f"  g      = {cs.g:12.6f}")
    print(f"  Q_ext  = {cs.Q_ext:12.6f}")
    print(f"  Q_sca  = {cs.Q_sca:12.6f}")
    print(f"  Q_abs  = {cs.Q_abs:12.6f}")
    print(f"  r_eff  = {cs.r_eff:12.4f}  (nm)")
    print(f"  N_dipo = {result.n_dipoles}")
    if result.validity:
        v = result.validity
        print(f"  |m|kd  = {v['m_k_d']:.4f}  {'OK' if v['valid'] else 'WARNING: > 1'}")
    print(f"{'='*50}")
