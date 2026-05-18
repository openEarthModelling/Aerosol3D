"""PyVista and matplotlib visualization for DDA optical results."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .datastructs import OpticalResult
    from .optics_export import AerosolOpticsData

logger = logging.getLogger(__name__)


def plot_phase_function_2d(
    result: OpticalResult,
    plane: str = "xz",
    log_scale: bool = True,
    save_path: str | None = None,
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
        raise ValueError(
            "OpticalResult has no phase_function. Re-run solve_optics with compute_phase_func=True."
        )

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
    """Plot orthogonal slices of ``|E|^2`` through the voxel grid.

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
        raise ValueError(
            "OpticalResult has no voxel_grid. Re-run solve_optics with compute_near_field=True."
        )
    if "E_intensity" not in result.voxel_grid.cell_data:
        raise ValueError(
            "voxel_grid has no E_intensity. Re-run solve_optics with compute_near_field=True."
        )

    grid = result.voxel_grid
    plotter = pv.Plotter(title=f"|E|^2 @ {result.cross_sections.wavelength} nm")

    slices = grid.slice_orthogonal(scalars="E_intensity")
    if log_scale:
        slices["E_intensity_log"] = np.log10(slices["E_intensity"] + 1e-30)
        plotter.add_mesh(slices, scalars="E_intensity_log", cmap="hot", show_scalar_bar=True)
    else:
        plotter.add_mesh(slices, scalars="E_intensity", cmap="hot", show_scalar_bar=True)

    if show:
        plotter.show()
    return plotter


def print_macroscopic(result, solve_time: float = None):
    """Print core aerosol macroscopic optical properties to stdout.

    Args:
        result: OpticalResult instance.
        solve_time: Optional solve time in seconds. If None, uses
            result.solve_time when available.
    """
    cs = result.cross_sections
    if solve_time is None:
        solve_time = getattr(result, "solve_time", None)
    print(f"{'=' * 50}")
    print(f"  Aerosol Optical Properties @ {cs.wavelength:.1f} nm")
    print(f"{'=' * 50}")
    print(f"  C_ext  = {cs.C_ext:12.4f}  nm\u00b2")
    print(f"  C_sca  = {cs.C_sca:12.4f}  nm\u00b2")
    print(f"  C_abs  = {cs.C_abs:12.4f}  nm\u00b2")
    print(f"  SSA    = {cs.SSA:12.6f}")
    print(f"  g      = {cs.g:12.6f}")
    print(f"  Q_ext  = {cs.Q_ext:12.6f}")
    print(f"  Q_sca  = {cs.Q_sca:12.6f}")
    print(f"  Q_abs  = {cs.Q_abs:12.6f}")
    print(f"  r_eff  = {cs.r_eff:12.4f}  nm")
    print(f"  N_dipo = {result.n_dipoles}")
    if solve_time is not None:
        print(f"  Time   = {solve_time:12.2f}  s")
    if result.validity:
        v = result.validity
        status = "OK" if v["valid"] else "WARNING: > 1"
        print(f"  |m|kd  = {v['m_k_d']:.4f}  {status}")
    print(f"{'=' * 50}")


def _ensure_matplotlib_agg():
    import matplotlib

    if "matplotlib.backends" not in sys.modules:
        matplotlib.use("Agg")


def plot_spectral_properties(data: AerosolOpticsData, output_dir: str):
    """Plot C_ext, C_sca, C_abs, SSA, g vs wavelength."""
    import matplotlib.pyplot as plt

    _ensure_matplotlib_agg()
    wl = data.wavelength_nm

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    props = [
        ("C_ext (nm²)", data.C_ext),
        ("C_sca (nm²)", data.C_sca),
        ("C_abs (nm²)", data.C_abs),
        ("SSA", data.SSA),
        ("g", data.g),
    ]

    for ax, (label, values) in zip(axes.flat, props):
        ax.plot(wl, values, "b-o", markersize=4)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    axes.flat[5].set_visible(False)
    fig.suptitle(f"Optical Properties ({data.solver})")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "spectral_properties.png", dpi=150)
    plt.close(fig)


def plot_phase_function(
    data: AerosolOpticsData,
    wl_idx: int,
    output_dir: str,
    polar: bool = True,
    log_scale: bool = True,
):
    """Plot single-wavelength P11 in polar and linear views."""
    import matplotlib.pyplot as plt

    _ensure_matplotlib_agg()
    wl = data.wavelength_nm[wl_idx]
    theta = data.theta_rad
    P11_azimuthal = np.mean(data.P11[wl_idx], axis=1)

    n_panels = 2 if polar else 1
    fig = plt.figure(figsize=(5 * n_panels, 5))

    ax = fig.add_subplot(1, n_panels, 1)
    if log_scale:
        ax.semilogy(np.degrees(theta), np.maximum(P11_azimuthal, 1e-30))
    else:
        ax.plot(np.degrees(theta), P11_azimuthal)
    ax.set_xlabel("Scattering angle (°)")
    ax.set_ylabel("P11" + (" (log)" if log_scale else ""))
    ax.set_title(f"P11 @ {wl:.0f} nm")
    ax.grid(True, alpha=0.3)

    if polar:
        ax = fig.add_subplot(1, n_panels, 2, projection="polar")
        if log_scale:
            ax.semilogy(theta, np.maximum(P11_azimuthal, 1e-30))
        else:
            ax.plot(theta, P11_azimuthal)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(-1)
        ax.set_title(f"P11 @ {wl:.0f} nm (polar)", va="bottom")

    fig.tight_layout()
    fig.savefig(Path(output_dir) / f"phase_function_{wl:.0f}nm.png", dpi=150)
    plt.close(fig)


def plot_optical_comparison(
    datasets: list,
    labels: list[str],
    output_dir: str,
):
    """Compare all scalar optical properties across datasets."""
    import matplotlib.pyplot as plt

    _ensure_matplotlib_agg()
    colors = ["b", "r", "g", "m", "c"]
    props = [
        ("C_ext (nm²)", "C_ext"),
        ("C_sca (nm²)", "C_sca"),
        ("C_abs (nm²)", "C_abs"),
        ("SSA", "SSA"),
        ("g", "g"),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True)

    for col, (label, attr) in enumerate(props):
        ax_top = axes[0, col]
        for i, (ds, name) in enumerate(zip(datasets, labels)):
            ax_top.plot(
                ds.wavelength_nm,
                getattr(ds, attr),
                f"{colors[i]}-o",
                label=name,
                markersize=4,
            )
        ax_top.set_ylabel(label)
        ax_top.set_title(label)
        ax_top.legend(fontsize=8)
        ax_top.grid(True, alpha=0.3)

        ax_bot = axes[1, col]
        ref = getattr(datasets[0], attr)
        for i in range(1, len(datasets)):
            rel = (getattr(datasets[i], attr) - ref) / (np.abs(ref) + 1e-12) * 100
            ax_bot.plot(
                datasets[i].wavelength_nm,
                rel,
                f"{colors[i]}-o",
                label=f"{labels[i]} vs {labels[0]}",
                markersize=4,
            )
        ax_bot.set_ylabel(f"Δ{label} (%)")
        ax_bot.set_xlabel("Wavelength (nm)")
        ax_bot.axhline(0, color="k", linewidth=0.5)
        ax_bot.legend(fontsize=8)
        ax_bot.grid(True, alpha=0.3)

    fig.suptitle("Optical Property Comparison")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "optical_comparison.png", dpi=150)
    plt.close(fig)


def plot_phase_function_comparison(
    datasets: list,
    labels: list[str],
    output_dir: str,
):
    """Compare P11 across datasets for each wavelength."""
    import matplotlib.pyplot as plt

    _ensure_matplotlib_agg()
    colors = ["b", "r", "g", "m", "c"]
    n_wl = len(datasets[0].wavelength_nm)

    for i_wl in range(n_wl):
        wl = datasets[0].wavelength_nm[i_wl]
        fig = plt.figure(figsize=(15, 4))

        ax = fig.add_subplot(131)
        for j, (ds, name) in enumerate(zip(datasets, labels)):
            P11_az = np.mean(ds.P11[i_wl], axis=1)
            ax.semilogy(
                np.degrees(ds.theta_rad),
                np.maximum(P11_az, 1e-30),
                f"{colors[j]}-",
                label=name,
                linewidth=1.5,
            )
        ax.set_xlabel("Scattering angle (°)")
        ax.set_ylabel("P11 (log)")
        ax.set_title(f"P11 @ {wl:.0f} nm")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(132, projection="polar")
        for j, (ds, name) in enumerate(zip(datasets, labels)):
            P11_az = np.mean(ds.P11[i_wl], axis=1)
            ax.semilogy(
                ds.theta_rad,
                np.maximum(P11_az, 1e-30),
                f"{colors[j]}-",
                label=name,
                linewidth=1.5,
            )
        ax.set_title("Polar view", va="bottom")

        ax = fig.add_subplot(133)
        ref_az = np.mean(datasets[0].P11[i_wl], axis=1)
        for j in range(1, len(datasets)):
            P11_az = np.mean(datasets[j].P11[i_wl], axis=1)
            if datasets[j].theta_rad.shape != datasets[0].theta_rad.shape or not np.allclose(
                datasets[j].theta_rad, datasets[0].theta_rad
            ):
                P11_az = np.interp(datasets[0].theta_rad, datasets[j].theta_rad, P11_az)
            rel = (P11_az - ref_az) / (np.abs(ref_az) + 1e-12) * 100
            ax.plot(
                np.degrees(datasets[0].theta_rad),
                rel,
                f"{colors[j]}-",
                label=f"{labels[j]} vs {labels[0]}",
                linewidth=1,
            )
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xlabel("Scattering angle (°)")
        ax.set_ylabel("Relative diff (%)")
        ax.set_title("(Δ) / ref × 100%")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(Path(output_dir) / f"p11_comparison_{wl:.0f}nm.png", dpi=150)
        plt.close(fig)


def plot_legendre_convergence(
    data,
    wl_idx: int,
    output_dir: str,
):
    """Plot original P11 vs Legendre reconstruction with increasing terms."""
    import matplotlib.pyplot as plt
    from numpy.polynomial.legendre import legval

    _ensure_matplotlib_agg()
    wl = data.wavelength_nm[wl_idx]
    theta = data.theta_rad
    P11_original = np.mean(data.P11[wl_idx], axis=1)
    moments = data.legendre_moments[wl_idx]
    mu = np.cos(theta)

    n_terms_list = [2, 4, 8, 16, min(32, len(moments))]
    n_terms_list = sorted(set(t for t in n_terms_list if t <= len(moments)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.semilogy(
        np.degrees(theta),
        np.maximum(P11_original, 1e-30),
        "k-",
        linewidth=2,
        label="Original P11",
    )
    for n in n_terms_list:
        P11_recon = legval(mu, moments[:n])
        ax.semilogy(np.degrees(theta), np.maximum(P11_recon, 1e-30), "--", label=f"{n} terms")
    ax.set_xlabel("Scattering angle (°)")
    ax.set_ylabel("P11 (log)")
    ax.set_title(f"Legendre Convergence @ {wl:.0f} nm")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    max_orders = list(range(2, len(moments) + 1))
    rms_errors = []
    for n in max_orders:
        P11_recon = legval(mu, moments[:n])
        rms = np.sqrt(np.mean((P11_recon - P11_original) ** 2))
        rms_errors.append(rms)
    ax.semilogy(max_orders, rms_errors, "b-o", markersize=3)
    ax.set_xlabel("Number of Legendre terms")
    ax.set_ylabel("RMS reconstruction error")
    ax.set_title("Convergence rate")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(Path(output_dir) / "legendre_convergence.png", dpi=150)
    plt.close(fig)


def plot_legendre_moments_spectrum(data, output_dir: str):
    """Heatmap of Legendre moments k_l vs wavelength."""
    import matplotlib.pyplot as plt

    _ensure_matplotlib_agg()
    moments = data.legendre_moments

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.pcolormesh(
        data.wavelength_nm,
        np.arange(data.n_legendre),
        moments.T,
        shading="auto",
        cmap="RdBu_r",
    )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Legendre order l")
    ax.set_title(f"Legendre Moments k_l ({data.solver})")
    fig.colorbar(im, ax=ax, label="k_l")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "legendre_moments_spectrum.png", dpi=150)
    plt.close(fig)


def generate_comparison_summary(
    datasets: list,
    labels: list[str],
) -> str:
    """Generate text summary of optical property differences."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"Optical Property Comparison: {' vs '.join(labels)}")
    lines.append("=" * 70)
    lines.append("")

    ref = datasets[0]
    idx_550 = np.argmin(np.abs(ref.wavelength_nm - 550.0))
    wl_550 = ref.wavelength_nm[idx_550]
    lines.append(f"Reference wavelength: {wl_550:.0f} nm")
    lines.append("")

    props = [
        ("C_ext", "C_ext", "nm²"),
        ("C_sca", "C_sca", "nm²"),
        ("C_abs", "C_abs", "nm²"),
        ("SSA", "SSA", ""),
        ("g", "g", ""),
    ]

    for label, attr, unit in props:
        ref_val = getattr(ref, attr)[idx_550]
        lines.append(f"  {label} ({unit}): ref = {ref_val:.6f}")
        for i in range(1, len(datasets)):
            val = getattr(datasets[i], attr)[idx_550]
            delta_pct = (val - ref_val) / (np.abs(ref_val) + 1e-12) * 100
            lines.append(f"    {labels[i]}: {val:.6f} (Δ = {delta_pct:+.2f}%)")

    lines.append("")

    if ref.legendre_moments is not None:
        lines.append("  Legendre moments (k_1 at 550 nm):")
        ref_k1 = ref.legendre_moments[idx_550, 1]
        lines.append(f"    {labels[0]}: {ref_k1:.6f}")
        for i in range(1, len(datasets)):
            k1 = datasets[i].legendre_moments[idx_550, 1]
            lines.append(f"    {labels[i]}: {k1:.6f}")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)
