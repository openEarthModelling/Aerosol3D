"""Stage 3: Compare DDA and Mie radiative transfer results.

Usage:
    python compare_results.py
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from config import (
    OPTICS_DDA_NC,
    OPTICS_MIE_NC,
    RT_DDA_NC,
    RT_MIE_NC,
    SUMMARY_TXT,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Scientific plot style
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def load_rt_results() -> tuple[xr.Dataset, xr.Dataset]:
    """Load radiative transfer results.

    Returns:
        (dda_ds, mie_ds) tuple of xarray Datasets.
    """
    logger.info("Loading RT results...")
    dda = xr.open_dataset(RT_DDA_NC)
    mie = xr.open_dataset(RT_MIE_NC)
    return dda, mie


def load_optics_results() -> tuple[xr.Dataset, xr.Dataset]:
    """Load optical property results.

    Returns:
        (dda_ds, mie_ds) tuple of xarray Datasets.
    """
    dda = xr.open_dataset(OPTICS_DDA_NC)
    mie = xr.open_dataset(OPTICS_MIE_NC)
    return dda, mie


def compute_relative_difference(dda: np.ndarray, mie: np.ndarray) -> np.ndarray:
    """Compute (DDA - Mie) / Mie * 100 with epsilon protection."""
    epsilon = 1e-12
    return (dda - mie) / (np.abs(mie) + epsilon) * 100.0


def plot_spectral_comparison(dda: xr.Dataset, mie: xr.Dataset, output_dir: Path):
    """Plot spectral comparison of irradiance and transmittance.

    Args:
        dda: DDA RT result Dataset.
        mie: Mie RT result Dataset.
        output_dir: Directory to save plots.
    """
    wl = dda["wavelength"].values

    # Extract variables
    dda_edir = dda["edir"].values if "edir" in dda else np.zeros_like(wl)
    mie_edir = mie["edir"].values if "edir" in mie else np.zeros_like(wl)
    dda_edn = dda["edn"].values if "edn" in dda else np.zeros_like(wl)
    mie_edn = mie["edn"].values if "edn" in mie else np.zeros_like(wl)
    dda_eup = dda["eup"].values if "eup" in dda else np.zeros_like(wl)
    mie_eup = mie["eup"].values if "eup" in mie else np.zeros_like(wl)

    dda_total = dda_edir + dda_edn
    mie_total = mie_edir + mie_edn

    # Figure 1: Direct irradiance
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax = axes[0]
    ax.plot(wl, dda_edir, "b-", label="DDA", linewidth=1.5)
    ax.plot(wl, mie_edir, "r--", label="Mie", linewidth=1.5)
    ax.set_ylabel("Direct irradiance")
    ax.set_title("Direct Irradiance (edir)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    rel = compute_relative_difference(dda_edir, mie_edir)
    ax.plot(wl, rel, "g-", linewidth=1.0)
    ax.axhline(0, color="k", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Relative diff. (%)")
    ax.set_title("Relative Difference: (DDA - Mie) / Mie x 100%")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "compare_direct_irradiance.png")
    plt.close(fig)
    logger.info("Saved: compare_direct_irradiance.png")

    # Figure 2: Diffuse downward
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax = axes[0]
    ax.plot(wl, dda_edn, "b-", label="DDA", linewidth=1.5)
    ax.plot(wl, mie_edn, "r--", label="Mie", linewidth=1.5)
    ax.set_ylabel("Diffuse downward")
    ax.set_title("Diffuse Downward Irradiance (edn)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    rel = compute_relative_difference(dda_edn, mie_edn)
    ax.plot(wl, rel, "g-", linewidth=1.0)
    ax.axhline(0, color="k", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Relative diff. (%)")
    ax.set_title("Relative Difference")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "compare_diffuse_downward.png")
    plt.close(fig)
    logger.info("Saved: compare_diffuse_downward.png")

    # Figure 3: Total downward
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax = axes[0]
    ax.plot(wl, dda_total, "b-", label="DDA", linewidth=1.5)
    ax.plot(wl, mie_total, "r--", label="Mie", linewidth=1.5)
    ax.set_ylabel("Total downward")
    ax.set_title("Total Downward Irradiance (edir + edn)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    rel = compute_relative_difference(dda_total, mie_total)
    ax.plot(wl, rel, "g-", linewidth=1.0)
    ax.axhline(0, color="k", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Relative diff. (%)")
    ax.set_title("Relative Difference")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "compare_total_downward.png")
    plt.close(fig)
    logger.info("Saved: compare_total_downward.png")

    # Figure 4: All relative differences combined
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wl, compute_relative_difference(dda_edir, mie_edir), "b-", label="Direct", linewidth=1.2)
    ax.plot(wl, compute_relative_difference(dda_edn, mie_edn), "r-", label="Diffuse down", linewidth=1.2)
    ax.plot(wl, compute_relative_difference(dda_total, mie_total), "g-", label="Total down", linewidth=1.2)
    ax.axhline(0, color="k", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Relative difference (%)")
    ax.set_title("DDA vs Mie: All Components")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "compare_relative_difference.png")
    plt.close(fig)
    logger.info("Saved: compare_relative_difference.png")

    # Figure 5: Transmittance (if available)
    # Compute from edir + edn (normalized)
    dda_trans = dda_total / (dda_edir + dda_edn + 1e-12)  # Simplified
    mie_trans = mie_total / (mie_edir + mie_edn + 1e-12)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wl, dda_trans, "b-", label="DDA", linewidth=1.5)
    ax.plot(wl, mie_trans, "r--", label="Mie", linewidth=1.5)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Transmittance")
    ax.set_title("Transmittance Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "compare_transmittance.png")
    plt.close(fig)
    logger.info("Saved: compare_transmittance.png")


def generate_summary(optics_dda: xr.Dataset, optics_mie: xr.Dataset,
                     rt_dda: xr.Dataset, rt_mie: xr.Dataset) -> str:
    """Generate text summary of all differences.

    Args:
        optics_dda: DDA optical properties.
        optics_mie: Mie optical properties.
        rt_dda: DDA RT results.
        rt_mie: Mie RT results.

    Returns:
        Summary string.
    """
    lines = []
    lines.append("=" * 78)
    lines.append("DDA vs Mie Radiative Transfer Difference Summary")
    lines.append("=" * 78)
    lines.append("")

    # Find index closest to 550 nm
    wl = optics_dda["wavelength_nm"].values
    idx_550 = np.argmin(np.abs(wl - 550.0))
    wl_550 = wl[idx_550]

    # Optical property differences @ 550 nm
    lines.append(f"Optical property differences (@ {wl_550:.0f} nm):")

    dda_g = float(optics_dda["g"].values[idx_550])
    mie_g = float(optics_mie["g"].values[idx_550])
    dg = dda_g - mie_g
    lines.append(f"  Delta g          =  DDA_g - Mie_g          =  {dg:.6f}")

    dda_ssa = float(optics_dda["SSA"].values[idx_550])
    mie_ssa = float(optics_mie["SSA"].values[idx_550])
    dssa = dda_ssa - mie_ssa
    lines.append(f"  Delta SSA        =  DDA_SSA - Mie_SSA      =  {dssa:.6f}")

    dda_qext = float(optics_dda["Q_ext"].values[idx_550])
    mie_qext = float(optics_mie["Q_ext"].values[idx_550])
    dq_ext = (dda_qext - mie_qext) / mie_qext * 100
    lines.append(f"  Delta Q_ext (%)  =  (DDA-Mie)/Mie x 100   =  {dq_ext:.2f}%")

    dda_qsca = float(optics_dda["Q_sca"].values[idx_550])
    mie_qsca = float(optics_mie["Q_sca"].values[idx_550])
    dq_sca = (dda_qsca - mie_qsca) / mie_qsca * 100
    lines.append(f"  Delta Q_sca (%)  =  (DDA-Mie)/Mie x 100   =  {dq_sca:.2f}%")
    lines.append("")

    # RT differences @ 550 nm
    rt_wl = rt_dda["wavelength"].values
    rt_idx = np.argmin(np.abs(rt_wl - wl_550))

    dda_edir = rt_dda["edir"].values[rt_idx] if "edir" in rt_dda else 0
    mie_edir = rt_mie["edir"].values[rt_idx] if "edir" in rt_mie else 0
    dda_edn = rt_dda["edn"].values[rt_idx] if "edn" in rt_dda else 0
    mie_edn = rt_mie["edn"].values[rt_idx] if "edn" in rt_mie else 0
    dda_total = dda_edir + dda_edn
    mie_total = mie_edir + mie_edn

    lines.append(f"Radiative transfer differences (@ {wl_550:.0f} nm):")
    if mie_edir > 0:
        rel_edir = (dda_edir - mie_edir) / mie_edir * 100
        lines.append(f"  Direct irradiance  relative diff =  {rel_edir:.2f}%")
    if mie_edn > 0:
        rel_edn = (dda_edn - mie_edn) / mie_edn * 100
        lines.append(f"  Diffuse downward   relative diff =  {rel_edn:.2f}%")
    if mie_total > 0:
        rel_total = (dda_total - mie_total) / mie_total * 100
        lines.append(f"  Total downward     relative diff =  {rel_total:.2f}%")
    lines.append("")

    # Full-band averages
    dda_edir_all = rt_dda["edir"].values if "edir" in rt_dda else np.zeros_like(rt_wl)
    mie_edir_all = rt_mie["edir"].values if "edir" in rt_mie else np.zeros_like(rt_wl)
    dda_edn_all = rt_dda["edn"].values if "edn" in rt_dda else np.zeros_like(rt_wl)
    mie_edn_all = rt_mie["edn"].values if "edn" in rt_mie else np.zeros_like(rt_wl)
    dda_total_all = dda_edir_all + dda_edn_all
    mie_total_all = mie_edir_all + mie_edn_all

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_edir_all = np.abs((dda_edir_all - mie_edir_all) / (mie_edir_all + 1e-12)) * 100
        rel_edn_all = np.abs((dda_edn_all - mie_edn_all) / (mie_edn_all + 1e-12)) * 100
        rel_total_all = np.abs((dda_total_all - mie_total_all) / (mie_total_all + 1e-12)) * 100

    lines.append("Full-band averages:")
    lines.append(f"  Mean |Delta direct|  =  {np.nanmean(rel_edir_all):.2f}%")
    lines.append(f"  Mean |Delta diffuse| =  {np.nanmean(rel_edn_all):.2f}%")

    max_diff_idx = np.nanargmax(rel_total_all)
    lines.append(f"  Max  |Delta total|   =  {rel_total_all[max_diff_idx]:.2f}% "
                 f"@ {rt_wl[max_diff_idx]:.0f} nm")
    lines.append("")

    # Conclusion
    lines.append("Conclusion:")
    max_optics_diff = max(abs(dq_ext), abs(dq_sca))
    if max_optics_diff < 1.0:
        lines.append("  DDA and Mie optical property differences are small (< 1%).")
        lines.append("  Radiative transfer results should be consistent.")
    elif max_optics_diff < 5.0:
        lines.append("  DDA and Mie optical property differences are moderate (1-5%).")
        lines.append("  Differences mainly come from phase function details.")
    else:
        lines.append("  DDA and Mie optical property differences are significant (> 5%).")
        lines.append("  Check DDA convergence and computation parameters.")
    lines.append("")
    lines.append("=" * 78)

    return "\n".join(lines)


def main():
    logger.info("Stage 3: Comparing DDA and Mie radiative transfer results")

    # Check prerequisites
    for path in [RT_DDA_NC, RT_MIE_NC, OPTICS_DDA_NC, OPTICS_MIE_NC]:
        if not path.exists():
            logger.error(f"Required file not found: {path}")
            logger.error("Run previous stages first:")
            logger.error("  python compute_optics.py")
            logger.error("  python run_radiative_transfer.py")
            return

    # Load data
    rt_dda, rt_mie = load_rt_results()
    optics_dda, optics_mie = load_optics_results()

    # Generate plots
    output_dir = RT_DDA_NC.parent
    plot_spectral_comparison(rt_dda, rt_mie, output_dir)

    # Generate summary
    summary = generate_summary(optics_dda, optics_mie, rt_dda, rt_mie)
    print("\n" + summary)

    # Save summary
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(summary)
    logger.info(f"Saved summary to {SUMMARY_TXT}")

    logger.info("Stage 3 complete.")


if __name__ == "__main__":
    main()
