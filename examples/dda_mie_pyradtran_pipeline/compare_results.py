"""Stage 3: Compare DDA and Mie radiative transfer results.

Usage:
    python compare_results.py
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from Aerosol3D.optics.optics_export import AerosolOpticsData
from Aerosol3D.optics.visualization import (
    plot_optical_comparison,
    plot_phase_function_comparison,
    generate_comparison_summary,
)

from config import (
    OPTICS_DDA_NC,
    OPTICS_MIE_NC,
    RT_DDA_NC,
    RT_MIE_NC,
    SUMMARY_TXT,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def rel_diff(a, b):
    return (a - b) / (np.abs(b) + 1e-12) * 100.0


def plot_spectral_comparison(dda: xr.Dataset, mie: xr.Dataset, output_dir: Path):
    """Plot spectral comparison of irradiance and transmittance."""
    wl = dda["wavelength"].values

    dda_edir = dda["edir"].values if "edir" in dda else np.zeros_like(wl)
    mie_edir = mie["edir"].values if "edir" in mie else np.zeros_like(wl)
    dda_edn = dda["edn"].values if "edn" in dda else np.zeros_like(wl)
    mie_edn = mie["edn"].values if "edn" in mie else np.zeros_like(wl)

    dda_total = dda_edir + dda_edn
    mie_total = mie_edir + mie_edn

    plots = [
        ("Direct irradiance (edir)", dda_edir, mie_edir, "compare_direct_irradiance"),
        ("Diffuse downward (edn)", dda_edn, mie_edn, "compare_diffuse_downward"),
        ("Total downward", dda_total, mie_total, "compare_total_downward"),
    ]

    for title, dda_v, mie_v, fname in plots:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axes[0].plot(wl, dda_v, "b-", label="DDA", linewidth=1.5)
        axes[0].plot(wl, mie_v, "r--", label="Mie", linewidth=1.5)
        axes[0].set_ylabel(title.split("(")[0].strip())
        axes[0].set_title(title)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(wl, rel_diff(dda_v, mie_v), "g-", linewidth=1.0)
        axes[1].axhline(0, color="k", linewidth=0.5)
        axes[1].set_xlabel("Wavelength (nm)")
        axes[1].set_ylabel("Relative diff. (%)")
        axes[1].set_title("Relative Difference")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / f"{fname}.png")
        plt.close(fig)
        logger.info(f"Saved: {fname}.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wl, rel_diff(dda_edir, mie_edir), "b-", label="Direct", linewidth=1.2)
    ax.plot(wl, rel_diff(dda_edn, mie_edn), "r-", label="Diffuse down", linewidth=1.2)
    ax.plot(wl, rel_diff(dda_total, mie_total), "g-", label="Total down", linewidth=1.2)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Relative difference (%)")
    ax.set_title("DDA vs Mie: All Components")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "compare_relative_difference.png")
    plt.close(fig)
    logger.info("Saved: compare_relative_difference.png")

    dda_trans = dda_total / (dda_total + 1e-12)
    mie_trans = mie_total / (mie_total + 1e-12)
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


def main():
    logger.info("Stage 3: Comparing DDA and Mie radiative transfer results")

    for path in [RT_DDA_NC, RT_MIE_NC, OPTICS_DDA_NC, OPTICS_MIE_NC]:
        if not path.exists():
            logger.error(f"Required file not found: {path}")
            logger.error("Run previous stages first")
            return

    rt_dda = xr.open_dataset(RT_DDA_NC)
    rt_mie = xr.open_dataset(RT_MIE_NC)
    optics_dda = AerosolOpticsData.from_netcdf(str(OPTICS_DDA_NC))
    optics_mie = AerosolOpticsData.from_netcdf(str(OPTICS_MIE_NC))

    output_dir = RT_DDA_NC.parent

    plot_optical_comparison([optics_dda, optics_mie], ["DDA", "Mie"], str(output_dir))
    plot_phase_function_comparison([optics_dda, optics_mie], ["DDA", "Mie"], str(output_dir))

    summary = generate_comparison_summary([optics_dda, optics_mie], ["DDA", "Mie"])
    print("\n" + summary)

    plot_spectral_comparison(rt_dda, rt_mie, output_dir)

    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(summary)
    logger.info(f"Saved summary to {SUMMARY_TXT}")

    logger.info("Stage 3 complete.")


if __name__ == "__main__":
    main()
