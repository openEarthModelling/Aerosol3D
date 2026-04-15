"""Example: Uncoated black carbon sphere — 3D visualization and DDA optical computation.

Usage:
    python black_carbon_sphere.py --no-optics
    python black_carbon_sphere.py
    python black_carbon_sphere.py --save
"""

import argparse
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).parent
OUTPUT_DIR = EXAMPLE_DIR / "output"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-optics", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Step 1: Create bare BC sphere ---
    from aerosol3d import AerosolParticle, Material, create_sphere, MixingState, save_screenshot

    soot = Material("soot", refractive_index=complex(1.95, 0.79), density=1.8)
    radius_nm = 50.0

    particle = AerosolParticle(
        name="bare_bc_sphere",
        mixing_state=MixingState.INTERNAL,
        unit="nm",
    )
    particle.add_mesh("core", create_sphere((0, 0, 0), radius_nm), soot)

    print(f"Particle: {particle}")
    print(f"Material: {soot},  n = {soot.refractive_index.real}, k = {soot.refractive_index.imag}")

    # --- Step 2: 3D visualization ---
    save_screenshot(
        particle,
        str(OUTPUT_DIR / "bc_sphere_3d.png"),
        colors={"core": "black"},
        opacity={"core": 0.9},
    )
    print(f"3D screenshot saved: {OUTPUT_DIR / 'bc_sphere_3d.png'}")

    # --- Step 3: DDA optical computation ---
    if args.no_optics:
        print("Optical computation skipped (--no-optics).")
        return

    from aerosol3d import solve_optics, SimulationConfig
    from aerosol3d.optics.visualization import print_macroscopic, plot_phase_function_2d

    config = SimulationConfig(
        wavelength=550.0,
        polarization=(1.0, 0.0, 0.0),
        propagation=(0.0, 0.0, 1.0),
        n_host=1.0,
        dipole_spacing=10.0,
    )

    print(f"\nRunning DDA solve ...")
    result = solve_optics(
        particle,
        config,
        voxel_size=10.0,
        compute_near_field=True,
        compute_phase_func=True,
    )
    print_macroscopic(result)

    if args.save and result.phase_function is not None:
        plot_phase_function_2d(
            result,
            save_path=str(OUTPUT_DIR / "bc_phase_function.png"),
            plane="xz",
            log_scale=True,
        )
        print(f"Phase function saved: {OUTPUT_DIR / 'bc_phase_function.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
