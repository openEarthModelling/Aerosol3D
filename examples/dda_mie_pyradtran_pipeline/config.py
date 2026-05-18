"""Shared configuration for the DDA-Mie-pyRadtran pipeline example."""

import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXAMPLE_DIR = Path(__file__).parent
OUTPUT_DIR = EXAMPLE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Particle configuration
# ---------------------------------------------------------------------------
PARTICLE_CONFIG = {
    "material": "black_carbon",
    "radius_nm": 200.0,
    "wavelengths_nm": np.linspace(400, 700, 7).tolist(),  # match DDA grid
}

# ---------------------------------------------------------------------------
# DDA configuration
# ---------------------------------------------------------------------------
DDA_CONFIG = {
    "precision": "medium",
    "solver": "DDA",
    "compute_phase_func": True,
}

# ---------------------------------------------------------------------------
# Mie configuration
# ---------------------------------------------------------------------------
MIE_CONFIG = {
    "solver": "MIE",
    "compute_phase_func": True,
}

# ---------------------------------------------------------------------------
# Radiative transfer scene configuration
# ---------------------------------------------------------------------------
SCENE_CONFIG = {
    "atmosphere": {"profile": "us", "altitude": 0.0},
    "source": {"type": "solar", "sza": 30.0},
    "wavelength": {"min_nm": 401.0, "max_nm": 699.0},
    "solver": {
        "method": "disort",
        "streams": 16,
        "disort_intcor": "moments",
        "pseudospherical": True,
        "deltam": True,
    },
    "surface": {"albedo": 0.1},
    "output": {
        "quantities": ["lambda", "edir", "edn", "eup"],
        "quantity": "transmittance",
        "format": "netcdf",
    },
}

# ---------------------------------------------------------------------------
# Aerosol vertical profile configuration
# ---------------------------------------------------------------------------
AEROSOL_PROFILE = {
    "altitude_grid_km": np.linspace(0, 10, 11).tolist(),  # 0, 1, ..., 10 km
    "scale_height_km": 1.5,
    "total_optical_depth_550": 0.3,
    "particle_density_kg_m3": 1800.0,  # black carbon density
}

# ---------------------------------------------------------------------------
# Output filenames
# ---------------------------------------------------------------------------
OPTICS_DDA_NC = OUTPUT_DIR / "optics_dda.nc"
OPTICS_MIE_NC = OUTPUT_DIR / "optics_mie.nc"
RT_DDA_NC = OUTPUT_DIR / "rt_dda.nc"
RT_MIE_NC = OUTPUT_DIR / "rt_mie.nc"
SUMMARY_TXT = OUTPUT_DIR / "summary.txt"
