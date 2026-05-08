"""Preset refractive index database for common aerosol materials.

Values at 550 nm from published literature. Users can override any value.
"""

from __future__ import annotations

from Aerosol3D.core.material import Material

# Refractive indices at 550 nm.
REFRACTIVE_INDEX: dict[str, dict] = {
    "black_carbon": {
        "refractive_index": complex(1.95, 0.79),
        "density": 1.8,
        "reference": "Bond & Bergstrom 2006",
    },
    "sulfate": {
        "refractive_index": complex(1.43, 0.00),
        "density": 1.8,
        "reference": "OPAC (Hess et al. 1998)",
    },
    "sea_salt": {
        "refractive_index": complex(1.50, 1.0e-8),
        "density": 2.2,
        "reference": "OPAC",
    },
    "mineral_dust": {
        "refractive_index": complex(1.53, 0.008),
        "density": 2.6,
        "reference": "OPAC",
    },
    "organic_carbon": {
        "refractive_index": complex(1.55, 0.01),
        "density": 1.5,
        "reference": "OPAC",
    },
    "water": {
        "refractive_index": complex(1.333, 0.00),
        "density": 1.0,
        "reference": "Hale & Querry 1973",
    },
    "volcanic_ash": {
        "refractive_index": complex(1.50, 0.01),
        "density": 2.5,
        "reference": "OPAC",
    },
}


def preset_material(name: str, **overrides) -> Material:
    """Create a Material using preset refractive index, with optional overrides.

    Args:
        name: One of the keys in REFRACTIVE_INDEX (e.g. "black_carbon").
        **overrides: Override any field (e.g. refractive_index=complex(2.0, 1.0)).

    Returns:
        Material instance.

    Raises:
        KeyError: If name is not in REFRACTIVE_INDEX.
    """
    from .core.material import Material

    if name not in REFRACTIVE_INDEX:
        available = ", ".join(sorted(REFRACTIVE_INDEX.keys()))
        raise KeyError(f"Unknown material '{name}'. Available: {available}")

    data = REFRACTIVE_INDEX[name].copy()
    data.pop("reference", None)
    data.update(overrides)
    return Material(name=name, **data)
