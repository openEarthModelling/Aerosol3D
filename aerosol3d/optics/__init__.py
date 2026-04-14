"""DDA optical property computation via CoupledElectricMagneticDipoles.jl."""

from .datastructs import SimulationConfig, OpticalResult  # noqa: F401
from .dda_solver import solve_optics  # noqa: F401
