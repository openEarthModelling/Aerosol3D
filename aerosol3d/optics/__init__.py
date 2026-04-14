"""DDA optical property computation via CoupledElectricMagneticDipoles.jl."""

from .datastructs import SimulationConfig, OpticalResult  # noqa: F401
from .dda_solver import solve_optics  # noqa: F401
from .visualization import plot_phase_function_2d, plot_near_field, print_macroscopic  # noqa: F401
