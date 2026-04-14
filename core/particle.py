from typing import List, Optional, Union
from enum import Enum, auto
from aerosol3d.core.component import Component

class MixingState(Enum):
    """Enumeration of possible mixing states for an aerosol particle."""
    INTERNAL = auto()      # Internally mixed (components combined within a single particle)
    EXTERNAL = auto()      # Externally mixed (physically separate entities)
    COATED = auto()        # Core-shell configuration
    AGGREGATED = auto()    # Fractal aggregate configuration

class AerosolParticle:
    """
    Class representing an aerosol particle composed of multiple components.
    
    This class maintains a list of Component objects and provides methods
    to manage their relationship and overall particle state.
    """
    
    def __init__(
        self,
        name: str = "AerosolParticle",
        mixing_state: MixingState = MixingState.INTERNAL
    ):
        """
        Initialize an AerosolParticle.

        Args:
            name: Identifiable name for the particle.
            mixing_state: Mixing state of the components (default is internal).
        """
        self.name = name
        self.mixing_state = mixing_state
        self._components: List[Component] = []

    def add_component(self, component: Component) -> None:
        """Add a geometric component to the particle."""
        if not isinstance(component, Component):
            raise TypeError("Expected an instance of Component.")
        self._components.append(component)

    @property
    def components(self) -> List[Component]:
        """Return the list of components composing the particle."""
        return self._components

    def set_mixing_state(self, state: MixingState) -> None:
        """Set the mixing state of the particle."""
        self.mixing_state = state

    def __len__(self) -> int:
        """Return the number of components in the particle."""
        return len(self._components)

    def __repr__(self) -> str:
        return (f"AerosolParticle(name={self.name}, "
                f"components={len(self)}, mixing_state={self.mixing_state.name})")

    def summary(self) -> str:
        """Generate a technical summary of the particle composition."""
        lines = [f"Aerosol Particle: {self.name}", f"Mixing State: {self.mixing_state.name}"]
        for i, comp in enumerate(self._components):
            lines.append(f"  [{i}] {comp}")
        return "\n".join(lines)
