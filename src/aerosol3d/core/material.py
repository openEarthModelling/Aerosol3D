class Material:
    """Material properties for an aerosol component."""

    _next_id: int = 1

    def __init__(self, name: str, refractive_index: complex, density: float):
        self.name = name
        self.refractive_index = refractive_index
        self.density = density
        self.id = Material._next_id
        Material._next_id += 1

    def __repr__(self) -> str:
        return f"Material(name={self.name!r}, id={self.id})"