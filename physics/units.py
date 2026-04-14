import pint

# Create a global unit registry for the Aerosol3D project
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

# Define common units used in aerosol physics to ensure consistency
ureg.define('micrometer = 1e-6 * meter = um = micron')
ureg.define('nanometer = 1e-9 * meter = nm')

def to_meters(value, unit_str):
    """
    Convert a value (scalar or array) from a given unit to meters.
    
    Args:
        value: The numerical value(s) to convert.
        unit_str (str): The unit of the value (e.g., 'nm', 'um', 'mm').
        
    Returns:
        float or np.ndarray: The value(s) in meters.
    """
    return Q_(value, unit_str).to(ureg.meter).magnitude
