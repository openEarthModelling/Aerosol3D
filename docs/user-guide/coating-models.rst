Coating Models
==============

Aerosol3D provides several algorithms for applying coatings to particles.
Each model represents a different physical coating process.

Distance-Based Coating
----------------------

Applies a coating of uniform thickness to all outer surfaces of the particle.
This represents condensation processes where material deposits uniformly.

Closed-Cell Model (CCM)
-----------------------

Models coating as a closed-cell foam structure. Suitable for organic coatings
that form a continuous shell with enclosed voids.

Coated-Aggregate Model (CAM)
----------------------------

Specifically designed for fractal aggregates. Places coating material at the
contact points between primary particles, then builds up a shell.

Potential-Based Coating
-----------------------

Uses a potential field to determine coating placement:

- **Edge coating**: Material deposits preferentially at edges and corners
- **Void coating**: Material fills internal voids within the particle

Choosing a Model
----------------

+-------------------+----------------------------------------+
| Model             | Use Case                               |
+-------------------+----------------------------------------+
| Distance-based    | Uniform condensation (sulfate on BC)   |
| CCM               | Organic foam-like coatings             |
| CAM               | Fractal aggregate coatings             |
| Potential edge    | Edge-selective deposition              |
| Potential void    | Internal void filling                  |
+-------------------+----------------------------------------+
