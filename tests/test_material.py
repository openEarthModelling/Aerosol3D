import pytest


class TestMaterial:
    def test_create_material(self):
        from Aerosol3D.core.material import Material
        m = Material(name="soot", refractive_index=complex(1.8, 0.7), density=1.8)
        assert m.name == "soot"
        assert m.refractive_index == complex(1.8, 0.7)
        assert m.density == 1.8

    def test_auto_increment_id(self):
        from Aerosol3D.core.material import Material
        Material._next_id = 1  # reset for testing (id=0 reserved for void)
        m1 = Material(name="soot", refractive_index=complex(1.8, 0.7), density=1.8)
        m2 = Material(name="sulfate", refractive_index=complex(1.4, 0.0), density=1.8)
        assert m1.id == 1
        assert m2.id == 2

    def test_repr(self):
        from Aerosol3D.core.material import Material
        Material._next_id = 1
        m = Material(name="soot", refractive_index=complex(1.8, 0.7), density=1.8)
        assert "soot" in repr(m)