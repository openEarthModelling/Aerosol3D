import pytest


class TestPresetMaterial:
    def test_creates_black_carbon(self):
        from Aerosol3D.materials import preset_material

        mat = preset_material("black_carbon")
        assert mat.name == "black_carbon"
        assert mat.refractive_index == complex(1.95, 0.79)
        assert mat.density == 1.8

    def test_creates_all_presets(self):
        from Aerosol3D.materials import REFRACTIVE_INDEX, preset_material

        for name in REFRACTIVE_INDEX:
            mat = preset_material(name)
            assert mat.name == name
            assert abs(mat.refractive_index.imag) >= 0

    def test_override_refractive_index(self):
        from Aerosol3D.materials import preset_material

        mat = preset_material("sulfate", refractive_index=complex(1.5, 0.01))
        assert mat.refractive_index == complex(1.5, 0.01)
        assert mat.density == 1.8

    def test_unknown_material_raises(self):
        from Aerosol3D.materials import preset_material

        with pytest.raises(KeyError, match="Unknown material"):
            preset_material("nonexistent")

    def test_refractive_index_dict_structure(self):
        from Aerosol3D.materials import REFRACTIVE_INDEX

        for name, data in REFRACTIVE_INDEX.items():
            assert "refractive_index" in data
            assert "density" in data
            assert isinstance(data["refractive_index"], complex)
            assert data["density"] > 0
