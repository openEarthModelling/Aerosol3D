def test_legendre_export_importable():
    from Aerosol3D.optics import compute_legendre_moments

    assert callable(compute_legendre_moments)


def test_optics_export_importable():
    from Aerosol3D.optics import AerosolOpticsData, from_optical_results

    assert AerosolOpticsData is not None
    assert callable(from_optical_results)
