def test_legendre_export_importable():
    from aerosol3d.optics import compute_legendre_moments
    assert callable(compute_legendre_moments)


def test_pyradtran_export_importable():
    from aerosol3d.optics import optical_results_to_pyradtran_data
    assert callable(optical_results_to_pyradtran_data)
