def test_optical_result_has_solver_field():
    from Aerosol3D.optics.datastructs import CrossSections, OpticalResult, SimulationConfig

    cs = CrossSections(
        wavelength=550.0,
        C_ext=100.0,
        C_sca=80.0,
        C_abs=20.0,
        Q_ext=2.0,
        Q_sca=1.6,
        Q_abs=0.4,
        SSA=0.8,
        g=0.7,
        r_eff=50.0,
    )
    cfg = SimulationConfig(wavelength=550.0)
    result = OpticalResult(config=cfg, cross_sections=cs)
    assert hasattr(result, "solver")
    assert result.solver == "DDA"
