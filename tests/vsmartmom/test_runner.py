"""Tests for VSmartMOMRunner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from Aerosol3D.vsmartmom.runner import VSmartMOMRunner


class TestVSmartMOMRunnerConstruction:
    """Test suite for runner construction and basic config."""

    def test_runner_construction(self) -> None:
        """Verify config fields are stored correctly."""
        runner = VSmartMOMRunner(
            julia_project="/path/to/project",
            julia_executable="/usr/bin/julia",
            cleanup_temp=False,
        )
        assert runner.julia_project == Path("/path/to/project")
        assert runner.julia_executable == "/usr/bin/julia"
        assert runner.cleanup_temp is False

    def test_runner_defaults(self) -> None:
        """Verify default values."""
        runner = VSmartMOMRunner(julia_project="/path/to/project")
        assert runner.julia_project == Path("/path/to/project")
        assert runner.julia_executable == "julia"
        assert runner.cleanup_temp is True


class TestVSmartMOMRunnerValidation:
    """Test suite for input validation."""

    def test_runner_validation_heights_conc_mismatch(self) -> None:
        """Heights length must equal number_conc length + 1."""
        runner = VSmartMOMRunner(julia_project="/tmp")
        heights = np.array([0.0, 1000.0, 2000.0])  # 3 heights -> expects 2 conc
        number_conc = np.array([1.0, 2.0, 3.0])  # 3 conc -> mismatch

        with pytest.raises(ValueError, match="heights.*number_conc"):
            runner._validate_inputs(heights, number_conc)

    def test_runner_validation_negative_conc(self) -> None:
        """All concentrations must be non-negative."""
        runner = VSmartMOMRunner(julia_project="/tmp")
        heights = np.array([0.0, 1000.0, 2000.0])
        number_conc = np.array([1.0, -0.5])

        with pytest.raises(ValueError, match="negative"):
            runner._validate_inputs(heights, number_conc)

    def test_runner_validation_non_increasing_heights(self) -> None:
        """Heights must be strictly increasing."""
        runner = VSmartMOMRunner(julia_project="/tmp")
        heights = np.array([0.0, 1000.0, 500.0])
        number_conc = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="strictly increasing"):
            runner._validate_inputs(heights, number_conc)

    def test_runner_validation_equal_heights(self) -> None:
        """Heights must be strictly increasing (no equal values)."""
        runner = VSmartMOMRunner(julia_project="/tmp")
        heights = np.array([0.0, 1000.0, 1000.0])
        number_conc = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="strictly increasing"):
            runner._validate_inputs(heights, number_conc)


class TestVSmartMOMRunnerJuliaCheck:
    """Test suite for Julia executable checking."""

    def test_runner_julia_not_found(self) -> None:
        """RuntimeError when Julia executable is not found."""
        runner = VSmartMOMRunner(julia_project="/tmp", julia_executable="nonexistent_julia_binary")

        with pytest.raises(RuntimeError, match="Julia executable.*not found"):
            runner._check_julia()

    def test_runner_julia_found(self) -> None:
        """No error when Julia executable exists."""
        runner = VSmartMOMRunner(julia_project="/tmp", julia_executable="python")
        # python is guaranteed to exist
        runner._check_julia()  # should not raise


class TestVSmartMOMRunnerRunRT:
    """Test suite for run_rt method."""

    def _make_bulk_mock(self, n_wl: int = 2, n_legendre: int = 4) -> MagicMock:
        """Create a mock BulkAerosolOpticsData with required attributes."""
        bulk = MagicMock()
        bulk.wavelength_nm = np.array([500.0, 600.0])
        bulk.C_ext = np.ones(n_wl) * 1.5
        bulk.SSA = np.array([0.9, 0.85])
        bulk.beta = np.ones((n_wl, n_legendre)) * 0.5
        return bulk

    @patch("Aerosol3D.vsmartmom.runner.VSmartMOMResult.from_netcdf")
    @patch("subprocess.run")
    @patch("Aerosol3D.vsmartmom.runner.serialize_input")
    @patch("shutil.which")
    def test_run_rt_success(
        self,
        mock_which: MagicMock,
        mock_serialize: MagicMock,
        mock_subprocess: MagicMock,
        mock_from_netcdf: MagicMock,
    ) -> None:
        """Happy path: validate inputs, serialize, run Julia, deserialize."""
        mock_which.return_value = "/usr/bin/julia"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        expected_result = MagicMock()
        mock_from_netcdf.return_value = expected_result

        runner = VSmartMOMRunner(julia_project="/path/to/julia/project", cleanup_temp=False)
        bulk = self._make_bulk_mock()
        heights = np.array([0.0, 1000.0, 2000.0])
        number_conc = np.array([100.0, 200.0])
        sza = 30.0
        vza = np.array([0.0, 30.0, 60.0])

        result = runner.run_rt(bulk, heights, number_conc, sza, vza)

        assert result is expected_result
        mock_serialize.assert_called_once()
        mock_subprocess.assert_called_once()
        mock_from_netcdf.assert_called_once()

    @patch("shutil.which")
    def test_run_rt_invalid_inputs(self, mock_which: MagicMock) -> None:
        """Validation errors propagate before Julia check."""
        mock_which.return_value = "/usr/bin/julia"

        runner = VSmartMOMRunner(julia_project="/tmp")
        bulk = self._make_bulk_mock()
        heights = np.array([0.0, 1000.0, 2000.0])
        number_conc = np.array([100.0, 200.0, 300.0])  # mismatch
        sza = 30.0
        vza = np.array([0.0, 30.0])

        with pytest.raises(ValueError, match="heights.*number_conc"):
            runner.run_rt(bulk, heights, number_conc, sza, vza)

    @patch("shutil.which")
    def test_run_rt_julia_missing(self, mock_which: MagicMock) -> None:
        """RuntimeError when Julia is missing."""
        mock_which.return_value = None

        runner = VSmartMOMRunner(julia_project="/tmp", julia_executable="missing_julia")
        bulk = self._make_bulk_mock()
        heights = np.array([0.0, 1000.0, 2000.0])
        number_conc = np.array([100.0, 200.0])
        sza = 30.0
        vza = np.array([0.0, 30.0])

        with pytest.raises(RuntimeError, match="Julia executable.*not found"):
            runner.run_rt(bulk, heights, number_conc, sza, vza)

    @patch("Aerosol3D.vsmartmom.runner.VSmartMOMResult.from_netcdf")
    @patch("subprocess.run")
    @patch("Aerosol3D.vsmartmom.runner.serialize_input")
    @patch("shutil.which")
    def test_run_rt_with_vaz(
        self,
        mock_which: MagicMock,
        mock_serialize: MagicMock,
        mock_subprocess: MagicMock,
        mock_from_netcdf: MagicMock,
    ) -> None:
        """vaz is passed through to serialize_input."""
        mock_which.return_value = "/usr/bin/julia"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        expected_result = MagicMock()
        mock_from_netcdf.return_value = expected_result

        runner = VSmartMOMRunner(julia_project="/path/to/julia/project", cleanup_temp=False)
        bulk = self._make_bulk_mock()
        heights = np.array([0.0, 1000.0, 2000.0])
        number_conc = np.array([100.0, 200.0])
        sza = 30.0
        vza = np.array([0.0, 30.0, 60.0])
        vaz = np.array([0.0, 90.0, 180.0])

        result = runner.run_rt(bulk, heights, number_conc, sza, vza, vaz=vaz)

        assert result is expected_result
        # Check that vaz was passed to serialize_input
        call_kwargs = mock_serialize.call_args.kwargs
        assert "vaz" in call_kwargs
        np.testing.assert_array_equal(call_kwargs["vaz"], vaz)

    @patch("Aerosol3D.vsmartmom.runner.VSmartMOMResult.from_netcdf")
    @patch("subprocess.run")
    @patch("Aerosol3D.vsmartmom.runner.serialize_input")
    @patch("shutil.which")
    def test_run_rt_tau_ref_scaling(
        self,
        mock_which: MagicMock,
        mock_serialize: MagicMock,
        mock_subprocess: MagicMock,
        mock_from_netcdf: MagicMock,
    ) -> None:
        """tau_ref override scales concentrations to match target optical depth."""
        mock_which.return_value = "/usr/bin/julia"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        expected_result = MagicMock()
        mock_from_netcdf.return_value = expected_result

        runner = VSmartMOMRunner(julia_project="/path/to/julia/project", cleanup_temp=False)
        bulk = self._make_bulk_mock(n_wl=1, n_legendre=4)
        bulk.wavelength_nm = np.array([500.0])
        bulk.C_ext = np.array([2.0])
        heights = np.array([0.0, 1000.0])
        number_conc = np.array([100.0])
        sza = 30.0
        vza = np.array([0.0, 30.0])
        tau_ref = 0.5

        result = runner.run_rt(bulk, heights, number_conc, sza, vza, tau_ref=tau_ref)

        assert result is expected_result
        # Check that scaled concentration was passed to serialize_input
        call_kwargs = mock_serialize.call_args.kwargs
        scaled_conc = call_kwargs["number_conc"]
        # Original tau = 100 * 2.0 * 1000 * 1e-6 = 0.2
        # Scale factor = 0.5 / 0.2 = 2.5
        # Scaled conc = 100 * 2.5 = 250
        assert scaled_conc[0] == pytest.approx(250.0)

    @patch("subprocess.run")
    @patch("Aerosol3D.vsmartmom.runner.serialize_input")
    @patch("shutil.which")
    def test_run_rt_subprocess_failure(
        self,
        mock_which: MagicMock,
        mock_serialize: MagicMock,
        mock_subprocess: MagicMock,
    ) -> None:
        """RuntimeError when Julia subprocess returns non-zero."""
        mock_which.return_value = "/usr/bin/julia"
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = b"Julia error"
        mock_subprocess.return_value = mock_result

        runner = VSmartMOMRunner(julia_project="/path/to/julia/project", cleanup_temp=False)
        bulk = self._make_bulk_mock()
        heights = np.array([0.0, 1000.0, 2000.0])
        number_conc = np.array([100.0, 200.0])
        sza = 30.0
        vza = np.array([0.0, 30.0])

        with pytest.raises(RuntimeError, match="vSmartMOM.*failed"):
            runner.run_rt(bulk, heights, number_conc, sza, vza)
