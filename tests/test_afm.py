import os
import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")
from unittest.mock import patch, MagicMock

from automet.base import BaseAnalyzer
from automet.afm import AFMAnalyzer


def _make_mock_ibw(shape=(64, 64, 4)):
    """Return a minimal igor2-style dict with random channel data."""
    rng = np.random.default_rng(42)
    wave_data = rng.normal(0, 5e-9, shape)
    note = b"ScanSize:5e-6\nScanRate:1.0\n"
    return {"wave": {"wData": wave_data, "note": note}}


@pytest.fixture
def analyzer_loaded(tmp_path):
    """AFMAnalyzer with load_data mocked so no real .ibw file is needed."""
    fake_path = str(tmp_path / "test.ibw")
    analyzer = AFMAnalyzer(fake_path)
    mock_ibw = _make_mock_ibw()
    with patch("automet.afm.igor2.binarywave.load", return_value=mock_ibw):
        analyzer.load_data()
    return analyzer


def test_afm_inherits_base(tmp_path):
    analyzer = AFMAnalyzer(str(tmp_path / "f.ibw"))
    assert isinstance(analyzer, BaseAnalyzer)


def test_load_data_returns_self(tmp_path):
    analyzer = AFMAnalyzer(str(tmp_path / "f.ibw"))
    with patch("automet.afm.igor2.binarywave.load", return_value=_make_mock_ibw()):
        result = analyzer.load_data()
    assert result is analyzer


def test_load_data_populates_channels(analyzer_loaded):
    a = analyzer_loaded
    assert a.height is not None
    assert a.defl is not None
    assert a.amp is not None
    assert a.phase is not None


def test_height_converted_to_nm(analyzer_loaded):
    """Height channel should be scaled from metres to nm (÷1e-9)."""
    raw_metres = _make_mock_ibw()["wave"]["wData"][:, :, 0]
    expected_nm = raw_metres / 1e-9
    np.testing.assert_allclose(analyzer_loaded.height, expected_nm)


def test_apply_median_filter_returns_self(analyzer_loaded):
    result = analyzer_loaded.apply_median_filter(size=3)
    assert result is analyzer_loaded


def test_apply_gaussian_filter_returns_self(analyzer_loaded):
    result = analyzer_loaded.apply_gaussian_filter(sigma=1)
    assert result is analyzer_loaded


def test_compute_roughness_returns_self(analyzer_loaded):
    result = analyzer_loaded.compute_roughness()
    assert result is analyzer_loaded


def test_compute_roughness_values_positive(analyzer_loaded):
    analyzer_loaded.compute_roughness()
    assert analyzer_loaded.Sa > 0
    assert analyzer_loaded.Sq > 0
    assert analyzer_loaded.Sq >= analyzer_loaded.Sa


def test_compute_spatially_filtered_roughness(analyzer_loaded):
    analyzer_loaded.compute_spatially_filtered_roughness(sigma=2.0)
    assert analyzer_loaded.Sa_r is not None
    assert analyzer_loaded.Sq_r is not None
    assert analyzer_loaded.waviness is not None
    assert analyzer_loaded.roughness is not None


def test_compute_psd_2d(analyzer_loaded):
    analyzer_loaded.compute_spatially_filtered_roughness(sigma=2.0)
    analyzer_loaded.compute_psd_2d()
    assert analyzer_loaded.PSD is not None
    assert analyzer_loaded.PSD.shape == analyzer_loaded.height.shape


def test_save_output_creates_file(analyzer_loaded, tmp_path):
    analyzer_loaded.compute_spatially_filtered_roughness(sigma=2.0)
    analyzer_loaded.compute_psd_2d()
    out = str(tmp_path / "psd.png")
    analyzer_loaded.save_output(out, dpi=72)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0
