import os
import pathlib

import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")
from unittest.mock import patch

from automet.base import BaseAnalyzer
from automet.profilometer import ProfilometerAnalyzer, ZonFile


# ---------------------------------------------------------------------------
# Lightweight ZonFile stand-in (no real .zon file or libzstd needed)
# ---------------------------------------------------------------------------

class _MockZon:
    """Minimal ZonFile mimic with configurable height data."""

    ZUNIT  = 1e-8
    XYUNIT = 2.36e-5

    def __init__(self, height_um: np.ndarray):
        self.path     = pathlib.Path('/fake/test.zon')
        self.w        = height_um.shape[1]
        self.h        = height_um.shape[0]
        self.metadata = {'ScanDateTime': '01/01/2025 12:00:00',
                         'MeterPerPixel': str(self.XYUNIT)}
        self.thumbnail_bytes = b''
        self._h_um  = height_um.astype(float)
        self._h_rel = self._h_um - np.nanmean(self._h_um)
        self.height_raw = (self._h_um / (self.ZUNIT * 1e6)).astype(np.uint32)

    @property
    def height_um(self):
        return self._h_um.copy()

    @property
    def height_rel_um(self):
        return self._h_rel.copy()

    @property
    def pixel_size_um(self):
        return self.XYUNIT * 1e6

    @property
    def scan_width_mm(self):
        return self.w * self.XYUNIT * 1e3

    @property
    def scan_height_mm(self):
        return self.h * self.XYUNIT * 1e3

    def roughness_stats(self, height_map=None, percentile_clip=1.0):
        if height_map is None:
            height_map = self._h_rel
        v    = height_map[~np.isnan(height_map)]
        mean = np.mean(v); vc = v - mean
        Sa   = float(np.mean(np.abs(vc)))
        Sq   = float(np.sqrt(np.mean(vc ** 2)))
        lo   = np.percentile(v, percentile_clip)
        hi   = np.percentile(v, 100 - percentile_clip)
        clip = v[(v >= lo) & (v <= hi)]
        Sz   = float(clip.max() - clip.min())
        return dict(Sa=Sa, Sq=Sq, Sz=Sz,
                    Sp=float(v.max() - mean), Sv=float(mean - v.min()),
                    Ssk=0.0, Sku=3.0)


def _flat_height(shape=(64, 64)):
    """Uniform background height map with small noise (µm)."""
    rng = np.random.default_rng(42)
    return np.full(shape, 5.0) + rng.normal(0, 0.01, shape)


def _die_height(shape=(80, 80)):
    """Height map with a raised rectangular die in the centre."""
    rng = np.random.default_rng(42)
    h = np.full(shape, 5.0) + rng.normal(0, 0.01, shape)
    h[15:65, 15:65] += 0.5   # raised die region (50×50 px)
    return h


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analyzer_loaded(tmp_path):
    """ProfilometerAnalyzer with load_data mocked (no real .zon file needed)."""
    a = ProfilometerAnalyzer(str(tmp_path / 'test.zon'))
    with patch('automet.profilometer.ZonFile', return_value=_MockZon(_flat_height())):
        a.load_data()
    return a


@pytest.fixture
def analyzer_roughness(analyzer_loaded):
    """ProfilometerAnalyzer after roughness computation."""
    return analyzer_loaded.compute_roughness()


@pytest.fixture
def analyzer_with_die(tmp_path):
    """ProfilometerAnalyzer loaded with a die-containing height map."""
    a = ProfilometerAnalyzer(
        str(tmp_path / 'die.zon'),
        smooth_sigma=1,
        morph_open_radius=1,
        morph_close_radius=2,
        min_die_area_frac=0.05,
    )
    with patch('automet.profilometer.ZonFile', return_value=_MockZon(_die_height())):
        a.load_data()
    return a


@pytest.fixture
def analyzer_die(analyzer_with_die):
    """ProfilometerAnalyzer after die detection."""
    return analyzer_with_die.detect_die()


@pytest.fixture
def analyzer_edges(analyzer_die):
    """ProfilometerAnalyzer after edge extraction."""
    return analyzer_die.extract_edges()


@pytest.fixture
def analyzer_edge_roughness(analyzer_edges):
    """ProfilometerAnalyzer after edge roughness measurement."""
    return analyzer_edges.measure_edge_roughness()


@pytest.fixture
def analyzer_per_side(analyzer_edge_roughness):
    """ProfilometerAnalyzer after per-side analysis."""
    return analyzer_edge_roughness.analyse_per_side()


# ---------------------------------------------------------------------------
# Inheritance & instantiation
# ---------------------------------------------------------------------------

def test_inherits_base(tmp_path):
    a = ProfilometerAnalyzer(str(tmp_path / 'f.zon'))
    assert isinstance(a, BaseAnalyzer)


def test_load_data_returns_self(tmp_path):
    a = ProfilometerAnalyzer(str(tmp_path / 'f.zon'))
    with patch('automet.profilometer.ZonFile', return_value=_MockZon(_flat_height())):
        result = a.load_data()
    assert result is a


def test_load_data_sets_zon(analyzer_loaded):
    assert analyzer_loaded._zon is not None


# ---------------------------------------------------------------------------
# Roughness
# ---------------------------------------------------------------------------

def test_compute_roughness_returns_self(analyzer_loaded):
    result = analyzer_loaded.compute_roughness()
    assert result is analyzer_loaded


def test_compute_roughness_sets_params(analyzer_roughness):
    params = analyzer_roughness.roughness_params
    assert params is not None
    for key in ('Sa', 'Sq', 'Sz', 'Sp', 'Sv', 'Ssk', 'Sku'):
        assert key in params


def test_compute_roughness_positive_sa_sq(analyzer_roughness):
    params = analyzer_roughness.roughness_params
    assert params['Sa'] > 0
    assert params['Sq'] > 0
    assert params['Sq'] >= params['Sa']


def test_roughness_stats_keys(analyzer_loaded):
    stats = analyzer_loaded._zon.roughness_stats()
    assert set(stats.keys()) == {'Sa', 'Sq', 'Sz', 'Sp', 'Sv', 'Ssk', 'Sku'}


# ---------------------------------------------------------------------------
# Die detection
# ---------------------------------------------------------------------------

def test_detect_die_returns_self(analyzer_with_die):
    result = analyzer_with_die.detect_die()
    assert result is analyzer_with_die


def test_detect_die_sets_mask(analyzer_die):
    assert analyzer_die.die_mask is not None
    assert analyzer_die.die_mask.dtype == bool


def test_detect_die_mask_shape(analyzer_die):
    z = analyzer_die._zon
    assert analyzer_die.die_mask.shape == (z.h, z.w)


def test_detect_die_finds_die(analyzer_die):
    """The raised die region should be at least 5% of the scan area."""
    coverage = analyzer_die.die_mask.sum() / analyzer_die.die_mask.size
    assert coverage >= 0.05


# ---------------------------------------------------------------------------
# Edge extraction
# ---------------------------------------------------------------------------

def test_extract_edges_returns_self(analyzer_die):
    result = analyzer_die.extract_edges()
    assert result is analyzer_die


def test_extract_edges_finds_contours(analyzer_edges):
    assert analyzer_edges.contours is not None
    assert len(analyzer_edges.contours) >= 1


def test_extract_edges_main_contour(analyzer_edges):
    assert analyzer_edges.main_contour is not None
    assert analyzer_edges.main_contour.ndim == 2
    assert analyzer_edges.main_contour.shape[1] == 2


# ---------------------------------------------------------------------------
# Edge roughness
# ---------------------------------------------------------------------------

def test_measure_edge_roughness_returns_self(analyzer_edges):
    result = analyzer_edges.measure_edge_roughness()
    assert result is analyzer_edges


def test_edge_result_keys(analyzer_edge_roughness):
    keys = {'edge_Ra', 'edge_Rq', 'edge_Rz',
            'edge_heights', 'residual', 'arc_s',
            'edge_dev_um', 'contour_mm'}
    assert keys.issubset(analyzer_edge_roughness.edge_result.keys())


def test_edge_roughness_positive(analyzer_edge_roughness):
    er = analyzer_edge_roughness.edge_result
    assert er['edge_Ra'] > 0
    assert er['edge_Rq'] > 0
    assert er['edge_Rz'] >= er['edge_Ra']


def test_edge_dev_is_array(analyzer_edge_roughness):
    dev = analyzer_edge_roughness.edge_result['edge_dev_um']
    assert isinstance(dev, np.ndarray)
    assert len(dev) > 0


# ---------------------------------------------------------------------------
# Per-side analysis
# ---------------------------------------------------------------------------

def test_analyse_per_side_returns_self(analyzer_edge_roughness):
    result = analyzer_edge_roughness.analyse_per_side()
    assert result is analyzer_edge_roughness


def test_per_side_has_sides(analyzer_per_side):
    assert analyzer_per_side.per_side is not None
    assert len(analyzer_per_side.per_side) > 0


def test_per_side_expected_keys(analyzer_per_side):
    for side_data in analyzer_per_side.per_side.values():
        for key in ('n_pts', 'len_mm', 'Ra_um', 'Rq_um', 'Rz_um',
                    'LER_sigma_um', 'LER_3sigma_um'):
            assert key in side_data


def test_per_side_positive_roughness(analyzer_per_side):
    for side_data in analyzer_per_side.per_side.values():
        assert side_data['Ra_um'] > 0
        assert side_data['LER_sigma_um'] >= 0
        assert np.isfinite(side_data['LER_sigma_um'])
        assert side_data['LER_3sigma_um'] == pytest.approx(
            3 * side_data['LER_sigma_um'], rel=1e-6)


# ---------------------------------------------------------------------------
# Print scorecard
# ---------------------------------------------------------------------------

def test_print_scorecard_returns_self(analyzer_roughness, capsys):
    result = analyzer_roughness.print_roughness_scorecard()
    assert result is analyzer_roughness


def test_print_scorecard_output(analyzer_roughness, capsys):
    analyzer_roughness.print_roughness_scorecard()
    out = capsys.readouterr().out
    assert 'PROFILOMETER ANALYSIS SCORECARD' in out
    assert 'Sa' in out


# ---------------------------------------------------------------------------
# Save methods
# ---------------------------------------------------------------------------

def test_save_output_creates_file(analyzer_roughness, tmp_path):
    out = str(tmp_path / 'overview.png')
    analyzer_roughness.save_output(out, dpi=72)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_save_edge_analysis_creates_file(analyzer_edge_roughness, tmp_path):
    out = str(tmp_path / 'edge.png')
    analyzer_edge_roughness.save_edge_analysis(out, dpi=72)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


# ---------------------------------------------------------------------------
# ZonFile roughness_stats (unit-level)
# ---------------------------------------------------------------------------

def test_zonfile_roughness_stats_via_mock():
    mock = _MockZon(_flat_height())
    stats = mock.roughness_stats()
    assert stats['Sa'] > 0
    assert stats['Sq'] >= stats['Sa']


def test_zonfile_roughness_custom_map():
    mock = _MockZon(_flat_height())
    custom = np.ones((10, 10)) * 2.0
    stats = mock.roughness_stats(height_map=custom)
    assert stats['Sa'] == pytest.approx(0.0, abs=1e-9)
    assert stats['Sq'] == pytest.approx(0.0, abs=1e-9)
