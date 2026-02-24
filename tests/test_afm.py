import os
import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")
from unittest.mock import patch, MagicMock

from automet.base import BaseAnalyzer
from automet.afm import AFMAnalyzer


# ---------------------------------------------------------------------------
# Mock IBW helpers
# ---------------------------------------------------------------------------

def _make_mock_ibw(shape=(64, 64, 4)):
    """Return a minimal igor2-style dict with random channel data."""
    rng = np.random.default_rng(42)
    wave_data = rng.normal(0, 5e-9, shape)
    note = b"ScanSize:5e-6\nScanRate:1.0\n"
    return {"wave": {"wData": wave_data, "note": note}}


def _make_mock_ibw_with_particles(shape=(64, 64, 4)):
    """Return a mock IBW dict with a few clear particle-like bright spots."""
    rng = np.random.default_rng(42)
    wave_data = rng.normal(0, 5e-9, shape)
    # Add two bright particle blobs well inside the border (rows 10:15, 40:43)
    wave_data[10:15, 10:15, 0] = 50e-9   # particle 1 (5x5 px)
    wave_data[40:43, 40:43, 0] = 40e-9   # particle 2 (3x3 px)
    note = b"ScanSize:5e-6\nScanRate:1.0\n"
    return {"wave": {"wData": wave_data, "note": note}}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analyzer_loaded(tmp_path):
    """AFMAnalyzer with load_data mocked so no real .ibw file is needed."""
    fake_path = str(tmp_path / "test.ibw")
    analyzer = AFMAnalyzer(fake_path)
    with patch("automet.afm.igor2.binarywave.load", return_value=_make_mock_ibw()):
        analyzer.load_data()
    return analyzer


@pytest.fixture
def analyzer_with_particles(tmp_path):
    """AFMAnalyzer loaded with particle-containing mock data."""
    fake_path = str(tmp_path / "test.ibw")
    analyzer = AFMAnalyzer(fake_path)
    with patch("automet.afm.igor2.binarywave.load",
               return_value=_make_mock_ibw_with_particles()):
        analyzer.load_data()
    return analyzer


@pytest.fixture
def analyzer_leveled(analyzer_with_particles):
    """AFMAnalyzer after plane leveling with explicit 5 um scan size."""
    return analyzer_with_particles.plane_level(scan_size_um=5.0)


@pytest.fixture
def analyzer_characterized(analyzer_leveled):
    """AFMAnalyzer after background characterization."""
    return analyzer_leveled.characterize_background()


@pytest.fixture
def analyzer_segmented(analyzer_characterized):
    """AFMAnalyzer after particle segmentation (Otsu method)."""
    return analyzer_characterized.segment_particles(method='Otsu')


@pytest.fixture
def analyzer_featured(analyzer_segmented):
    """AFMAnalyzer after feature extraction."""
    return analyzer_segmented.extract_particle_features()


# ---------------------------------------------------------------------------
# Inheritance & loading
# ---------------------------------------------------------------------------

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
    """Height channel should be scaled from metres to nm (div 1e-9)."""
    raw_metres = _make_mock_ibw()["wave"]["wData"][:, :, 0]
    expected_nm = raw_metres / 1e-9
    np.testing.assert_allclose(analyzer_loaded.height, expected_nm)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def test_apply_median_filter_returns_self(analyzer_loaded):
    result = analyzer_loaded.apply_median_filter(size=3)
    assert result is analyzer_loaded


def test_apply_gaussian_filter_returns_self(analyzer_loaded):
    result = analyzer_loaded.apply_gaussian_filter(sigma=1)
    assert result is analyzer_loaded


# ---------------------------------------------------------------------------
# Roughness metrics
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# PSD
# ---------------------------------------------------------------------------

def test_compute_psd_2d(analyzer_loaded):
    analyzer_loaded.compute_spatially_filtered_roughness(sigma=2.0)
    analyzer_loaded.compute_psd_2d()
    assert analyzer_loaded.PSD is not None
    assert analyzer_loaded.PSD.shape == analyzer_loaded.height.shape


# ---------------------------------------------------------------------------
# Plane-level correction
# ---------------------------------------------------------------------------

def test_plane_level_returns_self(analyzer_with_particles):
    result = analyzer_with_particles.plane_level(scan_size_um=5.0)
    assert result is analyzer_with_particles


def test_plane_level_sets_attributes(analyzer_leveled):
    assert analyzer_leveled.height_leveled is not None
    assert analyzer_leveled.scan_size_um == pytest.approx(5.0)
    assert analyzer_leveled.px_size_nm > 0
    assert analyzer_leveled.px_size_um > 0
    assert analyzer_leveled.height_leveled.shape == analyzer_leveled.height.shape


def test_plane_level_reads_scan_size_from_metadata(tmp_path):
    """plane_level() should auto-detect scan size from metadata key ScanSize (in m)."""
    fake_path = str(tmp_path / "test.ibw")
    analyzer = AFMAnalyzer(fake_path)
    with patch("automet.afm.igor2.binarywave.load",
               return_value=_make_mock_ibw_with_particles()):
        analyzer.load_data()
    # metadata has ScanSize:5e-6 (meters) -> 5.0 um
    analyzer.plane_level()
    assert analyzer.scan_size_um == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Background characterization
# ---------------------------------------------------------------------------

def test_characterize_background_returns_self(analyzer_leveled):
    result = analyzer_leveled.characterize_background()
    assert result is analyzer_leveled


def test_characterize_background_sets_attributes(analyzer_characterized):
    assert analyzer_characterized.height_smooth is not None
    assert analyzer_characterized.bg_mean is not None
    assert analyzer_characterized.bg_std is not None
    assert analyzer_characterized.bg_std > 0
    assert analyzer_characterized.height_smooth.shape == analyzer_characterized.height.shape


# ---------------------------------------------------------------------------
# Particle segmentation
# ---------------------------------------------------------------------------

def test_segment_particles_returns_self(analyzer_characterized):
    result = analyzer_characterized.segment_particles(method='Otsu')
    assert result is analyzer_characterized


def test_segment_particles_finds_particles(analyzer_segmented):
    assert analyzer_segmented.binary_final is not None
    assert analyzer_segmented.labeled_array is not None
    assert analyzer_segmented.n_particles >= 1


def test_segment_particles_all_methods(analyzer_characterized):
    for method in ('Otsu', 'mu+2s', 'mu+3s'):
        analyzer_characterized.segment_particles(method=method)
        assert analyzer_characterized.n_particles >= 0  # method runs without error


def test_segment_particles_invalid_method_raises(analyzer_characterized):
    with pytest.raises(ValueError):
        analyzer_characterized.segment_particles(method='invalid')


def test_labeled_array_shape(analyzer_segmented):
    assert analyzer_segmented.labeled_array.shape == analyzer_segmented.height.shape


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def test_extract_particle_features_returns_self(analyzer_segmented):
    result = analyzer_segmented.extract_particle_features()
    assert result is analyzer_segmented


def test_extract_particle_features_columns(analyzer_featured):
    expected_cols = [
        'Particle ID', 'Centroid x (um)', 'Centroid y (um)',
        'Area (nm2)', 'Eq Diameter (nm)', 'Max Height (nm)',
        'Mean Height (nm)', 'Aspect Ratio', 'Major Axis (nm)', 'Minor Axis (nm)',
    ]
    for col in expected_cols:
        assert col in analyzer_featured.particle_df.columns


def test_extract_particle_features_row_count(analyzer_featured):
    assert len(analyzer_featured.particle_df) == analyzer_featured.n_particles


def test_extract_particle_features_positive_values(analyzer_featured):
    df = analyzer_featured.particle_df
    assert (df['Area (nm2)'] > 0).all()
    assert (df['Eq Diameter (nm)'] > 0).all()


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------

def test_print_particle_scorecard_returns_self(analyzer_featured, capsys):
    result = analyzer_featured.print_particle_scorecard()
    assert result is analyzer_featured


def test_print_particle_scorecard_output(analyzer_featured, capsys):
    analyzer_featured.print_particle_scorecard()
    out = capsys.readouterr().out
    assert "AFM PARTICLE ANALYSIS SCORECARD" in out
    assert "Particles detected" in out


# ---------------------------------------------------------------------------
# Save methods
# ---------------------------------------------------------------------------

def test_save_output_creates_file(analyzer_loaded, tmp_path):
    analyzer_loaded.compute_spatially_filtered_roughness(sigma=2.0)
    analyzer_loaded.compute_psd_2d()
    out = str(tmp_path / "psd.png")
    analyzer_loaded.save_output(out, dpi=72)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_save_particle_map_creates_file(analyzer_featured, tmp_path):
    out = str(tmp_path / "particle_map.png")
    analyzer_featured.save_particle_map(out, dpi=72)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_save_particle_features_creates_csv(analyzer_featured, tmp_path):
    out = str(tmp_path / "features.csv")
    analyzer_featured.save_particle_features(out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0
