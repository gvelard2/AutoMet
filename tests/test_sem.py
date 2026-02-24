import os
import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")
from skimage import io as skio

from automet.base import BaseAnalyzer
from automet.sem import SEMAnalyzer


def _write_sem_image(path, height=200, width=300):
    """Write a synthetic SEM-like RGB image with periodic bright lines."""
    rng = np.random.default_rng(13)
    img = np.full((height, width, 3), 30, dtype=np.uint8)
    # Add 4 bright vertical bands (lines) that peak detection should find
    for center in [60, 110, 160, 210, 260]:
        if center < width:
            left = max(0, center - 8)
            right = min(width, center + 8)
            img[:, left:right, :] = 200
    img = img + rng.integers(0, 10, img.shape, dtype=np.uint8)
    skio.imsave(path, img)


@pytest.fixture
def sem_image(tmp_path):
    path = str(tmp_path / "test_sem.png")
    _write_sem_image(path)
    return path


def test_sem_inherits_base(sem_image):
    analyzer = SEMAnalyzer(sem_image)
    assert isinstance(analyzer, BaseAnalyzer)


def test_load_data_returns_self(sem_image):
    result = SEMAnalyzer(sem_image).load_data()
    assert isinstance(result, SEMAnalyzer)


def test_load_data_populates_image(sem_image):
    analyzer = SEMAnalyzer(sem_image, crop_bottom=200).load_data()
    assert analyzer.image is not None
    assert analyzer.image.ndim == 3
    assert analyzer.image.shape[0] == 200


def test_crop_bottom(sem_image):
    analyzer = SEMAnalyzer(sem_image, crop_bottom=100).load_data()
    assert analyzer.image.shape[0] == 100


def test_get_channel_shape(sem_image):
    analyzer = SEMAnalyzer(sem_image, crop_bottom=200).load_data()
    for color in ["R", "G", "B"]:
        ch = analyzer._get_channel(color, row=10)
        assert ch.shape == (analyzer.image.shape[1],)


def test_smooth_row_returns_array(sem_image):
    analyzer = SEMAnalyzer(sem_image, crop_bottom=200).load_data()
    smoothed = analyzer._smooth_row(row=10)
    assert len(smoothed) == analyzer.image.shape[1]


def test_compute_all_peak_widths_returns_self(sem_image):
    analyzer = SEMAnalyzer(sem_image, crop_bottom=200).load_data()
    result = analyzer.compute_all_peak_widths(prominence=30)
    assert result is analyzer


def test_compute_all_peak_widths_produces_df(sem_image):
    analyzer = SEMAnalyzer(sem_image, crop_bottom=200).load_data()
    analyzer.compute_all_peak_widths(prominence=30)
    assert analyzer.df is not None
    assert analyzer.df.shape[0] == analyzer.image.shape[0]


def test_compute_peak_stats_returns_self(sem_image):
    analyzer = SEMAnalyzer(sem_image, crop_bottom=200).load_data()
    analyzer.compute_all_peak_widths(prominence=30)
    result = analyzer.compute_peak_stats()
    assert result is analyzer


def test_compute_peak_stats_columns(sem_image):
    analyzer = (SEMAnalyzer(sem_image, crop_bottom=200)
                .load_data()
                .compute_all_peak_widths(prominence=30)
                .compute_peak_stats())
    assert "mean_width" in analyzer.df_stats.columns
    assert "std_dev" in analyzer.df_stats.columns
    assert len(analyzer.df_stats) > 0


def test_save_output_creates_file(sem_image, tmp_path):
    analyzer = (SEMAnalyzer(sem_image, crop_bottom=200)
                .load_data()
                .compute_all_peak_widths(prominence=30)
                .compute_peak_stats())
    out = str(tmp_path / "peak_stats.png")
    analyzer.save_output(out, dpi=72)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0
