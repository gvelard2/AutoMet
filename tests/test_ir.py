import os
import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")

from automet.base import BaseAnalyzer
from automet.ir import IRAnalyzer


def _write_asc(path, rows=30, cols=40):
    """Write a minimal IRBIS3-style .asc file with a synthetic hotspot."""
    rng = np.random.default_rng(7)
    temp = 20 + rng.normal(0, 0.5, (rows, cols))
    # inject a hotspot in the upper-left quadrant
    temp[5:10, 5:10] += 40.0

    lines = ["[Header]\n", "SomeKey=SomeValue\n", "[Data]\n"]
    for row in temp:
        lines.append(" ".join(f"{v:.3f}" for v in row) + "\n")

    with open(path, "w", encoding="cp1252") as f:
        f.writelines(lines)


@pytest.fixture
def asc_file(tmp_path):
    path = str(tmp_path / "test.asc")
    _write_asc(path, rows=30, cols=40)
    return path


def test_ir_inherits_base(asc_file):
    analyzer = IRAnalyzer(asc_file)
    assert isinstance(analyzer, BaseAnalyzer)


def test_load_data_returns_self(asc_file):
    result = IRAnalyzer(asc_file).load_data()
    assert isinstance(result, IRAnalyzer)


def test_load_data_populates_temp(asc_file):
    analyzer = IRAnalyzer(asc_file, crop_right=0).load_data()
    assert analyzer.temp is not None
    assert analyzer.temp.ndim == 2
    assert analyzer.temp.shape[0] > 0


def test_crop_right(asc_file):
    full = IRAnalyzer(asc_file, crop_right=0).load_data()
    cropped = IRAnalyzer(asc_file, crop_right=5).load_data()
    assert cropped.temp.shape[1] == full.temp.shape[1] - 5


def test_compute_half_means_returns_self(asc_file):
    analyzer = IRAnalyzer(asc_file, crop_right=0).load_data()
    result = analyzer.compute_half_means()
    assert result is analyzer


def test_find_hotspot_returns_self(asc_file):
    analyzer = IRAnalyzer(asc_file, crop_right=0).load_data()
    result = analyzer.find_hotspot(threshold=10)
    assert result is analyzer


def test_find_hotspot_centroid_in_bounds(asc_file):
    analyzer = IRAnalyzer(asc_file, crop_right=0).load_data().find_hotspot(threshold=10)
    h, w = analyzer.temp.shape
    assert 0 <= analyzer.cy < h
    assert 0 <= analyzer.cx < w


def test_compute_gradient_returns_self(asc_file):
    analyzer = IRAnalyzer(asc_file, crop_right=0).load_data()
    result = analyzer.compute_gradient()
    assert result is analyzer


def test_compute_gradient_shape(asc_file):
    analyzer = IRAnalyzer(asc_file, crop_right=0).load_data().compute_gradient()
    assert analyzer.grad_mag.shape == analyzer.temp.shape
    assert np.all(analyzer.grad_mag >= 0)


def test_save_output_creates_file(asc_file, tmp_path):
    analyzer = IRAnalyzer(asc_file, crop_right=0).load_data().compute_gradient()
    out = str(tmp_path / "gradient.png")
    analyzer.save_output(out, dpi=72)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0
