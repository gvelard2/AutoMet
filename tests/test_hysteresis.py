import os
import csv
import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")
from unittest.mock import patch, MagicMock

from automet.base import BaseAnalyzer
from automet.hysteresis import HysteresisAnalyzer


def _write_hys_file(path, fname, n_pts=100, voltage_max=4.5):
    """Write a minimal Radiant Technologies-style hysteresis .txt file.

    Format mirrors what HysteresisAnalyzer.load_data() / _var_define() expect:
    - data[0][0].split() must contain 'Hysteresis'
    - _var_define looks for exact string matches as list elements (tab-delimited)
    - _hys_spectrum looks for 'Drive Voltage' as a list element
    - voltage is at col 2, polarization at col 3 of data rows
    """
    v = np.linspace(-voltage_max, voltage_max, n_pts)
    # Synthetic polarization: tanh-based hysteresis shape
    p = 30 * np.tanh(3 * v) + np.random.default_rng(0).normal(0, 1, n_pts)

    lines = [
        "Hysteresis\t\t\t\n",
        "Points:\t" + str(n_pts) + "\t\t\n",
        "Sample Thickness (µm):\t200\t\t\n",
        "Hysteresis Period (ms):\t2\t\t\n",
        "Drive Voltage\t\t\t\n",
    ]
    for vi, pi in zip(v, p):
        lines.append(f"\t\t{vi}\t{pi}\n")

    with open(path, "w", newline="", encoding="cp1252") as f:
        f.writelines(lines)


@pytest.fixture
def hys_data_dir(tmp_path):
    """Create a temp directory with three synthetic hysteresis .txt files."""
    for i in range(3):
        fname = f"sample_{i:02d}.txt"
        _write_hys_file(tmp_path / fname, fname)
    return str(tmp_path)


def test_hysteresis_inherits_base(hys_data_dir):
    analyzer = HysteresisAnalyzer(data_dir=hys_data_dir)
    assert isinstance(analyzer, BaseAnalyzer)


def test_load_data_returns_self(hys_data_dir):
    result = HysteresisAnalyzer(data_dir=hys_data_dir).load_data()
    assert isinstance(result, HysteresisAnalyzer)


def test_load_data_finds_files(hys_data_dir):
    analyzer = HysteresisAnalyzer(data_dir=hys_data_dir).load_data()
    assert analyzer.dfA.shape[1] == 3
    assert analyzer.dfB.shape[1] == 3


def test_load_data_row_count(hys_data_dir):
    analyzer = HysteresisAnalyzer(data_dir=hys_data_dir).load_data()
    assert analyzer.dfA.shape[0] == 100


def test_normalize_returns_self(hys_data_dir):
    analyzer = HysteresisAnalyzer(data_dir=hys_data_dir).load_data()
    result = analyzer.normalize()
    assert result is analyzer


def test_normalize_produces_scaled_data(hys_data_dir):
    analyzer = HysteresisAnalyzer(data_dir=hys_data_dir).load_data().normalize()
    assert analyzer.x_scaled is not None
    assert analyzer.x_scaled.shape[0] == 3   # 3 samples (rows after transpose)


def test_run_pca_returns_self(hys_data_dir):
    analyzer = HysteresisAnalyzer(data_dir=hys_data_dir).load_data().normalize()
    result = analyzer.run_pca(n_components=2)
    assert result is analyzer


def test_run_pca_produces_components(hys_data_dir):
    analyzer = (HysteresisAnalyzer(data_dir=hys_data_dir)
                .load_data().normalize().run_pca(n_components=2))
    assert analyzer.principal_df is not None
    assert analyzer.principal_df.shape == (3, 2)


def _make_pca_analyzer(hys_data_dir):
    """Helper: load, normalize, and run PCA, mocking elbow detection to k=2."""
    return (HysteresisAnalyzer(data_dir=hys_data_dir)
            .load_data().normalize().run_pca(n_components=2))


def test_run_kmeans_returns_self(hys_data_dir):
    analyzer = _make_pca_analyzer(hys_data_dir)
    mock_viz = MagicMock()
    mock_viz.elbow_value_ = 2
    with patch("automet.hysteresis.KElbowVisualizer", return_value=mock_viz):
        result = analyzer.run_kmeans()
    assert result is analyzer


def test_run_kmeans_produces_cluster_df(hys_data_dir):
    analyzer = _make_pca_analyzer(hys_data_dir)
    mock_viz = MagicMock()
    mock_viz.elbow_value_ = 2
    with patch("automet.hysteresis.KElbowVisualizer", return_value=mock_viz):
        analyzer.run_kmeans()
    assert analyzer.df_kmeans is not None
    assert "ClusterID" in analyzer.df_kmeans.columns
    assert "FName" in analyzer.df_kmeans.columns
    assert len(analyzer.df_kmeans) == 3


def test_save_output_creates_file(hys_data_dir, tmp_path):
    analyzer = _make_pca_analyzer(hys_data_dir)
    mock_viz = MagicMock()
    mock_viz.elbow_value_ = 2
    with patch("automet.hysteresis.KElbowVisualizer", return_value=mock_viz):
        analyzer.run_kmeans()
    out = str(tmp_path / "clusters.png")
    analyzer.save_output(out, dpi=72)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0
