import os
import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")

from automet.base import BaseAnalyzer
from automet.xrd import XRDAnalyzer


SAMPLE_CSV = os.path.join(os.path.dirname(__file__), "data", "sample_xrd.csv")


@pytest.fixture
def sample_csv(tmp_path):
    """Write a minimal Panalytical-style CSV for testing."""
    content = (
        "Instrument,Test Diffractometer\n"
        "Operator,Test\n"
        "Sample,TestSample\n"
        "[Scan points]\n"
        "Angle,Intensity\n"
    )
    angles = np.linspace(10, 80, 200)
    # Gaussian peak at 44 degrees on a sloping background
    background = 500 + 2 * angles
    peak = 5000 * np.exp(-((angles - 44) ** 2) / (2 * 0.3 ** 2))
    intensities = background + peak + np.random.default_rng(0).normal(0, 10, len(angles))
    intensities = np.clip(intensities, 1, None)

    lines = [content]
    for a, i in zip(angles, intensities):
        lines.append(f"{a:.4f},{i:.2f}\n")

    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("".join(lines))
    return str(csv_file)


def test_xrd_inherits_base(sample_csv):
    analyzer = XRDAnalyzer(sample_csv)
    assert isinstance(analyzer, BaseAnalyzer)


def test_load_data_returns_self(sample_csv):
    analyzer = XRDAnalyzer(sample_csv)
    result = analyzer.load_data()
    assert result is analyzer


def test_load_data_populates_scan_df(sample_csv):
    analyzer = XRDAnalyzer(sample_csv).load_data()
    assert analyzer.scan_df is not None
    assert "Angle" in analyzer.scan_df.columns
    assert "Intensity" in analyzer.scan_df.columns
    assert len(analyzer.scan_df) > 0


def test_smooth_returns_self(sample_csv):
    analyzer = XRDAnalyzer(sample_csv).load_data()
    result = analyzer.smooth()
    assert result is analyzer


def test_smooth_produces_array(sample_csv):
    analyzer = XRDAnalyzer(sample_csv).load_data().smooth()
    assert analyzer.smoothed is not None
    assert len(analyzer.smoothed) == len(analyzer.scan_df)


def test_fit_baseline_returns_self(sample_csv):
    analyzer = XRDAnalyzer(sample_csv).load_data().smooth()
    result = analyzer.fit_baseline()
    assert result is analyzer


def test_fit_baseline_produces_corrected(sample_csv):
    analyzer = XRDAnalyzer(sample_csv).load_data().smooth().fit_baseline()
    assert analyzer.corrected is not None
    assert len(analyzer.corrected) == len(analyzer.scan_df)


def test_detect_peaks_returns_self(sample_csv):
    analyzer = XRDAnalyzer(sample_csv).load_data().smooth().fit_baseline()
    result = analyzer.detect_peaks(prominence=100)
    assert result is analyzer


def test_detect_peaks_finds_peak(sample_csv):
    analyzer = (XRDAnalyzer(sample_csv)
                .load_data().smooth()
                .fit_baseline()
                .detect_peaks(prominence=100))
    assert analyzer.peaks is not None
    assert len(analyzer.peaks) >= 1


def test_fit_peaks_returns_self(sample_csv):
    analyzer = (XRDAnalyzer(sample_csv)
                .load_data().smooth()
                .fit_baseline()
                .detect_peaks(prominence=100)
                .fit_peaks())
    assert isinstance(analyzer, XRDAnalyzer)


def test_save_output_creates_file(sample_csv, tmp_path):
    analyzer = (XRDAnalyzer(sample_csv)
                .load_data().smooth()
                .fit_baseline()
                .detect_peaks(prominence=100)
                .fit_peaks())
    out = str(tmp_path / "peak_fits.png")
    analyzer.save_output(out, dpi=72)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_pseudo_voigt_peak_at_center():
    x = np.linspace(-2, 2, 201)   # odd number → x=0 is exactly in the grid
    y = XRDAnalyzer.pseudo_voigt(x, x0=0, A=1, sigma=0.5, eta=0.5)
    peak_x = x[np.argmax(y)]
    assert abs(peak_x) < 0.02     # peak should be within one grid step of x=0
