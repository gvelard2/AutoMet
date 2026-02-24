import os
import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")

from automet.base import BaseAnalyzer
from automet.xrd import XRDAnalyzer


# ---------------------------------------------------------------------------
# Shared test fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_csv(tmp_path):
    """Write a minimal Panalytical-style CSV with a clear peak and usable baseline."""
    header = (
        "Instrument,Test Diffractometer\n"
        "Anode material,Cu\n"
        "Time per step,0.44\n"
        "Diffractometer system,XPERT-3\n"
        "[Scan points]\n"
        "Angle,Intensity\n"
    )
    rng = np.random.default_rng(0)
    angles = np.linspace(10, 80, 500)   # 500 pts, 10-80 deg; dense enough for rolling stats
    background = 500 + 2 * angles
    peak = 8000 * np.exp(-((angles - 44) ** 2) / (2 * 0.3 ** 2))
    intensities = background + peak + rng.normal(0, 10, len(angles))
    intensities = np.clip(intensities, 1, None)

    lines = [header]
    for a, i in zip(angles, intensities):
        lines.append(f"{a:.4f},{i:.2f}\n")

    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("".join(lines))
    return str(csv_file)


@pytest.fixture
def loaded(sample_csv):
    """XRDAnalyzer with data loaded."""
    return XRDAnalyzer(sample_csv).load_data()


@pytest.fixture
def preprocessed(loaded):
    """XRDAnalyzer with smoothing, polynomial baseline, and SNIP baseline applied."""
    return loaded.smooth().fit_baseline(mask_ranges=[(42, 46)]).fit_snip_baseline()


@pytest.fixture
def with_peaks(preprocessed):
    """XRDAnalyzer with peaks detected and fitted."""
    return preprocessed.detect_peaks(prominence=100).fit_peaks()


# ---------------------------------------------------------------------------
# Inheritance & loading
# ---------------------------------------------------------------------------

def test_xrd_inherits_base(sample_csv):
    assert isinstance(XRDAnalyzer(sample_csv), BaseAnalyzer)


def test_load_data_returns_self(sample_csv):
    a = XRDAnalyzer(sample_csv)
    assert a.load_data() is a


def test_load_data_populates_scan_df(loaded):
    assert loaded.scan_df is not None
    assert "Angle" in loaded.scan_df.columns
    assert "Intensity" in loaded.scan_df.columns
    assert len(loaded.scan_df) == 500


def test_load_data_parses_metadata(loaded):
    assert loaded.metadata.get("Anode material") == "Cu"
    assert loaded.metadata.get("Time per step") == "0.44"


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def test_smooth_returns_self(loaded):
    assert loaded.smooth() is loaded


def test_smooth_produces_array(loaded):
    loaded.smooth()
    assert loaded.smoothed is not None
    assert len(loaded.smoothed) == len(loaded.scan_df)


# ---------------------------------------------------------------------------
# Polynomial baseline
# ---------------------------------------------------------------------------

def test_fit_baseline_returns_self(loaded):
    loaded.smooth()
    assert loaded.fit_baseline() is loaded


def test_fit_baseline_produces_corrected(preprocessed):
    assert preprocessed.corrected is not None
    assert len(preprocessed.corrected) == len(preprocessed.scan_df)


def test_fit_baseline_stores_mask(preprocessed):
    assert preprocessed._is_baseline is not None
    assert preprocessed._mask_ranges is not None
    # mask_ranges=[(42,46)] — those points should be masked out
    two_theta = preprocessed.scan_df.Angle.values
    in_range = (two_theta >= 42) & (two_theta <= 46)
    assert not preprocessed._is_baseline[in_range].any()


# ---------------------------------------------------------------------------
# SNIP baseline
# ---------------------------------------------------------------------------

def test_fit_snip_returns_self(loaded):
    loaded.smooth()
    assert loaded.fit_snip_baseline() is loaded


def test_fit_snip_produces_arrays(preprocessed):
    assert preprocessed.baseline_snip is not None
    assert preprocessed.corrected_snip is not None
    assert len(preprocessed.baseline_snip) == len(preprocessed.scan_df)
    assert len(preprocessed.corrected_snip) == len(preprocessed.scan_df)


def test_fit_snip_baseline_is_nonnegative(preprocessed):
    # SNIP is a background estimator; it should be >= 0 for non-negative data
    assert np.all(preprocessed.baseline_snip >= 0)


# ---------------------------------------------------------------------------
# Peak detection & fitting
# ---------------------------------------------------------------------------

def test_detect_peaks_returns_self(preprocessed):
    assert preprocessed.detect_peaks(prominence=100) is preprocessed


def test_detect_peaks_finds_peak(preprocessed):
    preprocessed.detect_peaks(prominence=100)
    assert preprocessed.peaks is not None
    assert len(preprocessed.peaks) >= 1


def test_fit_peaks_returns_self(with_peaks):
    assert isinstance(with_peaks, XRDAnalyzer)


def test_pseudo_voigt_peak_at_center():
    x = np.linspace(-2, 2, 201)
    y = XRDAnalyzer.pseudo_voigt(x, x0=0, A=1, sigma=0.5, eta=0.5)
    peak_x = x[np.argmax(y)]
    assert abs(peak_x) < 0.02


# ---------------------------------------------------------------------------
# Noise analysis
# ---------------------------------------------------------------------------

def test_analyze_noise_regions_returns_self(preprocessed):
    noise_regions = [("10-20deg", 10, 20), ("25-40deg", 25, 40)]
    assert preprocessed.analyze_noise_regions(noise_regions=noise_regions) is preprocessed


def test_analyze_noise_regions_produces_df(preprocessed):
    noise_regions = [("10-20deg", 10, 20), ("25-40deg", 25, 40)]
    preprocessed.analyze_noise_regions(noise_regions=noise_regions)
    assert preprocessed.noise_df is not None
    for col in ["Region", "N points", "Mean (counts)", "Sigma / Sqrt Mu ratio", "Poisson-limited?"]:
        assert col in preprocessed.noise_df.columns


def test_analyze_noise_regions_ratio_positive(preprocessed):
    noise_regions = [("10-20deg", 10, 20)]
    preprocessed.analyze_noise_regions(noise_regions=noise_regions)
    assert preprocessed.noise_df["Sigma / Sqrt Mu ratio"].iloc[0] > 0


def test_analyze_noise_regions_skips_empty_range(preprocessed):
    # A range entirely inside the peak mask should produce no rows
    noise_regions = [("43-45deg", 43, 45)]   # masked out by fit_baseline
    preprocessed.analyze_noise_regions(noise_regions=noise_regions)
    assert len(preprocessed.noise_df) == 0


# ---------------------------------------------------------------------------
# SNR computation
# ---------------------------------------------------------------------------

def test_compute_snr_returns_self(with_peaks):
    assert with_peaks.compute_snr() is with_peaks


def test_compute_snr_produces_df(with_peaks):
    with_peaks.compute_snr()
    assert with_peaks.snr_df is not None
    for col in ["2theta (deg)", "Measured SNR", "Poisson-limit SNR", "Quality"]:
        assert col in with_peaks.snr_df.columns


def test_compute_snr_one_row_per_peak(with_peaks):
    with_peaks.compute_snr()
    assert len(with_peaks.snr_df) == len(with_peaks.peaks)


def test_compute_snr_quality_labels(with_peaks):
    with_peaks.compute_snr()
    valid = {"Reliable", "Marginal", "Below detection limit"}
    assert set(with_peaks.snr_df["Quality"]).issubset(valid)


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------

def test_print_scorecard_returns_self(with_peaks, capsys):
    result = with_peaks.compute_snr().print_scorecard()
    assert result is with_peaks


def test_print_scorecard_output(with_peaks, capsys):
    with_peaks.compute_snr().print_scorecard()
    out = capsys.readouterr().out
    assert "XRD DATA QUALITY SCORECARD" in out
    assert "Peaks detected" in out


# ---------------------------------------------------------------------------
# Save methods
# ---------------------------------------------------------------------------

def test_save_output_creates_file(with_peaks, tmp_path):
    out = str(tmp_path / "peak_fits.png")
    with_peaks.save_output(out, dpi=72)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_save_noise_distributions_creates_file(preprocessed, tmp_path):
    noise_regions = [
        ("10-20deg", 10, 20),
        ("25-35deg", 25, 35),
        ("47-57deg", 47, 57),
        ("60-70deg", 60, 70),
    ]
    out = str(tmp_path / "noise_dist.png")
    preprocessed.save_noise_distributions(out, noise_regions=noise_regions, dpi=72)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_save_snr_creates_file(with_peaks, tmp_path):
    with_peaks.compute_snr()
    out = str(tmp_path / "snr.png")
    with_peaks.save_snr(out, dpi=72)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0
