# AutoMet

**AutoMet** is a Python package for automated materials characterization analysis. It provides a unified, object-oriented interface for processing data from five common characterization techniques: XRD, AFM, infrared thermography, SEM, and ferroelectric hysteresis measurements.

Each analyzer follows a consistent API — `load_data()`, analysis methods, `plot_*()` methods, and `save_output()` — and can be run end-to-end with a single `.run()` call.

---

## Analyzers

| Module | Class | Instrument / Data Type | Key Output |
|---|---|---|---|
| `xrd.py` | `XRDAnalyzer` | Panalytical XRD `.csv` | Pseudo-Voigt peak fits |
| `afm.py` | `AFMAnalyzer` | Asylum Research AFM `.ibw` | 2D Power Spectral Density |
| `ir.py` | `IRAnalyzer` | IRBIS3 thermal imager `.asc` | Gradient magnitude map |
| `sem.py` | `SEMAnalyzer` | SEM image (`.png`, `.tif`, etc.) | Line-space peak width statistics |
| `hysteresis.py` | `HysteresisAnalyzer` | Radiant Technologies hysteresis `.txt` | PCA + K-Means cluster plots |

All analyzers inherit from `BaseAnalyzer` (`automet/base.py`) and share path utilities from `automet/utils.py`.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/gvelard2/AutoMet.git
cd AutoMet
```

### 2. Create and activate a conda environment

```bash
conda create -n automet_env python=3.10 -y
conda activate automet_env
```

### 3. Install the package with dependencies

```bash
pip install -e ".[dev]"
```

> The `[dev]` extra installs `pytest` for running the test suite.

### 4. (Optional) Register as a Jupyter / VS Code kernel

```bash
python -m ipykernel install --user --name automet_env --display-name "Python (automet_env)"
```

---

## Usage

### Quick start — full pipeline

```python
from automet import XRDAnalyzer, AFMAnalyzer, IRAnalyzer, SEMAnalyzer, HysteresisAnalyzer

# XRD
XRDAnalyzer("scan.csv").run()

# AFM
AFMAnalyzer("image.ibw").run()

# Infrared thermography
IRAnalyzer("frame.asc").run()

# SEM
SEMAnalyzer("sem_image.png").run()

# Ferroelectric hysteresis (point to a folder of .txt files)
HysteresisAnalyzer(data_dir="path/to/hys_data").run()
```

### Step-by-step example (XRD)

```python
from automet import XRDAnalyzer

analyzer = XRDAnalyzer("scan.csv")
analyzer.load_data()
analyzer.smooth(sigma=1)
analyzer.fit_baseline(mask_ranges=[(20, 25), (42, 48)], deg=3)
analyzer.detect_peaks(prominence=400, distance=20)
analyzer.fit_peaks()
analyzer.plot_scan()
analyzer.plot_preprocessing()
analyzer.plot_peak_fits()
analyzer.save_output("peak_fits.png")
```

All methods return `self`, so they can be chained:

```python
XRDAnalyzer("scan.csv").load_data().smooth().fit_baseline().detect_peaks().fit_peaks().run()
```

---

## Running Tests

```bash
conda activate automet_env
pytest tests/ -v
```

All 64 tests cover instantiation, `BaseAnalyzer` inheritance enforcement, method chaining, and `save_output()` behavior for each analyzer.

---

## Project Structure

```
AutoMet/
├── automet/
│   ├── __init__.py       # Package entry point; exposes all analyzers
│   ├── base.py           # Abstract BaseAnalyzer class
│   ├── utils.py          # Shared path and figure utilities
│   ├── xrd.py            # XRDAnalyzer
│   ├── afm.py            # AFMAnalyzer
│   ├── ir.py             # IRAnalyzer
│   ├── sem.py            # SEMAnalyzer
│   └── hysteresis.py     # HysteresisAnalyzer
├── tests/
│   ├── test_base.py
│   ├── test_utils.py
│   ├── test_xrd.py
│   ├── test_afm.py
│   ├── test_ir.py
│   ├── test_sem.py
│   └── test_hysteresis.py
└── pyproject.toml
```

---

## Dependencies

- `numpy`, `pandas`, `matplotlib`, `scipy`
- `scikit-image` — SEM image I/O and processing
- `scikit-learn` — StandardScaler, PCA, KMeans
- `seaborn` — cluster visualization
- `yellowbrick` — KElbowVisualizer for optimal k selection
- `igor2` — Asylum Research `.ibw` file parsing
