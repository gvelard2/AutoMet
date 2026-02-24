import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
from scipy.signal import find_peaks, peak_widths, savgol_filter

from automet.base import BaseAnalyzer
from automet.utils import resolve_path, save_figure


class SEMAnalyzer(BaseAnalyzer):
    """Analyzer for SEM images of e-beam lithography patterns.

    Extracts line-space metrics via per-row peak width analysis
    across all image rows and reports mean ± std dev per peak.
    """

    def __init__(self, file_path, crop_bottom=500):
        self.file_path = file_path
        self.crop_bottom = crop_bottom
        self.image = None
        self.df = None
        self.df_stats = None

    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------

    def load_data(self):
        """Load and crop the SEM image."""
        self.image = io.imread(self.file_path)
        self.image = self.image[:self.crop_bottom, :]
        print(f"Image loaded — shape: {self.image.shape}, dtype: {self.image.dtype}")
        return self

    # -------------------------------------------------------------------------
    # Line Scan Helpers
    # -------------------------------------------------------------------------

    def _get_channel(self, color, row):
        """Extract a single color channel across a given image row."""
        return self.image[row, :, {'R': 0, 'G': 1, 'B': 2}[color]].astype(float)

    def _smooth_row(self, row, window_length=31, polyorder=3):
        """Apply a Savitzky-Golay filter to a single image row (R channel)."""
        return savgol_filter(self._get_channel('R', row),
                             window_length=window_length,
                             polyorder=polyorder)

    # -------------------------------------------------------------------------
    # Peak Analysis
    # -------------------------------------------------------------------------

    def compute_all_peak_widths(self, prominence=5):
        """Compute peak widths across every row of the image."""
        results = []
        for row in range(self.image.shape[0]):
            smoothed = self._smooth_row(row)
            peaks, _ = find_peaks(smoothed, prominence=prominence)
            widths = peak_widths(smoothed, peaks, rel_height=0.5)[0][1:-1]
            results.append(widths)
        self.df = pd.DataFrame(results)
        self.df.dropna(axis=1, how='any', inplace=True)
        print(f"Peak width DataFrame shape: {self.df.shape}")
        return self

    def compute_peak_stats(self):
        """Compute mean width and standard deviation for each detected peak column."""
        stats = [[self.df[col].mean(), self.df[col].std()] for col in self.df.columns]
        self.df_stats = pd.DataFrame(stats, columns=['mean_width', 'std_dev'])
        print(f"Image mean peak width: {self.df_stats.mean_width.mean():.3f} px")
        print(f"Image mean std dev:    {self.df_stats.std_dev.mean():.3f} px")
        return self

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------

    def plot_image(self):
        """Display the cropped SEM image."""
        plt.figure(figsize=(8, 8))
        plt.imshow(self.image)
        plt.title("SEM Image (cropped)")
        plt.axis("off")
        plt.show()

    def plot_line_scan(self, row=100):
        """Plot RGB channel line scans for a given image row."""
        plt.figure(figsize=(8, 4))
        for color, c in zip(['R', 'G', 'B'], ['red', 'green', 'blue']):
            plt.plot(self._get_channel(color, row), color=c, label=color)
        plt.title(f"Line Scan (row {row})")
        plt.xlabel("Pixel index")
        plt.ylabel("Intensity")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_smoothed_peaks(self, row=100):
        """Plot the smoothed line scan with detected peaks marked."""
        smoothed = self._smooth_row(row)
        peaks, _ = find_peaks(smoothed, prominence=5, height=130, distance=40)
        plt.figure(figsize=(8, 4))
        plt.plot(smoothed, color='red', label='Smoothed')
        plt.plot(peaks, smoothed[peaks], 'x', color='black', label='Peaks')
        plt.title(f"Smoothed Line Scan with Peaks (row {row})")
        plt.xlabel("Pixel index")
        plt.ylabel("Intensity")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_peak_histogram(self, peak_idx=0):
        """Plot the width distribution for a single peak column."""
        peak_data = self.df[peak_idx]
        plt.figure(figsize=(6, 4))
        plt.hist(peak_data, bins=round(np.sqrt(len(peak_data))))
        plt.title(f"Peak {peak_idx} Width Distribution")
        plt.xlabel("Peak Width (pixels)")
        plt.ylabel("Frequency")
        print(f"Peak {peak_idx} — mean: {np.mean(peak_data):.2f}, std: {np.std(peak_data):.2f}")
        plt.show()

    def _build_peak_stats_figure(self):
        """Build and return the peak stats bar chart with error bars."""
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(self.df_stats))
        ax.bar(x, self.df_stats.mean_width, yerr=self.df_stats.std_dev,
               capsize=5, color='steelblue', edgecolor='black', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Peak {i}" for i in x])
        ax.set_xlabel("Peak")
        ax.set_ylabel("Mean Width (pixels)")
        ax.set_title("Peak Width Statistics (mean ± std dev)")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        return fig

    def plot_peak_stats(self):
        """Display the peak statistics bar chart."""
        fig = self._build_peak_stats_figure()
        plt.show()

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    def save_output(self, filename="peak_stats.png", dpi=150):
        """Save the peak statistics bar chart to a PNG.

        Args:
            filename: output file name (default: 'peak_stats.png').
            dpi: image resolution (default: 150).
        """
        out_path = resolve_path(filename, __file__)
        save_figure(self._build_peak_stats_figure(), out_path, dpi)

    # -------------------------------------------------------------------------
    # Full Pipeline
    # -------------------------------------------------------------------------

    def run(self):
        """Execute the full analysis pipeline and display all plots."""
        self.load_data()
        self.plot_image()
        self.plot_line_scan()
        self.plot_smoothed_peaks()
        self.compute_all_peak_widths()
        self.compute_peak_stats()
        self.plot_peak_histogram()
        self.plot_peak_stats()
        self.save_output("peak_stats.png")
        return self
