import re
import numpy as np
import matplotlib.pyplot as plt

from automet.base import BaseAnalyzer
from automet.utils import resolve_path, save_figure


class IRAnalyzer(BaseAnalyzer):
    """Analyzer for IRBIS3 infrared thermal imaging ASCII (.asc) files.

    Extracts temperature matrices, identifies hotspot centroids,
    computes gradient magnitude maps, and profiles radial temperature decay.
    """

    def __init__(self, file_path, crop_right=20):
        self.file_path = file_path
        self.crop_right = crop_right
        self.temp = None
        self.grad_mag = None
        self.mask = None
        self.cx = self.cy = None

    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------

    def load_data(self):
        """Parse an IRBIS3 ASCII .asc file into a temperature matrix."""
        data_started = False
        rows = []

        with open(self.file_path, "r", encoding="cp1252", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not data_started:
                    if line.lower() == "[data]":
                        data_started = True
                    continue
                if not line:
                    continue
                line = line.replace(",", ".")
                line = re.sub(r"[^0-9.\-+eE ]", " ", line)
                parts = line.split()
                if parts:
                    try:
                        rows.append([float(x) for x in parts])
                    except ValueError:
                        continue

        self.temp = np.array(rows, dtype=float)
        if self.crop_right > 0:
            self.temp = self.temp[:, :-self.crop_right]

        print("Temperature matrix shape:", self.temp.shape)
        return self

    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------

    def compute_half_means(self):
        """Compute and print mean temperature of the left and right halves."""
        h, w = self.temp.shape
        print(f"Left mean:  {self.temp[:, :w//2].mean():.3f} °C")
        print(f"Right mean: {self.temp[:, w//2:].mean():.3f} °C")
        return self

    def find_hotspot(self, threshold=10):
        """Identify the hotspot centroid using a threshold above ambient temperature."""
        T_ambient = np.median(self.temp)
        self.mask = self.temp > (T_ambient + threshold)
        coords = np.column_stack(np.nonzero(self.mask))
        self.cy, self.cx = coords.mean(axis=0)
        print(f"Hotspot centroid (y, x): {self.cy:.3f}, {self.cx:.3f}")
        return self

    def compute_gradient(self):
        """Compute the gradient magnitude of the temperature map."""
        gy, gx = np.gradient(self.temp)
        self.grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        return self

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------

    def plot_thermal(self):
        """Plot the raw thermal image."""
        plt.figure(figsize=(12, 10))
        plt.imshow(self.temp, cmap="viridis")
        plt.colorbar(label="Temperature (°C)", shrink=0.5)
        plt.title("Thermal Frame (ASCII Export)")
        plt.axis("off")
        plt.show()

    def plot_line_scan(self, row=None):
        """Plot a horizontal line scan across the thermal image."""
        if row is None:
            row = self.temp.shape[0] // 2
        plt.figure(figsize=(8, 4))
        plt.plot(self.temp[row, :])
        plt.title(f"Line Scan (row {row})")
        plt.xlabel("Pixel index")
        plt.ylabel("Temperature (°C)")
        plt.grid(True)
        plt.show()

    def plot_hotspot_mask(self):
        """Plot the binary hotspot mask with the centroid marked."""
        plt.figure(figsize=(12, 10))
        plt.imshow(self.mask, cmap="viridis")
        plt.scatter(self.cx, self.cy, color="red", s=80, label="Centroid")
        plt.colorbar(label="Binary Mask", shrink=0.5)
        plt.title("Hotspot Mask")
        plt.legend()
        plt.show()

    def _build_gradient_figure(self):
        """Build and return the gradient magnitude figure."""
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(self.grad_mag, cmap="viridis")
        fig.colorbar(im, ax=ax, label="|∇T| (°C/pixel)", shrink=0.7)
        ax.set_title("Gradient Magnitude Map")
        ax.axis("off")
        plt.tight_layout()
        return fig

    def plot_gradient(self):
        """Display the gradient magnitude map."""
        fig = self._build_gradient_figure()
        plt.show()

    def plot_radial_profile(self):
        """Plot the radial temperature profile centered on the hotspot centroid."""
        h, w = self.temp.shape
        r = np.sqrt((np.arange(w)[None, :] - self.cx) ** 2 +
                    (np.arange(h)[:, None] - self.cy) ** 2)
        r_flat, t_flat = r.ravel(), self.temp.ravel()
        bins = np.linspace(0, r_flat.max(), 50)
        digitized = np.digitize(r_flat, bins)
        radial_mean = np.array([t_flat[digitized == i].mean() for i in range(1, len(bins))])
        r_centers = 0.5 * (bins[:-1] + bins[1:])

        plt.figure(figsize=(6, 5))
        plt.plot(r_centers, radial_mean)
        plt.xlabel("Radius (pixels)")
        plt.ylabel("Temperature (°C)")
        plt.title("Radial Temperature Profile")
        plt.grid(True)
        plt.show()

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    def save_output(self, filename="gradient_magnitude.png", dpi=150):
        """Save the gradient magnitude map to a PNG.

        Args:
            filename: output file name (default: 'gradient_magnitude.png').
            dpi: image resolution (default: 150).
        """
        out_path = resolve_path(filename, __file__)
        save_figure(self._build_gradient_figure(), out_path, dpi)

    # -------------------------------------------------------------------------
    # Full Pipeline
    # -------------------------------------------------------------------------

    def run(self):
        """Execute the full analysis pipeline and display all plots."""
        self.load_data()
        self.plot_thermal()
        self.plot_line_scan()
        self.compute_half_means()
        self.find_hotspot()
        self.plot_hotspot_mask()
        self.compute_gradient()
        self.plot_gradient()
        self.save_output("gradient_magnitude.png")
        self.plot_radial_profile()
        return self
