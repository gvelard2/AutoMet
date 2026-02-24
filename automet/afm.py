import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import igor2
from matplotlib.colors import Normalize
from matplotlib import cm
from scipy.ndimage import median_filter, gaussian_filter
from scipy.ndimage import label as nd_label
from scipy.stats import norm
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects, binary_opening, disk
from skimage.segmentation import clear_border, find_boundaries
from skimage.color import label2rgb

from automet.base import BaseAnalyzer
from automet.utils import resolve_path, save_figure


class AFMAnalyzer(BaseAnalyzer):
    """Analyzer for Asylum Research AFM .ibw files.

    Extracts height, deflection, amplitude, and phase channels.
    Computes roughness metrics (Sa, Sq) and 2D Power Spectral Density.
    Provides plane-level correction, threshold-based particle segmentation,
    per-particle feature extraction, and population statistics.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.metadata = {}
        self.height = None
        self.defl = None
        self.amp = None
        self.phase = None
        self.filtered = None
        self.waviness = None
        self.roughness = None
        self.Sa = self.Sq = self.Sa_r = self.Sq_r = None
        self.PSD = None

        # Particle segmentation state
        self.scan_size_um = None     # scan size in microns
        self.px_size_nm = None       # nm per pixel
        self.px_size_um = None       # um per pixel
        self.height_leveled = None   # plane-corrected height map (nm)
        self.height_smooth = None    # mildly smoothed for segmentation
        self.bg_mean = None          # background mean height (nm)
        self.bg_std = None           # background std dev (nm)
        self.binary_final = None     # final binary particle mask
        self.labeled_array = None    # labeled connected-component array
        self.n_particles = None      # total particle count
        self.particle_df = None      # per-particle feature DataFrame

    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------

    def load_data(self):
        """Load .ibw file and extract channel data and metadata."""
        ibw = igor2.binarywave.load(self.file_path)
        self.data = ibw['wave']['wData']

        note = ibw['wave']['note'].decode('utf-8', errors='ignore')
        for line in note.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                self.metadata[key.strip()] = value.strip()

        self.height = self.data[:, :, 0] / 1e-9
        self.defl   = self.data[:, :, 1]
        self.amp    = self.data[:, :, 2]
        self.phase  = self.data[:, :, 3]

        print("Data shape:", self.data.shape)
        return self

    def get_metadata(self):
        """Return metadata as a formatted DataFrame."""
        return pd.DataFrame(self.metadata.items(), columns=["Field", "Value"])

    # -------------------------------------------------------------------------
    # Filtering
    # -------------------------------------------------------------------------

    def apply_median_filter(self, size=3):
        """Apply a median filter to the height channel."""
        self.filtered = median_filter(self.height.astype(float), size=size)
        return self

    def apply_gaussian_filter(self, sigma=1):
        """Apply a Gaussian filter to the height channel."""
        self.filtered = gaussian_filter(self.height.astype(float), sigma=sigma)
        return self

    # -------------------------------------------------------------------------
    # Roughness Metrics
    # -------------------------------------------------------------------------

    def compute_roughness(self):
        """Compute Sa (mean absolute) and Sq (RMS) roughness on raw height."""
        Z = self.height.astype(float)
        Z_mean = np.mean(Z)
        self.Sa = np.mean(np.abs(Z - Z_mean))
        self.Sq = np.sqrt(np.mean((Z - Z_mean) ** 2))
        print(f"Sa: {self.Sa.round(4)}")
        print(f"Sq: {self.Sq.round(4)}")
        return self

    def compute_spatially_filtered_roughness(self, sigma=2.0):
        """Decompose height into waviness and roughness via Gaussian low-pass filter."""
        Z = np.nan_to_num(self.height.astype(float))
        self.waviness = gaussian_filter(Z, sigma=sigma)
        self.roughness = Z - self.waviness
        r_mean = np.mean(self.roughness)
        self.Sa_r = np.mean(np.abs(self.roughness - r_mean))
        self.Sq_r = np.sqrt(np.mean((self.roughness - r_mean) ** 2))
        print(f"Spatially Filtered Sa: {self.Sa_r.round(4)}")
        print(f"Spatially Filtered Sq: {self.Sq_r.round(4)}")
        return self

    # -------------------------------------------------------------------------
    # PSD
    # -------------------------------------------------------------------------

    def compute_psd_2d(self):
        """Compute the 2D Power Spectral Density of the waviness map."""
        Z = self.waviness.astype(float) - np.mean(self.waviness)
        self.PSD = np.abs(np.fft.fftshift(np.fft.fft2(Z))) ** 2
        return self

    # -------------------------------------------------------------------------
    # Plane-Level Correction
    # -------------------------------------------------------------------------

    def plane_level(self, scan_size_um=None):
        """Fit and subtract a least-squares plane from the height channel.

        Removes tilt artifacts from sample mounting. Sets self.height_leveled,
        self.scan_size_um, self.px_size_nm, and self.px_size_um.

        Args:
            scan_size_um: scan size in microns. If None, attempts to read from
                          metadata (ScanSize / FastScanSize / SlowScanSize keys),
                          falling back to 1.0 um.
        """
        if scan_size_um is not None:
            self.scan_size_um = float(scan_size_um)
        else:
            self.scan_size_um = 1.0
            for key in ('ScanSize', 'FastScanSize', 'SlowScanSize'):
                if key in self.metadata:
                    try:
                        self.scan_size_um = float(self.metadata[key]) * 1e6
                        break
                    except ValueError:
                        pass

        n_rows, n_cols = self.height.shape
        self.px_size_nm = (self.scan_size_um * 1000) / n_cols
        self.px_size_um = self.scan_size_um / n_cols

        X, Y = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
        A = np.column_stack([np.ones(n_rows * n_cols), X.ravel(), Y.ravel()])
        coeffs, _, _, _ = np.linalg.lstsq(A, self.height.ravel(), rcond=None)
        plane = coeffs[0] + coeffs[1] * X + coeffs[2] * Y

        self.height_leveled = self.height - plane

        print(f"Scan size: {self.scan_size_um:.3f} um  |  "
              f"Pixel size: {self.px_size_nm:.3f} nm/px")
        print(f"Plane removed -- tilt: x={coeffs[1]:.4f} nm/px, y={coeffs[2]:.4f} nm/px")
        print(f"Leveled height range: {self.height_leveled.min():.3f} to "
              f"{self.height_leveled.max():.3f} nm")
        return self

    # -------------------------------------------------------------------------
    # Particle Segmentation
    # -------------------------------------------------------------------------

    def characterize_background(self, sigma=0.8):
        """Smooth the leveled height map and characterize the substrate background.

        Estimates background mean and std dev from pixels at or below the
        median height (the flat surface, not particles). Sets self.height_smooth,
        self.bg_mean, and self.bg_std.

        Args:
            sigma: Gaussian sigma for pre-segmentation smoothing (default: 0.8).
        """
        if self.height_leveled is None:
            self.plane_level()

        self.height_smooth = gaussian_filter(self.height_leveled.astype(float), sigma=sigma)

        median_h = np.median(self.height_smooth)
        bg_px = self.height_smooth[self.height_smooth <= median_h]
        self.bg_mean = bg_px.mean()
        self.bg_std = bg_px.std()

        thresh_2sig = self.bg_mean + 2 * self.bg_std
        thresh_3sig = self.bg_mean + 3 * self.bg_std

        print(f"Background mean:  {self.bg_mean:.3f} nm")
        print(f"Background Sq:    {self.bg_std:.3f} nm")
        print(f"Threshold mu+2s:  {thresh_2sig:.3f} nm")
        print(f"Threshold mu+3s:  {thresh_3sig:.3f} nm")
        return self

    def segment_particles(self, method='Otsu', min_particle_px=5):
        """Threshold, morphologically clean, and label surface particles.

        Supported methods: 'Otsu', 'mu+2s' (mean + 2*std), 'mu+3s' (mean + 3*std).
        Edge-touching particles are removed. Sets self.binary_final,
        self.labeled_array, and self.n_particles.

        Args:
            method: thresholding strategy -- 'Otsu', 'mu+2s', or 'mu+3s' (default: 'Otsu').
            min_particle_px: minimum particle area in pixels; smaller blobs are
                             discarded as noise (default: 5).
        """
        if self.height_smooth is None:
            self.characterize_background()

        thresh_map = {
            'Otsu':  threshold_otsu(self.height_smooth),
            'mu+2s': self.bg_mean + 2 * self.bg_std,
            'mu+3s': self.bg_mean + 3 * self.bg_std,
        }
        if method not in thresh_map:
            raise ValueError(f"method must be one of {list(thresh_map.keys())}")

        thresh = thresh_map[method]
        binary = self.height_smooth > thresh
        binary = binary_opening(binary, footprint=disk(1))
        binary = remove_small_objects(binary, min_size=min_particle_px)
        self.binary_final = clear_border(binary)

        self.labeled_array, self.n_particles = nd_label(self.binary_final)
        print(f"Method: {method}  (threshold = {thresh:.3f} nm)")
        print(f"Particles detected (after edge removal): {self.n_particles}")
        return self

    def extract_particle_features(self):
        """Extract per-particle geometric and height metrics using regionprops.

        Computes equivalent diameter, max/mean height, area, aspect ratio, and
        centroid position for each labeled particle. Sets self.particle_df.
        """
        if self.labeled_array is None:
            self.segment_particles()

        props = regionprops(self.labeled_array, intensity_image=self.height_leveled)
        records = []
        for p in props:
            area_nm2 = p.area * (self.px_size_nm ** 2)
            eq_diam_nm = np.sqrt(4 * area_nm2 / np.pi)

            particle_heights = self.height_leveled[self.labeled_array == p.label]
            max_h = particle_heights.max()
            mean_h = particle_heights.mean()

            major = p.major_axis_length * self.px_size_nm
            minor = p.minor_axis_length * self.px_size_nm
            aspect = major / minor if minor > 0 else np.nan

            cy, cx = p.centroid
            records.append({
                'Particle ID':       p.label,
                'Centroid x (um)':   round(cx * self.px_size_um, 4),
                'Centroid y (um)':   round(cy * self.px_size_um, 4),
                'Area (nm2)':        round(area_nm2, 1),
                'Eq Diameter (nm)':  round(eq_diam_nm, 2),
                'Max Height (nm)':   round(max_h, 3),
                'Mean Height (nm)':  round(mean_h, 3),
                'Aspect Ratio':      round(aspect, 3),
                'Major Axis (nm)':   round(major, 2),
                'Minor Axis (nm)':   round(minor, 2),
            })

        self.particle_df = pd.DataFrame(records)
        print(f"Extracted features for {len(self.particle_df)} particles")
        print(self.particle_df[['Eq Diameter (nm)', 'Max Height (nm)',
                                 'Area (nm2)', 'Aspect Ratio']].describe().round(3))
        return self

    def print_particle_scorecard(self):
        """Print a summary scorecard of particle population statistics."""
        if self.particle_df is None:
            self.extract_particle_features()

        diams = self.particle_df['Eq Diameter (nm)']
        heights = self.particle_df['Max Height (nm)']
        scan_area_um2 = self.scan_size_um ** 2
        density = len(self.particle_df) / scan_area_um2
        m, _ = np.polyfit(diams, heights, 1)

        width = 60
        print("\n" + "=" * width)
        print("     AFM PARTICLE ANALYSIS SCORECARD")
        print("=" * width)
        summary = {
            "Image size": (f"{self.height.shape[0]}x{self.height.shape[1]} px  "
                           f"({self.scan_size_um:.3f} x {self.scan_size_um:.3f} um)"),
            "Pixel size":                f"{self.px_size_nm:.3f} nm/px",
            "Background Sq (roughness)": f"{self.bg_std:.3f} nm",
            "Particles detected":        str(self.n_particles),
            "Particle density":          f"{density:.3f} /um2",
            "Mean eq. diameter":         f"{diams.mean():.2f} +/- {diams.std():.2f} nm",
            "Median eq. diameter":       f"{diams.median():.2f} nm",
            "Mean max height":           f"{heights.mean():.3f} +/- {heights.std():.3f} nm",
            "Height-diameter slope":     f"{m:.4f} nm/nm  (linear fit)",
        }
        for k, v in summary.items():
            print(f"  {k:<32} {v}")
        print("=" * width)
        return self

    # -------------------------------------------------------------------------
    # Plotting — channels / roughness / PSD
    # -------------------------------------------------------------------------

    def plot_channels(self):
        """Plot all four raw AFM data channels."""
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for ax, channel, name in zip(axes,
                                     [self.height, self.defl, self.amp, self.phase],
                                     ['Height', 'Deflection', 'Amplitude', 'Phase']):
            ax.imshow(channel, cmap='viridis')
            ax.set_title(f"AFM {name} Map")
        plt.tight_layout()
        plt.show()

    def plot_filter_comparison(self, vmin=-2, vmax=5):
        """Plot original height map alongside the filtered result."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        im0 = axes[0].imshow(self.height, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        axes[0].set_title("Original Height Map")
        plt.colorbar(im0, ax=axes[0]).set_label('Height (nm)')
        im1 = axes[1].imshow(self.filtered, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        axes[1].set_title("Filtered Height Map")
        plt.colorbar(im1, ax=axes[1]).set_label('Height (nm)', size=14, weight='bold')
        plt.tight_layout()
        plt.show()

    def plot_roughness_decomposition(self, vmin=-2, vmax=5):
        """Plot original, waviness, and roughness components side by side."""
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        for ax, data, name in zip(axes,
                                  [self.height, self.waviness, self.roughness],
                                  ['Original', 'Waviness', 'Roughness']):
            im = ax.imshow(data, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
            ax.set_title(name)
            fig.colorbar(im, ax=ax, shrink=0.5).set_label('Height (nm)')
        plt.tight_layout()
        plt.show()

    def _build_psd_figure(self, vmin=2, vmax=6):
        """Build and return the 2D PSD figure."""
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(np.log10(self.PSD + 1e-15), cmap='inferno', vmin=vmin, vmax=vmax)
        ax.set_title("2D Power Spectral Density (log scale)")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        return fig

    def plot_psd(self, vmin=2, vmax=6):
        """Display the 2D Power Spectral Density plot."""
        fig = self._build_psd_figure(vmin, vmax)
        plt.show()

    # -------------------------------------------------------------------------
    # Plotting — particle segmentation
    # -------------------------------------------------------------------------

    def plot_plane_correction(self):
        """Plot the raw height, fitted plane, and plane-leveled result side by side."""
        if self.height_leveled is None:
            self.plane_level()
        n_rows, n_cols = self.height.shape
        X, Y = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
        A = np.column_stack([np.ones(n_rows * n_cols), X.ravel(), Y.ravel()])
        coeffs, _, _, _ = np.linalg.lstsq(A, self.height.ravel(), rcond=None)
        plane = coeffs[0] + coeffs[1] * X + coeffs[2] * Y

        extent = [0, self.scan_size_um, 0, self.scan_size_um]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, img, title in zip(axes,
                                   [self.height, plane, self.height_leveled],
                                   ['Raw Height', 'Fitted Plane', 'Plane-Leveled']):
            im = ax.imshow(img, cmap='afmhot', origin='lower', extent=extent)
            ax.set_title(title)
            ax.set_xlabel('x (um)')
            ax.set_ylabel('y (um)')
            fig.colorbar(im, ax=ax, shrink=0.8).set_label('nm')
        plt.suptitle('Plane-Level Correction', fontsize=13)
        plt.tight_layout()
        plt.show()

    def plot_height_distribution(self):
        """Plot height histogram with background Gaussian fit and threshold candidates."""
        if self.height_smooth is None:
            self.characterize_background()

        thresh_2sig = self.bg_mean + 2 * self.bg_std
        thresh_3sig = self.bg_mean + 3 * self.bg_std
        otsu_thresh = threshold_otsu(self.height_smooth)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(self.height_smooth.ravel(), bins=120, color='steelblue',
                alpha=0.7, density=True, label='All pixels')

        x_fit = np.linspace(self.height_smooth.min(), self.height_smooth.max(), 400)
        ax.plot(x_fit, norm.pdf(x_fit, self.bg_mean, self.bg_std), 'r-', lw=2,
                label=f'Background fit (mean={self.bg_mean:.2f}, s={self.bg_std:.2f} nm)')
        ax.axvline(thresh_2sig, color='orange', ls='--', lw=1.5,
                   label=f'mean+2s = {thresh_2sig:.2f} nm')
        ax.axvline(thresh_3sig, color='red', ls='--', lw=1.5,
                   label=f'mean+3s = {thresh_3sig:.2f} nm')
        ax.axvline(otsu_thresh, color='purple', ls=':', lw=1.5,
                   label=f'Otsu = {otsu_thresh:.2f} nm')

        ax.set_xlabel('Height (nm)')
        ax.set_ylabel('Probability density')
        ax.set_title('Height Distribution -- Surface vs. Particle Pixels')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_threshold_comparison(self, min_particle_px=5):
        """Plot segmentation masks side by side for three threshold methods.

        Args:
            min_particle_px: minimum particle size for small-object removal (default: 5).
        """
        if self.height_smooth is None:
            self.characterize_background()

        extent = [0, self.scan_size_um, 0, self.scan_size_um]
        methods = {
            'mean+2s': self.bg_mean + 2 * self.bg_std,
            'mean+3s': self.bg_mean + 3 * self.bg_std,
            'Otsu':    threshold_otsu(self.height_smooth),
        }

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (name, thresh) in zip(axes, methods.items()):
            binary = self.height_smooth > thresh
            binary = binary_opening(binary, footprint=disk(1))
            binary = remove_small_objects(binary, min_size=min_particle_px)
            _, n = nd_label(binary)
            ax.imshow(self.height_leveled, cmap='afmhot', origin='lower', extent=extent)
            ax.imshow(np.ma.masked_where(~binary, binary),
                      cmap='cool', alpha=0.6, origin='lower', extent=extent)
            ax.set_title(f"{name}  (thresh={thresh:.2f} nm)\n{n} particles detected")
            ax.set_xlabel('x (um)')
            ax.set_ylabel('y (um)')

        plt.suptitle('Segmentation Threshold Comparison', fontsize=13)
        plt.tight_layout()
        plt.show()

    def plot_particle_labels(self):
        """Plot the leveled height map alongside the color-coded particle label overlay."""
        if self.labeled_array is None:
            self.segment_particles()

        extent = [0, self.scan_size_um, 0, self.scan_size_um]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        axes[0].imshow(self.height_leveled, cmap='afmhot', origin='lower', extent=extent)
        axes[0].set_title('Plane-Leveled Height Map')
        axes[0].set_xlabel('x (um)')
        axes[0].set_ylabel('y (um)')
        fig.colorbar(cm.ScalarMappable(
            norm=Normalize(self.height_leveled.min(), self.height_leveled.max()),
            cmap='afmhot'), ax=axes[0], shrink=0.85).set_label('nm')

        labeled_rgb = label2rgb(self.labeled_array, image=self.height_leveled,
                                bg_label=0, alpha=0.45, kind='overlay')
        axes[1].imshow(labeled_rgb, origin='lower', extent=extent)
        axes[1].set_title(f'Particle Labels  (n = {self.n_particles})')
        axes[1].set_xlabel('x (um)')
        axes[1].set_ylabel('y (um)')

        plt.tight_layout()
        plt.show()

    def plot_population_statistics(self):
        """Plot 2x2 panel: size distribution, height distribution, scatter, spatial map."""
        if self.particle_df is None:
            self.extract_particle_features()

        diams = self.particle_df['Eq Diameter (nm)']
        heights = self.particle_df['Max Height (nm)']
        extent = [0, self.scan_size_um, 0, self.scan_size_um]

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        ax = axes[0, 0]
        ax.hist(diams, bins=20, color='steelblue', alpha=0.8, edgecolor='white')
        ax.axvline(diams.mean(), color='red', ls='--', lw=1.5,
                   label=f'Mean = {diams.mean():.1f} nm')
        ax.axvline(diams.median(), color='orange', ls='--', lw=1.5,
                   label=f'Median = {diams.median():.1f} nm')
        ax.set_xlabel('Equivalent Diameter (nm)')
        ax.set_ylabel('Count')
        ax.set_title('Particle Size Distribution')
        ax.legend()

        ax = axes[0, 1]
        ax.hist(heights, bins=20, color='tomato', alpha=0.8, edgecolor='white')
        ax.axvline(heights.mean(), color='black', ls='--', lw=1.5,
                   label=f'Mean = {heights.mean():.2f} nm')
        ax.set_xlabel('Max Particle Height (nm)')
        ax.set_ylabel('Count')
        ax.set_title('Particle Height Distribution')
        ax.legend()

        ax = axes[1, 0]
        sc = ax.scatter(diams, heights,
                        c=self.particle_df['Aspect Ratio'], cmap='plasma',
                        alpha=0.8, edgecolors='k', linewidths=0.4, s=60)
        fig.colorbar(sc, ax=ax).set_label('Aspect Ratio')
        m, b = np.polyfit(diams, heights, 1)
        x_line = np.linspace(diams.min(), diams.max(), 100)
        ax.plot(x_line, m * x_line + b, 'k--', lw=1.2, label=f'slope={m:.3f}')
        ax.set_xlabel('Equivalent Diameter (nm)')
        ax.set_ylabel('Max Height (nm)')
        ax.set_title('Height vs. Diameter  (color = aspect ratio)')
        ax.legend()

        ax = axes[1, 1]
        ax.imshow(self.height_leveled, cmap='afmhot', origin='lower',
                  extent=extent, alpha=0.85)
        sc2 = ax.scatter(self.particle_df['Centroid x (um)'],
                         self.particle_df['Centroid y (um)'],
                         c=diams, cmap='cool',
                         s=diams * 2, edgecolors='white', linewidths=0.5, alpha=0.9)
        fig.colorbar(sc2, ax=ax).set_label('Eq. Diameter (nm)')
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')
        ax.set_title('Particle Map  (size & color = diameter)')

        plt.suptitle('Particle Population Analysis', fontsize=14)
        plt.tight_layout()
        plt.show()

    def _build_particle_map_figure(self):
        """Build and return the publication-style annotated particle map figure."""
        if self.particle_df is None:
            self.extract_particle_features()

        diams = self.particle_df['Eq Diameter (nm)']
        scan_area_um2 = self.scan_size_um ** 2
        density = len(self.particle_df) / scan_area_um2
        extent = [0, self.scan_size_um, 0, self.scan_size_um]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(self.height_leveled, cmap='afmhot', origin='lower', extent=extent)

        boundary_mask = find_boundaries(self.labeled_array, mode='outer')
        boundary_rgba = np.zeros((*boundary_mask.shape, 4))
        boundary_rgba[boundary_mask] = [0.2, 0.9, 0.2, 0.9]
        ax.imshow(boundary_rgba, origin='lower', extent=extent)

        for _, row in self.particle_df.iterrows():
            ax.text(row['Centroid x (um)'], row['Centroid y (um)'],
                    str(int(row['Particle ID'])),
                    color='white', fontsize=7, ha='center', va='center',
                    fontweight='bold')

        ax.set_xlabel('x (um)', fontsize=12)
        ax.set_ylabel('y (um)', fontsize=12)
        ax.set_title(f'AFM Particle Map -- {self.n_particles} particles detected\n'
                     f'Density: {density:.2f} /um2  |  '
                     f'Mean diameter: {diams.mean():.1f} nm', fontsize=11)

        sm = cm.ScalarMappable(
            cmap='afmhot',
            norm=Normalize(self.height_leveled.min(), self.height_leveled.max()))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.85).set_label('Height (nm)', fontsize=11)

        plt.tight_layout()
        return fig

    def plot_particle_map(self):
        """Display the publication-style annotated particle map."""
        fig = self._build_particle_map_figure()
        plt.show()

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    def save_output(self, filename="psd_2d.png", dpi=150):
        """Save the 2D PSD plot to a PNG.

        Args:
            filename: output file name (default: 'psd_2d.png').
            dpi: image resolution (default: 150).
        """
        out_path = resolve_path(filename, __file__)
        save_figure(self._build_psd_figure(), out_path, dpi)

    def save_particle_map(self, filename="particle_map.png", dpi=150):
        """Save the annotated particle map to a PNG.

        Args:
            filename: output file name (default: 'particle_map.png').
            dpi: image resolution (default: 150).
        """
        out_path = resolve_path(filename, __file__)
        save_figure(self._build_particle_map_figure(), out_path, dpi)

    def save_particle_features(self, filename="particle_features.csv"):
        """Save the per-particle feature table to a CSV file.

        Args:
            filename: output file name (default: 'particle_features.csv').
        """
        if self.particle_df is None:
            self.extract_particle_features()
        out_path = resolve_path(filename, __file__)
        self.particle_df.to_csv(out_path, index=False)
        print(f"Saved particle features to: {out_path}")
        return self

    # -------------------------------------------------------------------------
    # Full Pipeline
    # -------------------------------------------------------------------------

    def run(self):
        """Execute the full analysis pipeline and display all plots."""
        self.load_data()

        # Roughness / PSD pipeline
        self.plot_channels()
        self.apply_gaussian_filter(sigma=1)
        self.plot_filter_comparison()
        self.compute_roughness()
        self.compute_spatially_filtered_roughness(sigma=2)
        self.plot_roughness_decomposition()
        self.compute_psd_2d()
        self.plot_psd()
        self.save_output("psd_2d.png")

        # Particle segmentation pipeline
        self.plane_level()
        self.plot_plane_correction()
        self.characterize_background()
        self.plot_height_distribution()
        self.plot_threshold_comparison()
        self.segment_particles()
        self.plot_particle_labels()
        self.extract_particle_features()
        self.plot_population_statistics()
        self.plot_particle_map()
        self.save_particle_map("particle_map.png")
        self.save_particle_features("particle_features.csv")
        self.print_particle_scorecard()

        return self
