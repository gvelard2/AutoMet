import io
import re
import struct
import ctypes
import ctypes.util
import zipfile
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage
from scipy.stats import norm as _norm_dist
from skimage import filters, morphology, measure

from automet.base import BaseAnalyzer
from automet.utils import resolve_path, save_figure

# ── colour palette ──────────────────────────────────────────────────────────
_ACCENT  = '#58a6ff'
_ACCENT2 = '#ffa657'
_ACCENT3 = '#f78166'


# ── zstd decompression (ctypes binding to system libzstd) ───────────────────

def _load_libzstd():
    lib = ctypes.CDLL(ctypes.util.find_library('zstd'))
    lib.ZSTD_decompress.restype  = ctypes.c_size_t
    lib.ZSTD_decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                     ctypes.c_void_p, ctypes.c_size_t]
    lib.ZSTD_getFrameContentSize.restype  = ctypes.c_uint64
    lib.ZSTD_getFrameContentSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    return lib


def _zstd_decompress(data: bytes, lib, max_buf: int = 20 * 1024 * 1024) -> bytes:
    """Decompress a zstd-compressed byte string via libzstd ctypes binding."""
    src = (ctypes.c_uint8 * len(data))(*data)
    dst = (ctypes.c_uint8 * max_buf)()
    n   = lib.ZSTD_decompress(dst, max_buf, src, len(data))
    if n > max_buf:
        raise RuntimeError(f'Decompression output ({n} B) exceeded buffer ({max_buf} B).')
    return bytes(dst[:n])


# ─────────────────────────────────────────────────────────────────────────────
# ZonFile — Keyence VR-6000 .zon file parser
# ─────────────────────────────────────────────────────────────────────────────

class ZonFile:
    """Parser for Keyence VR-6000 .zon profilometer files.

    File layout
    -----------
    Bytes 0-3   : b'KPK1' magic
    Bytes 4-7   : BMP thumbnail size (uint32 LE)
    Bytes 8-8+N : BMP thumbnail
    Bytes 8+N-? : ZIP archive of zstd-compressed UUID-named entries

    Inside the ZIP
    --------------
    Each entry is a zstd frame. After decompression the first 16 bytes are a
    header: width(u32), height(u32), bytes_per_sample(u32), stride(u32).
    The RGBA entry (bps=4) encodes height as uint24 = R | G<<8 | B<<16.
    """

    ZUNIT  = 1e-8                       # metres per count (default)
    XYUNIT = 2.35974178415767e-5        # metres per pixel (default)

    _XML_TAGS = [
        'MeterPerPixel', 'MeterPerUnit', 'ScanDateTime',
        'LastSavedApplication', 'Profilometer', 'FileFormat',
        'Lower', 'Upper', 'Unit', 'ZOriginMeter', 'ZPositionMeter',
    ]

    def __init__(self, path: str):
        self.path = pathlib.Path(path)
        self._lib = _load_libzstd()
        self._raw = self.path.read_bytes()
        self._parse()

    # ── internal helpers ──────────────────────────────────────────────────

    def _decompress(self, name: str) -> bytes:
        return _zstd_decompress(self._zip.read(name), self._lib)

    def _parse(self):
        d = self._raw
        assert d[:4] == b'KPK1', 'Not a valid .zon file'
        bmp_size          = struct.unpack_from('<I', d, 4)[0]
        zip_start         = 8 + bmp_size
        self.thumbnail_bytes = d[8:zip_start]
        self._zip     = zipfile.ZipFile(io.BytesIO(d[zip_start:]))
        self._entries = {n: self._zip.getinfo(n).file_size
                         for n in self._zip.namelist()}
        self.metadata = self._parse_metadata()

        xy = self.metadata.get('MeterPerPixel')
        if xy:
            self.XYUNIT = float(xy)
        zu = self.metadata.get('MeterPerUnit')
        if zu:
            self.ZUNIT = float(zu)

        self.height_raw, self.w, self.h = self._load_height()

    def _parse_metadata(self) -> dict:
        meta = {}
        for name in self._entries:
            try:
                text = self._decompress(name).decode('utf-8-sig', errors='replace')
                for tag in self._XML_TAGS:
                    m = re.search(rf'<{tag}>([^<]+)</{tag}>', text)
                    if m:
                        meta[tag] = m.group(1).strip()
            except Exception:
                pass
        return meta

    def _load_height(self):
        best_name, best_size = None, 0
        for name, size in self._entries.items():
            if size > best_size:
                dec = self._decompress(name)
                hdr = struct.unpack_from('<4I', dec, 0)
                if hdr[2] == 4:          # bps == 4 -> RGBA -> height channel
                    best_name, best_size = name, size
        if best_name is None:
            raise RuntimeError('Could not locate height data entry in .zon file.')
        dec = self._decompress(best_name)
        w, h, _, stride = struct.unpack_from('<4I', dec, 0)
        raw4 = np.frombuffer(dec[16:], dtype=np.uint8).reshape(h, stride)
        rgba = raw4[:, :w * 4].reshape(h, w, 4)
        R = rgba[:, :, 0].astype(np.uint32)
        G = rgba[:, :, 1].astype(np.uint32)
        B = rgba[:, :, 2].astype(np.uint32)
        height_raw = R | (G << 8) | (B << 16)   # uint24 absolute height count
        return height_raw, w, h

    # ── public properties ─────────────────────────────────────────────────

    @property
    def height_um(self) -> np.ndarray:
        """Absolute height map in micrometres; invalid (zero-raw) pixels = NaN."""
        h = self.height_raw.astype(np.float64) * self.ZUNIT * 1e6
        h[self.height_raw == 0] = np.nan
        return h

    @property
    def height_rel_um(self) -> np.ndarray:
        """Height map relative to the scan mean, in micrometres."""
        h = self.height_um
        return h - np.nanmean(h)

    @property
    def pixel_size_um(self) -> float:
        return self.XYUNIT * 1e6

    @property
    def scan_width_mm(self) -> float:
        return self.w * self.XYUNIT * 1e3

    @property
    def scan_height_mm(self) -> float:
        return self.h * self.XYUNIT * 1e3

    def roughness_stats(self, height_map: np.ndarray = None,
                        percentile_clip: float = 1.0) -> dict:
        """Compute ISO 25178 areal roughness parameters.

        Args:
            height_map: 2D array to analyze. If None, uses the full relative height map.
            percentile_clip: percentile used to clip outliers when computing Sz (default: 1.0).

        Returns:
            dict with keys: Sa, Sq, Sz, Sp, Sv, Ssk, Sku (all in µm).
        """
        if height_map is None:
            height_map = self.height_rel_um
        v    = height_map[~np.isnan(height_map)]
        mean = np.mean(v)
        vc   = v - mean
        Sa   = np.mean(np.abs(vc))
        Sq   = np.sqrt(np.mean(vc ** 2))
        lo   = np.percentile(v, percentile_clip)
        hi   = np.percentile(v, 100 - percentile_clip)
        vc_  = v[(v >= lo) & (v <= hi)]
        Sz   = float(vc_.max() - vc_.min())
        Sp   = float(v.max() - mean)
        Sv   = float(mean - v.min())
        Ssk  = float(np.mean(vc ** 3) / Sq ** 3) if Sq > 0 else 0.0
        Sku  = float(np.mean(vc ** 4) / Sq ** 4) if Sq > 0 else 0.0
        return dict(Sa=float(Sa), Sq=float(Sq), Sz=Sz, Sp=Sp, Sv=Sv, Ssk=Ssk, Sku=Sku)


# ─────────────────────────────────────────────────────────────────────────────
# ProfilometerAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class ProfilometerAnalyzer(BaseAnalyzer):
    """Analyzer for Keyence VR-6000 optical profilometer .zon files.

    Parses height-map data, computes ISO 25178 areal roughness parameters,
    detects Si die boundaries via Otsu thresholding, and measures per-side
    line-edge roughness (LER) and edge height roughness profiles.
    """

    def __init__(self, file_path,
                 smooth_sigma: float = 3,
                 morph_open_radius: int = 5,
                 morph_close_radius: int = 10,
                 min_die_area_frac: float = 0.05,
                 edge_band_px: int = 30,
                 profile_step_px: int = 5):
        self.file_path = file_path
        # die detection parameters
        self.smooth_sigma       = smooth_sigma
        self.morph_open_radius  = morph_open_radius
        self.morph_close_radius = morph_close_radius
        self.min_die_area_frac  = min_die_area_frac
        # edge roughness parameters
        self.edge_band_px    = edge_band_px
        self.profile_step_px = profile_step_px

        # state
        self._zon            = None
        self.roughness_params = None
        self.die_mask        = None
        self.contours        = None
        self.main_contour    = None
        self.edge_result     = None
        self.per_side        = None

    # ── Data Loading ──────────────────────────────────────────────────────

    def load_data(self):
        """Load the .zon file and print scan metadata."""
        self._zon = ZonFile(self.file_path)
        z = self._zon
        print(f'File loaded: {z.path.name}')
        print(f'  Scan date  : {z.metadata.get("ScanDateTime", "n/a")}')
        print(f'  Resolution : {z.w} x {z.h} pixels')
        print(f'  XY pixel   : {z.pixel_size_um:.2f} µm/pixel')
        print(f'  Z unit     : {z.ZUNIT * 1e9:.1f} nm/count')
        print(f'  Scan area  : {z.scan_width_mm:.2f} x {z.scan_height_mm:.2f} mm')
        return self

    # ── Roughness ─────────────────────────────────────────────────────────

    def compute_roughness(self, height_map=None, percentile_clip: float = 1.0):
        """Compute ISO 25178 areal roughness parameters.

        If a die mask has been computed, applies it automatically (analyzes
        die pixels only) unless a custom height_map is supplied.

        Args:
            height_map: 2D array to analyze. If None, uses die-masked or full map.
            percentile_clip: percentile for Sz outlier clipping (default: 1.0).
        """
        if height_map is None and self.die_mask is not None:
            height_map = self._zon.height_rel_um.copy()
            height_map[~self.die_mask] = np.nan
        self.roughness_params = self._zon.roughness_stats(height_map, percentile_clip)
        print('ISO 25178 roughness parameters:')
        for k, v in self.roughness_params.items():
            print(f'  {k:4s} = {v:.4f} µm')
        return self

    # ── Die Detection ─────────────────────────────────────────────────────

    def detect_die(self):
        """Detect the Si die mask using Otsu thresholding and morphological cleanup.

        Fills NaN gaps, applies Gaussian smoothing, thresholds with Otsu,
        and removes small regions. If nothing is found, tries inverted contrast.
        Sets self.die_mask.
        """
        z = self._zon
        h = z.height_um.copy()
        nan_mask = np.isnan(h)
        h_filled = h.copy()
        h_filled[nan_mask] = np.nanmedian(h)

        h_smooth = ndimage.gaussian_filter(h_filled, sigma=self.smooth_sigma)
        thresh   = filters.threshold_otsu(h_smooth[~nan_mask])
        binary   = h_smooth > thresh
        binary[nan_mask] = False

        selem_open  = morphology.disk(self.morph_open_radius)
        selem_close = morphology.disk(self.morph_close_radius)
        cleaned = morphology.binary_opening(binary, selem_open)
        cleaned = morphology.binary_closing(cleaned, selem_close)

        min_area = self.min_die_area_frac * z.w * z.h
        labeled  = measure.label(cleaned)
        props    = measure.regionprops(labeled)
        self.die_mask = np.zeros_like(binary, dtype=bool)
        for p in props:
            if p.area >= min_area:
                self.die_mask[labeled == p.label] = True

        # fallback: try inverted contrast if nothing was found
        if not self.die_mask.any():
            binary   = ~binary
            cleaned  = morphology.binary_opening(binary, selem_open)
            cleaned  = morphology.binary_closing(cleaned, selem_close)
            labeled  = measure.label(cleaned)
            props    = measure.regionprops(labeled)
            for p in props:
                if p.area >= min_area:
                    self.die_mask[labeled == p.label] = True

        coverage = self.die_mask.sum() / self.die_mask.size * 100
        print(f'Die mask: {self.die_mask.sum():,} / {self.die_mask.size:,} '
              f'pixels ({coverage:.1f}% of scan)')
        return self

    # ── Edge Extraction ───────────────────────────────────────────────────

    def extract_edges(self):
        """Extract die perimeter contours from the die mask.

        Filters out short spurious contours and sets self.contours and
        self.main_contour (the longest one).
        """
        if self.die_mask is None:
            self.detect_die()

        contours  = measure.find_contours(self.die_mask.astype(float), level=0.5)
        min_len   = 0.01 * 2 * (self._zon.h + self._zon.w)
        self.contours = [c for c in contours if len(c) > min_len]
        self.main_contour = sorted(self.contours, key=len, reverse=True)[0]

        print(f'Found {len(self.contours)} edge contour(s)')
        for i, c in enumerate(self.contours):
            len_mm = len(c) * self._zon.pixel_size_um / 1000
            print(f'  Contour {i}: {len(c)} pts  (~{len_mm:.1f} mm perimeter)')
        return self

    # ── Edge Roughness ────────────────────────────────────────────────────

    def measure_edge_roughness(self):
        """Sample height profiles along the die edge and compute roughness metrics.

        Fits a quadratic baseline to the height profile along the perimeter and
        computes Ra, Rq, Rz of the residual. Also computes global lateral edge
        deviation (LER) as the perpendicular offset from a best-fit straight line.
        Sets self.edge_result.
        """
        if self.main_contour is None:
            self.extract_edges()

        z       = self._zon
        h_abs   = z.height_um
        px_um   = z.pixel_size_um
        rows    = self.main_contour[:, 0]
        cols    = self.main_contour[:, 1]

        idx    = np.arange(0, len(self.main_contour), self.profile_step_px)
        rows_s = rows[idx]
        cols_s = cols[idx]

        # sample height at each edge point
        edge_heights = np.array([
            h_abs[int(np.clip(round(r), 0, z.h - 1)),
                  int(np.clip(round(c), 0, z.w - 1))]
            for r, c in zip(rows_s, cols_s)
        ], dtype=float)

        arc_s  = np.arange(len(edge_heights)) * self.profile_step_px * px_um
        finite = np.isfinite(edge_heights)
        if finite.sum() > 10:
            poly     = np.polyfit(arc_s[finite], edge_heights[finite], deg=2)
            trend    = np.polyval(poly, arc_s)
            residual = edge_heights - trend
            residual[~finite] = np.nan
        else:
            residual = edge_heights - np.nanmean(edge_heights)

        res_valid = residual[np.isfinite(residual)]
        edge_Ra   = float(np.mean(np.abs(res_valid))) if len(res_valid) else np.nan
        edge_Rq   = float(np.sqrt(np.mean(res_valid ** 2))) if len(res_valid) else np.nan
        edge_Rz   = float(res_valid.max() - res_valid.min()) if len(res_valid) else np.nan

        # lateral edge deviation (perpendicular from best-fit straight line through contour)
        A = np.vstack([cols_s, np.ones_like(cols_s)]).T
        try:
            m, b   = np.linalg.lstsq(A, rows_s, rcond=None)[0]
            dev_px = (rows_s - (m * cols_s + b)) / np.sqrt(1 + m ** 2)
        except np.linalg.LinAlgError:
            dev_px = rows_s - rows_s.mean()
        dev_um = dev_px * px_um

        self.edge_result = dict(
            edge_Ra=edge_Ra, edge_Rq=edge_Rq, edge_Rz=edge_Rz,
            edge_heights=edge_heights, residual=residual, arc_s=arc_s,
            edge_dev_um=dev_um,
            contour_mm=np.column_stack([cols * px_um / 1000, rows * px_um / 1000]),
        )
        print(f'Edge  Ra = {edge_Ra:.3f} µm  |  '
              f'Rq = {edge_Rq:.3f} µm  |  Rz = {edge_Rz:.3f} µm')
        return self

    # ── Per-Side Analysis ─────────────────────────────────────────────────

    def analyse_per_side(self, band_frac: float = 0.02, trim_frac: float = 0.05):
        """Compute edge roughness independently for each side of the rectangular die.

        Splits the main contour into Top / Bottom / Left / Right segments using
        a bounding-box fraction, fits an axis-aligned best-fit line to each side,
        and computes Ra, Rq, Rz (height profile) and LER σ / 3σ (lateral deviation).
        Sets self.per_side.

        Args:
            band_frac: fraction of bounding-box dimension used to classify each side (default: 0.02).
            trim_frac: fraction trimmed from each end of each side to remove corner overlap (default: 0.05).
        """
        if self.main_contour is None:
            self.extract_edges()

        z      = self._zon
        px_mm  = z.XYUNIT * 1e3
        rows_c = self.main_contour[:, 0]
        cols_c = self.main_contour[:, 1]

        r_min, r_max = rows_c.min(), rows_c.max()
        c_min, c_max = cols_c.min(), cols_c.max()
        r_range = r_max - r_min
        c_range = c_max - c_min

        sides = {
            'Top':    rows_c < r_min + band_frac * r_range,
            'Bottom': rows_c > r_max - band_frac * r_range,
            'Left':   cols_c < c_min + band_frac * c_range,
            'Right':  cols_c > c_max - band_frac * c_range,
        }

        results = {}
        for side_name, mask_s in sides.items():
            side_rows = rows_c[mask_s]
            side_cols = cols_c[mask_s]
            if len(side_rows) < 20:
                continue

            x_span = (side_cols.max() - side_cols.min()) * px_mm
            y_span = (side_rows.max() - side_rows.min()) * px_mm

            if x_span >= y_span:
                # horizontal edge (Top / Bottom) — fit Y = mX + b
                sort_idx  = np.argsort(side_cols)
                side_rows = side_rows[sort_idx]; side_cols = side_cols[sort_idx]
                n_trim    = int(len(side_cols) * trim_frac)
                if n_trim > 0:
                    side_rows = side_rows[n_trim:-n_trim]
                    side_cols = side_cols[n_trim:-n_trim]
                sx_mm = side_cols * px_mm; sy_mm = side_rows * px_mm
                A = np.vstack([sx_mm, np.ones_like(sx_mm)]).T
                m, b  = np.linalg.lstsq(A, sy_mm, rcond=None)[0]
                dev_mm = (sy_mm - m * sx_mm - b) / np.sqrt(m ** 2 + 1)
            else:
                # vertical edge (Left / Right) — fit X = mY + b
                sort_idx  = np.argsort(side_rows)
                side_rows = side_rows[sort_idx]; side_cols = side_cols[sort_idx]
                n_trim    = int(len(side_rows) * trim_frac)
                if n_trim > 0:
                    side_rows = side_rows[n_trim:-n_trim]
                    side_cols = side_cols[n_trim:-n_trim]
                sx_mm = side_cols * px_mm; sy_mm = side_rows * px_mm
                A = np.vstack([sy_mm, np.ones_like(sy_mm)]).T
                m, b  = np.linalg.lstsq(A, sx_mm, rcond=None)[0]
                dev_mm = (sx_mm - m * sy_mm - b) / np.sqrt(m ** 2 + 1)

            dev_um = dev_mm * 1000

            # height profile roughness along the side
            h_abs  = z.height_um
            arc_s  = np.arange(len(side_rows)) * px_mm * 1000
            heights = np.array([
                h_abs[int(np.clip(round(r), 0, z.h - 1)),
                      int(np.clip(round(c), 0, z.w - 1))]
                for r, c in zip(side_rows, side_cols)
            ], dtype=float)

            finite = np.isfinite(heights)
            if finite.sum() > 10:
                poly     = np.polyfit(arc_s[finite], heights[finite], deg=2)
                trend    = np.polyval(poly, arc_s)
                residual = heights - trend
                residual[~finite] = np.nan
            else:
                residual = heights - np.nanmean(heights)

            res_v  = residual[np.isfinite(residual)]
            Ra = float(np.mean(np.abs(res_v))) if len(res_v) else np.nan
            Rq = float(np.sqrt(np.mean(res_v ** 2))) if len(res_v) else np.nan
            Rz = float(res_v.max() - res_v.min()) if len(res_v) else np.nan

            _, sig = _norm_dist.fit(dev_um)
            len_mm = np.sum(np.sqrt(
                np.diff(sx_mm, prepend=sx_mm[0]) ** 2 +
                np.diff(sy_mm, prepend=sy_mm[0]) ** 2))

            results[side_name] = {
                'n_pts':         len(side_rows),
                'len_mm':        round(len_mm, 4),
                'Ra_um':         round(Ra, 4),
                'Rq_um':         round(Rq, 4),
                'Rz_um':         round(Rz, 4),
                'LER_sigma_um':  round(sig, 4),
                'LER_3sigma_um': round(3 * sig, 4),
            }

        self.per_side = results
        df = (pd.DataFrame(results).T
                .reset_index()
                .rename(columns={'index': 'Side'}))
        print(df.to_string(index=False))
        return self

    # ── Plotting ──────────────────────────────────────────────────────────

    def _build_overview_figure(self):
        """Build and return the 4-panel height-map overview figure."""
        z     = self._zon
        h_rel = z.height_rel_um
        stats = self.roughness_params or z.roughness_stats()
        W_mm  = z.scan_width_mm
        H_mm  = z.scan_height_mm

        fig = plt.figure(figsize=(8, 6))
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.55)
        ax_map   = fig.add_subplot(gs[0, 0])
        ax_hist  = fig.add_subplot(gs[0, 1])
        ax_profx = fig.add_subplot(gs[1, 0])
        ax_profy = fig.add_subplot(gs[1, 1])

        fig.suptitle(
            f'VR6000 Optical Profilometer — Height Map Analysis\n'
            f'File: {z.path.name}  ·  {z.metadata.get("ScanDateTime", "")}',
            fontsize=10, fontweight='bold', y=0.98)

        extent = [0, W_mm, H_mm, 0]
        vmin   = np.nanpercentile(h_rel, 2)
        vmax   = np.nanpercentile(h_rel, 98)

        im = ax_map.imshow(h_rel, cmap='RdYlBu_r', aspect='equal',
                           extent=extent, origin='upper', vmin=vmin, vmax=vmax)
        ax_map.set_aspect('equal', adjustable='box', anchor='W')
        ax_map.set_title('Surface Height Map (relative to mean)', fontsize=8)
        ax_map.set_xlabel('X (mm)'); ax_map.set_ylabel('Y (mm)')
        divider = make_axes_locatable(ax_map)
        cax = divider.append_axes('right', size='3%', pad=0.05)
        plt.colorbar(im, cax=cax).set_label('Height (µm)')
        ax_map.text(0.02, 0.97,
                    f'Sa={stats["Sa"]:.1f} µm   Sq={stats["Sq"]:.1f} µm\n'
                    f'Sz={stats["Sz"]:.1f} µm   {z.w}×{z.h} px',
                    transform=ax_map.transAxes, va='top', fontsize=6, color='white',
                    bbox=dict(facecolor='black', alpha=0.7, boxstyle='round'))

        valid = h_rel[~np.isnan(h_rel)]
        lo = np.percentile(valid, 0.1); hi = np.percentile(valid, 99.9)
        ax_hist.hist(valid[(valid >= lo) & (valid <= hi)],
                     bins=250, color=_ACCENT, edgecolor='none', alpha=0.85)
        ax_hist.axvline(0, color=_ACCENT3, lw=1.5, ls='--', label='Mean')
        ax_hist.axvline( stats['Sa'], color=_ACCENT2, lw=1.2, ls=':',
                         label=f"±Sa={stats['Sa']:.0f} µm")
        ax_hist.axvline(-stats['Sa'], color=_ACCENT2, lw=1.2, ls=':')
        ax_hist.set_title('Height Distribution', fontsize=8)
        ax_hist.set_xlabel('Relative Height (µm)'); ax_hist.set_ylabel('Count')
        ax_hist.legend(fontsize=8); ax_hist.grid(True)

        x_ax  = np.linspace(0, W_mm, z.w)
        mid_y = z.h // 2
        px_   = h_rel[mid_y, :]; vx = ~np.isnan(px_)
        ax_profx.fill_between(x_ax[vx], px_[vx], alpha=0.25, color=_ACCENT)
        ax_profx.plot(x_ax[vx], px_[vx], lw=0.7, color=_ACCENT)
        ax_profx.axhline(0, color=_ACCENT3, lw=1, ls='--')
        ax_profx.set_title(f'X Cross-Section  (Y = {H_mm / 2:.1f} mm)', fontsize=8)
        ax_profx.set_xlabel('X (mm)'); ax_profx.set_ylabel('Height (µm)')
        ax_profx.grid(True); ax_profx.xaxis.set_minor_locator(AutoMinorLocator())

        y_ax  = np.linspace(0, H_mm, z.h)
        mid_x = z.w // 2
        py_   = h_rel[:, mid_x]; vy = ~np.isnan(py_)
        ax_profy.fill_between(y_ax[vy], py_[vy], alpha=0.25, color=_ACCENT2)
        ax_profy.plot(y_ax[vy], py_[vy], lw=0.7, color=_ACCENT2)
        ax_profy.axhline(0, color=_ACCENT3, lw=1, ls='--')
        ax_profy.set_title(f'Y Cross-Section  (X = {W_mm / 2:.1f} mm)', fontsize=8)
        ax_profy.set_xlabel('Y (mm)'); ax_profy.set_ylabel('Height (µm)')
        ax_profy.grid(True); ax_profy.xaxis.set_minor_locator(AutoMinorLocator())

        plt.tight_layout()
        return fig

    def plot_overview(self):
        """Display the 4-panel height-map overview."""
        fig = self._build_overview_figure()
        plt.show()

    def _build_edge_analysis_figure(self):
        """Build and return the 5-panel Si die edge roughness figure."""
        z     = self._zon
        h_rel = z.height_rel_um
        W_mm  = z.scan_width_mm
        H_mm  = z.scan_height_mm
        px_um = z.pixel_size_um

        fig = plt.figure(figsize=(20, 14))
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.38)
        ax_hmap  = fig.add_subplot(gs[0, 0])
        ax_mask  = fig.add_subplot(gs[0, 1])
        ax_edge  = fig.add_subplot(gs[0, 2])
        ax_hprof = fig.add_subplot(gs[1, :2])
        ax_ler   = fig.add_subplot(gs[1, 2])

        fig.suptitle(f'Si Die Edge Roughness Analysis — {z.path.name}',
                     fontsize=14, fontweight='bold', y=0.99)

        extent = [0, W_mm, H_mm, 0]
        vmin   = np.nanpercentile(h_rel, 2); vmax = np.nanpercentile(h_rel, 98)

        ax_hmap.imshow(h_rel, cmap='RdYlBu_r', aspect='auto',
                       extent=extent, origin='upper', vmin=vmin, vmax=vmax)
        for c in self.contours:
            ax_hmap.plot(c[:, 1] * px_um / 1000, c[:, 0] * px_um / 1000,
                         lw=1.2, color='lime', alpha=0.85)
        ax_hmap.set_title('(A) Height Map + Detected Edge')
        ax_hmap.set_xlabel('X (mm)'); ax_hmap.set_ylabel('Y (mm)')

        overlay = np.zeros((*self.die_mask.shape, 4))
        overlay[self.die_mask]  = [0.35, 0.65, 1.0, 0.55]
        overlay[~self.die_mask] = [0.05, 0.05, 0.05, 0.80]
        ax_mask.imshow(h_rel, cmap='gray', aspect='auto',
                       extent=extent, origin='upper', vmin=vmin, vmax=vmax, alpha=0.6)
        ax_mask.imshow(overlay, aspect='auto', extent=extent, origin='upper')
        for c in self.contours:
            ax_mask.plot(c[:, 1] * px_um / 1000, c[:, 0] * px_um / 1000,
                         lw=1.5, color='lime', alpha=0.9)
        ax_mask.set_title('(B) Die Mask (blue) + Edge Contour')
        ax_mask.set_xlabel('X (mm)'); ax_mask.set_ylabel('Y (mm)')

        dev     = self.edge_result['edge_dev_um']
        arc_s_s = np.arange(len(dev)) * self.profile_step_px * px_um
        ax_edge.scatter(arc_s_s / 1000, dev, s=1.5, c=_ACCENT, alpha=0.6)
        ax_edge.axhline(0,          color=_ACCENT3, lw=1.2, ls='--')
        ax_edge.axhline( dev.std(), color=_ACCENT2, lw=1, ls=':',
                         label=f'±1σ = {dev.std():.1f} µm')
        ax_edge.axhline(-dev.std(), color=_ACCENT2, lw=1, ls=':')
        ax_edge.set_title('(C) Lateral Edge Deviation (LER)')
        ax_edge.set_xlabel('Position along edge (mm)')
        ax_edge.set_ylabel('Lateral deviation (µm)')
        ax_edge.legend(fontsize=8); ax_edge.grid(True)

        arc_s  = self.edge_result['arc_s']
        resid  = self.edge_result['residual']
        fin    = np.isfinite(resid)
        Ra = self.edge_result['edge_Ra']
        Rq = self.edge_result['edge_Rq']
        Rz = self.edge_result['edge_Rz']
        ax_hprof.fill_between(arc_s[fin] / 1000, resid[fin], alpha=0.25, color='#3fb950')
        ax_hprof.plot(arc_s[fin] / 1000, resid[fin], lw=0.8, color='#3fb950')
        ax_hprof.axhline(0,   color=_ACCENT3, lw=1, ls='--')
        ax_hprof.axhline( Ra, color=_ACCENT2, lw=1, ls=':', label=f'±Ra = {Ra:.2f} µm')
        ax_hprof.axhline(-Ra, color=_ACCENT2, lw=1, ls=':')
        ax_hprof.set_title(f'(D) Edge Height Profile (detrended)  '
                           f'Ra={Ra:.2f} µm  Rq={Rq:.2f} µm  Rz={Rz:.2f} µm')
        ax_hprof.set_xlabel('Position along perimeter (mm)')
        ax_hprof.set_ylabel('Height residual (µm)')
        ax_hprof.legend(fontsize=9); ax_hprof.grid(True)

        mu, sigma = _norm_dist.fit(dev)
        xs = np.linspace(dev.min(), dev.max(), 200)
        ax_ler.hist(dev, bins=60, color=_ACCENT, edgecolor='none', alpha=0.85, density=True)
        ax_ler.plot(xs, _norm_dist.pdf(xs, mu, sigma), color=_ACCENT3, lw=2,
                    label=f'N({mu:.1f}, {sigma:.1f})')
        ax_ler.set_title(f'(E) LER Distribution  σ = {sigma:.2f} µm')
        ax_ler.set_xlabel('Lateral deviation (µm)'); ax_ler.set_ylabel('Density')
        ax_ler.legend(fontsize=9); ax_ler.grid(True)

        plt.tight_layout()
        return fig

    def plot_edge_analysis(self):
        """Display the 5-panel Si die edge roughness analysis figure."""
        fig = self._build_edge_analysis_figure()
        plt.show()

    def plot_per_side_roughness(self):
        """Display a grouped bar chart comparing roughness metrics across die sides."""
        if self.per_side is None:
            self.analyse_per_side()

        sides     = list(self.per_side.keys())
        Ra_vals   = [self.per_side[s]['Ra_um']         for s in sides]
        Rq_vals   = [self.per_side[s]['Rq_um']         for s in sides]
        LER_vals  = [self.per_side[s]['LER_sigma_um']  for s in sides]
        LER3_vals = [self.per_side[s]['LER_3sigma_um'] for s in sides]

        x = np.arange(len(sides)); w = 0.20
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - 1.5*w, Ra_vals,   w, label='Ra (height)',      color=_ACCENT,   alpha=0.85)
        ax.bar(x - 0.5*w, Rq_vals,   w, label='Rq (height)',      color=_ACCENT2,  alpha=0.85)
        ax.bar(x + 0.5*w, LER_vals,  w, label='LER σ (lateral)',  color='#3fb950', alpha=0.85)
        ax.bar(x + 1.5*w, LER3_vals, w, label='LER 3σ (lateral)', color='#f778ba', alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(sides)
        ax.set_ylabel('Roughness (µm)')
        ax.set_title('Per-Side Edge Roughness Comparison', fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, axis='y')
        plt.tight_layout()
        plt.show()

    def roi_analysis(self, x0_mm: float, x1_mm: float,
                     y0_mm: float, y1_mm: float,
                     title: str = 'ROI') -> dict:
        """Crop scan to a sub-region, display height map and distribution, return roughness stats.

        Args:
            x0_mm, x1_mm: X bounds in mm.
            y0_mm, y1_mm: Y bounds in mm.
            title: label for plot title.

        Returns:
            dict of ISO 25178 roughness parameters for the ROI.
        """
        z  = self._zon
        px = z.XYUNIT * 1e3   # mm per pixel
        c0 = int(x0_mm / px); c1 = int(x1_mm / px)
        r0 = int(y0_mm / px); r1 = int(y1_mm / px)
        h_roi = z.height_rel_um[r0:r1, c0:c1]
        stats = z.roughness_stats(h_roi)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        extent = [x0_mm, x1_mm, y1_mm, y0_mm]
        vmin = np.nanpercentile(h_roi, 2); vmax = np.nanpercentile(h_roi, 98)
        im = axes[0].imshow(h_roi, cmap='RdYlBu_r', aspect='auto',
                            extent=extent, origin='upper', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'{title} — Height Map')
        axes[0].set_xlabel('X (mm)'); axes[0].set_ylabel('Y (mm)')
        plt.colorbar(im, ax=axes[0]).set_label('Height (µm)')

        valid = h_roi[~np.isnan(h_roi)]
        axes[1].hist(valid, bins=150, color=_ACCENT, edgecolor='none', alpha=0.85)
        axes[1].axvline(0, color=_ACCENT3, lw=1.5, ls='--', label='Mean')
        axes[1].set_title(f'{title} — Height Distribution')
        axes[1].set_xlabel('Relative Height (µm)'); axes[1].set_ylabel('Count')
        axes[1].legend(fontsize=8); axes[1].grid(True)

        fig.suptitle(
            f'{title}  ({x0_mm:.1f}–{x1_mm:.1f} mm × {y0_mm:.1f}–{y1_mm:.1f} mm)',
            fontsize=11, y=1.02)
        plt.tight_layout()
        plt.show()

        print(f'Roughness in {title}:')
        for k, v in stats.items():
            print(f'  {k:4s} = {v:9.3f} µm')
        return stats

    # ── Scorecard ─────────────────────────────────────────────────────────

    def print_roughness_scorecard(self):
        """Print a formatted summary of all computed profilometer metrics."""
        if self.roughness_params is None:
            self.compute_roughness()
        z = self._zon
        width = 62
        print('\n' + '=' * width)
        print('     PROFILOMETER ANALYSIS SCORECARD')
        print('=' * width)
        summary = {
            'File':            z.path.name,
            'Scan date':       z.metadata.get('ScanDateTime', 'n/a'),
            'Resolution':      f'{z.w} x {z.h} px',
            'Scan area':       f'{z.scan_width_mm:.2f} x {z.scan_height_mm:.2f} mm',
            'Pixel size':      f'{z.pixel_size_um:.2f} µm/px',
        }
        for k, v in self.roughness_params.items():
            summary[f'  {k}'] = f'{v:.4f} µm'
        if self.edge_result:
            summary['Edge Ra'] = f'{self.edge_result["edge_Ra"]:.4f} µm'
            summary['Edge Rq'] = f'{self.edge_result["edge_Rq"]:.4f} µm'
            summary['Edge Rz'] = f'{self.edge_result["edge_Rz"]:.4f} µm'
            _, sig = _norm_dist.fit(self.edge_result['edge_dev_um'])
            summary['LER σ']  = f'{sig:.4f} µm'
            summary['LER 3σ'] = f'{3 * sig:.4f} µm'
        for k, v in summary.items():
            print(f'  {k:<32} {v}')
        print('=' * width)
        return self

    # ── Save ──────────────────────────────────────────────────────────────

    def save_output(self, filename='profilometer_overview.png', dpi=150):
        """Save the overview figure to a PNG file.

        Args:
            filename: output file name (default: 'profilometer_overview.png').
            dpi: image resolution (default: 150).
        """
        out_path = resolve_path(filename, __file__)
        save_figure(self._build_overview_figure(), out_path, dpi)

    def save_edge_analysis(self, filename='profilometer_edge_analysis.png', dpi=150):
        """Save the edge analysis figure to a PNG file.

        Args:
            filename: output file name (default: 'profilometer_edge_analysis.png').
            dpi: image resolution (default: 150).
        """
        out_path = resolve_path(filename, __file__)
        save_figure(self._build_edge_analysis_figure(), out_path, dpi)

    # ── Full Pipeline ─────────────────────────────────────────────────────

    def run(self):
        """Execute the full analysis pipeline and display all plots."""
        self.load_data()
        self.compute_roughness()
        self.plot_overview()
        self.save_output('profilometer_overview.png')

        self.detect_die()
        self.compute_roughness()        # recompute on die pixels only
        self.extract_edges()
        self.measure_edge_roughness()
        self.plot_edge_analysis()
        self.save_edge_analysis('profilometer_edge_analysis.png')
        self.analyse_per_side()
        self.plot_per_side_roughness()
        self.print_roughness_scorecard()

        return self
