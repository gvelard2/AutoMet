import os
import csv
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

from automet.base import BaseAnalyzer
from automet.utils import resolve_path, save_figure


class HysteresisAnalyzer(BaseAnalyzer):
    """Analyzer for Radiant Technologies ferroelectric hysteresis loop data.

    Applies PCA dimensionality reduction and K-Means clustering to
    identify groups of similar switching behavior across samples.
    """

    def __init__(self, data_dir=None, m_type='Hysteresis', freq_period=2, max_voltage=5):
        """
        Args:
            data_dir: path to folder containing .txt hysteresis files.
                      Defaults to 'SampleHysData' next to this module.
            m_type: measurement type filter (default: 'Hysteresis').
            freq_period: hysteresis period in ms (default: 2 = 1kHz).
            max_voltage: maximum drive voltage filter in V (default: 5).
        """
        if data_dir is None:
            data_dir = resolve_path("SampleHysData", __file__)
        self.data_dir = data_dir
        self.m_type = m_type
        self.freq_period = freq_period
        self.max_voltage = max_voltage

        self.dfA = pd.DataFrame()
        self.dfB = pd.DataFrame()
        self.x_scaled = None
        self.principal_df = None
        self.pca = None
        self.df_kmeans = None
        self.n_clusters = None

    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------

    @staticmethod
    def _var_define(data, vname, col):
        """Extract a scalar value from a named row in parsed file data."""
        for line in data:
            if vname in line and line[col]:
                return float(line[col])

    @staticmethod
    def _hys_spectrum(data, no_pts, s_thick):
        """Extract voltage and polarization arrays from parsed file data."""
        v_start, unit = None, 1
        for i, line in enumerate(data):
            if 'Drive Voltage' in line:
                v_start, unit = i + 1, 1
            elif 'Field (kV/cm)' in line:
                v_start, unit = i + 1, s_thick / 10

        voltage, polarization = [], []
        if v_start is not None and v_start < 60:
            for j in range(int(no_pts)):
                voltage.append(float(data[v_start + j][2]) * unit)
                polarization.append(float(data[v_start + j][3]))
        return voltage, polarization

    def load_data(self):
        """Scan data_dir for .txt files and load matching hysteresis measurements."""
        txt_files = glob.glob(os.path.join(self.data_dir, "**", "*.txt"), recursive=True)

        for filepath in txt_files:
            with open(filepath, encoding="cp1252", errors="ignore") as f:
                data = list(csv.reader(f, delimiter="\t"))

            if self.m_type not in data[0][0].split():
                continue

            no_pts  = self._var_define(data, 'Points:', 1)
            s_thick = self._var_define(data, 'Sample Thickness (µm):', 1)
            freq    = self._var_define(data, 'Hysteresis Period (ms):', 1)
            voltage, polarization = self._hys_spectrum(data, no_pts, s_thick)

            if voltage and freq == self.freq_period and max(voltage) < self.max_voltage:
                fname = os.path.basename(filepath)
                self.dfA = pd.concat([self.dfA, pd.DataFrame({fname: polarization})], axis=1)
                self.dfB = pd.concat([self.dfB, pd.DataFrame({fname: voltage})], axis=1)

        print(f"Loaded {self.dfA.shape[1]} hysteresis loops ({self.dfA.shape[0]} points each)")
        return self

    # -------------------------------------------------------------------------
    # Preprocessing & PCA
    # -------------------------------------------------------------------------

    def normalize(self):
        """Transpose and StandardScaler-normalize the polarization data."""
        data = self.dfA.T
        self.x_scaled = StandardScaler().fit_transform(data.loc[:, self.dfA.index].values)
        return self

    def run_pca(self, n_components=2):
        """Reduce dimensionality with PCA."""
        if self.x_scaled is None:
            self.normalize()
        self.pca = PCA(n_components=n_components)
        components = self.pca.fit_transform(self.x_scaled)
        self.principal_df = pd.DataFrame(
            data=components,
            columns=[f'principal component {i+1}' for i in range(n_components)]
        )
        print("Explained variance ratio:", np.round(self.pca.explained_variance_ratio_, 3))
        return self

    # -------------------------------------------------------------------------
    # K-Means Clustering
    # -------------------------------------------------------------------------

    def run_kmeans(self):
        """Determine optimal k via elbow method and fit K-Means on PCA components."""
        components = self.principal_df.values
        fig_elbow, ax_elbow = plt.subplots(figsize=(8, 5))
        visualizer = KElbowVisualizer(KMeans(), k=(1, self.dfA.shape[1]), timings=False, ax=ax_elbow)
        visualizer.fit(components)
        self.n_clusters = visualizer.elbow_value_
        plt.close(fig_elbow)

        km = KMeans(n_clusters=self.n_clusters)
        km.fit(components)
        self.df_kmeans = pd.concat([
            self.principal_df,
            pd.DataFrame(km.labels_, columns=['ClusterID']),
            pd.DataFrame(self.dfA.T.index, columns=['FName'])
        ], axis=1)
        print(f"Optimal k = {self.n_clusters}")
        return self

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------

    def plot_raw(self):
        """Plot all raw hysteresis loops."""
        plt.figure(figsize=(8, 5))
        for col in self.dfB.columns:
            plt.plot(self.dfB[col], self.dfA[col], alpha=0.5)
        plt.xlabel('Voltage (V)')
        plt.ylabel(u'Polarization (µC/cm²)')
        plt.title("Raw Hysteresis Loops")
        plt.grid(True)
        plt.show()

    def plot_normalized(self):
        """Plot StandardScaler-normalized polarization curves."""
        plt.figure(figsize=(8, 5))
        plt.plot(self.x_scaled.T)
        plt.xlabel('Voltage Points')
        plt.ylabel('Norm. Polarization (a.u.)')
        plt.title("Normalized Hysteresis Loops")
        plt.show()

    def plot_pca(self):
        """Scatter plot of data in principal component space."""
        plt.figure(figsize=(6, 5))
        plt.scatter(self.principal_df['principal component 1'],
                    self.principal_df['principal component 2'], c='black')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title("PCA Projection")
        plt.grid(True)
        plt.show()

    def plot_explained_variance(self):
        """Bar chart of explained variance per principal component."""
        labels = [f'PC{i+1}' for i in range(len(self.pca.explained_variance_ratio_))]
        plt.figure(figsize=(6, 4))
        plt.bar(labels, self.pca.explained_variance_ratio_ * 100, width=0.5)
        plt.ylabel('Explained Variance (%)')
        plt.title("PCA Explained Variance")
        plt.show()

    def plot_kmeans_pca(self):
        """Scatter plot of K-Means cluster assignments in PCA space."""
        plt.figure(figsize=(6, 5))
        sns.scatterplot(data=self.df_kmeans, x='principal component 1',
                        y='principal component 2', hue='ClusterID', palette='tab10')
        plt.title("K-Means Clusters in PCA Space")
        plt.grid(True)
        plt.show()

    def _build_cluster_figure(self):
        """Build and return the hysteresis loops grouped by cluster figure."""
        fig, axes = plt.subplots(1, self.n_clusters, figsize=(6 * self.n_clusters, 6))
        if self.n_clusters == 1:
            axes = [axes]

        for cluster_id in range(self.n_clusters):
            ax = axes[cluster_id]
            idx = self.df_kmeans.query(f'ClusterID == {cluster_id}').index
            for i, sample_idx in enumerate(idx):
                ax.scatter(self.dfB.iloc[:, sample_idx], self.dfA.iloc[:, sample_idx], alpha=0.5)
                ax.set_xlim(-4, 4)
                ax.set_ylim(-80, 80)
                ax.set_xlabel('Voltage (V)')
                ax.set_ylabel(u'Polarization (µC/cm²)')
                ax.set_title(f'Cluster {cluster_id}')
                ax.text(-3.5, 60, f'n = {len(idx)}', fontsize=12, fontweight='bold')
                ax.text(-3.5, 50 - (5 * i), self.df_kmeans.FName[sample_idx], fontsize=8)

        plt.tight_layout()
        return fig

    def plot_clusters(self):
        """Display hysteresis loops grouped by K-Means cluster."""
        fig = self._build_cluster_figure()
        plt.show()

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    def save_output(self, filename="cluster_loops.png", dpi=150):
        """Save the clustered hysteresis loop plot to a PNG.

        Args:
            filename: output file name (default: 'cluster_loops.png').
            dpi: image resolution (default: 150).
        """
        out_path = resolve_path(filename, __file__)
        save_figure(self._build_cluster_figure(), out_path, dpi)

    # -------------------------------------------------------------------------
    # Full Pipeline
    # -------------------------------------------------------------------------

    def run(self):
        """Execute the full analysis pipeline and display all plots."""
        self.load_data()
        self.plot_raw()
        self.normalize()
        self.plot_normalized()
        self.run_pca()
        self.plot_pca()
        self.plot_explained_variance()
        self.run_kmeans()
        self.plot_kmeans_pca()
        self.plot_clusters()
        self.save_output("cluster_loops.png")
        return self
