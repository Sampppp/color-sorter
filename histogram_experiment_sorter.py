import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
from framework import BaseFeatureExtractor, BaseClusterer, SortingFramework, JPGLoader, RAWLoader, ingest_images, run_sorting_pipeline

# --- Distance Metric Implementations ---

def chi_square_distance(a, b):
    """Computes Chi-Square distance between two histograms."""
    # a, b are (N, D) arrays
    # dist = sum((a - b)^2 / (a + b))
    # Add epsilon to avoid division by zero
    eps = 1e-10
    num = (a - b)**2
    den = a + b + eps
    return np.sum(num / den, axis=1)

def js_divergence(a, b):
    """Computes Jensen-Shannon Divergence between two histograms."""
    # a, b are (N, D) arrays
    eps = 1e-10
    m = 0.5 * (a + b)
    
    def kl_div(p, q):
        # p: (1, D), q: (n, D)
        # sum(p * log(p/q))
        # Use np.where to handle p=0 and avoid log(0)
        return np.sum(np.where(p > 0, p * np.log((p + eps) / (q + eps)), 0), axis=1)

    return 0.5 * kl_div(a, m) + 0.5 * kl_div(b, m)

# --- Feature Extractor ---

class HistogramExperimentExtractor(BaseFeatureExtractor):
    """
    Extractor for comparing different histogram configurations.
    Configs:
    - 'baseline': Global 1D RGB Histogram (8 bins/channel = 24D)
    - 'perceptual': Global 1D CIELAB Histogram (16 bins/channel = 48D)
    - 'spatial': 2x2 Grid CIELAB Histogram (16 bins/channel/grid = 192D)
    """
    def __init__(self, config: str = "baseline", bins: int = 8):
        self.config = config
        self.bins = bins

    def _compute_hist(self, image: np.ndarray, channels: int = 3) -> np.ndarray:
        """Computes a concatenated histogram for the given image."""
        hists = []
        for i in range(channels):
            # Calculate histogram for each channel
            # ranges must be a sequence, e.g., [0, 256] for 8-bit images
            hist = cv2.calcHist([image], [i], None, [self.bins], [0, 256])
            hists.append(hist.flatten())
        
        feat = np.concatenate(hists)
        # L1 Normalization: sum of all bins = 1.0
        norm = np.sum(feat)
        return feat / (norm + 1e-7)

    def extract(self, image: np.ndarray) -> np.ndarray:
        if self.config == "baseline":
            # RGB, 8 bins/channel
            return self._compute_hist(image)
        
        elif self.config == "perceptual":
            # CIELAB, 16 bins/channel
            img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
            # Use 16 bins as requested for perceptual
            self.bins = 16 
            feat = self._compute_hist(img_lab)
            self.bins = 8 # reset to default if needed, though usually fixed per instance
            return feat
            
        elif self.config == "spatial":
            # CIELAB, 2x2 Grid, 16 bins/channel/grid
            img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
            h, w, _ = img_lab.shape
            mid_h, mid_w = h // 2, w // 2
            
            # Split into 4 quadrants
            quads = [
                img_lab[0:mid_h, 0:mid_w],      # Top-Left
                img_lab[0:mid_h, mid_w:w],      # Top-Right
                img_lab[mid_h:h, 0:mid_w],      # Bottom-Left
                img_lab[mid_h:h, mid_w:w],      # Bottom-Right
            ]
            
            self.bins = 16
            grid_hists = [self._compute_hist(q) for q in quads]
            self.bins = 8
            
            return np.concatenate(grid_hists)
        
        else:
            raise ValueError(f"Unknown config: {self.config}")

    def get_centroid_name(self, centroid: np.ndarray) -> str:
        return f"hist_{self.config}"

# --- Clusterer ---

class DistanceMatrixClusterer(BaseClusterer):
    """
    Clusterer that uses a precomputed distance matrix.
    Supports custom metrics like Chi-Square and JS Divergence.
    """
    def __init__(self, metric: str = "cosine", threshold: float = 0.5, n_clusters: Optional[int] = None):
        self.metric = metric
        self.threshold = threshold
        self.n_clusters = n_clusters

    def _compute_dist_matrix(self, features: np.ndarray) -> np.ndarray:
        n = features.shape[0]
        dist_matrix = np.zeros((n, n))
        
        if self.metric == "cosine":
            # Use scipy's optimized cdist
            dist_matrix = cdist(features, features, metric='cosine')
        
        elif self.metric == "chi2":
            # Compute pairwise Chi-Square
            for i in range(n):
                dist_matrix[i, :] = chi_square_distance(features[i:i+1], features)
        
        elif self.metric == "jsd":
            # Compute pairwise JS Divergence
            for i in range(n):
                # JSD is often used as sqrt(JSD) to make it a metric
                dist_matrix[i, :] = np.sqrt(js_divergence(features[i:i+1], features))
        
        else:
            # Default to euclidean if unknown
            dist_matrix = cdist(features, features, metric='euclidean')
            
        return dist_matrix

    def cluster(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 1. Compute distance matrix
        dist_matrix = self._compute_dist_matrix(features)
        
        # 2. Agglomerative Clustering with precomputed metric
        # If n_clusters is provided, use it; otherwise use distance_threshold
        n_clusters = self.n_clusters
        threshold = self.threshold if n_clusters is None else None
        
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            distance_threshold=threshold,
            metric='precomputed',
            linkage='average'
        )
        labels = model.fit_predict(dist_matrix)
        
        # 3. Calculate centroids (mean of features for each cluster)
        unique_labels = np.unique(labels)
        centroids = []
        for label in unique_labels:
            cluster_points = features[labels == label]
            centroids.append(np.mean(cluster_points, axis=0))
            
        return labels, np.array(centroids)

# --- CLI Entry Point ---

import click

@click.command()
@click.option('--input', '-i', 'input_dir', type=click.Path(exists=True), help='Input directory.')
@click.option('--output-dir', help='Output directory.')
@click.option('--config', type=click.Choice(['baseline', 'perceptual', 'spatial']), default='baseline')
@click.option('--metric', type=click.Choice(['cosine', 'chi2', 'jsd']), default='cosine')
@click.option('--threshold', default=0.5, help='Clustering distance threshold.')
@click.option('--raw', is_flag=True, default=False)
@click.option('--verbose', '-v', is_flag=True, default=False)
def main(input_dir, output_dir, config, metric, threshold, raw, verbose):
    """Run the histogram experiment sorter."""
    extractor = HistogramExperimentExtractor(config=config)
    clusterer = DistanceMatrixClusterer(metric=metric, threshold=threshold)
    
    run_sorting_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        move=False,
        raw=raw,
        verbose=verbose,
        extractor=extractor,
        clusterer=clusterer,
        title=f"Experiment: {config} ({metric})"
    )

if __name__ == "__main__":
    main()