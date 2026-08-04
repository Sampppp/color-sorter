import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import click
from sklearn.preprocessing import StandardScaler
from framework import (
    BaseFeatureExtractor, 
    BaseClusterer, 
    KMeansClusterer, 
    SortingFramework, 
    common_options, 
    run_sorting_pipeline, 
    setup_logging
)

# --- Custom Clusterer with Scaling ---

class ScalingKMeansClusterer(BaseClusterer):
    """Clusters images using K-Means after standardizing features."""
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()

    def cluster(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Standardize features: (x - mean) / std
        scaled_features = self.scaler.fit_transform(features)
        
        # Use the KMeansClusterer from framework
        clusterer = KMeansClusterer(n_clusters=self.n_clusters)
        labels, scaled_centroids = clusterer.cluster(scaled_features)
        
        # Inverse transform centroids back to original space for meaningful naming
        centroids = self.scaler.inverse_transform(scaled_centroids)
        
        return labels, centroids

# --- Custom Feature Extractor ---

class CVFeatureExtractor(BaseFeatureExtractor):
    """
    Extracts a composite feature vector using Traditional CV:
    1. ORB Spatial Grid (16 elements)
    2. HSV Color Histogram (32 elements: 16 Hue, 16 Sat)
    3. Laplacian Variance (1 element)
    4. Canny Edge Density (1 element)
    """
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters

    def extract(self, image: np.ndarray) -> np.ndarray:
        # image is RGB
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # 1. ORB Spatial Grid Distribution
        orb = cv2.ORB_create()
        kp = orb.detect(gray, None)
        
        grid_features = np.zeros(16)
        if len(kp) > 0:
            cell_h, cell_w = h // 4, w // 4
            for k in kp:
                x, y = k.pt
                col = int(x // cell_w)
                row = int(y // cell_h)
                if 0 <= row < 4 and 0 <= col < 4:
                    grid_features[row * 4 + col] += 1
            # Normalize by total keypoints
            grid_features /= len(kp)

        # 2. HSV Color Histogram (16 bins Hue, 16 bins Saturation)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # Hue is 0-179, Saturation is 0-255
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist_h, hist_h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_s, hist_s, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        hsv_features = np.concatenate([hist_h.flatten(), hist_s.flatten()])

        # 3. Laplacian Variance (Sharpness)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # 4. Canny Edge Density
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (h * w)

        # Concatenate all into a single 1D vector
        # Vector structure: [grid(16), hsv(32), lap_var(1), edge_density(1)]
        return np.concatenate([
            grid_features, 
            hsv_features, 
            [lap_var], 
            [edge_density]
        ])

    def get_centroid_name(self, centroid: np.ndarray) -> str:
        """
        Heuristic to name the cluster based on the composite vector.
        Vector: [grid(16), hsv(32), lap_var(1), edge_density(1)]
        """
        lap_var = centroid[-2]
        edge_density = centroid[-1]
        
        # Simple heuristics for naming
        if lap_var < 100: # Threshold depends on image resolution/content
            return "blurry_theme"
        if edge_density > 0.1:
            return "complex_geometry"
        
        return "visual_theme"

# --- CLI Entry Point ---
@click.command()
@common_options()
@click.option('--clusters', '-c', default=5, type=int, help='Number of clusters for KMeans.')
def main(input_dir, output_dir, move, raw, verbose, clusters):
    """
    Sorts images based on a composite CV feature vector:
    ORB Spatial Grid + HSV Histogram + Laplacian Variance + Canny Edge Density.
    """
    extractor = CVFeatureExtractor(n_clusters=clusters)
    clusterer = ScalingKMeansClusterer(n_clusters=clusters)
    
    run_sorting_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        move=move,
        raw=raw,
        verbose=verbose,
        extractor=extractor,
        clusterer=clusterer,
        title="CV Feature Sorting Summary"
    )

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
